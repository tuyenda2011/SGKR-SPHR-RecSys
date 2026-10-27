"""Stream raw reviews to validate current train edges; never rewrite dataset files."""
import argparse
import gzip
import hashlib
import json
import math
import pickle
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.bundle import resolve_bundle, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    bundle = resolve_bundle(args.bundle)
    train = pd.read_parquet(bundle.train_dir / 'train.parquet')
    mapping_entry = next(entry for entry in bundle.manifest['artifacts'] if Path(entry['path']).name == 'mappings.pkl')
    with (bundle.train_dir / 'mappings.pkl').open('rb') as handle:
        if hashlib.sha256(handle.read()).hexdigest() != mapping_entry['sha256']:
            raise ValueError('Mappings differ from the pinned local bundle')
        handle.seek(0)
        # Only the project's pinned, hash-verified local mapping artifact is accepted.
        mappings = pickle.load(handle)  # nosec B301
    inverse_users = {value: key for key, value in mappings['user2id'].items()}
    inverse_items = {value: key for key, value in mappings['item2id'].items()}
    mapping_checks = {}
    for name in ('user2id', 'item2id'):
        mapping = mappings[name]
        mapping_checks[name] = {'count': len(mapping), 'contiguous_unique_indices': sorted(mapping.values()) == list(range(len(mapping))),
                                'blank_or_untrimmed_ids': sum(not isinstance(key, str) or not key.strip() or key != key.strip() for key in mapping)}
    lookup = {(inverse_users[int(user)], inverse_items[int(item)]): index
              for index, (user, item) in enumerate(zip(train.u_idx, train.i_idx))}
    expected_timestamps = train.timestamp.to_numpy()
    counts = np.zeros(len(train), dtype=np.int32)
    errors, ratings, text_hashes = Counter(), Counter(), {}
    matched = lines = long_text_rows = 0
    source = next(entry for entry in bundle.manifest['source_files'] if Path(entry['path']).name == 'reviews_Electronics_5.json.gz')
    raw_path = ROOT / source['path']
    assert sha256_file(raw_path) == source['sha256'], 'Raw snapshot differs from bundle source'
    with gzip.open(raw_path, 'rt', encoding='utf-8') as stream:
        for lines, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
            except (ValueError, TypeError):
                errors['raw_unparseable_lines'] += 1
                continue
            if not isinstance(record, dict):
                errors['raw_nonobject_lines'] += 1
                continue
            key = (record.get('reviewerID'), record.get('asin'))
            index = lookup.get(key)
            if index is None:
                continue  # Do not inspect withheld ratings, review text or event times.
            counts[index] += 1
            matched += 1
            rating = record.get('overall')
            valid_rating = isinstance(rating, (int, float)) and not isinstance(rating, bool) and math.isfinite(rating) and rating in (1, 2, 3, 4, 5)
            if not valid_rating:
                errors['matched_invalid_rating'] += 1
            else:
                ratings[str(int(rating))] += 1
                if rating < 4:
                    errors['raw_low_rating_on_processed_positive_pair'] += 1
            timestamp = record.get('unixReviewTime')
            valid_time = isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool) and math.isfinite(timestamp) and timestamp > 0
            if not valid_time:
                errors['matched_invalid_timestamp'] += 1
            else:
                if timestamp != expected_timestamps[index]:
                    errors['matched_timestamp_differs_from_processed'] += 1
                try:
                    date = datetime.fromtimestamp(timestamp, timezone.utc)
                    if date > datetime.now(timezone.utc):
                        errors['matched_future_timestamp'] += 1
                    written = datetime.strptime(record.get('reviewTime', ''), '%m %d, %Y').replace(tzinfo=timezone.utc).date()
                    if written != date.date():
                        errors['matched_date_fields_disagree'] += 1
                except (ValueError, TypeError, OverflowError, OSError):
                    errors['matched_date_unparseable'] += 1
            text = record.get('reviewText')
            if not isinstance(text, str) or not text.strip():
                errors['matched_missing_review_text'] += 1
                continue
            normalized = ' '.join(text.casefold().split())
            if len(normalized) >= 200:
                long_text_rows += 1
                digest = hashlib.blake2b(normalized.encode('utf-8'), digest_size=16).digest()
                if digest not in text_hashes:
                    text_hashes[digest] = [1, key[0], key[1], False, False]
                else:
                    group = text_hashes[digest]
                    group[0] += 1
                    group[3] |= group[1] != key[0]
                    group[4] |= group[2] != key[1]
            if lines % 300000 == 0:
                print(json.dumps({'raw_lines_seen': lines, 'matched_train_records': matched}), flush=True)
    groups = list(text_hashes.values())
    duplicates = [group for group in groups if group[0] > 1]
    metadata = mappings['item_metadata']
    asin_mismatches = sum(metadata.get(index, {}).get('original_id') != external for external, index in mappings['item2id'].items())
    report = {'scope': 'Read-only current train/raw audit. Raw JSON parse/IDs scanned globally; ratings, times and text inspected only for processed train pairs. No validation/test labels scored, no data changes.',
              'bundle_id': bundle.build_id, 'raw_source_sha256_matches': True, 'raw_lines': lines,
              'processed_train_edges': len(train), 'matched_raw_train_records': matched,
              'train_edges_missing_raw_match': int((counts == 0).sum()), 'train_pairs_with_multiple_raw_records': int((counts > 1).sum()),
              'processed_train_timestamp_min_utc': str(pd.to_datetime(train.timestamp.min(), unit='s', utc=True)),
              'processed_train_timestamp_max_utc': str(pd.to_datetime(train.timestamp.max(), unit='s', utc=True)),
              'train_rating_distribution': dict(ratings), 'error_and_flag_counts': dict(errors),
              'mapping_checks': mapping_checks, 'item_mapping_metadata_asin_mismatches': asin_mismatches,
              'invalid_asin_format_count': sum(re.fullmatch(r'[A-Z0-9]{10}', key) is None for key in mappings['item2id']),
              'long_text_reuse': {'minimum_normalized_characters': 200, 'eligible_train_rows': long_text_rows,
                                  'repeated_text_groups': len(duplicates), 'train_rows_in_repeated_groups': sum(group[0] for group in duplicates),
                                  'groups_spanning_users': sum(group[3] for group in duplicates),
                                  'groups_spanning_items': sum(group[4] for group in duplicates),
                                  'largest_group': max((group[0] for group in groups), default=0),
                                  'interpretation': 'Exact normalized text reuse flag only; not a spam/fraud label. Active graph-only models do not consume review text.'}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
