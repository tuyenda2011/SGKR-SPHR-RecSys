"""Import-attested source manifests and guarded atomic artifact writes."""
import ast
import hashlib
import sys
import tempfile
from pathlib import Path
from types import MappingProxyType


def source_identity(path):
    """Bind source bytes to the actual module code being executed, including pyc loads."""
    content = Path(path).read_bytes()
    executing = sys._getframe(1).f_code
    compiled = compile(content, executing.co_filename, "exec", dont_inherit=True, optimize=sys.flags.optimize)
    if compiled != executing:
        raise ValueError(f"Source changed while importing: {path}")
    return hashlib.sha256(content).hexdigest()


SOURCE_SHA256 = source_identity(__file__)


def capture_sources(root, paths, imported=None):
    root = Path(root).resolve()
    result = {Path(path).resolve().relative_to(root).as_posix(): hashlib.sha256(Path(path).read_bytes()).hexdigest()
              for path in dict.fromkeys(paths)}
    loaded = dict(imported or {})
    for module in list(sys.modules.values()):
        values = getattr(module, "__dict__", {})
        filename, digest = values.get("__file__"), values.get("SOURCE_SHA256")
        if filename and isinstance(digest, str):
            path = Path(filename).resolve()
            if path.is_relative_to(root):
                loaded[path.relative_to(root).as_posix()] = digest
    for name, digest in loaded.items():
        if name not in result:
            continue
        if result.get(name) != digest:
            raise ValueError(f"Imported source differs from disk: {name}")
    return MappingProxyType(result)


class SourceGuard:
    def __init__(self, root, hashes, *, config=None, archive=None):
        self.root = Path(root).resolve()
        self.hashes = MappingProxyType(dict(hashes))
        self.config = None if config is None else Path(config).resolve()
        self.archive = None if archive is None else Path(archive).resolve()

    def _path(self, name, archive=False):
        root = self.archive if archive else self.root
        if not archive and name in {"input_config.yaml", "effective_input_config.yaml"} and self.config is not None:
            return self.config
        path = (root/name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Invalid source identity path: {name}")
        return path

    def check(self):
        for name, expected in self.hashes.items():
            for archived in ((False, True) if self.archive is not None else (False,)):
                try:
                    actual = hashlib.sha256(self._path(name, archived).read_bytes()).hexdigest()
                except OSError as error:
                    raise ValueError(f"Pinned source unavailable: {name}") from error
                if actual != expected:
                    location = "archive" if archived else "disk"
                    raise ValueError(f"Pinned source changed on {location}: {name}")

    def archive_to(self, directory):
        directory = Path(directory).resolve()
        self.check()
        for name, expected in self.hashes.items():
            content = self._path(name).read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Pinned source changed while archiving: {name}")
            destination = (directory/name).resolve()
            if not destination.is_relative_to(directory):
                raise ValueError(f"Invalid source archive path: {name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        self.check()

    def atomic_write(self, path, writer):
        """Check before and after serialization, and leave any existing artifact intact on drift."""
        path = Path(path)
        self.check()
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            # A few numeric serializers require exclusive file creation.
            temporary.unlink()
            writer(temporary)
            self.check()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def verify_imported(hashes, imported):
    for name, expected in imported.items():
        if hashes.get(name) != expected:
            raise ValueError(f"Pinned source differs from imported code: {name}")


def attestation_equivalent(archived, current):
    """Narrow legacy recovery: only import-attestation statements may differ."""
    def normalized(path):
        tree = ast.parse(Path(path).read_bytes())
        kept = []
        for node in tree.body:
            if (isinstance(node, ast.ImportFrom) and node.module == "src.utils.source_identity"
                    and len(node.names) == 1 and node.names[0].name == "source_identity" and node.names[0].asname is None):
                continue
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "SOURCE_SHA256"
                    and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "source_identity" and len(node.value.args) == 1
                    and isinstance(node.value.args[0], ast.Name) and node.value.args[0].id == "__file__"
                    and not node.value.keywords):
                continue
            kept.append(node)
        tree.body = kept
        return ast.dump(tree, include_attributes=False)
    return normalized(archived) == normalized(current)


def verify_legacy_sources(root, hashes, imported, *, archive=None):
    """Keep old lineage; permit only verified, attestation-only source changes."""
    root = Path(root).resolve()
    archived = None if archive is None else SourceGuard(archive, hashes)
    if archived is not None:
        archived.check()
    compatibility = {}
    for name, current in imported.items():
        original = hashes.get(name)
        if original == current:
            continue
        if (original is None or archived is None
                or not attestation_equivalent(archived._path(name), root/name)):
            raise ValueError(f"Legacy source interpretation changed: {name}; verified source archive required")
        compatibility[name] = {"archived_sha256": original, "runtime_sha256": current,
                               "reason": "import-attestation-only"}
    return compatibility
