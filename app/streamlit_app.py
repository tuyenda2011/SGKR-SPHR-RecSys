"""Demo: Hệ thống khuyến nghị sản phẩm điện tử (SGKR, SPHR and three graph baselines).

  streamlit run app/streamlit_app.py

Three tabs: an existing user (top-10 of the five models side by side, SGKR's explanation), a new
user built from a product search (SGKR and SPHR answer at once; the baselines need retraining), and
the report figures and tables. Validation data only; the test split is never opened.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from app.recommender import (COLORS, DENSITIES, LABELS, MODELS, PREVIEW, PROPOSED, load_data, load_ranker,
                             recommend, sample_users)

FIGURES = ROOT / "reports/figures"
TABLES = ROOT / "reports/tables"
METRICS = ("NDCG@20", "NDCG@10", "Recall@20", "Recall@10")

CSS = """
<style>
html, body, .stApp, .stApp * , button, input, select, textarea {
  font-family: 'Times New Roman', Times, 'Liberation Serif', serif !important; }
.stApp { background: #fbfbf9; }
.stButton > button, .stTextInput input, div[data-baseweb="select"] > div { border-radius: 2px !important; }
.box { background: #fff; border: 1px solid #e3e5ea; border-radius: 2px; padding: 14px 16px; margin-bottom: 14px; }
.box.answer { border-style: dashed; border-color: #9aa1ad; }
.box h4, .card h4 { margin: 0 0 8px; font-size: 16px; font-weight: 700; }
.hist { list-style: none; margin: 0; padding: 0; }
.hist li { display: flex; gap: 10px; padding: 7px 8px; margin-bottom: 6px; background: #f7f8fa; border-radius: 2px; }
.code { flex: 0 0 24px; height: 24px; background: #e3e5ea; border-radius: 2px; display: flex;
  align-items: center; justify-content: center; font-weight: 700; font-size: 13.5px; color: #3d424c; }
.sub { display: block; color: #5a5f6b; font-size: 13.5px; }
.card { background: #fff; border: 1px solid #e3e5ea; border-radius: 2px; overflow: hidden; margin-bottom: 12px; }
.card.locked { background: #f0f1f4; border-style: dashed; border-color: #b6bbc4; }
.strip { height: 4px; }
.head { display: flex; justify-content: space-between; align-items: baseline; padding: 10px 12px 0; }
.head b { font-size: 16px; }
.ms { font-size: 13.5px; color: #5a5f6b; }
.kind { padding: 0 12px 8px; font-size: 13.5px; color: #5a5f6b; border-bottom: 1px solid #eef0f3; }
.recs { list-style: none; margin: 0; padding: 4px 0; }
.recs li { display: flex; gap: 8px; padding: 6px 12px; align-items: flex-start; }
.recs li.hit { background: #eef0f3; }
.rank { flex: 0 0 18px; color: #5a5f6b; font-size: 13.5px; }
.title { flex: 1; min-width: 0; font-size: 14.5px; }
.tick { flex: 0 0 20px; height: 20px; border-radius: 50%; background: #16181d; color: #fff;
  display: inline-flex; align-items: center; justify-content: center; font-size: 12px; }
.why { margin: 4px 12px 12px; padding: 10px 12px; background: #eef4fc; border-radius: 2px; font-size: 13.5px; }
.why div { display: flex; gap: 8px; align-items: center; margin-top: 4px; }
.empty { padding: 36px 12px; text-align: center; color: #5a5f6b; font-size: 14.5px; }
.legend { color: #3d424c; font-size: 14.5px; }
</style>
"""


def short(text, length=70):
    text = str(text)
    return text if len(text) <= length else text[:length - 1].rstrip() + "…"


def item_label(data, item):
    row = data.metadata.loc[int(item)]
    title = row.title if row.has_title else "Không rõ tên"
    brand = row.brand if row.has_brand else "Không rõ hãng"
    return str(title), str(brand), str(row.leaf)


@st.cache_resource(show_spinner="Đang nạp dữ liệu train và validation…")
def get_data():
    data = load_data()
    samples, eligible = sample_users(data)
    return data, samples, eligible


@st.cache_resource(show_spinner=False)
def get_ranker(model, density):
    return load_ranker(model, density, get_data()[0])


def rankers(density):
    with st.spinner(f"Đang nạp 5 mô hình ở mật độ {density}…"):
        return {model: get_ranker(model, density) for model in MODELS}


def card(model, items, data, ms=None, target=None, why=None, kind=None):
    rows = []
    for rank, item in enumerate(items, 1):
        title, brand, _ = item_label(data, item)
        hit = target is not None and int(item) == target
        tick = '<span class="tick" aria-label="Trúng">✓</span>' if hit else ""
        rows.append(f'<li class="{"hit" if hit else ""}"><span class="rank">{rank}</span><span class="title" '
                    f'title="{html.escape(title)}">{html.escape(short(title))}<span class="sub">'
                    f'{html.escape(brand)}</span></span>{tick}</li>')
    latency = f'<span class="ms">{ms:.0f} ms</span>' if ms is not None else ""
    kind = kind or ("Đề xuất" if model in PROPOSED else "Baseline")
    return (f'<div class="card"><div class="strip" style="background:{COLORS[model]}"></div>'
            f'<div class="head"><b>{LABELS[model]}</b>{latency}</div><div class="kind">{kind}</div>'
            f'<ol class="recs">{"".join(rows)}</ol>{why or ""}</div>')


def empty_card(model, message, detail="", locked=False):
    return (f'<div class="card{" locked" if locked else ""}"><div class="strip" style="background:{COLORS[model]}"></div>'
            f'<div class="head"><b>{LABELS[model]}</b></div><div class="kind">'
            f'{"Đề xuất" if model in PROPOSED else "Baseline"}</div><div class="empty"><b>{message}</b>'
            f'<span class="sub">{detail}</span></div></div>')


def why_box(ranker, history, top, codes, data):
    if not len(top):
        return ""
    parts = ranker.explain(history, top[0])
    total = sum(value for _, value in parts) or 1
    lines = "".join(f'<div><span class="code">{codes.get(item, "·")}</span><span style="flex:1">'
                    f'{html.escape(short(item_label(data, item)[0], 45))}</span><span>{value / total:.0%}</span></div>'
                    for item, value in parts)
    return (f'<div class="why"><b>Vì sao gợi ý #1?</b><br>Sản phẩm trong lịch sử đóng góp nhiều nhất:'
            f'{lines or "<div>Chỉ do độ phổ biến (bias) của sản phẩm.</div>"}</div>')


def history_html(data, items, codes):
    rows = []
    for item in items:
        title, brand, leaf = item_label(data, item)
        rows.append(f'<li><span class="code">{codes[int(item)]}</span><span>{html.escape(short(title, 60))}'
                    f'<span class="sub">{html.escape(brand)} · {html.escape(leaf)}</span></span></li>')
    return f'<ol class="hist">{"".join(rows)}</ol>'


def tab_existing(data, samples, eligible):
    left, right = st.columns([1, 3.6], gap="large")
    with left:
        st.markdown("#### Chọn user")
        options = [user for _, user in samples]
        groups = dict((user, group) for group, user in samples)

        def chosen():
            st.session_state.user = st.session_state.user_pick
        st.selectbox("User mẫu", options, key="user_pick", on_change=chosen,
                     format_func=lambda u: f"User {u} · {len(data.history[u])} sản phẩm (nhóm {groups[u]})")
        if st.button("Chọn ngẫu nhiên", use_container_width=True):
            st.session_state.user = int(np.random.default_rng().choice(eligible))
        user = st.session_state.get("user", options[0])
        st.markdown("#### Mật độ dữ liệu train")
        density = st.radio("Mật độ dữ liệu train", DENSITIES, horizontal=True, key="density",
                           format_func=str, label_visibility="collapsed")
        history = data.history[user]
        codes = {int(item): chr(65 + i) if i < 26 else str(i + 1) for i, item in enumerate(history)}
        st.markdown(f'<div class="box"><h4>Lịch sử mua · {len(history)} sản phẩm</h4>'
                    f'{history_html(data, history, codes)}</div>', unsafe_allow_html=True)
        target = data.target[user]
        title, brand, leaf = item_label(data, target)
        st.markdown(f'<div class="box answer"><h4>Đáp án · sản phẩm mua tiếp theo</h4>{html.escape(title)}'
                    f'<span class="sub">{html.escape(brand)} · {html.escape(leaf)}</span></div>',
                    unsafe_allow_html=True)
    with right:
        st.markdown(f'#### Top-10 gợi ý của 5 mô hình <span class="legend">&nbsp;&nbsp;<span class="tick">✓</span>'
                    f' trúng sản phẩm mua tiếp theo</span>', unsafe_allow_html=True)
        loaded = rankers(density)
        for column, model in zip(st.columns(5, gap="small"), MODELS):
            ranker = loaded[model]
            with column:
                if ranker is None:
                    st.markdown(empty_card(model, "Chưa train", f"ở mật độ {density}"), unsafe_allow_html=True)
                    continue
                top, ms = recommend(ranker, data, user=user)
                why = why_box(ranker, ranker.train_items(user), top, codes, data) if model == "sgkr" else ""
                st.markdown(card(model, top, data, ms, target, why), unsafe_allow_html=True)


def tab_new_user(data):
    density = st.session_state.get("density", 1.0)
    loaded = rankers(density)
    proposed = [loaded[m] for m in PROPOSED if loaded[m] is not None]
    picked = st.session_state.setdefault("picked", [])
    left, right = st.columns([1.25, 3.4], gap="large")
    with left:
        st.markdown("#### Tạo lịch sử cho user mới")
        query = st.text_input("Tìm sản phẩm theo tên", placeholder="ví dụ: headphones, memory card")
        searchable = data.metadata.labeled.to_numpy().copy()
        if proposed:
            searchable &= proposed[0].candidates  # only products the trained graph knows
        if query.strip():
            titles = data.metadata.title.where(searchable, "")
            matches = np.flatnonzero(titles.str.contains(query.strip(), case=False, regex=False).to_numpy())[:8]
            if not len(matches):
                st.caption("Không tìm thấy sản phẩm.")
            for item in matches:
                title, brand, leaf = item_label(data, item)
                text, action = st.columns([4, 1.3])
                text.markdown(f"{html.escape(short(title, 60))}<span class='sub'>{html.escape(brand)} · "
                              f"{html.escape(leaf)}</span>", unsafe_allow_html=True)
                if int(item) in picked:
                    action.caption("Đã thêm")
                elif action.button("+ Thêm", key=f"add{item}"):
                    picked.append(int(item))
                    st.rerun()
        st.markdown(f"#### Lịch sử đã chọn · {len(picked)} sản phẩm")
        for item in list(picked):
            text, action = st.columns([4, 1.3])
            text.markdown(html.escape(short(item_label(data, item)[0], 60)))
            if action.button("✕", key=f"del{item}", help="Bỏ sản phẩm này"):
                picked.remove(item)
                st.rerun()
        run = st.button("Gợi ý ngay", type="primary", use_container_width=True, disabled=not picked)
        if run:
            st.session_state.asked = list(picked)
    with right:
        st.markdown("#### User này chưa từng có trong dữ liệu train")
        st.markdown("SGKR và SPHR gợi ý ngay từ lịch sử mua; ba baseline phải train lại.")
        asked = st.session_state.get("asked")
        for column, model in zip(st.columns(5, gap="small"), MODELS):
            with column:
                if model not in PROPOSED:
                    st.markdown(empty_card(model, "Cần train lại", "Chưa có vector embedding cho user này",
                                           locked=True), unsafe_allow_html=True)
                elif loaded[model] is None:
                    st.markdown(empty_card(model, "Chưa train", f"ở mật độ {density}"), unsafe_allow_html=True)
                elif not asked:
                    st.markdown(empty_card(model, "Chọn sản phẩm rồi bấm “Gợi ý ngay”"), unsafe_allow_html=True)
                else:
                    top, ms = recommend(loaded[model], data, items=asked)
                    st.markdown(card(model, top, data, ms), unsafe_allow_html=True)


def tab_results():
    pick, dens, _ = st.columns([1, 1, 3])
    metric = pick.selectbox("Chỉ số", METRICS)
    density = dens.selectbox("Mật độ", DENSITIES, format_func=str)
    slug, pct = metric.lower().replace("@", ""), round(density * 100)
    figures = [(f"H1_density_{slug}.png", "Theo mật độ dữ liệu train"),
               (f"H2_{slug}_d{pct}.png", "Chỉ số kèm khoảng tin cậy 95%"),
               (f"H3_learning_curves_{slug}_d{pct}.png", "Đường học theo epoch"),
               (f"H5_user_groups_{slug}_d{pct}.png", "Theo nhóm độ dài lịch sử user")]
    figures = [(name, caption) for name, caption in figures if (FIGURES / name).exists()]
    if not figures:
        st.info("Chưa có hình cho lựa chọn này. Chạy `python scripts\\run_experiments.py` để train và vẽ hình.")
    for row in range(0, len(figures), 2):
        for column, (name, caption) in zip(st.columns(2, gap="large"), figures[row:row + 2]):
            column.image(str(FIGURES / name), caption=caption, use_container_width=True)
    table = TABLES / f"B2a_metrics_d{pct}.csv"
    st.markdown(f"#### Bảng chỉ số · mật độ {density}")
    if table.exists():
        st.dataframe(pd.read_csv(table, encoding="utf-8-sig"), hide_index=True, use_container_width=True)
    else:
        st.caption("Chưa có bảng cho mật độ này.")


def main():
    st.set_page_config(page_title="Hệ thống khuyến nghị sản phẩm điện tử", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("## Hệ thống khuyến nghị sản phẩm điện tử")
    if PREVIEW:
        st.warning(f"Chế độ xem thử: mô hình đọc từ `{PREVIEW}` (checkpoint train ngắn), "
                   "gợi ý chưa phản ánh kết quả cuối cùng.")
    data, samples, eligible = get_data()
    existing, new, results = st.tabs(["1 · User có sẵn", "2 · User mới", "3 · Kết quả"])
    with existing:
        tab_existing(data, samples, eligible)
    with new:
        tab_new_user(data)
    with results:
        tab_results()


if __name__ == "__main__":
    main()
