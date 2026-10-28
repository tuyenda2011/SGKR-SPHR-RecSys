"""Demo: Hệ thống khuyến nghị sản phẩm điện tử (SGKR, SPHR, three graph baselines and two closed-form baselines).

  streamlit run app/streamlit_app.py

Laid out for a 1920x1080 desktop. Three tabs:
  Khách hàng có sẵn   the top-10 of all seven models side by side (one column each, grouped by family) next to the
                      customer's history and next purchase; hovering a product lights it up in every column.
  Khách hàng mới      the same grid for a history built from a product search (graph baselines cannot serve it).
  Benchmark học thuật  key findings, leaderboard, charts, paired tests against SGKR, protocol and report figures.
Validation data only; the test split is never opened.
"""
from __future__ import annotations

import html
import sys
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app.benchmark import METRICS, by_density, leaderboard, significance
from app.recommender import (COLORS, DENSITIES, FAMILY_COLORS, GROUPS, KINDS, LABELS, MODELS, PREVIEW, RESULTS,
                             checkpoint_ndcg, load_data, load_ranker, recommend, sample_users)

FIGURES = ROOT / "reports/figures"
CATEGORY_TINTS = ("#fdecea", "#e8f0fc", "#e6f5ee", "#fdf3e1", "#f0ecfa", "#e3f4f6", "#fbeaf2", "#eef1e4")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');
html, body, .stApp, .stApp *:not([data-testid="stIconMaterial"]):not(.material-symbols-rounded) {
  font-family: 'Be Vietnam Pro', 'Segoe UI', system-ui, sans-serif; }
.stApp { background: #f6f7fa; }
.block-container { padding: .5rem 1.6rem .6rem !important; max-width: 1900px !important; }
header[data-testid="stHeader"] { background: transparent; }
.topbar { display: flex; align-items: center; gap: 10px; padding: 8px 18px; margin-bottom: 0; border-radius: 12px;
  background: linear-gradient(110deg, #1d63c2 0%, #4b55c9 55%, #6a4bbd 100%); box-shadow: 0 4px 14px rgba(45,80,180,.18); }
.topbar .ttl { font-size: 20px; font-weight: 700; color: #fff; margin-right: auto; }
.topbar .pill { background: rgba(255,255,255,.18); color: #fff; border-radius: 999px; padding: 3px 12px; font-size: 13px; }
.panel { background: #fff; border: 1px solid #e4e7ee; border-radius: 10px; padding: 10px 12px; margin-bottom: 10px; }
.panel h4 { margin: 0 0 8px; font-size: 14px; font-weight: 600; color: #1c2030; }
.panel.answer { border: 1.5px dashed #13865d; background: #f4fbf7; font-size: 13.5px; }
.panel.answer h4 { color: #13865d; }
.hist { list-style: none; margin: 0; padding: 0; max-height: 300px; overflow-y: auto; }
.hist li { display: flex; gap: 8px; padding: 6px 8px; margin-bottom: 5px; background: #f5f6f9; border-radius: 7px;
  font-size: 13px; line-height: 1.3; }
.code { flex: 0 0 22px; height: 22px; background: #1d63c2; color: #fff; border-radius: 5px; display: flex;
  align-items: center; justify-content: center; font-weight: 600; font-size: 12px; }
.sub { display: block; color: #5f6577; font-size: 12px; margin-top: 1px; }
.cat { display: inline-block; border-radius: 999px; padding: 0 8px; font-size: 11px; color: #2b3040; margin-top: 3px; }
.why { padding: 9px 11px; background: #e8f0fc; border-left: 4px solid #1d63c2; border-radius: 10px; font-size: 12.5px;
  line-height: 1.35; }
.why b:first-child { color: #1d63c2; }
.why div { display: flex; gap: 7px; align-items: center; margin-top: 5px; }
.summary { display: flex; align-items: center; gap: 10px; font-size: 13.5px; color: #3a4052; margin: 0 0 6px; }
.summary .score { border-radius: 999px; padding: 3px 12px; font-weight: 700; background: #eef0f5; color: #6b7183; }
.summary .score.ok { background: #13865d; color: #fff; }
.legend { display: flex; flex-direction: column; gap: 8px; font-size: 13px; color: #3a4052; margin-top: 6px; }
.legend span { display: inline-flex; align-items: center; gap: 8px; }
.legend i { width: 16px; height: 16px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; }
.legend i.hitkey { background: #13865d; color: #fff; }
.legend b.n { background: #eef0f5; border-radius: 999px; padding: 0 6px; font-size: 11.5px; }
.ic { width: 13px; height: 13px; vertical-align: -2px; margin-right: 3px; color: #13865d; }
.mxwrap { overflow-x: auto; }
table.mx { width: 100%; border-collapse: separate; border-spacing: 5px 4px; table-layout: fixed; margin-left: -5px; }
table.mx th.fam { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .04em;
  padding: 3px 8px; border-radius: 6px; text-align: left; }
table.mx th.m { background: #fff; border: 1px solid #e4e7ee; border-top: 4px solid; border-radius: 8px;
  padding: 6px 8px; text-align: left; vertical-align: top; }
table.mx th.m .name { display: flex; justify-content: space-between; align-items: center; gap: 6px; }
table.mx th.m b { font-size: 15px; }
table.mx th.m .meta { display: block; font-size: 11px; color: #5f6577; font-weight: 400; margin-top: 1px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
table.mx th:first-child, table.mx td.r { width: 30px; background: transparent !important; border: none !important; }
table.mx td.r { color: #5f6577; font-weight: 600; font-size: 12px; text-align: center; white-space: nowrap; }
table.mx td.c { background: #fff; border: 1px solid #e6e8ee; border-radius: 7px; padding: 3px 8px; height: 40px;
  vertical-align: top; font-size: 12px; line-height: 1.25; cursor: default;
  transition: background-color .2s ease, border-color .2s ease; }
table.mx td.c .t { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
  color: #1c2030; }
table.mx td.c .b { display: flex; justify-content: space-between; gap: 6px; color: #5f6577; font-size: 11px;
  margin-top: 1px; white-space: nowrap; overflow: hidden; }
table.mx td.c .n { flex: 0 0 auto; background: #eef0f5; color: #3a4052; border-radius: 999px; padding: 0 6px;
  font-weight: 600; }
table.mx td.c.hit { background: #e6f5ee; border-color: #13865d; box-shadow: inset 4px 0 0 #13865d; }
table.mx td.c.hit .t { font-weight: 600; }
table.mx td.off { background: #f1f2f5; border: 1px dashed #c4c8d2; border-radius: 8px; color: #6b7183;
  text-align: center; vertical-align: middle; font-size: 13px; padding: 10px; }
.finding { background: #fff; border: 1px solid #e4e7ee; border-left: 5px solid; border-radius: 10px;
  padding: 11px 16px; margin-bottom: 8px; font-size: 14.5px; color: #1c2030; }
.protocol { background: #fff; border: 1px solid #e4e7ee; border-radius: 10px; padding: 12px 18px; font-size: 14px; }
.protocol li { margin-bottom: 4px; }
/* Narrower desktops: drop the brand line so all ten ranks still fit. */
@media (max-width: 1500px) {
  table.mx td.c .br { display: none; }
  table.mx td.c { position: relative; }
  table.mx td.c .b { position: absolute; right: 5px; bottom: 3px; margin: 0; }
  table.mx td.c { height: 32px; padding-right: 30px; }
  .topbar .pill { display: none; }
}
@media (prefers-reduced-motion: reduce) { table.mx td.c { transition: none; } }
</style>
"""


# ---------- shared helpers ----------

def short(text, length=70):
    text = str(text)
    return text if len(text) <= length else text[:length - 1].rstrip() + "…"


def item_label(data, item):
    row = data.metadata.loc[int(item)]
    title = row.title if row.has_title else "Không rõ tên"
    brand = row.brand if row.has_brand else ""
    return str(title), str(brand), str(row.leaf)


def category_chip(leaf):
    if not leaf:
        return ""
    tint = CATEGORY_TINTS[zlib.crc32(leaf.encode()) % len(CATEGORY_TINTS)]
    return f'<span class="cat" style="background:{tint}">{html.escape(short(leaf, 28))}</span>'


def plain_label(model):
    return LABELS[model].replace(" (top-128)", "")


@st.cache_resource(show_spinner="Đang nạp dữ liệu train và validation…")
def get_data():
    data = load_data()
    samples, eligible = sample_users(data)
    return data, samples, eligible


@st.cache_resource(show_spinner=False)
def get_ranker(model, density):
    return load_ranker(model, density, get_data()[0])


def rankers(density):
    with st.spinner(f"Đang nạp 7 mô hình ở mật độ {density:.0%}…"):
        return {model: get_ranker(model, density) for model in MODELS}


@st.cache_data(show_spinner=False)
def model_ndcg(model, density):
    return checkpoint_ndcg(model, density)


CHECK = ('<svg class="ic" viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8.5l3.2 3L13 4.5" fill="none" '
         'stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>')
FAMILY_SHORT = {"Đề xuất": "Đề xuất", "Baseline · GNN": "GNN", "Baseline · closed-form": "Closed-form"}


def view_options(key):
    """Family filter and a 'Hiển thị' popover (display filter + legend), shared by both recommendation tabs."""
    kinds = st.pills("Nhóm mô hình", list(FAMILY_SHORT), selection_mode="multi", default=list(FAMILY_SHORT),
                     format_func=FAMILY_SHORT.get, key=f"{key}_families", label_visibility="collapsed")
    return set(kinds) or set(FAMILY_SHORT)


def display_popover():
    with st.popover("Hiển thị", icon=":material/tune:", use_container_width=True):
        hide = st.toggle("Ẩn sản phẩm không rõ tên", value=True, key="hide",
                         help="Chỉ lọc phần hiển thị. Tắt để top-10 trùng đúng với lúc đánh giá.")
        st.markdown(f'<div class="legend"><span><i class="hitkey">{CHECK}</i>Trúng sản phẩm mua tiếp theo</span>'
                    '<span><b class="n">×k</b>Số mô hình cùng gợi ý</span>'
                    '<span><i style="background:#fff1bf;border:1px solid #e0a800"></i>Rê chuột: sáng ở mọi cột</span></div>',
                    unsafe_allow_html=True)
    return hide


def matrix(data, tops, density, target=None, ms=None, off=None, kinds=None):
    """The models as columns (grouped by family, filtered by `kinds`) and ranks 1-10 as rows.

    tops: model -> top-10 item ids (missing model = not available); off: model -> message for a greyed column.
    Products recommended by several shown models carry an "×k" badge and light up together on hover.
    """
    ms, off = ms or {}, off or {}
    groups = [group for group in GROUPS if kinds is None or group[1] in kinds]
    shown = [model for _, _, models in groups for model in models]
    counts = Counter(int(item) for model, top in tops.items() if model in shown for item in top)
    shared = [item for item, count in counts.items() if count > 1]
    hover = "".join(f'table.mx:has(td[data-i="{i}"]:hover) td[data-i="{i}"]{{background:#fff1bf;'
                    f'border-color:#e0a800;}}' for i in shared)
    families, heads = ['<th></th>'], ['<th></th>']
    for _, kind, models in groups:
        accent, tint = FAMILY_COLORS[kind]
        families.append(f'<th class="fam" colspan="{len(models)}" style="color:{accent};background:{tint}">{kind}</th>')
        for model in models:
            ndcg = model_ndcg(model, density)
            meta = " · ".join(filter(None, [f"NDCG@20 {ndcg:.4f}" if ndcg is not None else "",
                                            f"{ms[model]:.0f} ms" if model in ms else ""]))
            heads.append(f'<th class="m" style="border-top-color:{COLORS[model]};background:{COLORS[model]}12">'
                         f'<span class="name"><b style="color:{COLORS[model]}">{plain_label(model)}</b></span>'
                         f'<span class="meta">{meta}</span></th>')
    rows = []
    for rank in range(10):
        cells = [f'<td class="r">{rank + 1}</td>']
        for _, _, models in groups:
            for model in models:
                if model in off:
                    if rank == 0:
                        cells.append(f'<td class="off" rowspan="10">{off[model]}</td>')
                    continue
                top = tops.get(model)
                if top is None or rank >= len(top):
                    cells.append('<td class="c"></td>')
                    continue
                item = int(top[rank])
                title, brand, _ = item_label(data, item)
                badge = f'<span class="n">×{counts[item]}</span>' if counts[item] > 1 else ""
                hit = " hit" if item == target else ""
                tip = html.escape(title + (f" · {brand}" if brand else ""))
                cells.append(f'<td class="c{hit}" data-i="{item}" title="{tip}">'
                             f'<span class="t">{CHECK if hit else ""}{html.escape(title)}</span>'
                             f'<span class="b"><span class="br">{html.escape(short(brand, 22))}</span>{badge}</span></td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    # Each column keeps at least 150 px; below that the table scrolls inside its own frame, not the page.
    return (f'<style>{hover}</style><div class="mxwrap"><table class="mx" style="min-width:{30 + 155 * len(shown)}px">'
            f"<tr>{''.join(families)}</tr><tr>{''.join(heads)}</tr>{''.join(rows)}</table></div>")


def history_panel(data, user):
    history = data.history[user]
    codes = {int(item): chr(65 + i) if i < 26 else str(i + 1) for i, item in enumerate(history)}
    rows = []
    for item in history:
        title, brand, leaf = item_label(data, item)
        rows.append(f'<li><span class="code">{codes[int(item)]}</span><span>{html.escape(short(title, 70))}'
                    f'<span class="sub">{html.escape(brand)}</span>{category_chip(leaf)}</span></li>')
    st.markdown(f'<div class="panel"><h4>Lịch sử mua ({len(history)})</h4>'
                f'<ol class="hist">{"".join(rows)}</ol></div>', unsafe_allow_html=True)
    target = data.target[user]
    title, brand, leaf = item_label(data, target)
    st.markdown(f'<div class="panel answer"><h4>Mua tiếp theo (đáp án)</h4>{html.escape(title)}'
                f'<span class="sub">{html.escape(brand)}</span>{category_chip(leaf)}</div>', unsafe_allow_html=True)
    return codes, target


def why_box(ranker, history, top, codes, data):
    if not len(top):
        return ""
    parts = ranker.explain(history, top[0])
    total = sum(value for _, value in parts) or 1
    lines = "".join(f'<div><span class="code">{codes.get(item, "·")}</span><span style="flex:1">'
                    f'{html.escape(short(item_label(data, item)[0], 42))}</span><b>{value / total:.0%}</b></div>'
                    for item, value in parts)
    return (f'<div class="why"><b>Vì sao SGKR gợi ý #1?</b>'
            f'{lines or "<div>Chỉ do độ phổ biến (bias) của sản phẩm.</div>"}</div>')


def density_radio(key, collapsed=False):
    return st.radio("Mật độ dữ liệu train", DENSITIES, horizontal=True, key=key, format_func=lambda d: f"{d:.0%}",
                    label_visibility="collapsed" if collapsed else "visible",
                    help="Tỷ lệ tương tác train được giữ lại; mỗi mật độ có checkpoint riêng của từng mô hình.")


# ---------- tab 1: existing customer ----------

def tab_existing(data, samples, eligible):
    options = [user for _, user in samples]
    user = st.session_state.get("user", options[0])
    if user not in options:
        options = [user, *options]
    pick, shuffle, dens_label, dens, fam, show = st.columns([1.8, 0.8, 0.45, 1.8, 1.6, 0.75],
                                                           vertical_alignment="center")

    def chosen():
        st.session_state.user = st.session_state.user_pick
    pick.selectbox("Khách hàng", options, index=options.index(user), key="user_pick", on_change=chosen,
                   label_visibility="collapsed", format_func=lambda u: f"Khách {u} · {len(data.history[u])} sản phẩm")
    if shuffle.button("Ngẫu nhiên", icon=":material/shuffle:", use_container_width=True):
        st.session_state.user = int(np.random.default_rng().choice(eligible))
        st.rerun()
    dens_label.markdown("<div style='text-align:right;font-size:14px;color:#3a4052'>Mật độ</div>",
                        unsafe_allow_html=True)
    with dens:
        density = density_radio("density", collapsed=True)
    with fam:
        kinds = view_options("existing")
    with show:
        hide = display_popover()
    loaded = rankers(density)
    left, right = st.columns([1, 4.3], gap="medium")
    with left:
        codes, target = history_panel(data, user)
    tops, ms = {}, {}
    for model in MODELS:
        if loaded[model] is not None:
            tops[model], ms[model] = recommend(loaded[model], data, user=user, hide_untitled=hide)
    with right:
        off ={m: f"Chưa train ở mật độ {density:.0%}" for m in MODELS if loaded[m] is None}
        st.markdown(matrix(data, tops, density, target, ms, off, kinds), unsafe_allow_html=True)
    with left:
        if "sgkr" in tops:
            ranker = loaded["sgkr"]
            st.markdown(why_box(ranker, ranker.train_items(user), tops["sgkr"], codes, data), unsafe_allow_html=True)


# ---------- tab 2: new customer ----------

def tab_new(data):
    # Density, display filter and family filter are set once, in the existing-customer tab.
    density = st.session_state.get("density", 1.0)
    hide = st.session_state.get("hide", True)
    kinds = set(st.session_state.get("existing_families") or FAMILY_SHORT)
    loaded = rankers(density)
    picked = st.session_state.setdefault("picked", [])
    left, right = st.columns([1, 4.3], gap="medium")
    with left:
        query = st.text_input("Tìm sản phẩm", placeholder="ví dụ: headphones, memory card")
        known = next((r.candidates for r in loaded.values() if r is not None), None)
        searchable = data.metadata.labeled.to_numpy().copy()
        if known is not None:
            searchable &= known  # only products the trained graph knows
        if query.strip():
            titles = data.metadata.title.where(searchable, "")
            matches = np.flatnonzero(titles.str.contains(query.strip(), case=False, regex=False).to_numpy())[:6]
            if not len(matches):
                st.caption("Không tìm thấy sản phẩm.")
            for item in matches:
                title, brand, _ = item_label(data, item)
                text, action = st.columns([4, 1.4], vertical_alignment="center")
                text.markdown(f"<span style='font-size:13px'>{html.escape(short(title, 60))}</span>"
                              f"<span class='sub'>{html.escape(brand)}</span>", unsafe_allow_html=True)
                if int(item) in picked:
                    action.caption("Đã thêm")
                elif action.button("+ Thêm", key=f"add{item}"):
                    picked.append(int(item))
                    st.rerun()
        st.markdown(f"**Đã chọn ({len(picked)})**")
        for item in list(picked):
            text, action = st.columns([4, 1.4], vertical_alignment="center")
            text.markdown(f"<span style='font-size:13px'>{html.escape(short(item_label(data, item)[0], 60))}</span>",
                          unsafe_allow_html=True)
            if action.button("✕", key=f"del{item}", help="Bỏ sản phẩm này"):
                picked.remove(item)
                st.rerun()
        if st.button("Gợi ý ngay", type="primary", use_container_width=True, disabled=not picked):
            st.session_state.asked = list(picked)
    with right:
        asked = st.session_state.get("asked")
        off = {m: "Chưa train ở mật độ này" for m in MODELS if loaded[m] is None}
        off.update({m: "Không hỗ trợ khách mới" for m in MODELS
                    if loaded[m] is not None and not loaded[m].supports_new_users})
        tops, ms = {}, {}
        if asked:
            for model in MODELS:
                if model not in off:
                    tops[model], ms[model] = recommend(loaded[model], data, items=asked, hide_untitled=hide)
            st.markdown(f'<div class="summary"><span class="score ok">{len(tops)}/7 hỗ trợ khách mới</span>'
                        f'{len(asked)} sản phẩm đã chọn</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="summary">Thêm sản phẩm rồi bấm <b>Gợi ý ngay</b>.</div>', unsafe_allow_html=True)
        st.markdown(matrix(data, tops, density, None, ms, off, kinds), unsafe_allow_html=True)


# ---------- tab 3: benchmark ----------

@st.cache_data(show_spinner="Đang chạy paired bootstrap 10.000 lần cho từng cặp…")
def cached_significance(density):
    return significance(density, get_data()[0].target)


def findings(table, tests, density):
    """Three plain-language conclusions computed from the leaderboard and the tests."""
    best = table.iloc[0]
    items = [(COLORS[best.model], f"<b>{best['Mô hình']}</b> có NDCG@20 cao nhất ở mật độ {density:.0%}: "
                                  f"<b>{best['NDCG@20']:.4f}</b>.")]
    if not tests.empty:
        better = tests[tests.p_holm < .05]
        ties = tests[(tests.p_holm >= .05) & (tests.verdict != "Mô hình này tốt hơn")]
        worse = tests[tests.verdict == "Mô hình này tốt hơn"]
        text = f"SGKR tốt hơn có ý nghĩa thống kê <b>{len(better)}/{len(tests)}</b> mô hình (p Holm &lt; 0,05)"
        if len(ties):
            text += "; chưa khác biệt có ý nghĩa với " + ", ".join(f"<b>{plain_label(m)}</b>" for m in ties.model)
        if len(worse):
            text += "; kém hơn " + ", ".join(f"<b>{plain_label(m)}</b>" for m in worse.model)
        items.append((COLORS["sgkr"], text + "."))
    sgkr = table[table.model == "sgkr"]
    gnn = table[table.model.isin(["lightgcn", "xsimgcl", "directau"])]
    if len(sgkr) and len(gnn):
        fastest = gnn.train_seconds.min()
        items.append(("#13865d", f"SGKR train trong <b>{sgkr.train_seconds.iloc[0] / 60:.1f} phút</b>, nhanh hơn "
                                 f"khoảng <b>{fastest / sgkr.train_seconds.iloc[0]:.0f}×</b> so với GNN nhanh nhất "
                                 f"({fastest / 60:.0f} phút), và gợi ý được cho khách mới."))
    for color, text in items:
        st.markdown(f'<div class="finding" style="border-left-color:{color}">{text}</div>', unsafe_allow_html=True)


def leaderboard_view(table):
    table = table.assign(**{"Train (phút)": table.train_seconds / 60})
    view = table[["Hạng", "Mô hình", "Nhóm", *METRICS, "vs SGKR", "Train (phút)", "inference_ms_per_user"]]
    view = view.rename(columns={"vs SGKR": "NDCG@20 so với SGKR (%)", "inference_ms_per_user": "Suy luận (ms/user)"})
    family = {kind: accent for kind, (accent, _) in FAMILY_COLORS.items()}
    styler = (view.style
              .format({**{m: "{:.4f}" for m in METRICS}, "NDCG@20 so với SGKR (%)": "{:+.1f}",
                       "Train (phút)": "{:.1f}", "Suy luận (ms/user)": "{:.3f}"})
              .highlight_max(subset=list(METRICS), props="font-weight:700; background-color:#e6f5ee")
              .map(lambda kind: f"color:{family.get(kind, '#1c2030')}; font-weight:600", subset=["Nhóm"])
              .map(lambda v: "color:#13865d" if v > 0 else "color:#c0392b" if v < 0 else "",
                   subset=["NDCG@20 so với SGKR (%)"])
              .apply(lambda row: ["background-color:#e8f0fc" if row["Mô hình"] == "SGKR" else ""] * len(row), axis=1))
    st.dataframe(styler, hide_index=True, use_container_width=True)
    st.caption("Tô xanh: cao nhất cột. Thời gian train: GNN chạy GPU, còn lại CPU.")


def charts(density, metric):
    frame = by_density()
    frame["Mô hình"] = frame.model.map(plain_label)
    colors = alt.Scale(domain=[plain_label(m) for m in MODELS], range=[COLORS[m] for m in MODELS])
    lines = (alt.Chart(frame).mark_line(point=True, strokeWidth=2.5)
             .encode(x=alt.X("density:Q", title="Mật độ dữ liệu train", axis=alt.Axis(format="%", values=list(DENSITIES))),
                     y=alt.Y(f"{metric}:Q", title=metric),
                     color=alt.Color("Mô hình:N", scale=colors, legend=alt.Legend(orient="bottom", columns=7)),
                     tooltip=["Mô hình", alt.Tooltip("density:Q", format="%"), alt.Tooltip(f"{metric}:Q", format=".4f")])
             .properties(title=f"{metric} theo mật độ", height=380))
    current = frame[frame.density == density].assign(minutes=lambda f: f.train_seconds / 60)
    scatter = (alt.Chart(current).mark_circle(size=170)
               .encode(x=alt.X("minutes:Q", scale=alt.Scale(type="log"), title="Thời gian train (phút, thang log)"),
                       y=alt.Y(f"{metric}:Q", title=metric),
                       color=alt.Color("Mô hình:N", scale=colors, legend=None),
                       tooltip=["Mô hình", alt.Tooltip("minutes:Q", format=".1f"), alt.Tooltip(f"{metric}:Q", format=".4f")])
               .properties(title=f"Chất lượng và chi phí · mật độ {density:.0%}", height=380))
    labels = scatter.mark_text(align="left", dx=10, fontSize=12).encode(text="Mô hình:N")
    left, right = st.columns(2, gap="large")
    left.altair_chart(lines, use_container_width=True)
    right.altair_chart(scatter + labels, use_container_width=True)


def significance_view(tests):
    if tests.empty:
        st.caption("Chưa đủ dự đoán validation để kiểm định ở mật độ này.")
        return
    view = pd.DataFrame({
        "So sánh": "SGKR − " + tests.model.map(plain_label), "Hiệu NDCG@20": tests.delta,
        "KTC 95%": [f"[{low:+.4f}, {high:+.4f}]" for low, high in tests.ci95],
        "p một phía": tests.p_one_sided, "p Holm": tests.p_holm, "Kết luận": tests.verdict, "Số user": tests.n_users})
    verdict = {"SGKR tốt hơn": "background-color:#e6f5ee;color:#13865d;font-weight:600",
               "Chưa khác biệt có ý nghĩa": "background-color:#fff8e6;color:#8a6d00;font-weight:600",
               "Mô hình này tốt hơn": "background-color:#fdecea;color:#c0392b;font-weight:600"}
    st.dataframe(view.style.format({"Hiệu NDCG@20": "{:+.5f}", "p một phía": "{:.4f}", "p Holm": "{:.4f}"})
                 .map(lambda v: verdict.get(v, ""), subset=["Kết luận"]),
                 hide_index=True, use_container_width=True)
    st.caption("Paired bootstrap 10.000 lần, một phía (SGKR > mô hình kia), hiệu chỉnh Holm; như bảng B3 của báo cáo.")


def figure_gallery(metric):
    slug = metric.lower().replace("@", "")
    figures = [(f"H1_density_{slug}.png", "Theo mật độ dữ liệu train"), (f"H2_{slug}.png", "Kèm khoảng tin cậy 95%"),
               (f"H4_retention_{slug}.png", "Mức giữ chất lượng khi dữ liệu thưa"),
               (f"H5_user_groups_{slug}.png", "Theo nhóm độ dài lịch sử user"),
               (f"H3_learning_curves_{slug}.png", "Đường học theo epoch"), ("H8_ndcg_at_k.png", "NDCG@k"),
               ("H6a_popularity_trajectory.png", "Độ phổ biến của gợi ý"), ("H6b_coverage.png", "Độ phủ catalog"),
               ("H7a_train_time.png", "Thời gian train"), ("H7b_inference_latency.png", "Độ trễ suy luận")]
    figures = [(name, caption) for name, caption in figures if (FIGURES / name).exists()]
    st.caption("Hình báo cáo: 5 mô hình gốc, chưa gồm EASE và GF-CF.")
    if not figures:
        st.caption("Chưa có hình. Chạy `python scripts\\make_report.py` để vẽ.")
    for row in range(0, len(figures), 3):
        for column, (name, caption) in zip(st.columns(3, gap="large"), figures[row:row + 3]):
            column.image(str(FIGURES / name), caption=caption, use_container_width=True)


def tab_benchmark(data):
    pick_density, pick_metric, _ = st.columns([2.2, 1.2, 3], vertical_alignment="bottom")
    with pick_density:
        density = density_radio("bench_density")
    metric = pick_metric.selectbox("Chỉ số cho biểu đồ và hình", METRICS)
    table = leaderboard(density)
    if table.empty:
        st.info("Chưa có kết quả cho mật độ này.")
        return
    tests = cached_significance(density)
    overview, trend, stats, report = st.tabs(["Tổng quan", "Biểu đồ", "Kiểm định thống kê", "Giao thức & hình báo cáo"])
    with overview:
        findings(table, tests, density)
        st.markdown(f"##### Bảng xếp hạng · mật độ {density:.0%}")
        leaderboard_view(table)
    with trend:
        charts(density, metric)
    with stats:
        st.markdown(f"##### SGKR so với từng mô hình · mật độ {density:.0%}")
        significance_view(tests)
    with report:
        st.markdown(
            '<div class="protocol"><b>Giao thức đánh giá</b><ul>'
            '<li>Tập <b>validation</b> (sản phẩm mua tiếp theo của mỗi user); tập test không được dùng.</li>'
            f'<li>Xếp hạng trên <b>toàn bộ catalog</b> ({data.num_items:,} sản phẩm), loại các sản phẩm user đã mua; '
            'cùng cohort user ấm cho mọi mô hình.</li>'
            '<li>Train ở 4 mật độ (25–100% tương tác), seed 42. Chỉ số: NDCG@10/20, Recall@10/20, MRR@20.</li>'
            '<li>Baseline GNN tối đa 200 epoch, lấy epoch tốt nhất; EASE và GF-CF là nghiệm đóng, '
            '<b>một cấu hình cố định theo bài gốc, không tinh chỉnh</b>.</li>'
            '<li>EASE trong bảng là ma trận đầy đủ; demo dùng bản rút gọn top-128 trọng số/sản phẩm.</li>'
            '</ul></div>', unsafe_allow_html=True)
        summary = RESULTS / "summary.csv"
        if summary.exists():
            st.download_button("Tải bảng kết quả đầy đủ (CSV)", summary.read_bytes(), "summary.csv", "text/csv")
        figure_gallery(metric)


def main():
    st.set_page_config(page_title="Hệ thống khuyến nghị sản phẩm điện tử", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    if PREVIEW:
        st.warning(f"Chế độ xem thử: mô hình đọc từ `{PREVIEW}` (checkpoint train ngắn), "
                   "gợi ý chưa phản ánh kết quả cuối cùng.")
    data, samples, eligible = get_data()
    pills = [f"{len(MODELS)} mô hình", f"{data.num_users:,} khách hàng", f"{data.num_items:,} sản phẩm"]
    st.markdown('<div class="topbar"><div class="ttl">Hệ thống khuyến nghị sản phẩm điện tử</div>'
                + "".join(f'<span class="pill">{p}</span>' for p in pills) + "</div>", unsafe_allow_html=True)
    existing, new, benchmark = st.tabs(["Khách hàng có sẵn", "Khách hàng mới", "Benchmark học thuật"])
    with existing:
        tab_existing(data, samples, eligible)
    with new:
        tab_new(data)
    with benchmark:
        tab_benchmark(data)


if __name__ == "__main__":
    main()
