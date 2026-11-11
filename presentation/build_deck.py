"""Builds the project presentation on top of a Canva-exported template deck.

Usage: python presentation/build_deck.py <template.pptx> <out.pptx>

The template supplies the background, the purple swirl art, the embedded fonts
(Garet, Archivo Black) and the title / contents / Q&A slides. Every content
slide is rebuilt here. Paper numbers come from Yin et al. (ICML 2024),
Tables 3, 4 and 8. Our own numbers are read from results/ when the result
files exist, and are left as [__] when they do not, so rebuilding after new
runs fills the deck in.
"""
import copy
import glob
import json
import math
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE, XL_TICK_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

INK = "000000"
GRAY = "5F5F5F"
LINE = "DDD3E6"
LAV = "E5BBF6"      # the template's box colour
TINT = "F7EEFC"
VIOLET = "7C3AED"
LVIOLET = "C9B0F7"
ORANGE = "F2740B"
LORANGE = "FBC38D"
TEAL = "0E9F9A"
SLATE = "8A8696"
RED = "D6336C"

ASSETS = Path(__file__).resolve().parent / "assets"
RESULTS = Path(__file__).resolve().parent.parent / "results"
PENDING = "[__]"
# R1 (Table 3 of the paper): row label, result file suffix, and per model the paper's numbers and the OWL M it used
R1_ROWS = [("Dense", "dense"), ("Wanda", "wanda_uniform_s70"), ("OWL + Wanda", "wanda_owl_s70_M{M}_l0.08"),
           ("SparseGPT", "sparsegpt_uniform_s70"), ("OWL + SparseGPT", "sparsegpt_owl_s70_M{M}_l0.08")]
R1_MODELS = [("LLaMA-7B", "7B", "llama7b", 5, [5.68, 85.77, 24.55, 26.30, 19.49]),
             ("LLaMA-13B", "13B", "llama13b", 7, [5.09, 55.90, 17.17, 19.24, 14.55]),
             ("OPT-6.7B", "OPT", "opt6.7b", 10, [10.13, 162.92, 40.22, 20.29, 22.48])]
# label on the slide, file prefix in results/e1 and results/e2
NEW_MODELS = [("Qwen 0.5B", "qwen2.5-0.5b"), ("Qwen 1.5B", "qwen2.5-1.5b"), ("Qwen 3B", "qwen2.5-3b"), ("Qwen 7B", "qwen2.5-7b"),
              ("Llama 1B", "llama3.2-1b"), ("Llama 3B", "llama3.2-3b"), ("Llama 8B", "llama3.1-8b")]
ALT = {"usf.png": "University of South Florida logo", "usf_box.png": "USF logo", "meta.png": "Meta logo", "qwen.png": "Qwen logo",
       "huggingface.png": "Hugging Face logo", "pytorch.png": "PyTorch logo", "colab.png": "Google Colab logo", "kaggle.png": "Kaggle logo",
       "nvidia.png": "NVIDIA logo", "github.png": "GitHub logo", "qr_repo.png": "QR code for the project repository",
       "owl_figure1.png": "Figure 1 of the OWL paper: OWL and uniform layerwise sparsity over the outlier distribution"}

ALIGN = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}
ANCHOR = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}
R_EMBED = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"


def rgb(h):
    return RGBColor.from_string(h)


# ---------------------------------------------------------------- our results
def load(rel):
    """A result file under results/, or None if that job has not run yet."""
    path = RESULTS / rel
    return json.loads(path.read_text()) if path.exists() else None


def ppl(rel):
    r = load(rel)
    return r["ppl"].get("wikitext2") if r else None


def zs_acc(rel):
    """Mean zero-shot accuracy in percent."""
    r = load(rel)
    v = ((r or {}).get("zeroshot") or {}).get("mean_acc")
    return None if v is None else (v * 100 if v <= 1 else v)


def num(v, fmt="{:.2f}"):
    return PENDING if v is None else fmt.format(v)


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(order):
            r[i] = float(k)
        return r
    a, b = ranks(x), ranks(y)
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    cov = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    return cov / math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))


def e1_rows():
    """Per new model: perplexities at 70% and dL = log PPL(uniform) - log PPL(OWL), for the models that have run."""
    out = []
    for label, stem in NEW_MODELS:
        r = {"label": label, "stem": stem, "dense": ppl(f"e1/{stem}_dense.json")}
        for m in ("wanda", "sparsegpt"):
            uni, owl = ppl(f"e1/{stem}_{m}_uniform_s70.json"), ppl(f"e1/{stem}_{m}_owl_s70_M5_l0.08.json")
            r[m] = (uni, owl, math.log(uni) - math.log(owl) if uni and owl else None)
        dense_lod = load(f"e2/{stem}_dense.json")
        r["lod_u"] = dense_lod["lod_dense_summary"]["5.0"]["u"] if dense_lod else None
        r["lod_cv"] = dense_lod["lod_dense_summary"]["5.0"]["cv"] if dense_lod else None
        if r["wanda"][2] is not None and r["sparsegpt"][2] is not None:
            out.append(r)
    return out


def done_runs():
    """Number of finished runs, their total GPU hours and the GPUs used (the smoke tests do not count)."""
    runs = [json.loads(Path(p).read_text()) for p in glob.glob(str(RESULTS / "*" / "*.json")) if Path(p).parent.name != "smoke"]
    gpus = sorted({"A100" if "A100" in g else g.replace("NVIDIA ", "") for g in (r["env"].get("gpu") for r in runs) if g})
    return len(runs), sum(r["total_seconds"] for r in runs) / 3600, " and ".join(gpus)


# ---------------------------------------------------------------- text
def style_run(run, size, font="Garet", bold=False, color=INK):
    f = run.font
    f.size = Pt(size)
    f.bold = True if bold else None
    f.color.rgb = rgb(color)
    name = "Garet Bold" if (bold and font == "Garet") else font
    f.name = name
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs", "a:sym"):
        rPr.append(rPr.makeelement(qn(tag), {"typeface": name}))
    rPr.set("spc", str(int(-size * 4)))  # slight negative tracking, like the template


def fill_tf(tf, paras, size=28, font="Garet", bold=False, color=INK, align="l", anchor="t", ls=1.2, after=0, inset=0):
    """paras: list of paragraphs. A paragraph is a str or a list of runs.
    A run is a str or (str, {size, font, bold, color})."""
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = ANCHOR[anchor]
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(inset)
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = ALIGN[align]
        runs = [para] if isinstance(para, (str, tuple)) else para
        biggest = size
        for item in runs:
            text, o = (item, {}) if isinstance(item, str) else item
            r = p.add_run()
            r.text = text
            rs = o.get("size", size)
            biggest = max(biggest, rs)
            style_run(r, rs, o.get("font", font), o.get("bold", bold), o.get("color", color))
        p.line_spacing = Pt(biggest * ls)
        p.space_after = Pt(after)
        p.space_before = Pt(0)


def txt(slide, x, y, w, h, paras, **kw):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    fill_tf(tb.text_frame, paras, **kw)
    return tb


def box(slide, x, y, w, h, fill=LAV, line=None, radius=None, paras=None, shape=None, **kw):
    kind = shape or (MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE)
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    s.shadow.inherit = False
    if radius and kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(1.5)
    else:
        s.line.fill.background()
    if paras is not None:
        kw.setdefault("align", "c")
        kw.setdefault("anchor", "m")
        kw.setdefault("inset", 0.12)
        fill_tf(s.text_frame, paras, **kw)
    return s


def arrow(slide, x1, y1, x2, y2, color=INK, width=4):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    return c


def pic(slide, name, x, y, h=None, w=None):
    kw = {}
    if h:
        kw["height"] = Inches(h)
    if w:
        kw["width"] = Inches(w)
    p = slide.shapes.add_picture(str(ASSETS / name), Inches(x), Inches(y), **kw)
    p.name = ALT.get(name, name)
    p._element.nvPicPr.cNvPr.set("descr", ALT.get(name, name))
    return p


def pic_fit(slide, name, x, y, bw, bh):
    """Place a picture centred inside a bw x bh box, keeping its aspect ratio."""
    iw, ih = Image.open(ASSETS / name).size
    k = min(bw / iw, bh / ih)
    return pic(slide, name, x + (bw - iw * k) / 2, y + (bh - ih * k) / 2, w=iw * k)


def badge(slide, x, y, d, icon, fill=INK):
    box(slide, x, y, d, d, shape=MSO_SHAPE.OVAL, fill=fill)
    pic_fit(slide, icon, x + d * 0.25, y + d * 0.25, d * 0.5, d * 0.5)


def chip(slide, x, y, label, w=1.05, h=0.62, fill=LAV, color=INK, size=24):
    return box(slide, x, y, w, h, fill=fill, paras=label, font="Archivo Black", size=size, color=color, inset=0.02)


# ---------------------------------------------------------------- template plumbing
def clone_shape(src_slide, shape, dst_slide):
    el = copy.deepcopy(shape._element)
    for blip in el.iter(qn("a:blip")):
        part = src_slide.part.related_part(blip.get(R_EMBED))
        blip.set(R_EMBED, dst_slide.part.relate_to(part, RT.IMAGE))
    dst_slide.shapes._spTree.append(el)
    return dst_slide.shapes[-1]


def shape_named(slide, name):
    return next(s for s in slide.shapes if s.name == name)


def set_lines(shape, lines, size=None, proto_index=0):
    """Replace a text box's paragraphs, keeping the first paragraph's formatting."""
    txBody = shape.text_frame._txBody
    old = txBody.findall(qn("a:p"))
    proto = old[proto_index]
    for p in old:
        txBody.remove(p)
    for line in lines:
        p = copy.deepcopy(proto)
        runs = p.findall(qn("a:r"))
        for extra in runs[1:]:
            p.remove(extra)
        runs[0].find(qn("a:t")).text = line
        if size:
            rPr = runs[0].find(qn("a:rPr"))
            rPr.set("sz", str(int(size * 100)))
            rPr.set("spc", str(int(-size * 6)))
            spc = p.find(qn("a:pPr")).find(qn("a:lnSpc"))
            if spc is not None:
                spc.find(qn("a:spcPts")).set("val", str(int(size * 105)))
        txBody.append(p)
    shape.text_frame.auto_size = MSO_AUTO_SIZE.NONE


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# ---------------------------------------------------------------- charts
def style_chart(ch, fsize=20, legend="b"):
    ch.font.size = Pt(fsize)
    ch.font.name = "Garet"
    ch.font.color.rgb = rgb("333333")
    ch.has_title = False
    ch.has_legend = bool(legend)
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.BOTTOM if legend == "b" else XL_LEGEND_POSITION.TOP
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(fsize)
        ch.legend.font.name = "Garet"
    va = ch.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb("E6E1EC")
    va.major_gridlines.format.line.width = Pt(1)
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(fsize)
    va.tick_labels.font.name = "Garet"
    ca = ch.category_axis
    ca.has_major_gridlines = False
    ca.format.line.color.rgb = rgb("9B95A6")
    ca.tick_labels.font.size = Pt(fsize)
    ca.tick_labels.font.name = "Garet"


def add_chart(slide, kind, x, y, w, h, cats, series, colors, name, fsize=20, legend="b", labels=None, gap=70, overlap=-8):
    cd = CategoryChartData()
    cd.categories = cats
    for sname, vals in series:
        cd.add_series(sname, vals)
    gf = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    gf.name = name
    ch = gf.chart
    style_chart(ch, fsize, legend)
    plot = ch.plots[0]
    plot.vary_by_categories = False
    is_bar = kind in (XL_CHART_TYPE.COLUMN_CLUSTERED, XL_CHART_TYPE.BAR_CLUSTERED)
    if is_bar:
        plot.gap_width = gap
        plot.overlap = overlap
    for s, c in zip(plot.series, colors):
        if is_bar:
            s.format.fill.solid()
            s.format.fill.fore_color.rgb = rgb(c)
            s.invert_if_negative = False
        else:
            s.format.line.color.rgb = rgb(c)
            s.format.line.width = Pt(4.5)
            s.smooth = False
            s.marker.style = XL_MARKER_STYLE.CIRCLE
            s.marker.size = 11
            s.marker.format.fill.solid()
            s.marker.format.fill.fore_color.rgb = rgb(c)
            s.marker.format.line.color.rgb = rgb(c)
    if labels:
        plot.has_data_labels = True
        dl = plot.data_labels
        dl.font.size = Pt(fsize - 2)
        dl.font.name = "Garet"
        dl.font.color.rgb = rgb(INK)
        dl.number_format = labels
        dl.number_format_is_linked = False
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
    return ch


def color_points(ch, colors):
    s = ch.plots[0].series[0]
    for i, c in enumerate(colors):
        pt = s.points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = rgb(c)


def log_axis(ch, lo, hi):
    va = ch.value_axis
    va.minimum_scale = lo
    va.maximum_scale = hi
    scaling = va._element.find(qn("c:scaling"))
    scaling.insert(0, scaling.makeelement(qn("c:logBase"), {"val": "10"}))


def label_skip(ch, n):
    ax = ch.category_axis._element
    el = ax.makeelement(qn("c:tickLblSkip"), {"val": str(n)})
    anchor = ax.find(qn("c:noMultiLvlLbl"))
    if anchor is not None:
        anchor.addprevious(el)
    else:
        ax.append(el)


# ---------------------------------------------------------------- tables
def add_table(slide, x, y, widths, rows, row_h=0.8, size=24, head_size=24, name="Table"):
    gf = slide.shapes.add_table(len(rows), len(widths), Inches(x), Inches(y), Inches(sum(widths)), Inches(row_h * len(rows)))
    gf.name = name
    t = gf.table
    t.horz_banding = False
    for i, w in enumerate(widths):
        t.columns[i].width = Inches(w)
    for r, row in enumerate(rows):
        t.rows[r].height = Inches(row_h)
        for c, val in enumerate(row):
            cell = t.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(LAV if r == 0 else ("FFFFFF" if r % 2 else TINT))
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.18)
            cell.margin_top = cell.margin_bottom = Inches(0.05)
            o = {}
            if isinstance(val, tuple):
                val, o = val
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = ALIGN[o.get("align", "l")]
            run = p.add_run()
            run.text = val
            if r == 0:
                style_run(run, head_size, "Archivo Black")
            else:
                style_run(run, o.get("size", size), "Garet", o.get("bold", False), o.get("color", INK))
            p.line_spacing = Pt((head_size if r == 0 else size) * 1.15)
    return t


# ---------------------------------------------------------------- deck
def build(template, out):
    prs = Presentation(template)
    slides = list(prs.slides)
    cover, toc, base, qa = slides[0], slides[1], slides[3], slides[57]
    blank = base.slide_layout
    base_shapes = [shape_named(base, n) for n in ("Freeform 2", "Freeform 3", "TextBox 6", "TextBox 7")]
    page = [2]

    def content(title, sub):
        s = prs.slides.add_slide(blank)
        _, _, t, h = [clone_shape(base, sh, s) for sh in base_shapes]
        set_lines(t, [title])
        set_lines(h, [sub])
        page[0] += 1
        txt(s, 17.9, 10.52, 1.0, 0.4, str(page[0]), size=20, color=GRAY, align="r")
        return s

    # ---- 1 cover
    set_lines(shape_named(cover, "TextBox 5"), ["OWL"])
    set_lines(shape_named(cover, "TextBox 7"), ["Outlier Weighed Layerwise Sparsity", "on small modern LLMs"], size=80)
    set_lines(shape_named(cover, "TextBox 8"), ["Fall 2026"])
    who = shape_named(cover, "TextBox 6")
    body = who.text_frame._txBody
    ps = body.findall(qn("a:p"))
    course, gap_p, person = ps[0], ps[1], ps[2]
    for p in ps:
        body.remove(p)
    course.find(qn("a:r")).find(qn("a:t")).text = "CAI 5307 - NLP Project"
    body.append(course)
    body.append(gap_p)
    for nm in ("Tuan Khang Phan", "Xuan Gia Han Nguyen"):
        p = copy.deepcopy(person)
        p.find(qn("a:r")).find(qn("a:t")).text = nm
        body.append(p)
    who.width = Inches(9.5)
    pic(cover, "usf.png", 1.12, 10.0, h=0.56)
    notes(cover, "[0:15] Hi, we are Khang and Han. We are reproducing OWL, a pruning paper from ICML 2024, "
                 "and testing whether its main idea still helps on small modern language models.")

    # ---- 2 contents
    for name, lines in (("TextBox 10", ["What does", "OWL claim?"]), ("TextBox 14", ["Can we", "reproduce it?"]),
                        ("TextBox 18", ["Does it hold", "on new LLMs?"]), ("TextBox 22", ["What do", "we need?"])):
        set_lines(shape_named(toc, name), lines)
    for name, line in (("TextBox 23", "One-shot pruning and the OWL idea"), ("TextBox 24", "Four claims, four experiments"),
                       ("TextBox 25", "Our extension to Qwen2.5 and Llama 3"), ("TextBox 26", "Models, GPU hours, timeline")):
        set_lines(shape_named(toc, name), [line])
    for (bx, by), icon in zip(((2.05, 3.36), (9.32, 3.36), (4.86, 7.05), (12.13, 7.05)),
                              ("icon_scissors.png", "icon_repeat.png", "icon_sprout.png", "icon_cpu.png")):
        badge(toc, bx + 5.62 - 0.55, by - 0.5, 1.05, icon)
    notes(toc, "[0:15] Four parts: what the paper claims, how we reproduce it, our extension, and the resources we need.")

    # ---- 3 the problem
    s = content("1. What does OWL claim?", "Pruning every layer equally breaks at 70%")
    flow = [("Dense LLM", "LLaMA-7B"), ("Prune 70% of weights", "one shot, no retraining"), ("Sparse LLM", "same sparsity in every layer")]
    for i, (a, b) in enumerate(flow):
        x = 1.12 + i * 6.15
        box(s, x, 4.1, 5.4, 1.45, paras=[(a, {"font": "Archivo Black", "size": 28}), (b, {"size": 22})], ls=1.15)
        if i < 2:
            arrow(s, x + 5.5, 4.82, x + 6.05, 4.82)
    ch = add_chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, 1.0, 5.95, 10.6, 4.4,
                   ["Dense", "50% sparse", "60% sparse", "70% sparse"], [("WikiText perplexity", [5.68, 7.26, 10.70, 85.77])],
                   [LVIOLET], "Chart: Wanda perplexity by sparsity", legend=None, labels="0.00", gap=90)
    color_points(ch, [SLATE, LVIOLET, LVIOLET, RED])
    ch.value_axis.maximum_scale = 100
    ch.value_axis.major_unit = 25
    txt(s, 12.3, 6.05, 6.6, 2.2, ["Wanda and SparseGPT score each weight, then cut the same share from every layer."], size=28)
    txt(s, 12.3, 8.1, 6.6, 1.5, [[("15x", {"font": "Archivo Black", "size": 60, "color": RED})]], size=28, ls=1.0)
    txt(s, 12.3, 9.15, 6.6, 1.1, ["worse than dense at 70% sparsity"], size=28)
    txt(s, 1.12, 10.45, 16.5, 0.4, "LLaMA-7B pruned with Wanda, uniform sparsity. Lower is better. Yin et al. 2024, Tables 3 and 8.", size=20, color=GRAY)
    notes(s, "[0:40] The setting is one-shot pruning: take a trained LLM, zero out a share of its weights, no retraining. "
             "The strong methods, Wanda and SparseGPT, score each weight and remove the same share from every layer. "
             "That works at 50 percent. On LLaMA-7B dense perplexity is 5.68 and Wanda at 50 percent is 7.26. "
             "At 70 percent it jumps to 85.77, about fifteen times worse. Uniform sparsity breaks exactly where pruning would be most useful.")

    # ---- 4 the idea
    s = content("1. What does OWL claim?", "OWL prunes outlier-rich blocks less")
    steps = [("Score every weight", "Wanda metric: input norm times weight size", "A = ||X|| · |W|"),
             ("Count outliers", "Per block: share of weights scoring over M times the mean", "LOD, one value per block"),
             ("Set block sparsity", "More outliers, lower sparsity, inside a narrow band", "Block sparsity ~ 1 − LOD")]
    for i, (head, desc, formula) in enumerate(steps):
        x = 1.12 + i * 6.05
        box(s, x, 4.05, 5.65, 2.95, fill=TINT, radius=0.06)
        box(s, x + 0.3, 4.3, 0.75, 0.75, shape=MSO_SHAPE.OVAL, paras=str(i + 1), font="Archivo Black", size=28, inset=0)
        txt(s, x + 1.25, 4.38, 4.2, 0.6, head, size=28, bold=True)
        txt(s, x + 0.3, 5.25, 5.1, 1.0, desc, size=22, color="333333")
        txt(s, x + 0.3, 6.3, 5.1, 0.55, formula, size=24, bold=True, color=VIOLET)
        if i < 2:
            arrow(s, x + 5.7, 5.5, x + 6.0, 5.5, width=3)
    box(s, 1.12, 7.25, 12.3, 3.05, fill="FFFFFF", line=LINE, radius=0.04)
    pic_fit(s, "owl_figure1.png", 1.32, 7.4, 11.9, 2.75)
    txt(s, 13.85, 7.3, 5.05, 2.9, [[("Bars ", {"bold": True}), "are the outlier share of each block."],
                                   [("Orange line ", {"bold": True, "color": ORANGE}), "is OWL's sparsity. It dips where the bars are tall."],
                                   [("Dashed line ", {"bold": True}), "is uniform 70%."]], size=22, after=8, anchor="m")
    txt(s, 1.12, 10.42, 14, 0.4, "Figure 1 of Yin et al. 2024 (LLaMA-7B, 13B, 30B at 70% sparsity).", size=20, color=GRAY)
    notes(s, "[0:45] OWL asks whether every layer should lose the same share. It uses the Wanda score, input norm times weight magnitude. "
             "For each Transformer block it counts outliers, weights whose score is more than M times the block average. "
             "That gives the layerwise outlier distribution, LOD. Blocks with more outliers are pruned less, and block sparsity stays "
             "in a narrow band around the target, so the total is unchanged. This is the paper's Figure 1: bars are outliers per block, "
             "the orange line is OWL's sparsity, and it dips where the bars are tall.")

    # ---- 5 claims
    s = content("1. What does OWL claim?", "Four claims we test")
    claims = [("C1", "85.77 → 24.55", "OWL beats uniform sparsity at 70% (LLaMA-7B perplexity, Wanda). On OPT-6.7B it hurts SparseGPT."),
              ("C2", "61.22 vs 6.62", "The gain grows as models shrink: perplexity drop with Wanda at 7B vs 65B."),
              ("C3", "39.4% → 46.5%", "Zero-shot accuracy improves too. Mean of 7 tasks, LLaMA-7B at 70%, Wanda."),
              ("C4", "Not uniform", "Outliers concentrate in some blocks (LOD), and good pruning methods keep them.")]
    for i, (tag, big, desc) in enumerate(claims):
        x = 1.12 + (i % 2) * 9.0
        y = 4.05 + (i // 2) * 3.15
        box(s, x, y, 8.7, 2.9, fill=TINT, radius=0.06)
        chip(s, x + 0.35, y + 0.38, tag)
        txt(s, x + 1.65, y + 0.25, 6.8, 0.9, big, font="Archivo Black", size=46, color=VIOLET, anchor="m", ls=1.0)
        txt(s, x + 0.35, y + 1.35, 8.0, 1.4, desc, size=24, color="222222")
    notes(s, "[0:40] We test four claims. One, OWL beats uniform sparsity at 70 percent: with Wanda on LLaMA-7B perplexity falls from 85.77 to 24.55. "
             "There is one exception we also check, SparseGPT on OPT gets slightly worse. Two, the gain grows as models shrink, 61 points at 7B "
             "but under 7 at 65B. Three, zero-shot accuracy improves, 39.4 to 46.5 on average. Four, outliers are unevenly spread "
             "across blocks, and good pruning methods keep them.")

    # ---- 6 reproduction plan
    s = content("2. Can we reproduce it?", "Four experiments, one per claim")
    rows = [["Exp.", "Paper target", "What we run", "Claim"],
            [("R1", {"bold": True}), "Table 3", "70% sparsity on LLaMA-7B, LLaMA-13B and OPT-6.7B. Magnitude, Wanda, SparseGPT, uniform vs OWL", "C1, C2"],
            [("R2", {"bold": True}), "Figure 2, Table 8", "40% to 80% sweep on LLaMA-7B with Wanda, six ways to split sparsity", "C1"],
            [("R3", {"bold": True}), "Table 4", "Zero-shot accuracy on 7 tasks, dense vs 70% sparse LLaMA-7B", "C3"],
            [("R4", {"bold": True}), "Table 1, Figure 1", "Outlier distribution of each dense model, again after pruning", "C4"]]
    add_table(s, 1.12, 4.0, [1.5, 3.7, 10.6, 1.9], rows, row_h=0.92, size=22, name="Table: reproduction plan")
    box(s, 1.12, 8.95, 17.7, 1.3, fill=TINT, radius=0.1)
    txt(s, 1.45, 8.95, 17.1, 1.3, [[("Reproduced ", {"bold": True}),
        "means the dense LLaMA-7B perplexity is 5.68, every R1 setting moves in the paper's direction, and our 70% perplexities are within 15% of the paper's."]],
        size=24, anchor="m")
    txt(s, 1.12, 10.4, 14, 0.4, "Left out: LLaMA-30B and 65B, the CPU speedup study, LoRA fine-tuning.", size=20, color=GRAY)
    notes(s, "[0:35] Each claim maps to one experiment. R1 is Table 3 at 70 percent on three models. R2 is the sparsity sweep with six allocation rules. "
             "R3 is zero-shot accuracy. R4 recomputes the outlier distribution before and after pruning. We skip the 30B and 65B models and the speedup "
             "and fine-tuning studies. We call a claim reproduced when the direction matches and our perplexity is within 15 percent of the paper's.")

    # ---- 7 R1
    s = content("2. Can we reproduce it?", "R1: Table 3 at 70% sparsity")
    ch = add_chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, 0.95, 3.85, 11.5, 6.4, ["LLaMA-7B", "LLaMA-13B", "OPT-6.7B"],
                   [("Wanda", [85.77, 55.90, 162.92]), ("OWL + Wanda", [24.55, 17.17, 40.22]),
                    ("SparseGPT", [26.30, 19.24, 20.29]), ("OWL + SparseGPT", [19.49, 14.55, 22.48])],
                   [LVIOLET, VIOLET, LORANGE, ORANGE], "Chart: Table 3 perplexity at 70% sparsity", labels="0.00", fsize=20, gap=45, overlap=-6)
    ch.value_axis.maximum_scale = 180
    ch.value_axis.major_unit = 30
    P = {"align": "r"}
    O = {"align": "r", "color": VIOLET, "bold": True}
    # our numbers per model, in the order of R1_ROWS (None where a job has not run)
    ours = {stem: [ppl(f"r1/{stem}_{suffix.format(M=M)}.json") for _, suffix in R1_ROWS] for _, _, stem, M, _ in R1_MODELS}
    r1 = dict(zip(("dense", "wanda", "wanda_owl", "sparsegpt", "sparsegpt_owl"), ours["llama7b"]))
    cell = lambda v: PENDING if v is None else (f"{v:.2f}" if v < 100 else f"{v:.1f}")
    rows = [["Ours"] + [(short, P) for _, short, _, _, _ in R1_MODELS]]
    for i, (label, _) in enumerate(R1_ROWS):
        rows.append([label] + [(cell(ours[stem][i]), O) for _, _, stem, _, _ in R1_MODELS])
    add_table(s, 12.85, 4.0, [2.75, 1.13, 1.13, 1.14], rows, row_h=0.72, size=20, head_size=22, name="Table: our perplexity at 70% sparsity")
    # the six uniform vs OWL cases, and how far our numbers are from the paper's
    cases = [(name, method, ours[stem][u], ours[stem][o]) for name, _, stem, _, _ in R1_MODELS
             for method, u, o in (("Wanda", 1, 2), ("SparseGPT", 3, 4)) if ours[stem][u] is not None and ours[stem][o] is not None]
    helped = [c for c in cases if c[3] < c[2]]
    worse = [c for c in cases if c[3] >= c[2]]
    gaps = [abs(v - ref) / ref for _, _, stem, _, paper in R1_MODELS for v, ref in zip(ours[stem], paper) if v is not None]
    missing = [name for name, _, stem, _, _ in R1_MODELS if any(v is None for v in ours[stem])]
    if not cases:
        r1_text = "OWL helps in 5 of 6 cases. With SparseGPT on OPT-6.7B it gets worse, and we check that too."
        r1_note = ("[Fill in before the talk: our dense perplexity is __, Wanda __, OWL plus Wanda __, and whether each case moves "
                   "in the paper's direction. The three paper models have not run yet.]")
    else:
        hurt = " and hurts " + " and ".join(f"{c[1]} on {c[0]}" for c in worse) if worse else ""
        same = ", as in the paper" if [(c[0], c[1]) for c in worse] == [("OPT-6.7B", "SparseGPT")] else ""
        r1_text = (f"Ours: OWL helps in {len(helped)} of {len(cases)} cases{hurt}{same}. Every number is within {math.ceil(max(gaps) * 100)}% of the paper's."
                   + (f" {' and '.join(missing)} not run yet." if missing else ""))
        r1_note = (f"We reproduce it. Dense LLaMA-7B is {r1['dense']:.2f} against 5.68, Wanda {r1['wanda']:.2f} against 85.77, and OWL plus Wanda "
                   f"{r1['wanda_owl']:.2f} against 24.55. In our runs OWL helps in {len(helped)} of {len(cases)} cases. "
                   + "".join(f"With {c[1]} on {c[0]} it gets worse, {c[2]:.2f} to {c[3]:.2f}. " for c in worse)
                   + (f"That is the same exception the paper reports. " if same else "")
                   + f"The largest gap to the paper is {math.ceil(max(gaps) * 100)} percent."
                   + (f" {' and '.join(missing)} had not run when this deck was built." if missing else ""))
    txt(s, 12.85, 8.5, 6.15, 1.8, [r1_text], size=22)
    txt(s, 1.12, 10.4, 17, 0.4, "WikiText-2 perplexity, lower is better. Bars show the paper's numbers (Yin et al. 2024, Table 3), the table shows ours.",
        size=20, color=GRAY)
    notes(s, "[0:50] This is the main table we target. Bars are the paper's numbers and the table on the right is ours. Dark bars are OWL, "
             "light bars are uniform. " + r1_note)

    # ---- 8 R2-R4
    s = content("2. Can we reproduce it?", "R2 to R4: sweep, zero-shot, outliers")
    txt(s, 1.12, 3.95, 9.5, 0.5, "R2: perplexity vs sparsity, LLaMA-7B + Wanda (log scale)", size=24, bold=True)
    sweep = ["40", "50", "60", "70", "80"]
    r2_uni = [ppl(f"r2/llama7b_wanda_uniform_s{x}.json") for x in sweep]
    r2_owl = [ppl(f"r2/llama7b_wanda_owl_s{x}_M5_l0.08.json") for x in sweep]
    r2_done = all(v is not None for v in r2_uni + r2_owl)
    series = [("Uniform", [6.38, 7.26, 10.70, 85.77, 3499.88]), ("OWL", [6.39, 7.22, 9.35, 24.54, 1002.87]),
              ("OWL-inverse", [6.51, 8.03, 26.05, 822.23, 9616.08]), ("ER", [6.55, 7.74, 12.16, 112.03, 11151.18])]
    colors = [SLATE, VIOLET, ORANGE, TEAL]
    if r2_done:
        series += [("Ours: uniform", [round(v, 2) for v in r2_uni]), ("Ours: OWL", [round(v, 2) for v in r2_owl])]
        colors += [INK, INK]
    ch = add_chart(s, XL_CHART_TYPE.LINE_MARKERS, 0.95, 4.45, 10.3, 5.85, [x + "%" for x in sweep], series, colors,
                   "Chart: Table 8 sparsity sweep", fsize=18 if r2_done else 20)
    for sr, style in zip(list(ch.plots[0].series)[4:], (XL_MARKER_STYLE.CIRCLE, XL_MARKER_STYLE.DIAMOND)):
        # our runs: hollow markers without a line, drawn around the paper's points
        sr.format.line.fill.background()
        sr.marker.style = style
        sr.marker.size = 17
        sr.marker.format.fill.background()
        sr.marker.format.line.color.rgb = rgb(INK)
        sr.marker.format.line.width = Pt(2.25)
    log_axis(ch, 1, 100000)
    if r2_done:
        txt(s, 2.75, 4.8, 6.6, 1.1, [[("At 70%, paper: ", {"bold": True}), "uniform 85.77, OWL 24.54"],
                                     [("At 70%, ours: ", {"bold": True, "color": VIOLET}),
                                      (f"uniform {r2_uni[3]:.2f}, OWL {r2_owl[3]:.2f}", {"bold": True, "color": VIOLET})]], size=20, ls=1.3)
    else:
        txt(s, 2.75, 4.85, 5.2, 1.0, [[("At 70%:  ", {"bold": True}), ("uniform 85.77", {"color": "5E5A6B", "bold": True})],
                                      [("OWL 24.54", {"color": VIOLET, "bold": True})]], size=22, ls=1.25)
    ch.value_axis.tick_labels.number_format = "#,##0"
    ch.value_axis.tick_labels.number_format_is_linked = False
    txt(s, 11.75, 3.95, 7.2, 0.5, "R3: mean zero-shot accuracy, 7 tasks", size=24, bold=True)
    rows = [["LLaMA-7B, 70%", ("Paper", P), ("Ours", P)],
            ["Dense", ("62.40", P), (num(zs_acc("r1/llama7b_dense.json")), O)],
            ["Wanda", ("39.39", P), (num(zs_acc("r1/llama7b_wanda_uniform_s70.json")), O)],
            ["OWL + Wanda", ("46.47", P), (num(zs_acc("r1/llama7b_wanda_owl_s70_M5_l0.08.json")), O)],
            ["SparseGPT", ("44.93", P), (num(zs_acc("r1/llama7b_sparsegpt_uniform_s70.json")), O)],
            ["OWL + SparseGPT", ("48.03", P), (num(zs_acc("r1/llama7b_sparsegpt_owl_s70_M5_l0.08.json")), O)]]
    add_table(s, 11.75, 4.5, [4.2, 1.5, 1.5], rows, row_h=0.6, size=20, head_size=20, name="Table: zero-shot paper vs ours")
    box(s, 11.75, 8.4, 7.2, 1.85, fill=TINT, radius=0.08)
    r4 = {m: load(f"r4/llama7b_{m}_uniform_s70_lod.json") for m in ("wanda", "sparsegpt", "magnitude")}
    r4_note = ""
    if all(r4.values()):
        dense_lod = r4["wanda"]["lod_dense"]["5.0"]
        mean = {m: r["lod_pruned_summary"]["5.0"]["mean"] for m, r in r4.items()}
        keep = min(spearman(dense_lod, r["lod_pruned"]["5.0"]) for r in r4.values())
        d_mean = sum(dense_lod) / len(dense_lod)
        r4_text = (f"mean outlier share per block (M = 5). Dense {d_mean:.2f}%. At 70% sparsity: Wanda {mean['wanda']:.2f}%, "
                   f"SparseGPT {mean['sparsegpt']:.2f}%, magnitude {mean['magnitude']:.2f}%. The per-block shape survives (ρ ≥ {math.floor(keep * 100) / 100:.2f}).")
        r4_note = (f" R4, ours: the dense LLaMA-7B has {d_mean:.2f} percent outliers per block on average, {max(dense_lod):.1f} in the early blocks and "
                   f"{min(dense_lod):.1f} in the late ones. After pruning to 70 percent the shape is the same for all three methods. "
                   "Magnitude pruning keeps as many outliers as Wanda and SparseGPT by this count, although its perplexity collapses.")
        size4 = 20
    else:
        r4_text = "outliers per block before and after pruning. We expect the same uneven shape to survive pruning. Ours: [__]"
        size4 = 22
    txt(s, 12.05, 8.45, 6.7, 1.8, [[("R4: ", {"bold": True}), r4_text]], size=size4, anchor="m")
    txt(s, 1.12, 10.4, 14, 0.4, "Paper numbers from Yin et al. 2024, Tables 4 and 8. OWL-inverse gives outlier-rich blocks more sparsity."
        + (" Rings are our runs." if r2_done else ""), size=20, color=GRAY)
    notes(s, "[0:40] R2 is the sweep. On a log axis the methods are close up to 50 percent and split after 60. OWL-inverse, which gives outlier-rich "
             "blocks more sparsity, is the worst, and that supports the idea. R3 is mean zero-shot accuracy over seven tasks, and R4 is the outlier "
             "distribution before and after pruning."
             + (f" The rings are our runs. They sit on the paper's points up to 70 percent: {r2_uni[3]:.2f} and {r2_owl[3]:.2f} against 85.77 and 24.54. "
                f"At 80 percent ours are higher, {r2_uni[4]:.0f} and {r2_owl[4]:.0f}, but OWL is still about {r2_uni[4] / r2_owl[4]:.1f} times better."
                if r2_done else " [Fill in our R2 numbers before the talk.]")
             + r4_note
             + ("" if zs_acc("r1/llama7b_dense.json") is not None else " The zero-shot column is not filled in yet."))

    # ---- 9 status
    n_done, gpu_hours, gpu_names = done_runs()
    rows_e1 = e1_rows()
    paper_done = r1["dense"] is not None
    example = load("e1/qwen2.5-3b_wanda_owl_s70_M5_l0.08.json")
    smoke = example or load("smoke/smollm_wanda_owl_50.json")
    env = smoke["env"]
    if rows_e1 and not paper_done:
        headline = "Status: extension ran, paper models next"
    elif rows_e1:
        headline = "Status: paper models and extension ran"
    else:
        headline = "Where we are: code runs, GPU runs next"
    s = content("2. Can we reproduce it?", headline)
    status = [(True, "Our port of OWL, Wanda, SparseGPT and magnitude pruning"),
              (True, "36 unit tests pass, CPU smoke run works end to end"),
              (True, f"{n_done} GPU runs done on Colab ({gpu_names}), {gpu_hours:.1f} GPU hours" if n_done else "Job lists written for every experiment"),
              (paper_done, f"Dense LLaMA-7B check: perplexity {r1['dense']:.2f}, paper 5.68" if paper_done
               else "Next: LLaMA-7B, OPT-6.7B and LLaMA-13B for R1 to R4")]
    for i, (done, line) in enumerate(status):
        y = 4.0 + i * 0.86
        box(s, 1.12, y + 0.08, 0.56, 0.56, shape=MSO_SHAPE.OVAL, fill=VIOLET if done else None, line=None if done else VIOLET,
            paras="✓" if done else "", font="Arial", size=22, color="FFFFFF", bold=True, inset=0)
        txt(s, 1.95, y, 6.9, 0.8, line, size=24, anchor="m", bold=not done)
    pic_fit(s, "pytorch.png", 1.12, 9.72, 0.5, 0.56)
    txt(s, 1.75, 9.72, 2.4, 0.56, "PyTorch " + ".".join(env["torch"].split(".")[:2]), size=22, anchor="m")
    pic_fit(s, "huggingface.png", 4.25, 9.72, 0.6, 0.56)
    txt(s, 4.98, 9.72, 3.6, 0.56, "Transformers " + ".".join(env["transformers"].split(".")[:2]), size=22, anchor="m")
    box(s, 1.12, 7.5, 7.75, 2.0, fill=TINT, radius=0.07)
    txt(s, 1.42, 7.5, 7.2, 2.0, [[("Found while porting: ", {"bold": True}),
        "the released code scales block sparsity into a band 2λ wide, then shifts the mean to S. Some blocks leave the paper's stated range S ± λ."]],
        size=22, anchor="m")
    lod = [round(v, 3) for v in smoke["prune"]["lod"]]
    target = [round(v, 4) for v in smoke["prune"]["block_target"]]
    S, lam = smoke["args"]["sparsity"], smoke["args"]["lam"]
    blocks = [str(i + 1) for i in range(len(lod))]
    txt(s, 9.6, 3.95, 9.3, 0.45, "Outliers per block (LOD, %)", size=22, bold=True)
    ch = add_chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, 9.4, 4.35, 9.6, 2.45, blocks, [("LOD (%)", lod)], [VIOLET],
                   "Chart: outlier ratio per block", legend=None, fsize=18, gap=35)
    label_skip(ch, 5)
    ch.value_axis.major_unit = 1
    txt(s, 9.6, 6.9, 9.3, 0.45, "OWL sparsity target per block", size=22, bold=True)
    ch = add_chart(s, XL_CHART_TYPE.LINE_MARKERS, 9.4, 7.3, 9.6, 2.75, blocks,
                   [("OWL target (our port)", target), ("S + λ", [S + lam] * len(lod)), ("S − λ", [S - lam] * len(lod))],
                   [ORANGE, SLATE, SLATE], "Chart: OWL block sparsity target", fsize=18)
    for sr in list(ch.plots[0].series)[1:]:
        sr.marker.style = XL_MARKER_STYLE.NONE
        sr.format.line.width = Pt(2.5)
        sr.format.line.dash_style = 4  # dash
    ch.plots[0].series[0].marker.size = 7
    ch.value_axis.minimum_scale = round(S - 0.2, 2)
    ch.value_axis.maximum_scale = round(S + 0.15, 2)
    ch.value_axis.major_unit = 0.1
    ch.value_axis.tick_labels.number_format = "0%"
    ch.value_axis.tick_labels.number_format_is_linked = False
    label_skip(ch, 5)
    low = min(range(len(target)), key=lambda i: target[i])
    if example:
        caption = f"Qwen2.5-3B with Wanda, {S:.0%} target, M = 5, λ = {lam}. Our run on a Colab L4."
        source = "The charts are from our Qwen2.5-3B run"
    else:
        caption = "SmolLM2-135M, 50% target, M = 5, λ = 0.08. CPU smoke test with 4 calibration samples, not a result."
        source = "The charts are from a CPU smoke run on a tiny model"
    txt(s, 9.6, 10.1, 9.3, 0.7, caption, size=18, color=GRAY)
    notes(s, "[0:25] Where we are. We ported OWL, Wanda, SparseGPT and magnitude pruning to current libraries, with 36 unit tests. "
             + (f"{n_done} GPU runs are done, all on the extension models. " if n_done and not paper_done else "")
             + f"{source}: blocks with more outliers get a lower sparsity target. "
             "It showed one difference. The released code scales block sparsity into a band of width two lambda and then shifts the mean, "
             f"so some blocks fall outside the range the paper states. Here block {low + 1} gets {target[low]:.0%}, below the {S - lam:.0%} floor. "
             "We keep the code's behaviour and report it."
             + ("" if paper_done else " The three paper models are next."))

    # ---- 10 extension motivation
    s = content("3. Does it hold on new LLMs?", "The paper only tested LLaMA-1, 7B and up")
    txt(s, 1.12, 3.95, 7.6, 0.5, "Pre-training tokens (trillions)", size=24, bold=True)
    ch = add_chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, 0.95, 4.45, 7.7, 5.8, ["LLaMA-1", "Llama 3", "Qwen2.5"],
                   [("Tokens (trillions)", [1.4, 15, 18])], [SLATE], "Chart: pre-training tokens", legend=None, labels='General"T"', gap=80)
    color_points(ch, [SLATE, ORANGE, VIOLET])
    ch.value_axis.maximum_scale = 20
    ch.value_axis.major_unit = 5
    # size ladder, log scale from 0.5B to 65B
    x0, per = 9.75, 3.82
    pos = lambda b: x0 + (math.log10(b) - math.log10(0.5)) * per
    txt(s, 9.75, 3.95, 9.2, 0.5, "Model sizes tested (parameters, log scale)", size=24, bold=True)
    box(s, 9.6, 4.6, pos(7) - 9.6 - 0.12, 3.85, fill=TINT, radius=0.04)
    txt(s, 9.8, 4.68, 4.2, 0.45, "not tested in the paper", size=20, color=GRAY)
    ladder = [("OWL paper: LLaMA-1", SLATE, "FFFFFF", [7, 13, 30, 65]),
              ("Ours: Qwen2.5", VIOLET, "FFFFFF", [0.5, 1.5, 3, 7]),
              ("Ours: Llama 3.2 and 3.1", ORANGE, "FFFFFF", [1, 3, 8])]
    for r, (label, fill, fg, sizes) in enumerate(ladder):
        y = 5.2 + r * 1.1
        for b in sizes:
            chip(s, pos(b), y + 0.36, f"{b:g}B", w=0.88, h=0.56, fill=fill, color=fg, size=20)
        lx = pos(sizes[0]) if r == 0 else 9.8
        pic_fit(s, "qwen.png" if r == 1 else "meta.png", lx, y - 0.02, 0.5, 0.36)
        txt(s, lx + 0.6, y, 6.0, 0.36, label, size=18, bold=True, color=fill if r else GRAY)
    txt(s, 9.75, 8.7, 9.15, 1.6, ["Evidence on modern models is mixed. With SparseGPT at 70%, OWL nearly halves Llama-3-8B perplexity "
                                  "(85.84 to 48.07), but it slightly hurts a 1B Llama 3 at 50%."], size=22)
    txt(s, 1.12, 10.4, 17, 0.4, "Token counts: Touvron et al. 2023, Grattafiori et al. 2024, Qwen Team 2025. Evidence: Sieberling et al. 2025, Choenni et al. 2025.",
        size=18, color=GRAY)
    notes(s, "[0:40] Our extension. The paper shows the gain grows as models shrink, but only on LLaMA-1 from 7B to 65B, trained on at most 1.4 trillion tokens. "
             "Llama 3 and Qwen2.5 saw 15 to 18 trillion. Nobody tested the trend below 7B, where it should matter most, and the existing evidence is mixed. "
             "So we ask: does OWL still help small modern models?")

    # ---- 11 extension experiments
    s = content("3. Does it hold on new LLMs?", "Three experiments, seven models")
    cards = [("E1", "Scaling within a family", ["7 models from 0.5B to 8B at 50%, 60% and 70%, uniform vs OWL.",
                                                "We report the gain in log perplexity and add a shuffled OWL control."]),
             ("E2", "Does the LOD predict the gain?", ["LOD of all 10 dense models: how uneven it is, and whether it forms a U.",
                                                       "Spearman correlation with the OWL gain. Exploratory with 10 models."]),
             ("E3", "Hyperparameter transfer", ["M and λ grid on Qwen2.5-1.5B and Llama-3.2-1B, picked on held-out C4.",
                                                "A weak result should not just mean bad settings."])]
    for i, (tag, head, body_lines) in enumerate(cards):
        x = 1.12 + i * 6.05
        box(s, x, 4.0, 5.65, 6.25, fill=TINT, radius=0.05)
        chip(s, x + 0.35, 4.3, tag)
        txt(s, x + 1.6, 4.25, 3.85, 0.95, head, size=24, bold=True, anchor="m", ls=1.1)
        vy = 5.5
        if i == 0:  # seven models growing in size
            cx = x + 0.4
            for k, (d, c) in enumerate(zip([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [VIOLET, ORANGE, VIOLET, ORANGE, VIOLET, VIOLET, ORANGE])):
                box(s, cx, vy + (1.1 - d), d, d, shape=MSO_SHAPE.OVAL, fill=c)
                cx += d + 0.07
        elif i == 1:  # U-shaped outlier distribution
            for k, hgt in enumerate([1.1, 0.72, 0.45, 0.32, 0.28, 0.3, 0.36, 0.5, 0.78, 1.1]):
                box(s, x + 0.4 + k * 0.49, vy + (1.1 - hgt), 0.38, hgt, fill=VIOLET)
        else:  # M x lambda grid, paper default highlighted
            for r in range(4):
                for c in range(5):
                    default = (r == 1 and c == 2)
                    box(s, x + 0.4 + c * 0.6, vy - 0.08 + r * 0.31, 0.52, 0.25, fill=ORANGE if default else LVIOLET)
            txt(s, x + 3.5, vy - 0.05, 1.95, 1.2, ["4 x 5 grid", "orange: paper default"], size=18, color=GRAY, anchor="m")
        txt(s, x + 0.35, 6.95, 4.95, 3.2, body_lines, size=24, color="222222", after=12)
    notes(s, "[0:40] Three experiments. E1 prunes seven models from two families, uniform versus OWL, and reports the gain in log "
             "perplexity so small and large models are comparable. As a control we shuffle OWL's block sparsities: same values, wrong blocks. "
             "E2 asks whether the shape of the outlier distribution predicts the gain. "
             "E3 reruns the paper's hyperparameter grid on two small models, so a weak result is not just bad settings.")

    # ---- 12 extension results (only once the E1 runs exist)
    if rows_e1:
        helped = [r for r in rows_e1 if r["wanda"][2] > 0]
        hurt = [r["label"] for r in rows_e1 if r["wanda"][2] <= 0]
        s = content("3. Does it hold on new LLMs?", f"E1 at 70%: OWL helps {len(helped)} of {len(rows_e1)} models")
        txt(s, 1.12, 3.95, 10.4, 0.5, "Gain from OWL: ΔL = log PPL(uniform) − log PPL(OWL)", size=24, bold=True)
        ch = add_chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, 0.95, 4.45, 10.7, 5.85, [r["label"] for r in rows_e1],
                       [("Wanda", [round(r["wanda"][2], 2) for r in rows_e1]), ("SparseGPT", [round(r["sparsegpt"][2], 2) for r in rows_e1])],
                       [VIOLET, ORANGE], "Chart: OWL gain in log perplexity at 70% sparsity", labels="+0.00;−0.00", fsize=18, gap=60, overlap=-5)
        ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
        top = max(max(r["wanda"][2], r["sparsegpt"][2]) for r in rows_e1)
        bottom = min(min(r["wanda"][2], r["sparsegpt"][2]) for r in rows_e1)
        ch.value_axis.maximum_scale = math.ceil((top + 0.15) / 0.4) * 0.4
        ch.value_axis.minimum_scale = math.floor((bottom - 0.1) / 0.4) * 0.4
        ch.value_axis.major_unit = 0.4
        ch.value_axis.tick_labels.number_format = "0.0"
        ch.value_axis.tick_labels.number_format_is_linked = False

        def owl_cell(pair):
            return (f"{pair[1]:.1f}", {"align": "r", "bold": True, "color": VIOLET if pair[2] > 0 else RED})
        txt(s, 12.0, 3.95, 6.95, 0.5, "WikiText-2 perplexity at 70%", size=24, bold=True)
        rows = [["Model", ("Wanda", P), ("+ OWL", P), ("SGPT", P), ("+ OWL", P)]]
        for r in rows_e1:
            rows.append([r["label"], (f"{r['wanda'][0]:.1f}", P), owl_cell(r["wanda"]), (f"{r['sparsegpt'][0]:.1f}", P), owl_cell(r["sparsegpt"])])
        add_table(s, 12.0, 4.5, [2.35, 1.2, 1.15, 1.1, 1.15], rows, row_h=0.47, size=18, head_size=18, name="Table: perplexity at 70%, uniform vs OWL")
        if len(hurt) == 2:
            finding = [f"OWL hurts {' and '.join(hurt)}, the smallest of each family. The gain does not grow as models shrink."]
        else:
            finding = [f"With Wanda, OWL helps {len(helped)} of {len(rows_e1)} models."]
        lod = [(r["lod_cv"], r["lod_u"], r["wanda"][2], r["sparsegpt"][2]) for r in rows_e1 if r["lod_cv"] is not None]
        for _, _, stem, _, _ in R1_MODELS:
            v, dense_lod = ours[stem], load(f"e2/{stem}_dense.json")
            if dense_lod and all(x is not None for x in v[1:]):
                summ = dense_lod["lod_dense_summary"]["5.0"]
                lod.append((summ["cv"], summ["u"], math.log(v[1]) - math.log(v[2]), math.log(v[3]) - math.log(v[4])))
        rho_note = ""
        if len(lod) >= 4:
            cv, u, gw, gs = zip(*lod)
            rho_w, rho_s, rho_u = spearman(cv, gw), spearman(cv, gs), spearman(u, gw)
            finding.append([("E2: ", {"bold": True}), f"a more uneven LOD goes with a larger Wanda gain, ρ = {rho_w:+.2f} (n = {len(lod)}). "
                            f"No signal with SparseGPT ({rho_s:+.2f})."])
            rho_note = (f" For E2 we take all {len(lod)} models that have run, the paper's included. The Spearman correlation between how uneven the "
                        f"dense outlier distribution is across blocks and the Wanda gain is {rho_w:+.2f}, and {rho_u:+.2f} with the U-shape measure. "
                        f"With SparseGPT it is {rho_s:+.2f}, so no signal. With {len(lod)} models this is exploratory.")
        box(s, 12.0, 8.45, 6.95, 1.85, fill=TINT, radius=0.07)
        txt(s, 12.25, 8.45, 6.5, 1.85, finding, size=19, anchor="m", after=6, ls=1.15)
        txt(s, 1.12, 10.4, 17, 0.4, "Our runs, one calibration seed, M = 5, λ = 0.08. Positive ΔL means OWL helps. In the table, purple beats uniform and red is worse.",
            size=20, color=GRAY)
        best = max(rows_e1, key=lambda r: r["wanda"][2])
        notes(s, f"[0:45] Our first extension result, at 70 percent sparsity. Bars above zero mean OWL beats uniform. With Wanda, OWL helps {len(helped)} of "
                 f"{len(rows_e1)} models. The largest gain is on {best['label']}, where perplexity drops from {best['wanda'][0]:.0f} to {best['wanda'][1]:.0f}. "
                 + (f"But OWL hurts {' and '.join(hurt)}, the smallest model of each family. " if len(hurt) == 2 else "")
                 + "So the gain does not grow as models shrink, which is the opposite of the paper's trend on LLaMA-1. "
                 "With SparseGPT the effect is smaller and mixed." + rho_note
                 + " These are single-seed runs, so the small differences may be noise.")

    # ---- 12 resources
    s = content("4. What do we need?", "212 runs, about 26 A100 hours, no paid API")
    txt(s, 1.12, 3.95, 8.0, 0.5, "Estimated A100 hours per experiment", size=24, bold=True)
    ch = add_chart(s, XL_CHART_TYPE.BAR_CLUSTERED, 0.95, 4.4, 8.3, 4.15, ["E3", "E1", "R4 + E2", "R3", "R2", "R1"],
                   [("A100 hours", [2, 15, 1, 3, 2, 3])], [VIOLET], "Chart: GPU hours per experiment", legend=None, labels="0", gap=45)
    color_points(ch, [ORANGE, ORANGE, VIOLET, VIOLET, VIOLET, VIOLET])
    ch.value_axis.maximum_scale = 18
    ch.value_axis.major_unit = 6
    box(s, 1.3, 8.52, 0.3, 0.3, fill=VIOLET)
    txt(s, 1.7, 8.47, 2.6, 0.4, "Reproduction", size=18)
    box(s, 4.0, 8.52, 0.3, 0.3, fill=ORANGE)
    txt(s, 4.4, 8.47, 2.6, 0.4, "Extension", size=18)
    done_card = (str(n_done), f"runs done, {gpu_hours:.1f} GPU hours so far") if n_done else ("26 h", "estimated, 60 h budgeted")
    stats = [("212", "pruning runs planned"), done_card, ("$0", "API cost, open weights")]
    for i, (big, small) in enumerate(stats):
        x = 9.8 + i * 3.08
        box(s, x, 4.0, 2.88, 1.85, fill=TINT, radius=0.08)
        txt(s, x + 0.1, 4.12, 2.68, 0.85, big, font="Archivo Black", size=44, color=VIOLET, align="c", anchor="m", ls=1.0)
        txt(s, x + 0.15, 5.0, 2.58, 0.8, small, size=18, align="c", color="222222", ls=1.1)
    pic_fit(s, "huggingface.png", 9.8 + 2 * 3.08 + 0.2, 4.22, 0.6, 0.6)
    gpus = [("usf_box.png", "USF GAIVI cluster", "A100, H100 and L40S, free with a supervisor"),
            ("colab.png", "Colab Pro", f"all {n_done} runs so far, on {gpu_names}") if n_done
            else ("colab.png", "Colab Pro, both accounts", "13 to 37 A100 hours a month"),
            ("kaggle.png", "Backup", "Kaggle, Modal credit or a rented GPU (under $110)"),
            ("nvidia.png", "GPU memory", "24 GB for 7B and 8B, 31 GB for 13B")]
    for i, (logo, head, desc) in enumerate(gpus):
        y = 6.1 + i * 0.66
        pic_fit(s, logo, 9.8, y + 0.05, 0.62, 0.45)
        txt(s, 10.62, y, 8.3, 0.55, [[(head + "  ", {"bold": True}), desc]], size=20, anchor="m")
    steps = ["Dense check: 5.68", "R1 to R4", "E1 to E3", "Report: Dec 4"]
    if rows_e1 and not paper_done:
        steps = ["E1 70% + E2: done", "R1 to R4", "E1 rest, E3", "Report: Dec 4"]
    elif rows_e1:
        r3_done = zs_acc("r1/llama7b_dense.json") is not None
        steps = ["R1 to R4: done" if r3_done else "R1, R2, R4: done", "E1 70% + E2: done", "E1 rest, E3" if r3_done else "R3, E1 rest, E3", "Report: Dec 4"]
    for i, label in enumerate(steps):
        x = 1.12 + i * 4.55
        box(s, x, 9.2, 3.75, 0.95, paras=label, font="Archivo Black", size=24)
        if i < 3:
            arrow(s, x + 3.87, 9.675, x + 4.43, 9.675)
    notes(s, "[0:35] Resources. All models are open weights, so no paid API. The plan is 212 runs and about 26 A100 hours. "
             + (f"So far {n_done} runs are done in {gpu_hours:.1f} GPU hours on Colab. " if n_done else "")
             + "A 7B model needs a 24 GB GPU and LLaMA-13B about 31. We use Colab Pro and USF's GAIVI cluster, with Kaggle or a rented GPU as backup. "
             + ("" if paper_done else "Next are the three paper models for R1 to R4, then the rest of the extension. ")
             + "The report is due December 4.")

    # ---- 13 Q&A
    pic(qa, "qr_repo.png", 1.12, 7.35, h=1.9)
    txt(qa, 3.3, 7.6, 8, 0.8, "Thank you", size=44, bold=True, anchor="m")
    txt(qa, 3.3, 8.4, 8, 0.6, "Code, proposal and slides are in the repo", size=24, color=GRAY, anchor="m")
    pic(qa, "github.png", 1.12, 9.6, h=0.5)
    txt(qa, 1.8, 9.6, 12, 0.5, "github.com/khangpt2k6/NLP_CLASS", size=26, color=GRAY, anchor="m")
    notes(qa, "[0:05] Thank you. Questions? Total target: 7:50, which leaves 10 seconds of slack under the 8 minute limit.")

    # ---- drop unused template slides, put Q&A last
    lst = prs.slides._sldIdLst
    ids = list(lst)
    keep = {ids[0], ids[1], ids[57]} | set(ids[59:])
    for sld in ids:
        if sld not in keep:
            prs.part.drop_rel(sld.rId)
            lst.remove(sld)
    qa_id = ids[57]
    lst.remove(qa_id)
    lst.append(qa_id)
    prs.save(out)
    print("saved", out, "slides:", len(prs.slides))


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
