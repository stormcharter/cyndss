"""Атлас графів: по одній схемі на машину — еталон проти знайдених ланцюгів.

Навіщо. Таблиця з числами каже, ЯКА частка кроків збіглась, але не каже, ЩО
саме система відтворила, а що вигадала. Атлас робить це видимим: зелений вузол
покриває крок еталона, сірий — ні. Одразу видно і головний дефект методу —
ланцюги в 1–2 переходи проти еталона в 2–4 кроки.

Читає `results/raw/*.json` (їх пише run_experiment.py) і складає один HTML.
Далі його можна віддати у браузер або надрукувати в PDF.

Запуск:
    python experiments/htb_reconstruction/run_experiment.py --split demo
    python experiments/htb_reconstruction/build_atlas.py
"""

import html
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "results" / "raw"
OUT = HERE / "results" / "atlas.html"

NODE_H = 23
GAP_X = 22
ROW_H = 33
PAD = 8
CHAR_W = 7.1


def _w(label: str) -> float:
    return max(58.0, len(label) * CHAR_W + 18)


def _chip(x, y, label, fill, stroke, text_fill, dashed=False):
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    return (
        f'<rect x="{x:.1f}" y="{y}" width="{_w(label):.1f}" height="{NODE_H}" rx="6" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"{dash}/>'
        f'<text x="{x + _w(label) / 2:.1f}" y="{y + NODE_H / 2 + 4:.1f}" '
        f'text-anchor="middle" font-size="11.5" fill="{text_fill}">{html.escape(label)}</text>'
    )


def _arrow(x1, y):
    return (f'<path d="M{x1:.1f},{y + NODE_H / 2:.1f} L{x1 + GAP_X - 8:.1f},'
            f'{y + NODE_H / 2:.1f}" stroke="#9a7bff" stroke-width="1.4" '
            f'marker-end="url(#ah)" fill="none"/>')


def _row(chain, y, gold_tokens, label):
    """Один ланцюг рядком. Зелений вузол = покриває крок еталона."""
    parts = [f'<text x="0" y="{y + NODE_H / 2 + 4:.1f}" font-size="10.5" '
             f'fill="#7b7391" font-family="JetBrains Mono, monospace">{html.escape(label)}</text>']
    x = 86.0
    for i, node in enumerate(chain):
        hit = node in gold_tokens
        parts.append(_chip(x, y, node,
                           "#e6f6ee" if hit else "#f4f2fa",
                           "#1f8a5f" if hit else "#d8d2e8",
                           "#12704c" if hit else "#4a4460"))
        x += _w(node)
        if i < len(chain) - 1:
            parts.append(_arrow(x, y))
            x += GAP_X
    return "".join(parts), x


def _gold_row(gold, y):
    """Еталон: множина токенів на крок — показуємо через «/»."""
    parts = [f'<text x="0" y="{y + NODE_H / 2 + 4:.1f}" font-size="10.5" '
             f'fill="#b26a00" font-family="JetBrains Mono, monospace">еталон</text>']
    x = 86.0
    for i, step in enumerate(gold):
        label = " / ".join(step[:3])
        parts.append(_chip(x, y, label, "#fdf4e3", "#b26a00", "#8f5400", dashed=True))
        x += _w(label)
        if i < len(gold) - 1:
            parts.append(_arrow(x, y))
            x += GAP_X
    return "".join(parts), x


def machine_svg(data: dict) -> str:
    gold = [step for step in data["reference"]["gold_steps_scored"]]
    gold_tokens = {tok for step in gold for tok in step}
    chains = data["chains"]["c_bidirectional_production"][:5]

    rows, width, y = [], 0.0, PAD
    if gold:
        svg, end = _gold_row(gold, y)
        rows.append(svg)
        width = max(width, end)
        y += ROW_H + 6
    if chains:
        for i, chain in enumerate(chains, 1):
            svg, end = _row(chain, y, gold_tokens, f"ланцюг {i}")
            rows.append(svg)
            width = max(width, end)
            y += ROW_H
    else:
        rows.append(f'<text x="86" y="{y + 18}" font-size="12" fill="#c0392b">'
                    'ланцюгів не знайдено</text>')
        y += ROW_H

    height = y + PAD
    vb_w = max(width, 640)
    # width задаємо ЯВНО: без нього svg{width:100%} розтягує схему на всю
    # колонку, блок роздувається і на сторінку лізе одна машина замість трьох.
    return (f'<svg class="atlas" width="{vb_w:.0f}" height="{height:.0f}" '
            f'viewBox="0 0 {vb_w:.0f} {height:.0f}" '
            f'xmlns="http://www.w3.org/2000/svg" role="img" '
            f'aria-label="Еталон і знайдені ланцюги для {html.escape(data["machine"])}">'
            '<defs><marker id="ah" markerWidth="8" markerHeight="8" refX="7" refY="4" '
            'orient="auto"><path d="M0,1 L7,4 L0,7 z" fill="#9a7bff"/></marker></defs>'
            '<g font-family="Commissioner, sans-serif">' + "".join(rows) + "</g></svg>")


def machine_block(data: dict) -> str:
    m = data["metrics"]
    verdict, cls = ("не влучив", "bad")
    cov = m.get("c_coverage") or 0.0
    topk = m.get("c_coverage_topk") or 0.0
    if cov >= 0.5 or topk >= 0.6:
        verdict, cls = ("основний ланцюг відтворено частково", "ok")
    elif topk > 0:
        verdict, cls = ("окремі ланки збіглись", "part")
    stats = (f'покриття {cov:.2f} · по top-5 {topk:.2f} · '
             f'точність {(m.get("c_precision") or 0):.2f} · '
             f'кроків еталона {len(data["reference"]["gold_steps_scored"])} · '
             f'ланцюг {m.get("c_hops") or 0} переход(ів)')
    return (
        f'<section class="mach">'
        f'<h3>{html.escape(data["machine"])} '
        f'<span class="goal">ціль: <code>{html.escape(data["reference"]["goal"] or "—")}</code></span>'
        f'<span class="v {cls}">{verdict}</span></h3>'
        f'<div class="stats">{stats}</div>'
        f'{machine_svg(data)}'
        f'</section>'
    )


def build() -> str:
    files = sorted(RAW.glob("*.json"))
    if not files:
        raise SystemExit(f"немає сирих даних у {RAW} — спершу запусти run_experiment.py")
    blocks, scored = [], 0
    for path in files:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if not data["reference"]["gold_steps_scored"]:
            continue          # нічим оцінювати — у атлас не тягнемо
        scored += 1
        blocks.append(machine_block(data))
    head = f"""<!doctype html>
<html lang="uk"><head><meta charset="utf-8">
<title>ONYX · Атлас графів реконструкції</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Commissioner:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{{--paper:#fbfaff;--panel:#fff;--line:#e2ddef;--ink:#1a1626;--ink-2:#4a4460;
--ink-3:#7b7391;--accent:#6d4aff;--good:#1f8a5f;--warn:#b26a00;--bad:#c0392b}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);
font-family:"Commissioner",system-ui,sans-serif;font-size:14px;line-height:1.55;font-weight:350}}
.wrap{{max-width:960px;margin:0 auto;padding:40px 28px 60px}}
header{{border-bottom:2px solid var(--ink);padding-bottom:18px;margin-bottom:26px}}
.eyebrow{{font-family:"JetBrains Mono",monospace;font-size:11px;letter-spacing:.14em;
text-transform:uppercase;color:var(--accent);font-weight:500}}
h1{{font-size:26px;margin:.3em 0 .2em;font-weight:700;letter-spacing:-.015em}}
.lead{{color:var(--ink-2);max-width:66ch;margin:0}}
.legend{{display:flex;flex-wrap:wrap;gap:16px;margin-top:16px;font-size:12.5px;color:var(--ink-2)}}
.legend i{{display:inline-block;width:14px;height:14px;border-radius:4px;vertical-align:-2px;
margin-right:6px;border:1.2px solid}}
.sw-gold{{background:#fdf4e3;border-color:var(--warn)}}
.sw-hit{{background:#e6f6ee;border-color:var(--good)}}
.sw-miss{{background:#f4f2fa;border-color:#d8d2e8}}
.mach{{border:1px solid var(--line);border-radius:9px;background:var(--panel);
padding:14px 16px;margin:0 0 14px}}
.mach h3{{margin:0 0 4px;font-size:16px;font-weight:650;display:flex;
align-items:baseline;gap:12px;flex-wrap:wrap}}
.goal{{font-size:12.5px;color:var(--ink-3);font-weight:400}}
code{{font-family:"JetBrains Mono",monospace;font-size:.9em;background:#f4f2fa;
padding:1px 5px;border-radius:4px}}
.v{{margin-left:auto;font-size:11.5px;padding:1px 9px;border-radius:20px;
border:1px solid currentColor;font-weight:500}}
.v.ok{{color:var(--good)}} .v.part{{color:var(--warn)}} .v.bad{{color:var(--bad)}}
.stats{{font-family:"JetBrains Mono",monospace;font-size:11px;color:var(--ink-3);
margin-bottom:6px}}
svg.atlas{{display:block;max-width:100%;height:auto;overflow:visible}}
@page{{size:A4 landscape;margin:12mm}}
@media print{{
  *{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
  body{{font-size:9.6pt}} .wrap{{max-width:none;padding:0}}
  /* Рамка+фон у друці лишають «привид» порожньої коробки, коли блок не
     влазить і переїжджає на наступну сторінку. Замінюємо на просту лінію. */
  .mach{{break-inside:avoid;border:none;border-top:1px solid var(--line);
         border-radius:0;background:none;padding:9px 0 4px;margin:0}}
  h1{{font-size:19pt}}
}}
</style></head><body><div class="wrap">
<header>
<div class="eyebrow">ONYX CTI · додаток до експерименту</div>
<h1>Атлас графів реконструкції — {scored} демонстраційних машин</h1>
<p class="lead">Для кожної машини: прихований еталонний ланцюг (пунктир) і те, що
знайшов двофронтовий обхід у бойовій формі (<code>hypotheses_from</code>, top-5).
Зелений вузол покриває крок еталона, сірий — ні. Цільовий крок з еталона
відкинуто: він подавався на вхід.</p>
<div class="legend">
<span><i class="sw-gold"></i>крок еталона</span>
<span><i class="sw-hit"></i>вузол покрив крок</span>
<span><i class="sw-miss"></i>вузол не з еталона</span>
</div>
</header>
"""
    return head + "\n".join(blocks) + "\n</div></body></html>\n"


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover
        pass
    OUT.write_text(build(), encoding="utf-8")
    print(f"Записано: {OUT}")
