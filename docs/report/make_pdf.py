"""REPORT.md → docs/report/REPORT.pdf  (GIF 는 정지 프레임으로).   python docs/report/make_pdf.py"""
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[2]
src = (ROOT / "REPORT.md").read_text(encoding="utf-8")


def gif_table(m):
    head = [c.strip() for c in m.group(1).strip("|").split("|")]
    imgs = re.findall(r"!\[\]\(([^)]+)\.gif\)", m.group(2))
    out = []
    for cap, g in zip(head, imgs):
        png = "docs/report/pdf/" + Path(g).name + "_frames.png"
        out.append(f'<figure><img src="{png}"><figcaption>{cap} (GIF 의 정지 프레임 — 움직이는 버전은 GitHub REPORT.md)</figcaption></figure>')
    return "\n".join(out) + "\n"


src = re.sub(r"^(\|[^\n]*\|)\n\|--\|--\|\n(\| !\[\][^\n]*\|)\n", gif_table, src, flags=re.M)
src = re.sub(r"(?m)^((?!\s*(?:\d+\.|-|\|)\s)[^\n]+)\n(?=(?:\d+\.|-) )", r"\1\n\n", src)
html_body = markdown.markdown(src, extensions=["tables", "fenced_code", "sane_lists"])
css = """
@page { size: A4; margin: 16mm 14mm 16mm 14mm; }
body { font-family: 'Noto Sans CJK KR', 'Noto Sans KR', sans-serif; font-size: 9.6pt; line-height: 1.55; color: #222; }
h1 { font-size: 18pt; border-bottom: 2px solid #2a78d6; padding-bottom: 6px; margin-top: 0; }
h2 { font-size: 13pt; color: #1c5cab; margin-top: 22px; border-bottom: 1px solid #ddd; padding-bottom: 3px; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 12px; font-size: 8.6pt; page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 4px 6px; vertical-align: top; }
th { background: #eef3fb; }
pre { background: #f6f6f4; border: 1px solid #e2e2dd; padding: 8px 10px; font-size: 7.8pt; line-height: 1.35;
      white-space: pre-wrap; page-break-inside: avoid; font-family: 'Noto Sans Mono CJK KR', monospace; }
code { font-family: 'Noto Sans Mono CJK KR', monospace; font-size: 8.4pt; background: #f2f2ef; padding: 0 2px; }
pre code { background: none; padding: 0; }
img { max-width: 100%; }
figure { margin: 8px 0 14px; page-break-inside: avoid; text-align: center; }
figcaption { font-size: 8pt; color: #666; margin-top: 3px; }
blockquote { border-left: 4px solid #2a78d6; background: #f4f8fd; margin: 8px 0; padding: 6px 12px; }
p > img { display: block; margin: 6px auto; page-break-inside: avoid; }
em { color: #555; }
"""
html = f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><base href="{ROOT.as_uri()}/"><style>{css}</style></head><body>{html_body}</body></html>'
tmp = ROOT / "docs/report/_report.html"
tmp.write_text(html, encoding="utf-8")

from playwright.sync_api import sync_playwright  # noqa: E402

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.goto(tmp.as_uri()); pg.wait_for_load_state("networkidle")
    pg.pdf(path=str(ROOT / "docs/report/REPORT.pdf"), format="A4", print_background=True,
           display_header_footer=True, header_template="<span></span>",
           footer_template='<div style="font-size:7pt;color:#888;width:100%;text-align:center;">'
                           '<span class="pageNumber"></span> / <span class="totalPages"></span></div>',
           margin={"top": "16mm", "bottom": "16mm", "left": "14mm", "right": "14mm"})
    b.close()
tmp.unlink()
print("docs/report/REPORT.pdf")
