"""同じディレクトリの実施資料.mdから、1区切り1ページの配布PDFを作る。"""
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

HERE = Path(__file__).resolve().parent
pdfmetrics.registerFont(TTFont("JapaneseSans", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"))
BODY = ParagraphStyle("body", fontName="JapaneseSans", fontSize=11.2,
                      leading=18.2, spaceAfter=9, wordWrap="CJK", alignment=TA_LEFT,
                      textColor=colors.HexColor("#233743"))
TITLE = ParagraphStyle("title", parent=BODY, fontSize=19, leading=27, spaceAfter=23,
                       textColor=colors.HexColor("#145b75"))

def footer(canvas, _doc):
    canvas.setStrokeColor(colors.HexColor("#c8d7dd"))
    canvas.line(45, 42, A4[0] - 45, 42)
    canvas.setFont("JapaneseSans", 9)
    canvas.setFillColor(colors.HexColor("#637682"))
    canvas.drawString(45, 28, "DAS 対面3人の試行 ／ 2026年11月12日")
    canvas.drawRightString(A4[0] - 45, 28, str(canvas.getPageNumber()))

story = []
for i, page in enumerate((HERE / "実施資料.md").read_text().split("---PAGE---")):
    if i:
        story.append(PageBreak())
    blocks = page.strip().split("\n\n")
    for block in blocks:
        title = block.startswith("# ")
        text = block.removeprefix("# ") if title else block
        story.append(Paragraph(escape(text).replace("\n", "<br/>"), TITLE if title else BODY))
    story.append(Spacer(1, 3))
SimpleDocTemplate(str(HERE / "実施資料_2026-11-12.pdf"), pagesize=A4,
                  leftMargin=45, rightMargin=45, topMargin=42, bottomMargin=58,
                  title="対面3人の試行：実施資料", author="松井 玲").build(story, onFirstPage=footer, onLaterPages=footer)
