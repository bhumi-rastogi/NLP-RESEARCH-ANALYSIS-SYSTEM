import io
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable


# Brand accent colour used across the PDF
_ACCENT_HEX  = "#5b21b6"
_LINK_HEX    = "#1d4ed8"
_DARK_HEX    = "#111827"


def _build_styles() -> dict:
    """Construct and return a dict of custom ReportLab paragraph styles."""
    base  = getSampleStyleSheet()
    styles = {}

    styles["brand_label"] = ParagraphStyle(
        "BrandLabel",
        parent    = base["Normal"],
        fontSize  = 9,
        textColor = colors.HexColor(_ACCENT_HEX),
        spaceAfter= 4,
        alignment = TA_CENTER,
    )
    styles["doc_title"] = ParagraphStyle(
        "DocTitle",
        parent    = base["Title"],
        fontSize  = 20,
        textColor = colors.HexColor(_DARK_HEX),
        spaceAfter= 10,
        alignment = TA_CENTER,
    )
    styles["section_heading"] = ParagraphStyle(
        "SectionHeading",
        parent     = base["Heading2"],
        fontSize   = 13,
        textColor  = colors.HexColor(_ACCENT_HEX),
        spaceBefore= 18,
        spaceAfter = 6,
    )
    styles["body_text"] = ParagraphStyle(
        "BodyText",
        parent    = base["Normal"],
        fontSize  = 10,
        leading   = 16,
        textColor = colors.HexColor(_DARK_HEX),
        spaceAfter= 6,
        alignment = TA_LEFT,
    )
    styles["bullet_item"] = ParagraphStyle(
        "BulletItem",
        parent    = base["Normal"],
        fontSize  = 10,
        leading   = 14,
        leftIndent= 18,
        textColor = colors.HexColor(_DARK_HEX),
        spaceAfter= 5,
    )
    styles["ref_item"] = ParagraphStyle(
        "RefItem",
        parent    = base["Normal"],
        fontSize  = 9,
        leading   = 13,
        leftIndent= 18,
        textColor = colors.HexColor(_LINK_HEX),
        spaceAfter= 4,
    )
    return styles


def generate_pdf_report(report_data: dict) -> bytes:
    """
    Build a formatted PDF from a structured research report dictionary.
    Returns the raw PDF as bytes, suitable for Streamlit's download_button.
    """
    output_buf = io.BytesIO()

    pdf_doc = SimpleDocTemplate(
        output_buf,
        pagesize     = A4,
        leftMargin   = 2 * cm,
        rightMargin  = 2 * cm,
        topMargin    = 2 * cm,
        bottomMargin = 2 * cm,
    )

    style = _build_styles()
    elements = []

    # Header / brand line
    elements.append(Paragraph("NexusResearch — AI-Powered Research Report", style["brand_label"]))
    elements.append(Spacer(1, 0.2 * cm))

    # Report title
    report_title = report_data.get("title", "Research Report")
    elements.append(Paragraph(report_title, style["doc_title"]))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor(_ACCENT_HEX)))
    elements.append(Spacer(1, 0.4 * cm))

    # Abstract section
    elements.append(Paragraph("Abstract", style["section_heading"]))
    elements.append(Paragraph(report_data.get("abstract", "Not available."), style["body_text"]))

    # Key findings
    elements.append(Paragraph("Key Findings", style["section_heading"]))
    findings = report_data.get("key_findings", [])
    if findings:
        for idx, point in enumerate(findings, start=1):
            elements.append(Paragraph(f"{idx}.  {point}", style["bullet_item"]))
    else:
        elements.append(Paragraph("No key findings were extracted.", style["body_text"]))

    # Conclusion
    elements.append(Paragraph("Conclusion", style["section_heading"]))
    elements.append(Paragraph(report_data.get("conclusion", "Not available."), style["body_text"]))

    # References / sources
    elements.append(Paragraph("References", style["section_heading"]))
    sources = report_data.get("sources", [])
    if sources:
        for idx, src in enumerate(sources, start=1):
            src_title = src.get("title", "Unknown Source")
            src_url   = src.get("url", "")
            if src_url:
                link_text = f"{idx}.  <a href='{src_url}' color='{_LINK_HEX}'>{src_title}</a>"
                elements.append(Paragraph(link_text, style["ref_item"]))
            else:
                elements.append(Paragraph(f"{idx}.  {src_title}", style["ref_item"]))
    else:
        elements.append(Paragraph("No references available.", style["body_text"]))

    pdf_doc.build(elements)
    output_buf.seek(0)
    return output_buf.read()
