"""majec_visualize --save_pdf joins per-sample pages with pypdf. PyPDF2 was
dropped: it is deprecated and has an unfixed advisory (CVE-2023-36464)."""

import io

import pytest


def test_merge_pdfs_joins_pages_in_order(tmp_path):
    pytest.importorskip("plotly")
    pypdf = pytest.importorskip("pypdf")

    from majec.analysis.visualization import merge_pdfs

    blobs = []
    for width in (100, 200, 300):  # page width marks the order
        w = pypdf.PdfWriter()
        w.add_blank_page(width=width, height=50)
        buf = io.BytesIO()
        w.write(buf)
        blobs.append(buf.getvalue())
    out = merge_pdfs(blobs, tmp_path / "joined.pdf")
    pages = pypdf.PdfReader(out).pages
    assert [round(float(p.mediabox.width)) for p in pages] == [100, 200, 300]
