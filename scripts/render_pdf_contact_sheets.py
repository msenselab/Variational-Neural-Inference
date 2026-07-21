from pathlib import Path

import fitz
from PIL import Image, ImageDraw


PDF = Path(r"D:\Variational-Neural-Inference\references\MLNeuralDataAnalysis slides\pdf\13_slds_new.pdf")
OUT = Path(r"D:\Variational-Neural-Inference\Presentation\slds_pdf_review")
GROUPS = {
    "phenomenon_to_dynamics": list(range(3, 13)),
    "vector_fields_and_lds": list(range(17, 26)),
    "slds_rslds": list(range(27, 33)),
    "neuroscience_results": list(range(33, 45)),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(PDF)
    for name, pages in GROUPS.items():
        thumbs = []
        for page_no in pages:
            page = doc[page_no - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(1.15, 1.15), alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            image.thumbnail((560, 315))
            tile = Image.new("RGB", (580, 350), "white")
            tile.paste(image, ((580 - image.width) // 2, 25))
            ImageDraw.Draw(tile).text((10, 6), f"Page {page_no}", fill="black")
            thumbs.append(tile)
        cols = 2
        rows = (len(thumbs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 580, rows * 350), (225, 225, 225))
        for i, tile in enumerate(thumbs):
            sheet.paste(tile, ((i % cols) * 580, (i // cols) * 350))
        sheet.save(OUT / f"{name}.png")


if __name__ == "__main__":
    main()
