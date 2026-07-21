import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Presentation" / "notebook_figures_review"
OUT.mkdir(parents=True, exist_ok=True)

SELECTIONS = {
    ROOT / "VAE-Tutorial" / "Part2_Dynamics" / "03a_hmm_foundations.ipynb": [4, 8, 10, 12, 14],
    ROOT / "VAE-Tutorial" / "Part2_Dynamics" / "03_hmm_lds.ipynb": [9, 22, 24, 32, 34],
    ROOT / "VAE-Tutorial" / "Part3_Variational" / "05_variational_em.ipynb": [10, 22, 34, 36, 49],
}

for notebook, cells in SELECTIONS.items():
    data = json.loads(notebook.read_text(encoding="utf-8"))
    for cell_index in cells:
        cell = data["cells"][cell_index]
        for output_index, output in enumerate(cell.get("outputs", [])):
            png = output.get("data", {}).get("image/png")
            if png:
                encoded = "".join(png) if isinstance(png, list) else png
                name = f"{notebook.stem}_cell{cell_index}_{output_index}.png"
                (OUT / name).write_bytes(base64.b64decode(encoded))
                print(OUT / name)
