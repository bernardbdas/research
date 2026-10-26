import glob
import json


def check_lines():
    for nb in sorted(glob.glob("notebooks/*.ipynb")):
        with open(nb, encoding="utf-8") as f:
            data = json.load(f)
        found = False
        for i, cell in enumerate(data.get("cells", [])):
            if cell.get("cell_type") != "markdown":
                continue
            src = "".join(cell.get("source", []))
            for line in src.splitlines():
                if any(
                    sym in line
                    for sym in [
                        "$",
                        "\\",
                        "approx",
                        "alpha",
                        "beta",
                        "entropy",
                        "Dirichlet",
                        "sigma",
                        "epsilon",
                    ]
                ):
                    if not found:
                        print(f"=== {nb} ===")
                        found = True
                    print(f"  Cell {i}: {line}")


if __name__ == "__main__":
    check_lines()
