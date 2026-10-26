import glob
import json


def inspect_all():
    for nb in sorted(glob.glob("notebooks/*.ipynb")):
        with open(nb, encoding="utf-8") as f:
            data = json.load(f)
        for i, cell in enumerate(data.get("cells", [])):
            if cell.get("cell_type") != "markdown":
                continue
            src = "".join(cell.get("source", []))
            for line_no, line in enumerate(src.splitlines()):
                # Unbalanced dollar signs
                # Ignore escaped dollars if any
                line_no_escaped = line.replace(r"\$", "")
                num_dollars = line_no_escaped.count("$")
                if num_dollars % 2 != 0:
                    print(f"[UNBALANCED $] {nb} Cell {i} L{line_no}: {line}")

                # Check for stray LaTeX commands outside math blocks
                # We can check if \text, \sum, \approx, \alpha, \mu, \beta, etc appear outside $...$
                parts = line.split("$")
                # Even indices are OUTSIDE math blocks, odd indices are INSIDE math blocks
                for idx, part in enumerate(parts):
                    if idx % 2 == 0:
                        # Outside math
                        for kw in [
                            r"\sum",
                            r"\alpha",
                            r"\approx",
                            r"\mu",
                            r"\beta",
                            r"\text",
                            r"\log",
                            r"\frac",
                            r"\in",
                            r"\sigma",
                            r"\Delta",
                        ]:
                            if kw in part:
                                print(
                                    f"[OUTSIDE MATH] {nb} Cell {i} L{line_no}: keyword '{kw}' found outside $ in: {line}"
                                )

                # Suspicious broken math patterns like " = -", "=10$", "($ classes)"
                suspicious = [" = -", "=10$", "($ classes", "=-$", "  ="]
                for s in suspicious:
                    if s in line:
                        print(f"[SUSPICIOUS] {nb} Cell {i} L{line_no} ('{s}'): {line}")


if __name__ == "__main__":
    inspect_all()
