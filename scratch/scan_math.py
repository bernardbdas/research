import glob
import json
import re


def scan_all_math():
    for nb in sorted(glob.glob("notebooks/*.ipynb")):
        with open(nb, encoding="utf-8") as f:
            data = json.load(f)
        for i, cell in enumerate(data.get("cells", [])):
            if cell.get("cell_type") != "markdown":
                continue
            src = "".join(cell.get("source", []))

            # Check for dollar sign issues
            lines = src.splitlines()
            for l_idx, line in enumerate(lines):
                # Count unescaped dollars
                clean_line = line.replace(r"\$", "")
                cnt = clean_line.count("$")
                if cnt % 2 != 0:
                    print(f"[{nb} Cell {i} L{l_idx}] ODD DOLLAR COUNT ({cnt}): {line}")

                # Check for common LaTeX commands that might be outside dollar signs
                # Split by unescaped $
                tokens = clean_line.split("$")
                # Even indices are text, odd indices are math
                for t_idx, token in enumerate(tokens):
                    if t_idx % 2 == 0:
                        # In text
                        for cmd in [
                            r"\alpha",
                            r"\beta",
                            r"\gamma",
                            r"\theta",
                            r"\lambda",
                            r"\mu",
                            r"\sigma",
                            r"\sum",
                            r"\prod",
                            r"\approx",
                            r"\in",
                            r"\leq",
                            r"\geq",
                            r"\neq",
                            r"\mathbb",
                            r"\mathcal",
                            r"\mathbf",
                            r"\text",
                            r"\frac",
                            r"\sqrt",
                            r"\delta",
                            r"\epsilon",
                            r"\nabla",
                        ]:
                            if cmd in token:
                                print(
                                    f"[{nb} Cell {i} L{l_idx}] COMMAND OUTSIDE MATH: '{cmd}' in '{token.strip()}' | Full line: {line}"
                                )

                # Specifically check for unformatted math variables
                if re.search(r"\b[A-Za-z]=\d+\$", line) or re.search(
                    r"\b[A-Za-z] = -\$", line
                ):
                    print(f"[{nb} Cell {i} L{l_idx}] BROKEN VAR EQUALITY: {line}")


if __name__ == "__main__":
    scan_all_math()
