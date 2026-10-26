import glob
import json
import re
import subprocess

katex_path = "/Users/bernard/.npm/_npx/8c2cfac42696c54b/node_modules/katex"


def check_all_notebooks():
    nbs = sorted(glob.glob("notebooks/*.ipynb"))
    errors = []

    for nb in nbs:
        with open(nb, encoding="utf-8") as f:
            data = json.load(f)

        for i, cell in enumerate(data.get("cells", [])):
            if cell.get("cell_type") != "markdown":
                continue
            src = "".join(cell.get("source", []))

            # Check control chars
            for char_code, char_name in [
                (7, "\\a"),
                (8, "\\b"),
                (12, "\\f"),
                (11, "\\v"),
            ]:
                if chr(char_code) in src:
                    errors.append(
                        (
                            nb,
                            i,
                            f"Contains control char ASCII {char_code} ({char_name})",
                        )
                    )

            # Match math
            math_blocks = re.findall(r"\$\$([\s\S]*?)\$\$|\$([^$\n]+?)\$", src)
            for b1, b2 in math_blocks:
                math = b1 if b1 else b2
                # Skip trivial non-math like currency $10
                if not any(c in math for c in ["\\", "_", "^", "{", "=", "+", "-"]):
                    continue
                # Test with node katex
                res = subprocess.run(
                    [
                        "node",
                        "-e",
                        f"""
                try {{
                    const katex = require('{katex_path}');
                    katex.renderToString({json.dumps(math)}, {{throwOnError: true}});
                }} catch(e) {{
                    console.error(e.message);
                    process.exit(1);
                }}
                """,
                    ],
                    capture_output=True,
                    text=True,
                )
                if res.returncode != 0:
                    errors.append(
                        (
                            nb,
                            i,
                            f"KaTeX parse error in '{math[:50]}': {res.stderr.strip()[:100]}",
                        )
                    )

    print(f"Total errors found: {len(errors)}")
    for nb, i, err in errors:
        print(f"[{nb} Cell {i}] {err}")


if __name__ == "__main__":
    check_all_notebooks()
