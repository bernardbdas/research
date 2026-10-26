import json

with open("notebooks/01-data-partitioning.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

nb["cells"][5]["source"] = [
    "## **3. [BENCHMARK] Generating Partition Regimes**\n",
    "\n",
    "We instantiate $K=10$ client partitions under IID, Dirichlet ($\\alpha=0.5$), and Pathological ($2$ classes/client) allocations.\n",
]

with open("notebooks/01-data-partitioning.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("[✓] Written cleanly from file!")
