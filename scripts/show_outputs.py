"""Print the text outputs of an executed notebook (handy for checking results without Jupyter)."""
import sys
import nbformat

sys.stdout.reconfigure(encoding="utf-8")

for path in sys.argv[1:]:
    nb = nbformat.read(path, 4)
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        for o in c.outputs:
            if o.get("output_type") == "error":
                print(f"[cell {i}] ERROR {o['ename']}: {o['evalue']}")
            elif "text" in o:
                print(f"[cell {i}] {o['text'].rstrip()}")
            elif "data" in o and "image/png" not in o["data"] and "text/plain" in o["data"]:
                print(f"[cell {i}] {o['data']['text/plain'][:400]}")
