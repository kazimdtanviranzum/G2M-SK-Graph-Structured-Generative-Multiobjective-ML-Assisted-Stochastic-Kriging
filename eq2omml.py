"""Convert the manuscript's LaTeX display equations to OMML (editable Word math).

Reads the EQS dict from eqs.py, converts each equation with pandoc, and writes
eqs_omml/eqNN.xml containing a single <m:oMath> element for injection.
"""
import os, re, subprocess, sys, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("eqs", os.path.join(HERE, "eqs.py.src"))

# eqs.py renders PNGs on import; instead parse the dict out of the source text.
src = open(os.path.join(HERE, "eqs.py")).read()
m = re.search(r"EQS = \{(.*?)\n\}", src, re.S)
body = m.group(1)
pairs = re.findall(r"\"(eq\d+)\":\s*r\"\$(.*?)\$\"", body, re.S)
assert len(pairs) == 30, f"expected 30 equations, found {len(pairs)}"

outdir = os.path.join(HERE, "eqs_omml")
os.makedirs(outdir, exist_ok=True)

fail = []
for name, tex in pairs:
    md = f"${tex}$\n"
    p = subprocess.run(["pandoc", "-f", "markdown", "-t", "docx",
                        "-o", os.path.join(outdir, "_tmp.docx")],
                       input=md.encode(), capture_output=True)
    if p.returncode != 0:
        fail.append((name, p.stderr.decode()[:300]))
        continue
    xml = subprocess.run(["unzip", "-p", os.path.join(outdir, "_tmp.docx"),
                          "word/document.xml"], capture_output=True).stdout.decode()
    mm = re.search(r"<m:oMath>.*</m:oMath>", xml, re.S)
    if not mm:
        fail.append((name, "no oMath in output"))
        continue
    open(os.path.join(outdir, name + ".xml"), "w").write(mm.group(0))

if fail:
    for n, e in fail:
        print("FAILED", n, e)
    sys.exit(1)
print(f"{len(pairs)} equations converted to OMML")
