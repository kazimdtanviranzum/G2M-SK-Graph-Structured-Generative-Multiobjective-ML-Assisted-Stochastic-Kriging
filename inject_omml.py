"""Replace EQOMML_eqNN_END placeholder runs in the built docx with native
editable OMML equations (eqs_omml/eqNN.xml), producing the final manuscript.

Usage: python3 inject_omml.py G2M-SK_manuscript_v3_raw.docx G2M-SK_manuscript_revised.docx
"""
import os, re, shutil, subprocess, sys, zipfile

src = sys.argv[1] if len(sys.argv) > 1 else "G2M-SK_manuscript_v3_raw.docx"
dst = sys.argv[2] if len(sys.argv) > 2 else "G2M-SK_manuscript_revised.docx"
work = "_inject_unpacked"

if os.path.exists(work):
    shutil.rmtree(work)
with zipfile.ZipFile(src) as z:
    z.extractall(work)

path = os.path.join(work, "word", "document.xml")
xml = open(path, encoding="utf-8").read()

# 1. ensure the math namespace is declared on the document root
root = re.search(r"<w:document[^>]*>", xml).group(0)
if "xmlns:m=" not in root:
    new_root = root[:-1] + ' xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
    xml = xml.replace(root, new_root, 1)

# 2. swap each placeholder run for its OMML equation
missing, injected = [], 0
def repl(m):
    global injected
    name = m.group(1)
    f = os.path.join("eqs_omml", name + ".xml")
    if not os.path.exists(f):
        missing.append(name)
        return m.group(0)
    injected += 1
    return open(f, encoding="utf-8").read()

pattern = re.compile(r"<w:r>(?:(?!</w:r>).)*?EQOMML_(eq\d+)_END(?:(?!</w:r>).)*?</w:r>", re.S)
xml = pattern.sub(repl, xml)

leftover = re.findall(r"EQOMML_eq\d+_END", xml)
if missing or leftover:
    print("MISSING:", missing, "LEFTOVER:", leftover)
    sys.exit(1)

open(path, "w", encoding="utf-8").write(xml)

if os.path.exists(dst):
    os.remove(dst)
cwd = os.getcwd()
os.chdir(work)
subprocess.run(["zip", "-Xrq", os.path.join(cwd, dst), "."], check=True)
os.chdir(cwd)
shutil.rmtree(work)
print(f"injected {injected} equations -> {dst}")
