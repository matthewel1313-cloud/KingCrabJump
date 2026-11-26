"""Tile PNGs into one contact sheet: python sheet.py OUT.png A.png B.png ... [--cols 3 --w 320]"""
import sys
from PIL import Image
args = sys.argv[1:]; cols, w = 3, 320
if "--cols" in args: i = args.index("--cols"); cols = int(args[i+1]); del args[i:i+2]
if "--w" in args: i = args.index("--w"); w = int(args[i+1]); del args[i:i+2]
out, files = args[0], args[1:]
ims = [Image.open(f).convert("RGB") for f in files]
h = int(w * ims[0].height / ims[0].width)
rows = (len(ims) + cols - 1) // cols
S = Image.new("RGB", (w * cols, h * rows), "white")
for k, im in enumerate(ims):
    S.paste(im.resize((w, h)), ((k % cols) * w, (k // cols) * h))
S.save(out)
