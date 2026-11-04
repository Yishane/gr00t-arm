"""Contact sheet of stills/*.png -> stills/sheet.png (for checking)."""
import glob, sys
from PIL import Image, ImageDraw
fs = sorted(glob.glob("stills/t*.png"))
cols = 3
tw, th = 640, 360
sheet = Image.new("RGB", (cols * tw, ((len(fs) + cols - 1) // cols) * th), "grey")
for i, f in enumerate(fs):
    im = Image.open(f).resize((tw - 4, th - 4))
    sheet.paste(im, ((i % cols) * tw + 2, (i // cols) * th + 2))
    ImageDraw.Draw(sheet).text(((i % cols) * tw + 8, (i // cols) * th + 6), f[8:-4], fill="red")
sheet.save("stills/sheet.png")
