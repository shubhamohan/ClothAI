"""Parametric 2D garment templates (front/back/side + pseudo-3D turntable) painted with procedural fabric."""
import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter


W, H, CX = 600, 900, 300
NECK = [0.10, 0.30, 0.50, 0.75]     # neckline depth levels
SLEEVE = [0.0, 0.30, 0.65, 1.0]     # sleeveless, cap, 3/4, full
FLARE = [0.10, 0.40, 0.70, 1.0]     # skirt/hem flare levels

# hem = hem line (y), wh = waist half-width, fmax = max extra hem width,
# nk/sl/fl = default level indices, extra = special details
GARMENTS = {
    "T-Shirt":       dict(hem=520, wh=92, fmax=6,   nk=0, sl=1, fl=0, extra=None),
    "Shirt":         dict(hem=560, wh=90, fmax=10,  nk=1, sl=3, fl=0, extra="collar"),
    "Jacket":        dict(hem=580, wh=92, fmax=14,  nk=1, sl=3, fl=0, extra="open"),
    "Kurti":         dict(hem=700, wh=78, fmax=40,  nk=1, sl=2, fl=0, extra="border"),
    "Modern Dress":  dict(hem=700, wh=62, fmax=110, nk=1, sl=1, fl=1, extra=None),
    "Short Frock":   dict(hem=600, wh=62, fmax=150, nk=1, sl=0, fl=2, extra=None),
    "Long Frock":    dict(hem=840, wh=62, fmax=190, nk=1, sl=0, fl=2, extra=None),
    "Skirt & Top":   dict(hem=700, wh=64, fmax=120, nk=1, sl=1, fl=1, extra="belt"),
    "Saree":         dict(hem=850, wh=64, fmax=60,  nk=1, sl=1, fl=0, extra="saree"),
    "Anarkali Suit": dict(hem=830, wh=60, fmax=230, nk=1, sl=3, fl=2, extra="border"),
    "Lehenga Choli": dict(hem=850, wh=62, fmax=240, nk=1, sl=1, fl=2, extra="lehenga"),
    "Festive Gown":  dict(hem=860, wh=60, fmax=210, nk=2, sl=0, fl=2, extra="border"),
}
SKIN = (232, 216, 202)


def _geometry(design, back):
    sp = GARMENTS[design["garment"]]
    sh_y, waist_y = 170, 390
    depth = NECK[design["neck"]] * (0.35 if back else 1.0) * 110 + 8
    wh = sp["wh"]
    hh = wh + FLARE[design["flare"]] * sp["fmax"]
    hem = sp["hem"]
    pts = [(CX - 38, sh_y), (CX - 110, sh_y + 16), (CX - 98, 265), (CX - wh, waist_y), (CX - hh, hem)]
    pts += [(CX + t * hh, hem + 14 * (1 - t * t)) for t in np.linspace(-1, 1, 9)]
    pts += [(CX + hh, hem), (CX + wh, waist_y), (CX + 98, 265), (CX + 110, sh_y + 16), (CX + 38, sh_y)]
    pts += [(CX + 38 * t, sh_y + depth * (1 - t * t) ** 0.9) for t in np.linspace(1, -1, 11)]
    sleeves = []
    L = SLEEVE[design["sleeve"]] * 330
    if L > 0:
        for s in (-1, 1):
            out = 118 + 0.2 * L
            ey = sh_y + 16 + max(L, 40)
            sleeves.append([(CX + s * 110, sh_y + 16), (CX + s * out, ey),
                            (CX + s * (out - 52), ey), (CX + s * 98, 265)])
    return dict(body=pts, sleeves=sleeves, wh=wh, hh=hh, hem=hem, waist_y=waist_y,
                sh_y=sh_y, depth=depth, extra=sp["extra"])


def _mask(polys, size=(W, H)):
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    for p in polys:
        d.polygon(p, fill=255)
    return m


def _garment_layer(design, back):
    g = _geometry(design, back)
    pal, tile = design["_palette"], design["_tile"]
    arr = np.asarray(tile.convert("RGB"))
    fabric = np.tile(arr, (math.ceil(H / 256), math.ceil(W / 256), 1))[:H, :W].astype(float)
    mask = _mask([g["body"]] + g["sleeves"])
    # soft edge shading so the cloth looks rounded
    edge = np.asarray(mask.filter(ImageFilter.GaussianBlur(11)), float) / 255
    fabric *= (0.72 + 0.28 * edge)[..., None]
    # fabric folds on flared skirts
    folds = Image.new("L", (W, H), 0)
    fd = ImageDraw.Draw(folds)
    for k in range(-5, 6):
        fd.line([(CX + k * g["wh"] / 5.5, g["waist_y"]), (CX + k * g["hh"] / 5.5, g["hem"])], fill=70, width=3)
    folds = folds.filter(ImageFilter.GaussianBlur(3))
    fabric *= (1 - np.asarray(folds, float) / 255 * 0.45)[..., None]
    img = Image.fromarray(np.clip(fabric, 0, 255).astype(np.uint8))

    def paint(shape_mask, color):
        img.paste(color, mask=ImageChops.multiply(shape_mask, mask))

    ex = g["extra"]
    if ex in ("border", "lehenga", "saree"):
        strip = _mask([[(0, g["hem"] - 30), (W, g["hem"] - 30), (W, g["hem"] + 20), (0, g["hem"] + 20)]])
        paint(strip, pal["accent"])
        paint(_mask([[(0, g["hem"] - 20), (W, g["hem"] - 20), (W, g["hem"] - 14), (0, g["hem"] - 14)]]), pal["gold"])
    if ex == "belt":
        paint(_mask([[(0, g["waist_y"] - 10), (W, g["waist_y"] - 10), (W, g["waist_y"] + 12), (0, g["waist_y"] + 12)]]), pal["dark"])
    d = ImageDraw.Draw(img)
    if ex in ("saree", "lehenga") and not back:
        s = 1 if ex == "saree" else -1
        band = [(CX - s * 105, 190), (CX - s * 50, 180), (CX + s * 150, 640), (CX + s * 85, 660)]
        img.paste(pal["accent"], mask=_mask([band]))
        d.line([(CX - s * 78, 185), (CX + s * 118, 650)], fill=pal["gold"], width=4)
    if ex == "collar" and not back:
        sy = g["sh_y"]
        for s in (-1, 1):
            d.polygon([(CX + s * 38, sy), (CX, sy + 62), (CX + s * 14, sy - 20)], fill=pal["dark"])
    if ex == "open" and not back:
        d.line([(CX, g["sh_y"] + g["depth"]), (CX, g["hem"] + 14)], fill=pal["dark"], width=4)
        for s in (-1, 1):
            d.polygon([(CX + s * 38, g["sh_y"]), (CX, g["sh_y"] + 110), (CX + s * 8, g["sh_y"] + 20)], fill=pal["dark"])
    outline = ImageChops.subtract(mask, mask.filter(ImageFilter.MinFilter(5)))
    img.paste(tuple(int(c * 0.45) for c in pal["base"]), mask=outline)
    return img, mask, g


def render_garment(design, angle=0):
    """Return an RGB image. angle: 0 front, 90 side, 180 back, any value = pseudo-3D turntable."""
    a = angle % 360
    back = 90 < a < 270
    sx = max(0.42, abs(math.cos(math.radians(a))))
    garment, mask, g = _garment_layer(design, back)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    # mannequin: head, neck, legs, stand
    ld.ellipse([CX - 40, 70, CX + 40, 150], fill=SKIN)
    ld.rectangle([CX - 16, 140, CX + 16, 190], fill=SKIN)
    if g["hem"] < 830:
        for s in (-1, 1):
            ld.rectangle([CX + s * 28 - 15, g["hem"], CX + s * 28 + 15, 860], fill=SKIN)
    ld.ellipse([CX - 120, 860, CX + 120, 890], fill=(205, 195, 215))
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow.paste((60, 40, 90, 90), mask=mask.filter(ImageFilter.GaussianBlur(8)).transform((W, H), Image.AFFINE, (1, 0, -6, 0, 1, -8)))
    layer = Image.alpha_composite(layer, shadow)
    layer.paste(garment, mask=mask)
    layer = layer.resize((max(1, int(W * sx)), H), Image.LANCZOS)
    bg = Image.new("RGBA", (W, H))
    bd = ImageDraw.Draw(bg)
    for y in range(H):
        t = y / H
        bd.line([(0, y), (W, y)], fill=(int(253 - 8 * t), int(242 - 0 * t), int(248 - 4 * t), 255))
    bg.alpha_composite(layer, ((W - layer.width) // 2, 0))
    return bg.convert("RGB")