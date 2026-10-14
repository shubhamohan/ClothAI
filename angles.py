"""Take the fabric (texture + colours) from an AI-generated image and show it on garment templates."""
import base64
import colorsys
import io

from PIL import Image

from garments import FLARE, GARMENTS, SLEEVE, render_garment

KEYWORDS = [("lehenga", "Lehenga Choli"), ("anarkali", "Anarkali Suit"), ("saree", "Saree"), ("sari", "Saree"),
            ("kurti", "Kurti"), ("gown", "Festive Gown"), ("long frock", "Long Frock"), ("frock", "Short Frock"),
            ("t-shirt", "T-Shirt"), ("tshirt", "T-Shirt"), ("jacket", "Jacket"), ("shirt", "Shirt"),
            ("skirt", "Skirt & Top"), ("dress", "Modern Dress")]


def guess_garment(prompt):
    p = (prompt or "").lower()
    return next((g for k, g in KEYWORDS if k in p), "Modern Dress")


def _hsv(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h % 1, min(max(s, 0), 1), min(max(v, 0), 1))
    return (int(r * 255), int(g * 255), int(b * 255))


def _palette(img):
    q = img.convert("RGB").quantize(6, method=Image.MEDIANCUT)
    pal = q.getpalette()
    cols = [tuple(pal[i * 3:i * 3 + 3]) for _, i in sorted(q.getcolors(), reverse=True)]
    # skip near-white / near-grey studio background when picking the main fabric colour
    bg = lambda c: colorsys.rgb_to_hsv(*[x / 255 for x in c])[1] < 0.12 and max(c) > 200
    base = next((c for c in cols if not bg(c)), cols[0])
    dist = lambda a, b: sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
    sat = lambda c: colorsys.rgb_to_hsv(*[x / 255 for x in c])[1]
    others = [c for c in cols if dist(c, base) > 60 and not bg(c)] or [tuple(255 - x for x in base)]
    accent = max(others, key=lambda c: sat(c) * 0.7 + dist(c, base) / 441 * 0.3)
    h, s, v = colorsys.rgb_to_hsv(*[x / 255 for x in base])
    return dict(base=base, light=_hsv(h, s * 0.4, v + 0.3), dark=_hsv(h, s + 0.1, v * 0.6),
                accent=accent, gold=(212, 175, 55))


def _best_crop(img):
    """Slide a window over the middle of the image and keep the one with the least background."""
    import numpy as np
    im = img.convert("RGB")
    w, h = im.size
    a = np.asarray(im.resize((96, 96)), float) / 255
    mx, mn = a.max(2), a.min(2)
    sat = (mx - mn) / (mx + 1e-6)
    bad = ((sat < 0.12) & (mx > 0.78)) | (mx < 0.12)          # white/grey backdrop or black outline
    best, best_xy = None, (0.38, 0.38)
    for y0 in np.arange(0.30, 0.56, 0.04):
        for x0 in np.arange(0.28, 0.56, 0.04):
            win = bad[int(y0 * 96):int((y0 + 0.2) * 96), int(x0 * 96):int((x0 + 0.2) * 96)]
            sc = (win.mean(), -a[int(y0 * 96):int((y0 + 0.2) * 96), int(x0 * 96):int((x0 + 0.2) * 96)].std())
            if best is None or sc < best:
                best, best_xy = sc, (x0, y0)
    x0, y0 = best_xy
    return im.crop((int(x0 * w), int(y0 * h), int((x0 + 0.2) * w), int((y0 + 0.2) * h)))


def extract_fabric(img):
    """Pick a clean fabric patch from the image and mirror-tile it so it repeats seamlessly."""
    crop = _best_crop(img)
    patch = crop.resize((128, 128), Image.LANCZOS)
    tile = Image.new("RGB", (256, 256))
    tile.paste(patch, (0, 0))
    tile.paste(patch.transpose(Image.FLIP_LEFT_RIGHT), (128, 0))
    tile.paste(patch.transpose(Image.FLIP_TOP_BOTTOM), (0, 128))
    tile.paste(patch.transpose(Image.ROTATE_180), (128, 128))
    pal = _palette(crop)
    buf = io.BytesIO()
    tile.save(buf, "JPEG", quality=88)
    return dict(tile=tile, pal=pal, b64="data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())


def design_for(garment, fabric):
    sp = GARMENTS[garment]
    return dict(garment=garment, neck=sp["nk"], sleeve=sp["sl"], flare=sp["fl"],
                _tile=fabric["tile"], _palette=fabric["pal"])


def render_views(garment, fabric):
    d = design_for(garment, fabric)
    return [render_garment(d, a) for a in (0, 90, 180)]