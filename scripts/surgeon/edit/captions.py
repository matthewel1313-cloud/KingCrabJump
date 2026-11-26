"""Caption overlays: bold white text with a black outline, colour emoji inline.

Text is set in Inter Bold (SIL OFL); emoji in Noto Color Emoji (SIL OFL),
which ships only as 109 px bitmaps, so emoji are drawn at 109 px and scaled
to the line height.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFilter, ImageFont

TEXT_FONT = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
EMOJI_FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"


def is_emoji(ch: str) -> bool:
    cp = ord(ch)
    return (cp >= 0x1F000 or 0x2600 <= cp <= 0x27BF or cp in (0x2705, 0xFE0F, 0x200D)
            or unicodedata.category(ch) == "So")


def runs(text: str):
    out, cur, kind = [], "", None
    for ch in text:
        k = "e" if is_emoji(ch) else "t"
        if ch == "️":
            k = "e"
        if kind is not None and k != kind:
            out.append((kind, cur))
            cur = ""
        cur += ch
        kind = k
    if cur:
        out.append((kind, cur))
    return out


@lru_cache(maxsize=8)
def tfont(size: int):
    return ImageFont.truetype(TEXT_FONT, size)


@lru_cache(maxsize=1)
def efont():
    return ImageFont.truetype(EMOJI_FONT, 109)


def emoji_img(s: str, h: int) -> Image.Image:
    s = s.replace("️", "")
    f = efont()
    imgs = []
    for ch in s:
        im = Image.new("RGBA", (136, 128), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((0, 0), ch, font=f, embedded_color=True)
        bbox = im.getbbox() or (0, 0, 1, 1)
        im = im.crop(bbox)
        sc = h / im.height
        imgs.append(im.resize((max(1, int(im.width * sc)), h), Image.LANCZOS))
    w = sum(i.width for i in imgs) + 6 * (len(imgs) - 1)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    x = 0
    for i in imgs:
        out.paste(i, (x, 0), i)
        x += i.width + 6
    return out


def line_img(text: str, size: int, stroke: int) -> Image.Image:
    f = tfont(size)
    asc, desc = f.getmetrics()
    h = asc + desc + 2 * stroke
    parts = []
    for kind, s in runs(text):
        if kind == "t":
            w = int(f.getlength(s)) + 2 * stroke
            im = Image.new("RGBA", (max(1, w), h), (0, 0, 0, 0))
            ImageDraw.Draw(im).text((stroke, stroke), s, font=f, fill=(255, 255, 255, 255),
                                    stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
            parts.append(im)
        else:
            eh = int(size * 1.05)
            em = emoji_img(s, eh)
            pad = Image.new("RGBA", (em.width + 8, h), (0, 0, 0, 0))
            pad.paste(em, (4, max(0, (h - eh) // 2 - stroke // 2)), em)
            parts.append(pad)
    w = sum(p.width for p in parts)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    x = 0
    for p in parts:
        out.paste(p, (x, 0), p)
        x += p.width
    return out


def caption(text: str, size: int = 92, stroke: int = 9, max_w: int = 960) -> Image.Image:
    """A caption block (possibly several lines, split on '\\n'), with a soft
    drop shadow, centred."""
    lines = [line_img(t, size, stroke) for t in text.split("\n")]
    for i, im in enumerate(lines):
        if im.width > max_w:
            sc = max_w / im.width
            lines[i] = im.resize((max_w, int(im.height * sc)), Image.LANCZOS)
    gap = int(size * 0.1)
    W = max(i.width for i in lines)
    H = sum(i.height for i in lines) + gap * (len(lines) - 1)
    out = Image.new("RGBA", (W + 40, H + 40), (0, 0, 0, 0))
    y = 20
    for im in lines:
        out.paste(im, (20 + (W - im.width) // 2, y), im)
        y += im.height + gap
    shadow = Image.new("RGBA", out.size, (0, 0, 0, 0))
    a = out.split()[3].point(lambda v: int(v * 0.45))
    shadow.putalpha(a.filter(ImageFilter.GaussianBlur(8)))
    final = Image.new("RGBA", out.size, (0, 0, 0, 0))
    final.paste(shadow, (4, 6), shadow)
    final.alpha_composite(out)
    return final


def place(frame: Image.Image, cap: Image.Image, cy: int, scale: float = 1.0,
          alpha: float = 1.0) -> None:
    """Composite `cap` centred at x, vertically centred at `cy`."""
    if scale != 1.0:
        cap = cap.resize((max(1, int(cap.width * scale)), max(1, int(cap.height * scale))),
                         Image.BILINEAR)
    if alpha < 1.0:
        a = cap.split()[3].point(lambda v: int(v * alpha))
        cap = cap.copy()
        cap.putalpha(a)
    x = (frame.width - cap.width) // 2
    y = int(cy - cap.height / 2)
    frame.alpha_composite(cap, (x, y))
