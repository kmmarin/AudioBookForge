"""Cover art: normalize uploads and draw a plain cover when a book has none."""
from __future__ import annotations

import hashlib
import textwrap
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
PALETTE = [((31, 58, 82), (232, 163, 23)), ((58, 44, 74), (226, 186, 120)),
           ((28, 74, 64), (238, 214, 140)), ((86, 40, 36), (240, 200, 150)),
           ((40, 48, 58), (120, 190, 220))]


def normalize(data: bytes, size: int = 1400) -> bytes:
    image = Image.open(BytesIO(data))
    image = image.convert("RGB")
    image.thumbnail((size, size))
    out = BytesIO()
    image.save(out, "JPEG", quality=88, optimize=True)
    return out.getvalue()


def _font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size=size)


def generate(title: str, author: str = "", size: int = 1400) -> bytes:
    seed = int(hashlib.sha1(title.encode()).hexdigest(), 16)
    background, accent = PALETTE[seed % len(PALETTE)]
    image = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(image)
    margin = size // 10
    draw.rectangle([margin, margin, margin + size // 7, margin + 14], fill=accent)

    title_font = _font(FONT_BOLD, size // 11)
    lines = textwrap.wrap(title or "Untitled", width=16)[:5]
    y = margin + 80
    for line in lines:
        draw.text((margin, y), line, font=title_font, fill=(245, 245, 240))
        y += int(size / 11 * 1.18)
    if author:
        draw.text((margin, size - margin - size // 22), author[:40],
                  font=_font(FONT_REGULAR, size // 24), fill=accent)
    out = BytesIO()
    image.save(out, "JPEG", quality=88)
    return out.getvalue()
