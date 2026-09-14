"""Create CPU contact sheets to select real paintings for presentation examples."""

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input.rglob("*.jpg"))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    catalog = []
    for i, path in enumerate(files, 1):
        with Image.open(path) as image:
            catalog.append(
                {"id": i, "path": str(path.resolve()), "name": path.stem, "size": list(image.size)}
            )
    for page in range(math.ceil(len(catalog) / 20)):
        sheet = Image.new("RGB", (1600, 1500), "#f3efe5")
        draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(catalog[page * 20 : (page + 1) * 20]):
            x, y = (j % 5) * 320, (j // 5) * 375
            with Image.open(row["path"]) as original:
                thumb = ImageOps.exif_transpose(original).convert("RGB")
                thumb.thumbnail((300, 325))
                sheet.paste(thumb, (x + (320 - thumb.width) // 2, y + 8))
            draw.text(
                (x + 10, y + 336),
                f"{row['id']:02}   {row['size'][0]} x {row['size'][1]}",
                fill="#263b32",
                font=font,
            )
        sheet.save(args.output / f"contact_{page + 1:02}.jpg", quality=92)
    (args.output / "catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Catalogued {len(catalog)} real input paintings")


if __name__ == "__main__":
    main()
