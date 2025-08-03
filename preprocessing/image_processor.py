import os

from PIL import Image
from typing import Optional

def process_image(image_path: str, output_dir: str, size: tuple[int, int] = (512, 512)) -> Optional[str]:
    if not os.path.exists(image_path):
        return None

    os.makedirs(output_dir, exist_ok=True)

    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail(size)
        filename = os.path.splitext(os.path.basename(image_path))[0] + ".jpg"
        output_path = os.path.join(output_dir, filename)
        img.save(output_path, format="JPEG", quality=90)
        return output_path
    except Exception as e:
        print(f"Image processing error {image_path}: {e}")
        return None
