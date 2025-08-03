import json

from pathlib import Path

from config import RAW_JSON, PROCESSED_JSON, RAW_IMAGES_DIR, PROCESSED_IMAGES_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from preprocessing.image_processor import process_image
from preprocessing.text_cleaner import clean_text

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks

def preprocess_articles():
    if not Path(RAW_JSON).exists():
        raise FileNotFoundError(f"File {RAW_JSON} not found")

    with open(RAW_JSON, "r", encoding="utf-8") as f:
        articles = json.load(f)

    Path(PROCESSED_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    processed_articles = []

    for article in articles:
        title = article.get("title", "").strip()
        content = clean_text(article.get("content", ""))

        chunks = chunk_text(content)

        image_filename = article.get("image_filename")
        processed_image_path = None

        if image_filename:
            raw_image_path = Path(RAW_IMAGES_DIR) / image_filename
            if raw_image_path.exists():
                processed_image_path = process_image(str(raw_image_path), str(PROCESSED_IMAGES_DIR))

        processed_articles.append({
            "issue": article.get("issue"),
            "title": title,
            "url": article.get("url"),
            "chunks": chunks,
            "image_url": article.get("image_url"),
            "image_path": processed_image_path
        })

    with open(PROCESSED_JSON, "w", encoding="utf-8") as f:
        json.dump(processed_articles, f, ensure_ascii=False, indent=2)

    print(f"Processing has been finished! Results in: {PROCESSED_JSON}")


if __name__ == "__main__":
    preprocess_articles()
