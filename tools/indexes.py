import json
import re

from pathlib import Path

import faiss
import numpy as np

from config import (
    TEXT_EMBEDDINGS_PATH,
    IMAGE_EMBEDDINGS_PATH,
    TEXT_INDEX_PATH,
    IMAGE_INDEX_PATH,
    UNIFIED_METADATA_PATH
)

def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-')

def build_separate_indexes():
    text_data = json.loads(Path(TEXT_EMBEDDINGS_PATH).read_text(encoding="utf-8"))
    image_data = json.loads(Path(IMAGE_EMBEDDINGS_PATH).read_text(encoding="utf-8"))

    grouped_issues = {}
    text_ids, image_ids, types = [], [], []

    text_counter = {}
    text_vectors = []
    for txt in text_data:
        issue = txt["metadata"]["issue"]
        title = txt["metadata"]["title"]
        title_slug = slugify(title)
        key = (issue, title_slug)
        text_counter[key] = text_counter.get(key, 0)

        item_id = f"{issue}_{title_slug}_chunk_{text_counter[key]}"
        text_counter[key] += 1

        text_ids.append(item_id)
        types.append("text")
        text_vectors.append(txt["embedding"])

        issue_str = str(issue)
        grouped_issues.setdefault(issue_str, {}).setdefault(title, {"text": [], "image": []})
        grouped_issues[issue_str][title]["text"].append({
            "id": item_id,
            "chunk": txt["metadata"].get("chunk"),
            "url": txt["metadata"].get("url"),
            "content_type": "text"
        })

    image_counter = {}
    image_vectors = []
    for img in image_data:
        issue = img["metadata"]["issue"]
        title = img["metadata"]["title"]
        title_slug = slugify(title)
        key = (issue, title_slug)
        image_counter[key] = image_counter.get(key, 0)

        item_id = f"{issue}_{title_slug}_image_{image_counter[key]}"
        image_counter[key] += 1

        image_ids.append(item_id)
        types.append("image")
        image_vectors.append(img["embedding"])

        issue_str = str(issue)
        grouped_issues.setdefault(issue_str, {}).setdefault(title, {"text": [], "image": []})
        grouped_issues[issue_str][title]["image"].append({
            "id": item_id,
            "image_path": img["metadata"].get("image_path"),
            "content_type": "image"
        })

    if text_vectors:
        text_vectors = np.array(text_vectors, dtype="float32")
        faiss.normalize_L2(text_vectors)
        text_index = faiss.IndexFlatIP(text_vectors.shape[1])
        text_index.add(text_vectors)
        faiss.write_index(text_index, str(TEXT_INDEX_PATH))
        print(f"Text index saved: {TEXT_INDEX_PATH}")

    if image_vectors:
        image_vectors = np.array(image_vectors, dtype="float32")
        faiss.normalize_L2(image_vectors)
        image_index = faiss.IndexFlatIP(image_vectors.shape[1])
        image_index.add(image_vectors)
        faiss.write_index(image_index, str(IMAGE_INDEX_PATH))
        print(f"Image index saved: {IMAGE_INDEX_PATH}")

    with open(UNIFIED_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "issues": grouped_issues,
            "ids": {
                "text": text_ids,
                "image": image_ids
            },
            "types": types
        }, f, ensure_ascii=False, indent=2)

    print(f"Unified metadata saved: {UNIFIED_METADATA_PATH}")

def run_index_building():
    print("Building separate text and image indexes with grouped metadata...")
    build_separate_indexes()
    print("Index building completed!")


if __name__ == "__main__":
    run_index_building()
