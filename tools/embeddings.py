import json
import os

from pathlib import Path

from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel

from config import (
    IMAGE_EMBEDDING_MODEL,
    TEXT_EMBEDDING_MODEL,
    PROCESSED_JSON,
    TEXT_EMBEDDINGS_PATH,
    IMAGE_EMBEDDINGS_PATH
)


class MultimodalEmbeddings:
    def __init__(self):
        self.client = OpenAI()
        self.clip_model = CLIPModel.from_pretrained(IMAGE_EMBEDDING_MODEL)
        self.clip_processor = CLIPProcessor.from_pretrained(IMAGE_EMBEDDING_MODEL)
        self.processed_data = None
        self.text_embeddings = []
        self.image_embeddings = []

    def load_data(self):
        data_path = Path(PROCESSED_JSON)
        self.processed_data = json.loads(data_path.read_text(encoding="utf-8"))
        print(f"Loaded {len(self.processed_data)} articles")

    def embed_text_openai(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=TEXT_EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding

    def embed_text_clip(self, text: str) -> list[float]:
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
        text_features = self.clip_model.get_text_features(**inputs)
        return text_features.detach().numpy().flatten().tolist()

    def process_article(self, article: dict):
        issue = article["issue"]
        title = article["title"]
        url = article["url"]

        for idx, chunk in enumerate(article.get("chunks", [])):
            weighted_text = (title + " ") * 3 + chunk
            embedding = self.embed_text_openai(weighted_text)
            self.text_embeddings.append({
                "id": f"{issue}_chunk_{idx}",
                "embedding": embedding,
                "metadata": {
                    "issue": issue,
                    "title": title,
                    "url": url,
                    "chunk": chunk,
                    "content_type": "text"
                }
            })

        img_path = article.get("image_path")
        if img_path:
            embedding = self.embed_text_clip(title)  # не картинка, а title
            self.image_embeddings.append({
                "id": f"{issue}_image",
                "embedding": embedding,
                "metadata": {
                    "issue": issue,
                    "title": title,
                    "url": url,
                    "image_path": os.path.basename(img_path) if img_path else None,
                    "content_type": "image"
                }
            })

    def create_embeddings(self):
        print("Creating separate text and image embeddings...")
        for article in self.processed_data:
            self.process_article(article)
        print(f"Created {len(self.text_embeddings)} text embeddings and {len(self.image_embeddings)} image embeddings")

    def save_embeddings(self):
        Path(TEXT_EMBEDDINGS_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(IMAGE_EMBEDDINGS_PATH).parent.mkdir(parents=True, exist_ok=True)

        with open(TEXT_EMBEDDINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.text_embeddings, f, ensure_ascii=False, indent=2)
        print(f"Text embeddings saved: {TEXT_EMBEDDINGS_PATH}")

        with open(IMAGE_EMBEDDINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.image_embeddings, f, ensure_ascii=False, indent=2)
        print(f"Image embeddings saved: {IMAGE_EMBEDDINGS_PATH}")

    def run_indexing(self):
        self.load_data()
        self.create_embeddings()
        self.save_embeddings()


def main():
    indexer = MultimodalEmbeddings()
    indexer.run_indexing()


if __name__ == "__main__":
    main()
