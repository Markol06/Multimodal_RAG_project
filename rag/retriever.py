import json

import faiss
import numpy as np

from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel

from config import (
    TEXT_INDEX_PATH,
    IMAGE_INDEX_PATH,
    UNIFIED_METADATA_PATH,
    IMAGE_EMBEDDING_MODEL,
    TEXT_EMBEDDING_MODEL,
    TOP_K,
)

class MultimodalRetriever:
    def __init__(self):
        self.client = OpenAI()

        self.text_index = faiss.read_index(str(TEXT_INDEX_PATH))
        self.image_index = faiss.read_index(str(IMAGE_INDEX_PATH))

        with open(UNIFIED_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata_data = json.load(f)

        self.issues = metadata_data.get("issues", {})
        self.ids = metadata_data.get("ids", {})
        self.types = metadata_data.get("types", [])

        self.clip_model = CLIPModel.from_pretrained(IMAGE_EMBEDDING_MODEL)
        self.clip_processor = CLIPProcessor.from_pretrained(IMAGE_EMBEDDING_MODEL)

        print(f"Loaded text index with {len(self.ids['text'])} text chunks")
        print(f"Loaded image index with {len(self.ids['image'])} images")

    def embed_text_openai(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model=TEXT_EMBEDDING_MODEL,
            input=text
        )
        vector = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(vector)
        return vector

    def embed_text_clip(self, text: str) -> np.ndarray:
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
        text_features = self.clip_model.get_text_features(**inputs)
        vector = text_features.detach().numpy().astype("float32")
        faiss.normalize_L2(vector)
        return vector

    def search_multimodal(self, query: str, top_k: int = TOP_K) -> dict:
        query_vector_text = self.embed_text_openai(query)
        D, I = self.text_index.search(query_vector_text, top_k)

        if I.size == 0:
            return {"text": [], "images": [], "query": query, "total_results": 0, "context": ""}

        text_results = []
        for rank, idx in enumerate(I[0]):
            if idx == -1:
                continue
            text_id = self.ids["text"][idx]

            found_meta = None
            for issue_num, articles in self.issues.items():
                for title, content in articles.items():
                    for txt in content.get("text", []):
                        if txt["id"] == text_id:
                            found_meta = {
                                "issue": issue_num,
                                "title": title,
                                **txt
                            }
                            break
                    if found_meta:
                        break
                if found_meta:
                    break

            if found_meta:
                text_results.append({
                    "id": text_id,
                    "score": float(D[0][rank]),
                    "rank": rank + 1,
                    **found_meta
                })

        if not text_results:
            return {"text": [], "images": [], "query": query, "total_results": 0, "context": ""}

        main_article_title = text_results[0]["title"]
        main_issue = text_results[0]["issue"]

        article_images = self.issues[main_issue][main_article_title].get("image", [])

        image_results = []
        markdown_image = ""
        if article_images:
            title_vector = self.embed_text_clip(main_article_title)

            image_indices = []
            for img in article_images:
                if img["id"] in self.ids["image"]:
                    image_indices.append(self.ids["image"].index(img["id"]))

            if image_indices:
                sub_image_index = faiss.IndexFlatIP(self.image_index.d)
                sub_embeds = np.zeros((len(image_indices), self.image_index.d), dtype="float32")
                for i, idx in enumerate(image_indices):
                    self.image_index.reconstruct(idx, sub_embeds[i])
                sub_image_index.add(sub_embeds)

                D_img, I_img = sub_image_index.search(title_vector, len(image_indices))

                for rank, local_idx in enumerate(I_img[0]):
                    global_idx = image_indices[local_idx]
                    img_id = self.ids["image"][global_idx]
                    img_meta = next((img for img in article_images if img["id"] == img_id), None)
                    if img_meta:
                        image_results.append({
                            "id": img_id,
                            "score": float(D_img[0][rank]),
                            "rank": rank + 1,
                            **img_meta
                        })

                if image_results and "image_path" in image_results[0]:
                    markdown_image = f"![{main_article_title}]({image_results[0]['image_path']})\n\n"

        context_parts = []
        if markdown_image:
            context_parts.append(markdown_image)
        for tr in text_results:
            context_parts.append(f"Chunk: {tr.get('chunk', '')}")

        context = "\n".join(context_parts)
        main_image = image_results[0]["image_path"] if image_results and "image_path" in image_results[0] else None

        return {
            "text": text_results,
            "images": image_results,
            "query": query,
            "total_results": len(text_results) + len(image_results),
            "context": context,
            "main_image": main_image
        }

retriever_instance = MultimodalRetriever()
