from pathlib import Path
from typing import Optional

import numpy as np

import faiss

from config import TOP_K

class VectorStore:
    def __init__(self):
        self.text_index = None
        self.image_index = None
        self.text_metadata = []
        self.image_metadata = []

    def load_from_file(
        self,
        text_index_path: Path,
        image_index_path: Path,
        text_metadata: Optional[list[dict]] = None,
        image_metadata: Optional[list[dict]] = None
    ) -> None:
        if text_index_path.exists():
            self.text_index = faiss.read_index(str(text_index_path))
        if image_index_path.exists():
            self.image_index = faiss.read_index(str(image_index_path))
        if text_metadata:
            self.text_metadata = text_metadata
        if image_metadata:
            self.image_metadata = image_metadata

    def search_text(self, query_vector: list[float], top_k: int = TOP_K) -> list[dict]:
        if self.text_index is None:
            raise ValueError("Text index not loaded")
        D, I = self.text_index.search(np.array([query_vector], dtype="float32"), top_k)
        return [{"score": float(D[0][i]), **self.text_metadata[I[0][i]]} for i in range(len(I[0]))]

    def search_image(self, query_vector: list[float], top_k: int = TOP_K) -> list[dict]:
        if self.image_index is None:
            raise ValueError("Image index not loaded")
        D, I = self.image_index.search(np.array([query_vector], dtype="float32"), top_k)
        return [{"score": float(D[0][i]), **self.image_metadata[I[0][i]]} for i in range(len(I[0]))]
