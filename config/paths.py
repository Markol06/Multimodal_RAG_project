from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_JSON = BASE_DIR / "data" / "raw" / "News.json"
PROCESSED_JSON = BASE_DIR / "data" / "processed" / "News_processed.json"

RAW_IMAGES_DIR = BASE_DIR / "data" / "raw" / "images"
PROCESSED_IMAGES_DIR = BASE_DIR / "data" / "processed" / "images"

TEXT_EMBEDDINGS_PATH = BASE_DIR / "data" / "embeddings" / "text_embeddings.json"
IMAGE_EMBEDDINGS_PATH = BASE_DIR / "data" / "embeddings" / "image_embeddings.json"

TEXT_INDEX_PATH = BASE_DIR / "data" / "indexes" / "text.index"
IMAGE_INDEX_PATH = BASE_DIR / "data" / "indexes" / "image.index"

UNIFIED_METADATA_PATH = BASE_DIR / "data" / "indexes" / "unified_metadata.json"

QUERY_EXPANSION_PATH = BASE_DIR / "evaluation" / "generated_test_queries.json"
