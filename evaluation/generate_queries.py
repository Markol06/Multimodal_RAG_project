import json
import re

from openai import OpenAI

from config import UNIFIED_METADATA_PATH, CHAT_MODEL, GENERATE_QUERIES_PROMPT, TEMPERATURE_CREATIVE

client = OpenAI()

def load_metadata(metadata_path: str):
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_unique_articles_from_ids(metadata: dict):
    articles = {}
    for chunk_id in metadata.get("ids", {}).get("text", []):
        article_title = chunk_id.rsplit("_chunk_", 1)[0]
        article_title = re.sub(r"^\d+_", "", article_title)
        articles.setdefault(article_title, []).append(chunk_id)
    return articles

def generate_queries(title: str):
    readable_title = title.replace("-", " ")
    prompt = GENERATE_QUERIES_PROMPT.format(readable_title=readable_title)
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE_CREATIVE
        )
        raw_content = response.choices[0].message.content.strip()

        raw_content = re.sub(r"^```json\s*|\s*```$", "", raw_content.strip(), flags=re.MULTILINE)

        queries = json.loads(raw_content)

    except Exception as e:
        print(f"Failed to generate query for '{title}': {e}")
        queries = {
            "direct": readable_title,
            "paraphrased": f"About {readable_title.lower()}",
            "noisy": f"Something related to {readable_title.lower()}"
        }

    return {k: v.replace("-", " ") for k, v in queries.items()}

def main():
    metadata = load_metadata(UNIFIED_METADATA_PATH)
    articles = extract_unique_articles_from_ids(metadata)

    test_queries = {}
    for article_title, chunk_ids in articles.items():
        queries = generate_queries(article_title)
        test_queries[queries["direct"]] = chunk_ids
        test_queries[queries["paraphrased"]] = chunk_ids
        test_queries[queries["noisy"]] = chunk_ids

    with open("generated_test_queries.json", "w", encoding="utf-8") as f:
        json.dump(test_queries, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(test_queries)} queries for {len(articles)} unique articles")

if __name__ == "__main__":
    main()
