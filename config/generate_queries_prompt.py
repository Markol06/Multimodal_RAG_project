GENERATE_QUERIES_PROMPT = """
You are a search query generator.
Given the article title: "{readable_title}", generate 3 search queries:
1. Direct title-based query (close to original wording).
2. Paraphrased query with the same meaning.
3. Related but noisy query (different wording, partial topic relevance).

Respond ONLY with valid JSON in the format:
{"direct": "...", "paraphrased": "...", "noisy": "..."}
"""