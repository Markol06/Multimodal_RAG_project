from openai import OpenAI

from config import RAG_PROMPT, SYSTEM_PROMPT, CHAT_MODEL, TEMPERATURE_STRICT, TOP_K
from rag.retriever import retriever_instance

client = OpenAI()

def generate_answer(query: str, top_k: int = TOP_K) -> dict:
    results = retriever_instance.search_multimodal(query, top_k=top_k)

    if results.get("text"):
        results["text"] = sorted(
            results["text"],
            key=lambda r: r["score"],
            reverse=True
        )

    context_parts = []
    if results.get("text"):
        for i, r in enumerate(results["text"], 1):
            title = r.get("title", "Unknown")
            issue = r.get("issue", "")
            chunk = r.get("chunk", "")
            score = r.get("score", 0)
            context_parts.append(f"{i}. Article: {title} (Issue {issue})")
            context_parts.append(f"   Content: {chunk}")
            context_parts.append(f"   Relevance Score: {score:.4f}")
            context_parts.append("")

    context = "\n".join(context_parts) if context_parts else "No relevant context found."

    prompt_with_context = RAG_PROMPT.format(context=context, query=query)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_with_context}
        ],
        temperature=TEMPERATURE_STRICT
    )

    answer_text = response.choices[0].message.content

    if "I could not find this information in the provided context." in answer_text:
        results["main_image"] = None
        results["images"] = []
        results["text"] = []

    return {
        "query": query,
        "answer": answer_text,
        "results": results,
        "total_results": results.get("total_results", 0),
        "text_count": len(results.get("text", [])),
        "image_count": len(results.get("images", [])),
        "main_image": results.get("main_image")
    }
