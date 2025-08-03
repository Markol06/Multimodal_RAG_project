from config import TOP_K
from rag import generate_answer

def main():
    query = input("Enter your search query: ").strip()
    if not query:
        print("Query cannot be empty.")
        return

    result = generate_answer(query=query, top_k=TOP_K)

    print("GPT Answer")
    print(result["answer"])

    print("Top 5 Text Results")
    if result["results"]["text"]:
        for idx, item in enumerate(result["results"]["text"], 1):
            print(f"{idx}. {item.get('title', 'No title')}")
            print(f"   Score: {item.get('score', 0):.4f}")
            print(f"   Content: {item.get('chunk', '')}\n")
    else:
        print("No text results found.")

    print("Relevant Images")
    if result["results"]["images"]:
        for idx, item in enumerate(result["results"]["images"], 1):
            print(f"{idx}. Image - {item.get('title', 'No title')}")
            print(f"   Score: {item.get('score', 0):.4f}")
            print(f"   Path: {item.get('image_path', '')}\n")
    else:
        print("No image results found.")

if __name__ == "__main__":
    main()
