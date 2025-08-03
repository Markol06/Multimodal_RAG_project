import os
import re
import traceback

from PIL import Image

import streamlit as st

from rag import generate_answer

st.set_page_config(page_title="Multimodal RAG Search", layout="wide")

st.title("Multimodal RAG Search")
st.write("Search for your desired article.")

query = st.text_input("Enter your search query:")

def split_into_paragraphs(text: str, sentences_per_paragraph: int = 3) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    paragraphs = []
    for i in range(0, len(sentences), sentences_per_paragraph):
        paragraph = " ".join(sentences[i:i + sentences_per_paragraph])
        paragraphs.append(paragraph)
    return paragraphs


def load_image_from_repo(image_path: str):
    if not image_path:
        return None

    repo_path = os.path.join("data", "processed", "images", image_path)

    if os.path.exists(repo_path):
        return Image.open(repo_path)
    return None

if st.button("Search") and query.strip():
    with st.spinner("Searching and generating answer..."):
        try:
            result = generate_answer(query=query, top_k=5)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.text(traceback.format_exc())
            st.stop()

    st.subheader("GPT Answer")

    col_left, col_right = st.columns(2)

    with col_left:
        answer_text = result["answer"].strip()

        parts = answer_text.split("\n", 1)
        summary = parts[0].replace("### Summary", "").strip()
        full_article = parts[1].strip() if len(parts) > 1 else ""

        st.markdown("### Summary")
        st.write(summary)

        if full_article:
            for paragraph in split_into_paragraphs(full_article, sentences_per_paragraph=3):
                st.markdown(paragraph)

    with col_right:
        if result["results"]["images"]:
            first_image_path = result["results"]["images"][0].get("image_path", None)
            if first_image_path:
                st.image(os.path.join("data", "processed", "images", first_image_path), use_container_width=True)
            else:
                st.info("No image available.")
        else:
            st.info("No image available.")

    st.subheader("Search Results")

    col_text, col_img = st.columns(2)

    with col_text:
        st.markdown("### Top 5 Text Results")
        if result["results"]["text"]:
            for idx, item in enumerate(result["results"]["text"], 1):
                with st.container():
                    st.markdown(f"**{idx}. {item.get('title', 'No title')}**")
                    st.markdown(f"*Score:* `{item.get('score', 0):.4f}`")
                    st.write(item.get("chunk", ""))
                    st.markdown("---")
        else:
            st.info("No text results found.")

    with col_img:
        st.markdown("### Relevant Images")
        if result["results"]["images"]:
            for idx, item in enumerate(result["results"]["images"], 1):
                with st.container():
                    st.markdown(f"*Score:* `{item.get('score', 0):.4f}`")
                    img = load_image_from_repo(item.get("image_path", ""))
                    if img:
                        st.image(img, use_container_width=True)
                    else:
                        st.warning("No image path available.")
                    st.markdown("---")
        else:
            st.info("No image results found.")
