import re

def clean_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text
