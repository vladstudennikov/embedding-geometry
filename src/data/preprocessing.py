import re

# This is quite simple preprocessing function, you can expand it as needed (e.g., removing stop words, stemming, etc.)
def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text