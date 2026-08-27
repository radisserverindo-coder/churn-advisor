from pathlib import Path
import re
import requests


# Folder: project/knowledge/
BASE_DIR = Path(__file__).resolve().parent

# Knowledge base file: project/knowledge/retention_knowledge.txt
KNOWLEDGE_FILE = BASE_DIR / "retention_knowledge.txt"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_knowledge():
    """Load the retention knowledge base and split it into chunks."""
    if not KNOWLEDGE_FILE.exists():
        return []

    text = KNOWLEDGE_FILE.read_text(encoding="utf-8").strip()

    if not text:
        return []

    # Split on numbered sections such as:
    # 1. ...
    # 2. ...
    # 3. ...
    sections = re.split(r"(?m)(?=^\s*\d+\.\s+)", text)

    chunks = []
    for section in sections:
        section = section.strip()
        if section:
            chunks.append(section)

    # If the knowledge file does not use numbered sections,
    # keep the whole text as one usable chunk.
    if not chunks:
        chunks = [text]

    return chunks


def keyword_score(query, text):
    """Calculate a simple keyword relevance score."""
    query_words = {
        word.lower()
        for word in re.findall(r"\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*\b", query)
        if len(word) > 2
    }

    text_words = {
        word.lower()
        for word in re.findall(r"\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*\b", text)
    }

    if not query_words:
        return 0

    return len(query_words.intersection(text_words))


def retrieve_chunks(query, top_k=3):
    """Retrieve the most relevant knowledge chunks."""
    chunks = load_knowledge()

    if not chunks:
        return []

    scored_chunks = []

    for chunk in chunks:
        score = keyword_score(query, chunk)

        scored_chunks.append(
            {
                "text": chunk,
                "score": score,
            }
        )

    scored_chunks.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Only return genuinely relevant chunks.
    relevant = [
        item
        for item in scored_chunks
        if item["score"] > 0
    ]

    return relevant[:top_k]


def ask_openrouter(
    question,
    context="",
    model="openai/gpt-4o-mini",
    api_key=None,
):
    """Send the question to OpenRouter."""
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY belum tersedia.")

    if context:
        prompt = f"""
You are a Telecom Customer Retention AI Advisor.

Answer the user's question using the provided knowledge base.

IMPORTANT:
- Do not invent company policies.
- Use the knowledge base when relevant.
- Give practical retention recommendations.
- If the knowledge base does not contain enough information,
  clearly say so.

KNOWLEDGE BASE:
{context}

USER QUESTION:
{question}
"""
    else:
        prompt = f"""
You are a Telecom Customer Retention AI Advisor.

Answer the user's question clearly and concisely.

USER QUESTION:
{question}
"""

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.2,
        },
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    answer = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    return {
        "answer": answer,
        "usage": usage,
        "model": data.get("model", model),
    }
