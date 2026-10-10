import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EMBEDDING_MODEL = "text-embedding-3-small"


def embedding_text(title: str | None, content: str) -> str:
    """Text to embed for a document. Including the title gives the vector
    the document's subject as well as its body, which matters when the
    content is long and the title carries the regulation's name."""
    if title:
        return f"{title}\n{content}"
    return content


def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding
