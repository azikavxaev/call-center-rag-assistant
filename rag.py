"""Retrieval and answer generation, shared by the Lambda handler and the local server.

Uses only the standard library (plus boto3, which the Lambda runtime provides) so the
Lambda deployment package needs no third-party dependencies.
"""
import json
import math
import os
import re
import urllib.error
import urllib.request

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faq_meta.json")
TOP_K = 3

REJECTION_MESSAGE = "This question is not related to the provided call center knowledge base, or there is no answer in the sources."
ERROR_MESSAGE = "Sorry, the answer service is temporarily unavailable. Please try again in a moment."

# Cosine distance (0 = identical, 2 = opposite). Measured with text-embedding-3-small:
# on-topic English questions score ~0.38-0.64, Azerbaijani ~0.59-0.82, off-topic ~0.79+.
# The model still rejects questions the retrieved sources don't answer.
MAX_DISTANCE = 0.75

SYSTEM_PROMPT = f"""
You are an assistant that answers based on the internal FAQ for a call center.
RULES (MANDATORY):
Answer based ONLY on the provided SOURCES.
Do not use general knowledge. Do not fabricate anything.
Answer in the same language as the question.
If the SOURCES answer the question:
- Answer it, and at the end list the sources you used in the format "(Source 1)" or "(Source 1, Source 2)".
- Do NOT add the "not related" sentence below and do NOT ask a clarifying question.
Only if the answer is not in the SOURCES or the topic is not relevant:
- Start your reply with exactly this sentence: "{REJECTION_MESSAGE}"
- Then ask 1 clarifying question. Do not cite any sources.
"""


class OpenAIUnavailable(Exception):
    pass


def load_env_file(path=".env"):
    """Minimal .env loader for local runs; existing environment variables win."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_api_key = None


def api_key():
    # Local runs use OPENAI_API_KEY; on Lambda the key is read once from Secrets Manager.
    global _api_key
    if _api_key is None:
        _api_key = os.environ.get("OPENAI_API_KEY")
        if not _api_key and os.environ.get("OPENAI_SECRET_ARN"):
            import boto3
            secret = boto3.client("secretsmanager").get_secret_value(SecretId=os.environ["OPENAI_SECRET_ARN"])
            _api_key = secret["SecretString"].strip()
        if not _api_key:
            raise OpenAIUnavailable("OPENAI_API_KEY is not set")
    return _api_key


def _openai(path, payload, timeout=40):
    request = urllib.request.Request(
        f"https://api.openai.com/v1/{path}",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        raise OpenAIUnavailable(f"{e.code} {e.read().decode(errors='replace')[:300]}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise OpenAIUnavailable(repr(e)) from e


def _normalize(vector):
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def embed(texts):
    data = _openai("embeddings", {"model": EMBED_MODEL, "input": texts})["data"]
    return [_normalize(d["embedding"]) for d in sorted(data, key=lambda d: d["index"])]


_index = None


def index():
    global _index
    if _index is None:
        with open(INDEX_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        if meta["embed_model"] != EMBED_MODEL:
            raise RuntimeError(f"{INDEX_PATH} was built with {meta['embed_model']}; re-run embedder.py")
        _index = (meta["items"], meta["embeddings"])
    return _index


def retrieve(question, top_k=TOP_K):
    query = embed([question])[0]
    items, vectors = index()
    scored = sorted(
        ((1.0 - sum(a * b for a, b in zip(query, v)), item) for item, v in zip(items, vectors)),
        key=lambda pair: pair[0],
    )[:top_k]
    return [{"rank": rank, "distance": d, "item": item} for rank, (d, item) in enumerate(scored, start=1)]


def build_context(results):
    parts = []
    for r in results:
        it = r["item"]
        parts.append(
            f"Source {r['rank']}\n"
            f"Topic: {it['topic']}\n"
            f"Question #: {it['qnum']}\n"
            f"Question:\n{it['question']}\n"
            f"Answer:\n{it['answer']}"
        )
    return "\n\n-------------------\n\n".join(parts)


def generate(question, results):
    user_prompt = f"QUESTION:\n{question}\n\nSOURCES:\n{build_context(results)}\n\nTASK:\nAnswer based on the sources above."
    response = _openai("responses", {
        "model": CHAT_MODEL,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,  # low for call-center accuracy
        "max_output_tokens": 600,
    })
    return "".join(
        part["text"]
        for item in response.get("output", []) if item.get("type") == "message"
        for part in item.get("content", []) if part.get("type") == "output_text"
    ).strip()


def cited_sources(answer, results):
    # Sources the model cited as "(Source N)"; fall back to the closest matches.
    cited = {int(n) for n in re.findall(r"Source\s+(\d+)", answer)}
    picked = [r for r in results if r["rank"] in cited]
    if not picked:
        best = results[0]["distance"]
        picked = [r for r in results if r["distance"] <= best + 0.05]
    return [
        {
            "rank": r["rank"],
            "topic": r["item"]["topic"],
            "qnum": r["item"]["qnum"],
            "question": r["item"]["question"],
            "answer": r["item"]["answer"],
        }
        for r in picked
    ]


def answer(question):
    """Returns {"answer", "sources", "rejected", "error"} for one question."""
    try:
        results = retrieve(question)
        print("DISTANCES =", [round(r["distance"], 3) for r in results])
        if not results or results[0]["distance"] > MAX_DISTANCE:
            return {"answer": f"{REJECTION_MESSAGE} Please clarify the question.", "sources": [], "rejected": True, "error": False}
        text = generate(question, results)
    except OpenAIUnavailable as e:
        print("OPENAI ERROR:", e)
        return {"answer": ERROR_MESSAGE, "sources": [], "rejected": False, "error": True}
    rejected = text.startswith(REJECTION_MESSAGE)
    return {
        "answer": text,
        "sources": [] if rejected else cited_sources(text, results),
        "rejected": rejected,
        "error": False,
    }
