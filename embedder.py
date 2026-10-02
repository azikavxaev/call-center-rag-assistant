import re
import fitz  # PyMuPDF
import json
from typing import List, Dict, Any

import rag

pdf_path = 'FAQ_RAG.pdf'

def extract_pdf_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    parts = []
    for page in doc:
        t = page.get_text("text")
        if t:
            parts.append(t)
    doc.close()

    text = "\n".join(parts)
    text = text.replace("\r\n", "\n").replace("\u00A0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def load_faq_from_pdf_with_topic_delimiter(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Format:
      Topic title
      1. Question?
      Answer...
      *****
      ...
      **********   <-- end of topic
    """
    raw = extract_pdf_text(pdf_path)
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]

    question_re = re.compile(r"^(\d+)\.\s*(.+)$")   # "1. Question..."
    qa_delim_re = re.compile(r"^\*{3,}$")            # "*****"
    topic_end_re = re.compile(r"^\*{10,}$")          # "**********"

    items: List[Dict[str, Any]] = []

    topic_index = 0
    current_topic = None

    current_qnum = None
    current_question = None
    answer_lines: List[str] = []

    def flush_question():
        nonlocal current_qnum, current_question, answer_lines
        if current_qnum is None:
            return
        answer = "\n".join(answer_lines).strip()
        if not answer:
            raise ValueError(
                f"Empty answer for t{topic_index}_q{current_qnum}"
            )
        items.append({
            "id": f"t{topic_index}_q{int(current_qnum)}",
            "topic": current_topic,
            "qnum": int(current_qnum),
            "question": current_question.strip() if current_question else "",
            "answer": answer
        })
        current_qnum = None
        current_question = None
        answer_lines = []

    for ln in lines:
        # End of topic
        if topic_end_re.match(ln):
            flush_question()
            current_topic = None
            continue

        # End of one Q/A
        if qa_delim_re.match(ln):
            flush_question()
            continue

        # Question line
        m = question_re.match(ln)
        if m:
            flush_question()
            if current_topic is None:
                raise ValueError(f"Question found before topic: {ln}")
            current_qnum = m.group(1)
            current_question = m.group(2)
            answer_lines = []
            continue

        # Topic title (only when not inside a question)
        if current_qnum is None and current_topic is None:
            topic_index += 1
            current_topic = ln
            continue

        # Answer text
        if current_qnum is not None:
            answer_lines.append(ln)

    flush_question()
    return items

def make_embed_text(it):
    return f"MÖVZU: {it['topic']}\nSUAL: {it['question']}\nCAVAB: {it['answer']}\nID: {it['id']}"

if __name__ == "__main__":
    # 1. Load and parse the FAQ
    faq_items = load_faq_from_pdf_with_topic_delimiter(pdf_path)
    print(f"Loaded {len(faq_items)} items from {pdf_path}")

    # 2. Prepare texts for embedding
    corpus_texts = [make_embed_text(it) for it in faq_items]

    # 3. Generate normalized embeddings with OpenAI
    rag.load_env_file()
    embeddings = rag.embed(corpus_texts)

    # 4. Save items and embeddings together; rag.py loads this file at runtime
    with open(rag.INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "embed_model": rag.EMBED_MODEL,
                "items": faq_items,
                "corpus_texts": corpus_texts,
                "embeddings": [[round(x, 6) for x in v] for v in embeddings],
            },
            f,
            ensure_ascii=False,
        )
    print(f"Saved {len(embeddings)} embeddings ({len(embeddings[0])} dims) to {rag.INDEX_PATH}")