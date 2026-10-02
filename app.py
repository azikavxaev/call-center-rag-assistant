from dotenv import load_dotenv
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("Current folder:", os.getcwd())
print(".env exists:", os.path.exists(".env"))
load_dotenv()
import json
import re
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
import json, numpy as np, torch
import streamlit as st
from openai import OpenAI, OpenAIError
import chromadb

st.set_page_config(
    page_title="📞 Call Center AI Assistant",
    page_icon="📞",
    layout="wide"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REJECTION_MESSAGE = "This question is not related to the provided call center knowledge base, or there is no answer in the sources."

# Cosine distance (0 = identical, 2 = opposite). Measured with bge-base-en-v1.5:
# on-topic English questions score ~0.21-0.33, off-topic ones ~0.49+.
MAX_DISTANCE = 0.42

SYSTEM_PROMPT = f"""
You are an assistant that answers based on the internal FAQ for a call center.
RULES (MANDATORY):
Answer based ONLY on the provided SOURCES.
Do not use general knowledge. Do not fabricate anything.
If the SOURCES answer the question:
- Answer it, and at the end list the sources you used in the format "(Source 1)" or "(Source 1, Source 2)".
- Do NOT add the "not related" sentence below and do NOT ask a clarifying question.
Only if the answer is not in the SOURCES or the topic is not relevant:
- Start your reply with exactly this sentence: "{REJECTION_MESSAGE}"
- Then ask 1 clarifying question. Do not cite any sources.
"""

ERROR_MESSAGE = "Sorry, the answer service is temporarily unavailable. Please try again in a moment."

# Streamlit re-runs this script on every message; cache_resource loads these once per server process.
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_resources():
    with open("faq_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    collection = chromadb.PersistentClient(path="./chroma_db").get_collection("call_center_faq")
    embedder = SentenceTransformer(meta["embed_model"], device="cpu")
    items_by_id = {str(it["id"]): it for it in meta["items"]}
    return collection, embedder, items_by_id, OpenAI()

collection, embedder, items_by_id, client = load_resources()

def retrieve(query: str, top_k: int = 3):

    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["distances", "documents","metadatas"]
    )

    retrieved = []

    ids = results["ids"][0]
    distances = results["distances"][0]


    for rank, (faq_id, distance) in enumerate(
        zip(ids, distances),
        start=1
    ):

        item = items_by_id[str(faq_id)]

        retrieved.append(
            {
                "rank": rank,
                "distance": float(distance),
                "item": item
            }
        )

    return retrieved

def answer_with_openai(query: str, results):
    context = build_context(results)
    user_prompt = f"""
QUESTION:
{query}
SOURCES:
{context}
TASKS:
ANSWER BASED ON UPPER MENTIONED SOURCES.
"""
    # Generator of text chunks for st.write_stream; API errors become ERROR_MESSAGE.
    try:
        stream = client.responses.create(
            model="gpt-4o-mini",   # or the model you use internally
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,  # low for call-center accuracy
            stream=True
        )
        for event in stream:
            if event.type == "response.output_text.delta":
                yield event.delta
    except OpenAIError as e:
        print("OPENAI ERROR:", repr(e))
        yield "\n\n" + ERROR_MESSAGE

def fixed_reply(text):
    yield text

def answer_or_escalate(query, results, max_distance=MAX_DISTANCE):
    if not results:
        print("NO RESULTS")
        return fixed_reply(f"{REJECTION_MESSAGE} Please clarify the question.")
    top1 = results[0]["distance"]
    top2 = results[1]["distance"] if len(results) > 1 else -1.0
    margin = top2 - top1
    print("TOP1 =", top1)
    print("TOP2 =", top2)
    # Gate 1: absolute similarity too low
    if top1 > max_distance:
        print("FAILED: score gate")
        return fixed_reply(f"{REJECTION_MESSAGE} Please clarify the question.")
    # Gate 2: retrieval is uncertain (top results almost equal)
    #if margin < min_margin:
       # print("FAILED: margin gate")
        #return ("Sualınızın dəqiq cavabını mənbələrdə müəyyən edə bilmədim. "
               # "Zəhmət olmasa sualı bir az daha konkret yazın.")
    #if not passes_overlap_gate(query, results, min_overlap=1):
        #print("FAILED: overlap gate")
        #return ("Bu sual təqdim etdiyim çağrı mərkəzi məlumat bazasına aid deyil və ya mənbələrdə cavab yoxdur. "
            #"Zəhmət olmasa sualı dəqiqləşdirin.")
    # If passes gates -> call OpenAI using ONLY the retrieved sources
    return answer_with_openai(query, results)

def token_set(s: str):
    s = s.lower()
    s = re.sub(r"[^a-zəğıöşüç0-9\s]", " ", s)
    toks = [t for t in s.split() if len(t) >= 3]
    return set(toks)

def passes_overlap_gate(query: str, results, min_overlap=1):
    q = token_set(query)
    src = token_set(" ".join(r["item"]["question"] + " " + r["item"]["answer"] for r in results))
    return len(q & src) >= min_overlap

#def build_context(results):
    #parts = []
    #for r in results:
        #it = r["item"]
        #parts.append(
            #f"[Mənbə {r['rank']} | {it['id']}]\n"
            #f"SUAL: {it['question']}\n"
            #f"CAVAB: {it['answer']}"
        #)
    #return "\n\n---\n\n".join(parts)'''

def build_context(results):
    parts = []
    for r in results:
        it = r["item"]
        parts.append(
            f"""
Source {r['rank']}
Topic: {it['topic']}
Question #: {it['qnum']}
Question:
{it['question']}
Answer:
{it['answer']}
"""
        )
    return "\n\n-------------------\n\n".join(parts)

#st.title("📞 Call Center AI Assistant")
#query = st.text_input(
#    "Sualınızı yazın:"
#)

#placeholder = st.empty()
#current_text = ""

#if st.button("Soruş"):
#    answer = answer_or_escalate(query)
#    st.subheader("Cavab")
#    # st.write(answer)
#    for word in answer.split():
#        current_text += word + " "
#        placeholder.markdown(current_text)
#        time.sleep(0.03)
st.title("📞 Call Center AI Assistant")
st.caption("Answers based on internal call center FAQ")

# -----------------------------------
# Chat history
# -----------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------------
# Display previous messages
# -----------------------------------
def render_sources(sources):
    for s in sources:
        with st.expander(f"📄 Source {s['rank']} — {s['topic']} · Q{s['qnum']}"):
            st.markdown(f"**{s['question']}**")
            st.text(s["answer"])

def cited_sources(answer, results):
    # Sources the model cited as "(Source N)"; fall back to the closest matches.
    cited = {int(n) for n in re.findall(r"Source\s+(\d+)", answer)}
    picked = [r for r in results if r["rank"] in cited]
    if not picked:
        best_distance = results[0]["distance"]
        picked = [r for r in results if r["distance"] <= best_distance + 0.1]
    return [
        {
            "rank": r["rank"],
            "topic": r["item"]["topic"],
            "qnum": r["item"]["qnum"],
            "question": r["item"]["question"],
            "answer": r["item"]["answer"],
            "distance": r["distance"]
        }
        for r in picked
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

# -----------------------------------
# User input
# -----------------------------------
query = st.chat_input("Ask any question...")

if query:
    # Show user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    # Retrieve
    results = retrieve(query)
    # Generate and stream the answer
    with st.chat_message("assistant"):
        answer = st.write_stream(answer_or_escalate(query, results))
        is_rejection = (
            answer.strip().startswith(REJECTION_MESSAGE)
            or ERROR_MESSAGE in answer
        )
        sources = [] if is_rejection else cited_sources(answer, results)
        # Show source only for valid answers
        if not is_rejection:
            render_sources(sources)
    # Save assistant message
    if not is_rejection:
        message = {
            "role": "assistant",
            "content": answer,
            "sources": sources
            }
    else:
        message = {
            "role": "assistant",
            "content": answer
        }
    st.session_state.messages.append(message)

