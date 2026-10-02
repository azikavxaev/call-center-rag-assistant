import chromadb
import json
import numpy as np

with open("faq_meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

faq_items = meta["items"]
embeddings = np.load("faq_embeddings.npy")

client = chromadb.PersistentClient(path="./chroma_db")

try:
    client.delete_collection("call_center_faq")
except:
    pass

collection = client.create_collection(
    name="call_center_faq",
    metadata={"hnsw:space": "cosine"}
)

collection.add(
    ids=[str(item["id"]) for item in faq_items],

    documents=[
        f"{item['question']}\n{item['answer']}"
        for item in faq_items
    ],

    metadatas=[
        {
            "topic": str(item["topic"]),
            "qnum": str(item["qnum"])
        }
        for item in faq_items
    ],

    embeddings=embeddings.tolist()
)

print("COUNT =", collection.count())