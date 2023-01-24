
from fastapi import FastAPI
from docarray import DocumentArray, Document
from clip_client import Client
from annlite import AnnLite
import json

app = FastAPI()

c = Client(
    'grpcs://api.clip.jina.ai:2096', credential={'Authorization': '991e3e1eb7c84a1242644521a948e6be'}
)

@app.on_event("startup")
def startup_event():
    global ann
    ann = AnnLite(n_dim=768,metric='cosine', data_path='data_test', columns={'price': 'str'})

@app.get("/search")
def search(text: str):
    query = c.encode([Document(text=text)])
    ann.search(query, limit=5)

    results = []
    for q in query:
        for k, m in enumerate(q.matches):
            results.append(json.dumps({'text': m.text, 'price': m.tags['price']}))
    
    return results