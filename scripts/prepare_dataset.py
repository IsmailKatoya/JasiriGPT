import os
import re
import json

INPUT_PATH = "data/kenya_constitution_2010.txt"
OUTPUT_PATH = "data/constitution_chunks.json"

CHUNK_SIZE = 1200  # characters per chunk
OVERLAP = 200

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # collapse whitespace
    text = text.replace("\x0c", "")   # remove pagebreaks
    return text.strip()

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # overlap prevents losing context

    return chunks

def process():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Input not found: {INPUT_PATH}")
        return
    
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        raw = f.read()

    print("🧹 Cleaning text...")
    cleaned = clean_text(raw)

    print("✂️ Chunking text...")
    chunks = chunk_text(cleaned)

    print(f"📦 Total chunks created: {len(chunks)}")

    print("💾 Saving JSON dataset...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"✅ Dataset ready: {OUTPUT_PATH}")

if __name__ == "__main__":
    process()
