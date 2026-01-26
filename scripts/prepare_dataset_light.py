import os
import json

INPUT_PATH = "data/kenya_constitution_2010.txt"
OUTPUT_PATH = "data/constitution_chunks.json"

CHUNK_SIZE = 2000  # characters per chunk
OVERLAP = 0        # no overlap, fastest

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Input file not found: {INPUT_PATH}")
        return

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"✅ Done! {len(chunks)} chunks saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
