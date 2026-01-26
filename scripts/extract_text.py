import fitz  # PyMuPDF
import os

PDF_PATH = "data/kenya_constitution_2010.pdf"
OUTPUT_PATH = "data/kenya_constitution_2010.txt"

def extract_text():
    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF not found: {PDF_PATH}")
        return

    print("📄 Opening PDF...")
    doc = fitz.open(PDF_PATH)
    all_text = ""

    for i, page in enumerate(doc):
        print(f"Extracting page {i+1}/{len(doc)}...")
        all_text += page.get_text()

    doc.close()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(all_text)

    print(f"✅ Extraction complete! Text saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    extract_text()
