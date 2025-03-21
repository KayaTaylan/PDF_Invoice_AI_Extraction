import fitz
import json
import os
from fastai.text.all import *

# Load the trained AI model
learn = load_learner("../fastai_invoice_model.pkl")

pdf = r"C:\Users\Taylan Kaya\Desktop\Programming_VCost\.ongoing\GAEB_to_Excel\Application\Resources\Data\V01H1 NR-52 AR 52 - D+S + MAG Prüfung.pdf"

def extract_text_pymupdf(pdf_path):
    extracted_data = []
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")  # Extract text with correct encoding

        if "Seite zur Rechnung" not in text and "Aufmaß" not in text:
            extracted_data.append({"page": page_num, "text": text})

    json_output = json.dumps(extracted_data, indent=4, ensure_ascii=False)
    json_file_path = pdf.replace(".pdf", "_pymupdf.json")

    with open(json_file_path, "w", encoding="utf-8") as f:
        f.write(json_output)

extract_text_pymupdf(pdf)
