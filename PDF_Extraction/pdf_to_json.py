import fitz
import json
import os

pdf_path = r"C:\Users\Taylan Kaya\Desktop\Programming_VCost\.ongoing\GAEB_to_Excel\Application\Resources\Data\V01H1 NR-52 AR 52 - D+S + MAG Prüfung.pdf"

def extract_text_pymupdf(pdf_path):
    extracted_data = []
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")  # Extract text with correct encoding

        extracted_data.append({"page": page_num, "text": text})

    return extracted_data

result = extract_text_pymupdf(pdf_path)
json_output = json.dumps(result, indent=4, ensure_ascii=False)

json_file_path = pdf_path.replace(".pdf", "_pymupdf.json")
with open(json_file_path, "w", encoding="utf-8") as f:
    f.write(json_output)

print(f"Extracted data saved to: {json_file_path}")