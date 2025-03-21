import json
from fastai.text.all import load_learner

# Load the trained model
model_path = "../../fastai_invoice_model.pkl"
learn = load_learner(model_path)

def load_extracted_json(json_path):
    """Load pre-extracted text content from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)  # ✅ Loads extracted text directly

def extract_entities_from_text(text):
    """Use AI model to extract structured entities from invoice text."""
    tokens = text.split("\n")
    predictions = learn.predict(tokens)  # 🔥 What AI thinks

    structured_data = []
    for token, label in zip(tokens, predictions[0]):
        print(f"🔍 Token: '{token}' → AI Predicted Label: '{label}'")  # ✅ Debugging

        if label != "O":
            structured_data.append({"text": token, "label": label})

    return structured_data



# 🔹 Load extracted JSON data (instead of a PDF file)
json_file_path = "../../Training_Data/gpt_generated_test.json"  # ✅ Path to pre-extracted JSON file
extracted_text_data = load_extracted_json(json_file_path)

# 🔹 Extract entities from each page
invoice_results = []
for page in extracted_text_data:
    structured_page = {
        "page": page["page"],
        "extracted_data": extract_entities_from_text(page["text"])
    }
    invoice_results.append(structured_page)

# 🔹 Save structured output
output_path = "structured_invoice_data.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(invoice_results, f, ensure_ascii=False, indent=4)

print(f"✅ Invoice data extracted and saved to {output_path}")
