import json
import re
import random
from tqdm import tqdm
from multiprocessing import Pool, freeze_support

# Load the JSON file
with open("../Training_Data/test_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def load_leistungsbeschreibung_words(filename="../Training_Data/leistungsbeschreibung_words.txt"):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

leistungsbeschreibung_words = load_leistungsbeschreibung_words()

# List of possible units of measurement (only for LV-MENGE)
units_of_measurement = [
    "m³", "psch", "St", "m³Wo", "StWo", "h", "m3Wo", "m2", "m2Wo", "stgm", "m", "mWo"
]

# Regex pattern to extract numbers and units
unit_pattern = re.compile(r"(\d+[,.]?\d*)\s*([a-zA-Z²³μ]*)")

def randomize_value(value, label):
    """Randomize numerical values or modify text based on label while preserving or replacing units for LV-MENGE."""
    if label == "ORDNUNGSZAHL":
        num_parts = random.randint(1, 5)  # Random number of parts (1 to 5)
        parts = []
        for _ in range(num_parts):
            part_type = random.choice(["number", "formatted"])
            if part_type == "number":
                part = str(random.randint(10, 9999))  # 2-4 digit number
            else:
                part = f"{str(random.randint(1, 999)).zfill(3)}.{str(random.randint(1, 99)).zfill(2)}"  # 002.01 format
            parts.append(part)
        return ".".join(parts)
    elif label == "LV-MENGE":
        match = unit_pattern.match(value)
        if match:
            num, unit = match.groups()
            new_num = str(random.randint(1, 999))  # Random integer
            new_unit = random.choice(units_of_measurement)  # Random unit
            return f"{new_num} {new_unit}".strip()
        return str(random.randint(1, 999))
    elif label in ["MENGE-ALT", "RE-MENGE", "GESAMTBETRAG"]:
        return str(random.randint(1, 999))  # Random integer without unit
    elif label in ["EINHEITSPREIS", "ALTLEISTUNG", "NEULEISTUNG"]:
        return f"{random.uniform(1, 999):.2f}"  # Random float with 2 decimals, no unit
    elif label == "LEISTUNGSBESCHREIBUNG":
        return random.choice(leistungsbeschreibung_words)  # Random word from list
    return value  # Return unchanged if no specific rule

def process_example(example):
    """Process each example by splitting, randomizing, and reconstructing text with correct label positions."""
    original_text = example["text"].strip().split("\n")
    labels = example["labels"]

    # Extract label names in order of appearance
    label_names = [label["label"] for label in labels]

    # Assign each part of the string to its respective label
    values = {label_names[i]: original_text[i] for i in range(len(original_text))}

    # Randomize each value
    randomized_values = {key: randomize_value(value, key) for key, value in values.items()}

    # Reconstruct the text with new values and correct newline positions
    new_text = "\n".join(randomized_values.values())

    # Calculate new start and end positions
    new_labels = []
    end_previous = 0
    for key in randomized_values:
        start_new = end_previous + 3 if end_previous > 0 else 0
        end_new = start_new + len(randomized_values[key]) - 1  # Subtract 1 to correct end position
        new_labels.append({"start": start_new, "end": end_new, "label": key})
        end_previous = end_new

    return {"text": new_text, "labels": new_labels}

def generate_synthetic_data_parallel(example_data, num_variations=5):
    """Generates multiple randomized versions of training data in parallel with a progress bar."""
    with Pool() as pool:
        tasks = example_data * num_variations  # Duplicate dataset
        synthetic_data = list(tqdm(pool.imap(process_example, tasks), total=len(tasks), desc="Generating Data"))
    return synthetic_data

if __name__ == "__main__":
    freeze_support()

    # Extract the "text" content and label positions
    if data and isinstance(data, list):
        synthetic_dataset = generate_synthetic_data_parallel(data, num_variations=1000)

        # Save to JSON file incrementally
        with open("../Training_Data/synthetic_data.json", "w", encoding="utf-8") as f:
            f.write("[\n")  # Start JSON array
            for i, data_entry in enumerate(synthetic_dataset):
                json.dump(data_entry, f, ensure_ascii=False, indent=2)
                if i < len(synthetic_dataset) - 1:
                    f.write(",\n")  # Add a comma between entries
            f.write("\n]")  # End JSON array

        print("Synthetic data saved to synthetic_data.json")
    else:
        print("No valid data found in test_data.json.")