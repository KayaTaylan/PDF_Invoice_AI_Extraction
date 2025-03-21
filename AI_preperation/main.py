import json
from fastai.text.all import *
from sklearn.model_selection import train_test_split
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Prevent multiprocessing issues on Windows

    # Load training dataset
    with open("../Training_Data/synthetic_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Prepare data for FastAI
    texts, labels = [], []
    for item in data:
        texts.append(item["text"])
        label_dict = {label["label"]: (label["start"], label["end"]) for label in item["labels"]}
        labels.append(label_dict)

    # 🔹 Split into training & validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Extract all unique labels from the training set
    all_labels = set()
    for label_dict in train_labels:
        all_labels.update(label_dict.keys())

    # ✅ Ensure "O" is included in the dataset
    all_labels.add("O")

    all_labels = sorted(list(all_labels))  # Sort for consistency
    print(f"✅ Training on {len(all_labels)} unique entity labels: {all_labels}")


    def tokenize_with_labels(text, label_dict):
        """Tokenize text while ensuring correct label alignment."""

        tokens = text.split("\n")  # ✅ Tokenize by newlines
        token_labels = ["O"] * len(tokens)  # ✅ Default label as "O"

        for label, (start, end) in label_dict.items():
            for i, token in enumerate(tokens):
                token_start = text.find(token)
                token_end = token_start + len(token)

                if start <= token_start and token_end <= end:
                    token_labels[i] = label  # ✅ Assign correct entity label

        return " ".join(tokens), " ".join(token_labels)


    # 🔹 Convert training & validation data into FastAI format
    train_data = [tokenize_with_labels(txt, lbl) for txt, lbl in zip(train_texts, train_labels)]
    val_data = [tokenize_with_labels(txt, lbl) for txt, lbl in zip(val_texts, val_labels)]

    # 🔹 Use `MultiCategoryBlock` with defined labels
    ner_block = DataBlock(
        blocks=(TextBlock.from_df(0, seq_len=256), MultiCategoryBlock(vocab=all_labels)),  # ✅ Ensure "O" is included
        get_x=ColReader(0),
        get_y=ColReader(1, label_delim=" "),  # ✅ Ensure labels are space-separated
        splitter=RandomSplitter(valid_pct=0.2)
    )

    # 🔹 Create DataLoaders
    dls = ner_block.dataloaders(train_data, bs=16)  # ✅ Adjust batch size

    # 🔹 Create Model with Proper Metrics
    learn = text_classifier_learner(
        dls, AWD_LSTM, metrics=[accuracy_multi]  # ✅ Correct metric for multi-label classification
    )

    # 🔹 Train the model
    learn.fine_tune(5, cbs=[ProgressCallback(), ShowGraphCallback()])

    # 🔹 Save the trained model
    learn.export("../fastai_invoice_model.pkl")

    print("✅ Model training complete. Saved as fastai_invoice_model.pkl")
