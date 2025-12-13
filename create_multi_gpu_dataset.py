import os

import datasets
from transformers import AutoTokenizer

if __name__ == "__main__":
    local_dir = "edu_fineweb_10b"
    dataset_name = "Amod/mental_health_counseling_conversations"
    # subdataset_name = "sample-10BT"
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    DATADIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(DATADIR, exist_ok=True)

    # dataset = datasets.load_dataset(dataset_name, subdataset_name, split="train")
    dataset = datasets.load_dataset(dataset_name, split="train")

    def tokenize_content(row):
        sentence = row["Context"] + " " + row["Response"]
        encoded_input = tokenizer.encode(
            sentence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=1024,
        )[0]

        return {"tokens": encoded_input}

    dataset = dataset.map(tokenize_content, remove_columns=dataset.column_names)

    # save dataset
    dataset.save_to_disk(DATADIR)
