import itertools
import os

import datasets
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

if __name__ == "__main__":
    N_SPLITS = 4
    local_dir = "edu_fineweb_10b"
    # dataset_name = "Amod/mental_health_counseling_conversations"
    dataset_name = "HuggingFaceFW/fineweb-edu"
    subdataset_name = "sample-10BT"
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    DATADIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(DATADIR, exist_ok=True)

    dataset = datasets.load_dataset(dataset_name, subdataset_name, split="train")
    dataset = dataset.shuffle(seed=42).select(range(100_000))
    # dataset = datasets.load_dataset(dataset_name, split="train")

    def tokenize_content(row):
        # sentence = row["Context"] + " " + row["Response"]
        sentence = row["text"]
        encoded_input = tokenizer.encode(
            sentence,
            # return_tensors="pt",
            # padding="max_length",
            # truncation=True,
            # max_length=1024,
        )

        return {"tokens": encoded_input}

    dataset = dataset.map(
        tokenize_content, remove_columns=dataset.column_names, num_proc=170, batched=False
    )

    token_list = [token for tokens in dataset["tokens"] for token in tokens]
    tokens = torch.tensor(token_list, dtype=torch.long)

    truncate_tokens_point = (tokens.shape[0] // 1024) * 1024
    tokens = tokens[: truncate_tokens_point + 1]

    train_inputs = tokens[:-1]
    target_inputs = tokens[1:]

    train_inputs = train_inputs.view(-1, 1024)
    target_inputs = target_inputs.view(-1, 1024)

    data_chunk = train_inputs.shape[0] // N_SPLITS
    for i in range(1, N_SPLITS + 1):
        start_pos = (i - 1) * data_chunk
        end_pos = i * data_chunk
        train_chunk = train_inputs[start_pos:end_pos, :]
        target_chunk = target_inputs[start_pos:end_pos, :]
        # save train and target chunk to file
        torch.save(
            {"train": train_chunk, "target": target_chunk},
            os.path.join(DATADIR, f"split_{i}.pt"),
        )
