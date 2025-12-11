import code
import time

import torch
from sklearn.model_selection import train_test_split
from torch.functional import F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from gpt import GPT2, GPTConfig


class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        input_tensors = self.inputs[idx]
        target_tensors = self.targets[idx]
        return input_tensors, target_tensors


if __name__ == "__main__":
    # Settings
    EPOCHS = 20
    LEARNING_RATE = 1e-3

    torch.set_float32_matmul_precision("high")

    # Load Model
    # Initializing vocav to 50304 that is power of 2
    config = GPTConfig(vocab_size=50304)
    model = GPT2(config)
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    # model = torch.compile(model)

    # Read the dataset
    with open("./tiny_shakespeare.txt", "r") as f:
        text = f.read()

    tokens = tokenizer.encode(text, return_tensors="pt")
    keep_tokens = 1024 * 330 + 1
    tokens = tokens[:, :keep_tokens]
    input = tokens[:, :-1].view(-1, 1024)
    targets = tokens[:, 1:].view(-1, 1024)

    X_train, X_test, y_train, y_test = train_test_split(input, targets, test_size=0.1)
    train_dataset = CustomDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataset = CustomDataset(X_test, y_test)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        train_loss = 0
        model.train()
        print("_" * 200)
        print(f"Epoch: {epoch + 1}")
        for i, (input, targets) in enumerate(
            tqdm(train_dataloader, desc="Training Step: ")
        ):
            t0 = time.time()
            input = input.to(device)
            targets = targets.to(device)
            B, T = input.size()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                predictions = model(input)
                # code.interact(local=locals())
            optimizer.zero_grad()
            # config.vocab_size is the output possibilities for the whole Vocabulary
            step_loss = F.cross_entropy(
                predictions.view(-1, config.vocab_size), targets.view(-1)
            )
            train_loss += step_loss.item()
            step_loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            t1 = time.time()
            time_diff = t1 - t0
            dt = time_diff * 1000
            tokens_per_sec = (B * T) / time_diff
            print(f"\ndt: {dt}, tok/sec: {tokens_per_sec}")
            if i % 10 == 0:
                print(f"Step Loss {i + 1}: {step_loss}")
        train_loss = train_loss / len(train_dataloader)
        print(f"Train Loss: {train_loss}")

        eval_loss = 0
        model.eval()
        for i, (input, targets) in enumerate(
            tqdm(val_dataloader, desc="Evaluation Step: ")
        ):
            input = input.to(device)
            targets = targets.to(device)
            B, T = input.size()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                predictions = model(input)
            step_loss = F.cross_entropy(
                predictions.view(-1, config.vocab_size), targets.view(-1)
            )
            eval_loss += step_loss.item()
        eval_loss = eval_loss / len(val_dataloader)
        print(f"Eval Loss: {eval_loss}")
