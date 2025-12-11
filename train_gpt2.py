import torch
from sklearn.model_selection import train_test_split
from torch.functional import F
from torch.optim import AdamW, Muon
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from gpt import GPT2


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

    # Load Model
    model = GPT2()
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

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

    optimizer = AdamW(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        train_loss = 0
        model.train()
        print("_" * 200)
        print(f"Epoch: {epoch + 1}")
        for i, (input, targets) in enumerate(
            tqdm(train_dataloader, desc="Training Step: ")
        ):
            input = input.to(device)
            targets = targets.to(device)
            B, T = input.size()
            predictions = model(input)
            optimizer.zero_grad()
            # 50257 is the output possibilities for the whole Vocabulary
            step_loss = F.cross_entropy(predictions.view(-1, 50257), targets.view(-1))
            train_loss += step_loss.item()
            step_loss.backward()
            optimizer.step()
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
            predictions = model(input)
            step_loss = F.cross_entropy(predictions.view(-1, 50257), targets.view(-1))
            eval_loss += step_loss.item()
        eval_loss = eval_loss / len(val_dataloader)
        print(f"Eval Loss: {eval_loss}")
