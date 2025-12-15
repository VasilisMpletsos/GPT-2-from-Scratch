import os
import time

import torch
import torch.distributed as dist
from sklearn.model_selection import train_test_split
from torch.distributed import destroy_process_group, init_process_group
from torch.functional import F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

# from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from gpt import GPT2, CosineLearningDecay, GPTConfig

ddp = int(os.environ.get("RANK", -1)) != -1

best_val_loss = float("inf")

# Check if DDP is available on the machine
if ddp:
    assert torch.cuda.is_available(), "We need CUDA GPU in order to run DDP"
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device=device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = "cpu"
    if torch.cuda.is_available():
        device = "gpu"
    print(f"Using device {device} for training")


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


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


# Settings
EPOCHS = 200
LEARNING_RATE = 1e-5
BATCH_SIZE = 16
GRAD_ACCUMULATION_STEPS = 4
gpt2_paper_training_settings = {"betas": (0.9, 0.05), "eps": 1e-8}
gpt2_paper_lr_scheduler_settings = {
    "max_lr": LEARNING_RATE,
    "min_lr": 1e-6,
    "max_steps": EPOCHS // 4,
    "warmup_steps": 0,
}

if master_process:
    writer = SummaryWriter(log_dir="./logs/gpt2_from_scratch_fine_web")

torch.set_float32_matmul_precision("high")

# Load Model
# Initializing vocav to 50304 that is power of 2
config = GPTConfig(vocab_size=50304)
model = GPT2(config)
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# # Read the dataset
# with open("./tiny_shakespeare.txt", "r") as f:
#     text = f.read()

# tokens = tokenizer.encode(text, return_tensors="pt")
# keep_tokens = 1024 * 330 + 1
# tokens = tokens[:, :keep_tokens]
# input = tokens[:, :-1].view(-1, 1024)
# targets = tokens[:, 1:].view(-1, 1024)
local_dir = "edu_fineweb_10b"
tokens = torch.load(f"./{local_dir}/split_{ddp_rank + 1}.pt")
input = tokens["train"]
targets = tokens["target"]

X_train, X_test, y_train, y_test = train_test_split(input, targets, test_size=0.02)
train_dataset = CustomDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = CustomDataset(X_test, y_test)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    **gpt2_paper_training_settings,
    fused=False,
)
lr_scheduler = CosineLearningDecay(
    **gpt2_paper_lr_scheduler_settings, optimizer=optimizer
)

model.train()
device_type = "cuda" if "cuda" in device else "cpu"
grad_acc_step = 0
optimizer.zero_grad()
train_dataloader_size = len(train_dataloader)
val_dataloader_size = len(train_dataloader)
for epoch in range(EPOCHS):
    train_loss = 0
    if master_process:
        print("_" * 200)
        print(f"Epoch: {epoch + 1}")

    # Change Learning Weight
    lr = lr_scheduler.update_lr(epoch)

    # Iterate whole training examples
    for i, (input, targets) in enumerate(
        tqdm(train_dataloader, desc="Training Step: ")
    ):
        t0 = time.time()
        input = input.to(device)
        targets = targets.to(device)
        B, T = input.size()
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            predictions = model(input)
            # code.interact(local=locals())
        # config.vocab_size is the output possibilities for the whole Vocabulary
        # Division by GRAD_ACCUMULATION_STEPS is needed in order to include percentage due to accumulation
        step_loss = F.cross_entropy(
            predictions.view(-1, config.vocab_size), targets.view(-1)
        )
        train_loss += step_loss.detach()
        if ddp:
            model.require_backward_grad_sync = (
                grad_acc_step == GRAD_ACCUMULATION_STEPS - 1
            )
        (step_loss / GRAD_ACCUMULATION_STEPS).backward()
        if ddp:
            dist.all_reduce(step_loss, op=dist.ReduceOp.AVG)
        grad_acc_step += 1
        if master_process:
            if i % 50 == 0:
                writer.add_scalar(
                    "Step_Loss/train", step_loss, epoch * train_dataloader_size + i
                )

        # if device.startswith("cuda"):
        #     torch.cuda.synchronize()
        t1 = time.time()
        time_diff = t1 - t0
        dt = time_diff * 1000
        tokens_per_sec = (B * T * 4) / time_diff

        if grad_acc_step == GRAD_ACCUMULATION_STEPS:
            grad_acc_step = 0
            # as mentioned in the paper `we clip the global norm of the gradient at 1.0`
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            # if master_process:
            #     print(f"\ndt: {dt}, tok/sec: {tokens_per_sec}")

        if i % 7000 == 0:
            validation_loss = 0
            with torch.no_grad():
                for i, (input, targets) in enumerate(
                    tqdm(val_dataloader, desc="Validation Step: ")
                ):
                    input = input.to(device)
                    targets = targets.to(device)
                    B, T = input.size()
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        predictions = model(input)
                    step_loss = F.cross_entropy(
                        predictions.view(-1, config.vocab_size), targets.view(-1)
                    )
                    validation_loss += step_loss.detach()
            if ddp:
                dist.all_reduce(validation_loss, op=dist.ReduceOp.AVG)
            validation_loss = validation_loss / val_dataloader_size
            if master_process:
                writer.add_scalar(
                    "Step_Loss/val",
                    validation_loss,
                    epoch * train_dataloader_size + i,
                )

                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    torch.save(
                        model.module.state_dict() if ddp else model.state_dict(),
                        f"./gpt2_from_scratch_fine_web_best_model.pth",
                    )
                    print(f"New best model saved with validation loss: {best_val_loss}")

    train_loss = train_loss / train_dataloader_size
    if ddp:
        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
    if master_process:
        writer.add_scalar("Epoch_Loss/train", train_loss.item(), epoch)

if master_process:
    writer.close()
