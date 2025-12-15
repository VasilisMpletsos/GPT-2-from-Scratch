import os
import time

import datasets
import torch
import torch.distributed as dist
from sklearn.model_selection import train_test_split
from torch.distributed import destroy_process_group, init_process_group
from torch.functional import F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from gpt import GPT2, CosineLearningDecay, GPTConfig

NUM_GPUS = 8
TEST_SIZE = 500

# ================== CONFIGURE DDP ==================

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

# ================== LOAD DATASET ==================

# Load dataset before tokenization
local_dir = "edu_fineweb_10b"
dataset_name = "HuggingFaceFW/fineweb-edu"
subdataset_name = "sample-10BT"

dataset = datasets.load_dataset(
    dataset_name, subdataset_name, split="train", streaming=True
)


def tokenize(row):
    sentence = row["text"]
    encoded_input = tokenizer.encode(
        sentence,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=1025,
    )[0]

    input = encoded_input[:-1]
    target = encoded_input[1:]

    return {"input": input, "target": target}


train_dataset = (
    dataset.skip(NUM_GPUS * TEST_SIZE)
    .shuffle()
    .map(tokenize, remove_columns=dataset.column_names)
)
eval_dataset = (
    dataset.skip(ddp_rank * TEST_SIZE)
    .take(TEST_SIZE)
    .map(tokenize, remove_columns=dataset.column_names)
)
BATCH_SIZE = 16
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
)
val_dataloader = DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
)

# ================== LOAD MODEL ==================

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# Settings
EPOCHS = 200
LEARNING_RATE = 1e-5

GRAD_ACCUMULATION_STEPS = 4
gpt2_paper_training_settings = {"betas": (0.9, 0.05), "eps": 1e-8}
gpt2_paper_lr_scheduler_settings = {
    "max_lr": LEARNING_RATE,
    "min_lr": 1e-6,
    "max_steps": 50000,
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
tokenizer.pad_token = tokenizer.eos_token
model = model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

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
global_step = 0
for epoch in range(EPOCHS):
    train_loss = 0
    if master_process:
        print("_" * 200)
        print(f"Epoch: {epoch + 1}")

    # Iterate whole training examples
    train_batch_count = 0
    for batch in train_dataloader:
        train_batch_count += 1
        input = batch["input"].to(device)
        targets = batch["target"].to(device)

        B, T = input.size()

        # Change Learning Weight
        lr = lr_scheduler.update_lr(global_step)
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
            if global_step % 50 == 0:
                writer.add_scalar("Step_Loss/train", step_loss, global_step)

        if grad_acc_step == GRAD_ACCUMULATION_STEPS:
            grad_acc_step = 0
            # as mentioned in the paper `we clip the global norm of the gradient at 1.0`
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            # if master_process:
            #     print(f"\ndt: {dt}, tok/sec: {tokens_per_sec}")

        if master_process:
            if global_step % 100 == 0 and master_process:
                print(
                    f"Epoch {epoch + 1} | Step {global_step} | Train Loss: {step_loss.item():.4f} | LR: {lr:.6f}"
                )

        if global_step % TEST_SIZE == 0:
            validation_loss = 0
            val_batch_count = 0
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_batch_count += 1
                    input = val_batch["input"].to(device)
                    targets = val_batch["target"].to(device)
                    B, T = input.size()
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        predictions = model(input)
                    step_loss = F.cross_entropy(
                        predictions.view(-1, config.vocab_size), targets.view(-1)
                    )
                    validation_loss += step_loss.detach()
            if ddp:
                validation_loss = validation_loss / val_batch_count
                dist.all_reduce(validation_loss, op=dist.ReduceOp.AVG)
            if master_process:
                writer.add_scalar(
                    "Step_Loss/val",
                    validation_loss,
                    global_step,
                )

                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    torch.save(
                        model.module.state_dict() if ddp else model.state_dict(),
                        f"./gpt2_from_scratch_fine_web_best_model.pth",
                    )
                    print(f"New best model saved with validation loss: {best_val_loss}")

        global_step += 1

    if ddp:
        train_loss = train_loss / train_batch_count
        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
    if master_process:
        writer.add_scalar("Epoch_Loss/train", train_loss.item(), epoch)

if master_process:
    writer.close()
