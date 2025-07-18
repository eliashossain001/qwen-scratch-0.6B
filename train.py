#!/usr/bin/env python
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import config
from src.model import QwenModel
from src.dataset import get_dataloader
from src.utils import save_checkpoint, load_checkpoint

def build_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train():
    device = config["device"]
    model = QwenModel(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    scheduler = build_scheduler(optimizer, config["warmup_steps"], config["total_steps"])

    dataloader = get_dataloader(
        file_path="data/book.txt",
        seq_len=config["seq_len"],
        batch_size=config["batch_size"]
    )

    global_step = 0
    model.train()

    for epoch in range(1000):
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch}"):
            x, y = x.to(device), y.to(device)

            # Manually checkpoint each TransformerBlock
            hidden = model.token_emb(x) + model.pos_emb(
                torch.arange(x.size(1), device=x.device).unsqueeze(0)
            )
            for block in model.layers:
                hidden = torch.utils.checkpoint.checkpoint(block, hidden)
            hidden = model.norm(hidden)
            logits = model.head(hidden)

            loss = F.cross_entropy(
                logits.view(-1, config["vocab_size"]),
                y.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            if global_step % 100 == 0:
                print(f"Step {global_step} | Loss {loss.item():.4f}")
            if global_step % config["save_every"] == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, config["save_dir"])
            if global_step >= config["total_steps"]:
                save_checkpoint(model, optimizer, scheduler, global_step, config["save_dir"])
                return

if __name__ == "__main__":
    train()
