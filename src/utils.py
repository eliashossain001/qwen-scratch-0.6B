# ==================== utils.py ====================
import os
import torch

def save_checkpoint(model, optimizer, scheduler, step, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        'step': step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
    }
    path = os.path.join(save_dir, f'ckpt_{step}.pt')
    torch.save(ckpt, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(model, optimizer, scheduler, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    step = ckpt['step']
    print(f"Loaded checkpoint from step {step}")
    return step