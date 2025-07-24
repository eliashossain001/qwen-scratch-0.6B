import torch
from src.model import QwenModel
from src.config import config
from transformers import PreTrainedTokenizerFast
from tokenizers import ByteLevelBPETokenizer

# Set device
device = config["device"]

# ✅ Explicit checkpoint path
checkpoint_path = "/home/research/cipher-aegis/elias/qwen-scratch-0.6B/checkpoints/ckpt_30000.pt"

# Load tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=None,
    tokenizer_object=ByteLevelBPETokenizer.from_file(
        vocab_filename="data/tokenizer/vocab.json",
        merges_filename="data/tokenizer/merges.txt"
    ),
    unk_token="<unk>",
    pad_token="<pad>",
    bos_token="<s>",
    eos_token="</s>"
)

# Load model
model = QwenModel(config).to(device)
model.eval()

# ✅ Load only model weights from checkpoint
ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt["model_state"])
print(f"✅ Loaded model weights from: {checkpoint_path}", flush=True)

# Prompt loop
print("\n💬 Enter a prompt (e.g., 'The key to happiness is...')\nType 'exit' to quit.", flush=True)
while True:
    prompt = input(">> ")
    if prompt.strip().lower() == "exit":
        break

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Generate tokens
    max_new_tokens = 100
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print("\n📘 Output:\n" + output + "\n", flush=True)
