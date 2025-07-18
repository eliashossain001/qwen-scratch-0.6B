# Qwen Scratch Project

A from-scratch PyTorch implementation of a Qwen-style Transformer language model trained on a single PDF book.

---

## 📂 Folder Structure

```
qwen_scratch_project/
├── checkpoints/              # Auto-populated: saved model checkpoints (ckpt_<step>.pt)
├── data/
│   ├── book.pdf             # Original PDF book
│   ├── book.txt             # Extracted text file for training
│   └── tokenizer/           # Byte-Level BPE files
│       ├── vocab.json       # BPE vocabulary
│       └── merges.txt       # BPE merge rules
├── src/                     # Project source code
│   ├── __init__.py          # Marks src as a Python package
│   ├── config.py            # Model & training hyperparameters
│   ├── dataset.py           # TextDataset & DataLoader factory
│   ├── model.py             # RotaryEmbedding, RMSNorm, GQA, Transformer, QwenModel
│   └── utils.py             # Checkpoint save & load utilities
├── extract_text.py          # Script: extract text from PDF → book.txt
├── train_tokenizer.py       # Script: train Byte-Level BPE tokenizer
├── train.py                 # Main training script: pre-training loop & checkpointing
├── test_tokenizer.py        # (Optional) Validate tokenizer encode/decode
├── requirements.txt         # Python dependencies
└── README.md                # Project overview & instructions
```

---

## 🚀 Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Extract Text from PDF

```bash
python extract_text.py   # reads data/book.pdf → data/book.txt
```

### 3. Train the Tokenizer

```bash
python train_tokenizer.py  # trains on data/book.txt ➞ data/tokenizer/{vocab.json, merges.txt}
```

(Optional test)

```bash
python test_tokenizer.py   # verify sample encode/decode
```

### 4. Configure Hyperparameters

Edit `src/config.py` to adjust:

* `vocab_size`, `hidden_size`, `num_hidden_layers`, etc.
* `batch_size`, `seq_len`, `learning_rate`, `total_steps`, `save_every`

### 5. Launch Pre-Training

```bash
python train.py
```

This will:

1. Load `data/book.txt` and your BPE tokenizer.
2. Build a Qwen-style Transformer on GPU.
3. Run an AdamW training loop with warmup scheduler.
4. Periodically save checkpoints under `checkpoints/`.

---

## 🔍 Project Details

* **Model architecture**: 20-layer Transformer with grouped-query attention, RMSNorm, SiLU-based feed-forward, custom Rotary Position Embeddings.
* **Tokenizer**: Byte-Level BPE (30k vocab) trained on your PDF text.
* **Training settings**:

  * Context length: 1024 tokens
  * Batch size: 1 (reduce OOM)
  * Learning rate: 2e-4 with linear warmup & cosine decay
  * Checkpoint every 5000 steps

---

## 📈 Monitoring & Checkpoints

* Check console logs for `Step <n> | Loss <x.xxxx>`.
* Checkpoints saved to `checkpoints/ckpt_<step>.pt` automatically.
* To resume from a checkpoint, call `utils.load_checkpoint(...)` in `train.py` before the loop.

---

## 🛠️ Customization Tips

* **Memory trade-offs**: reduce `seq_len`, `hidden_size`, or enable gradient checkpointing in `train.py`.
* **Longer context**: increase `max_position_embeddings` (adjust `pos_emb`) but watch GPU RAM.
* **Additional scripts**: add evaluation or sampling scripts by loading `QwenModel` and calling `.generate()` methods.

---

## 📚 References

* Qwen architecture diagram by Sebastian Raschka
* Byte-Level BPE: Hugging Face Tokenizers docs
* Rotary Position Embedding: Su et al., 2021

---

*EMA* (Elias Hossain) • July 2025
