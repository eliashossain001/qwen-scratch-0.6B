# Qwen Scratch 0.6B LLM
This repository provides a complete end-to-end workflow—including PDF text extraction, Byte-Level BPE tokenizer training, model definition with custom Rotary Position Embeddings and grouped-query attention, and a streamlined training loop with checkpointing followed by the original Qwen3 0.6B LLM. Designed for clarity and modularity, it’s ideal for experimenting with Transformer architectures on small, self-contained datasets, rapid iteration, and educational purposes. Whether you’re exploring novel position-encoding schemes or learning the nuts and bolts of language model pretraining, this project gives you a hands-on framework to build, train, and extend your own LLM from scratch.

---


## 🧠 Architecture Comparison

![1750423589554](https://github.com/user-attachments/assets/665f2734-e34d-4ec6-86dd-5f7712da9688)

**Figure: Architecture comparison between LLaMA-3 8B and Qwen Scratch 0.6B (this project).**  
This implementation replicates a **Qwen-style 0.6B parameter model**, intentionally scaled for single-GPU environments and educational experimentation.

```bash

| Feature                | LLaMA-3 8B         | Qwen Scratch 0.6B (This Project) |
|------------------------|--------------------|----------------------------------|
| **# Parameters**       | ~8 Billion         | ~0.6 Billion                     |
| **Layers**             | 32                 | 20                               |
| **Hidden Size**        | 8192               | 3072                             |
| **Attention Heads**    | 32                 | 16                               |
| **Attention Type**     | Grouped Query (GQA)| Grouped Query (GQA)              |
| **Norm Type**          | RMSNorm            | RMSNorm                          |
| **Position Embeddings**| RoPE               | RoPE                             |
| **Context Length**     | 8K                 | 1024 (default)                   |

```


> **Why 0.6B?**  
> This model size was chosen to balance **training feasibility** and **architectural completeness**. It includes all modern transformer components—like Grouped Query Attention (GQA), Rotary Position Embeddings (RoPE), and RMSNorm—while remaining trainable from scratch using a single GPU on limited data. It’s perfect for:
> 
> - Prototyping new architectural ideas (e.g., alternative normalization or position encoding)  
> - Educational deep dives into LLM training internals  
> - Testing small-scale capabilities of instruction-following or code modeling tasks




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
* **Tokenizer**: Byte-Level BPE (30k vocab) trained on a PDF text.
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

## 👨‍💼 Author

**Elias Hossain**  
_Machine Learning Researcher | PhD Student | AI x Reasoning Enthusiast_

[![GitHub](https://img.shields.io/badge/GitHub-EliasHossain001-blue?logo=github)](https://github.com/EliasHossain001)
