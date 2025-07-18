# ==================== src/config.py ====================
config = {
    # Model architecture
    "vocab_size": 30000,
    "hidden_size": 512,              # ↓ reduced from 1024
    "intermediate_size": 1536,       # ↓ reduced from 3072
    "num_attention_heads": 8,        # ↓ reduced from 16
    "num_hidden_layers": 20,         # ↓ reduced from 28
    "max_position_embeddings": 4096,
    "rotary_pct": 0.25,
    "rotary_base": 10000,
    "rms_norm_eps": 1e-6,

    # Training hyperparameters
    "batch_size": 1,     # ↓ reduced from 8
    "seq_len": 1024,     # ↓ new: shorter context to save memory
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_steps": 1000,
    "total_steps": 50000,

    # Device
    "device": "cuda" if __import__('torch').cuda.is_available() else "cpu",

    # Checkpointing
    "save_dir": "./checkpoints",
    "save_every": 5000,
}
