from tokenizers import ByteLevelBPETokenizer

# 1. Initialize
tokenizer = ByteLevelBPETokenizer()

# 2. Train on your extracted text
tokenizer.train(
    files=["data/book.txt"],
    vocab_size=30_000,
    min_frequency=2,
    special_tokens=[
        "<s>", "<pad>", "</s>", "<unk>", "<mask>",
    ],
)

# 3. Save to disk (creates vocab.json & merges.txt)
tokenizer.save_model("data/tokenizer")
print("Tokenizer files written to data/tokenizer/")
