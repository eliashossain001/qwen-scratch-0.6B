from tokenizers import ByteLevelBPETokenizer

tok = ByteLevelBPETokenizer("data/tokenizer/vocab.json", "data/tokenizer/merges.txt")
sample = "Once upon a time in a PDF book…"
ids = tok.encode(sample).ids
print("Token IDs:", ids)
print("Decoded:", tok.decode(ids))
