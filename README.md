# 🎶 RaGa: A Music Transformer for MIDI Generation

RaGa is a PyTorch-based Transformer model for symbolic music generation.  
It can learn from MIDI files, tokenize them into event sequences, train on these tokens, and generate new music.  

## ✨ Features
- **MIDI → Tokens → MIDI** conversion with a custom event vocabulary
- **Preprocessing pipeline**: raw MIDI folder → tokenized dataset (`processed_data.pt`)
- **Transformer model**: configurable embedding, attention heads, layers, feed-forward size
- **Training loop**: checkpointing, resume, and validation split
- **Generation**: greedy, argmax, or top-k sampling with temperature
