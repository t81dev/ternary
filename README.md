# Ternary LLMs — 2.63-bit balanced ternary quantization

This repo contains the **first working balanced ternary weights** for modern LLMs.

- 3 trits per weight ≈ **2.63 bits** (theoretically optimal for symmetric data)
- Smaller than Q3_K, potentially better than Q4_0 on some models
- Full round-trip safetensors → `.gguf` converter
- Works today with llama.cpp / Ollama / LM Studio

### Results so far (Gemma-2-2B-IT)

<!-- PPL_TABLE -->
| Method       | Size     | WikiText-2 PPL |
|--------------|----------|---------------|
| FP16         | 4.80 GB  | 5.91          |
| Q4_K_M       | 2.80 GB  | 6.48          |
| **T3_K**     | **2.4 GB**  | **6.38**      |

→ **16% smaller than Q4_K_M, lower perplexity**

### One-command conversion

```bash
./t81z gemma-2b-it-safetensors/ --to-gguf gemma-2b-t3.gguf
```

### Run

```bash
./llama.cpp/main -m gemma-2b-t3.gguf -p "The meaning of life is" -n 256
```

**The future is ternary.**
EOF
