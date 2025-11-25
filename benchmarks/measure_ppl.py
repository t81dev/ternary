#!/usr/bin/env python3
"""
measure_ppl.py - Compute WikiText-2 perplexity for any GGUF model using llama.cpp
Usage: python3 measure_ppl.py path/to/model.gguf
"""

import subprocess
import sys
import os
import re
from pathlib import Path

# Download wikitest-2-raw-v1 if not exists
WIKI_URL = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/test.txt"
WIKI_FILE = Path("benchmarks/wikitext2-test.txt")

def download_wikitext2():
    if WIKI_FILE.exists():
        return
    print("Downloading WikiText-2 test set...")
    os.makedirs(WIKI_FILE.parent, exist_ok=True)
    subprocess.run(["curl", "-s", "-L", WIKI_URL, "-o", str(WIKI_FILE)], check=True)

def run_llama_cpp(model_path: str):
    cmd = [
        "./llama.cpp/main",           # adjust if your binary has different name/location
        "-m", model_path,
        "--file", str(WIKI_FILE),
        "--temp", "0.0",
        "--log-disable",
        "-p", " ",
        "-n", "0",                    # predict until EOF
        "--verbose-prompt"
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout + result.stderr

def extract_perplexity(output: str) -> float:
    # llama.cpp prints: "perplexity = X.XXXXX"
    match = re.search(r"perplexity\s*=\s*([0-9.]+)", output, re.IGNORECASE)
    if not match:
        print("Perplexity not found! Raw output:")
        print(output[-1000:])
        sys.exit(1)
    return float(match.group(1))

def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    model_path = Path(sys.argv[1])
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    download_wikitext2()

    print(f"Measuring perplexity for {model_path.name} on WikiText-2...")
    output = run_llama_cpp(str(model_path))
    ppl = extract_perplexity(output)

    size_gb = model_path.stat().st_size / (1024**3)

    print("\n" + "="*50)
    print(f"MODEL:     {model_path.name}")
    print(f"SIZE:      {size_gb:.2f} GB")
    print(f"WikiText-2 Perplexity: {ppl:.4f}")
    print("="*50)

    # Auto-update README table
    readme_path = Path("README.md")
    if readme_path.exists():
        update_readme_table(model_path.name, size_gb, ppl)

def update_readme_table(model_name: str, size_gb: float, ppl: float):
    import textwrap
    new_row = f"| **{model_name}** | **{size_gb:.2f} GB** | **{ppl:.2f}** |"
    
    readme = Path("README.md").read_text()
    marker = "<!-- PPL_TABLE -->"
    if marker not in readme:
        print("Warning: README marker not found!")
        return

    lines = readme.splitlines()
    table_start = next(i for i, line in enumerate(lines) if marker in line)
    table_end = next(i for i, line in enumerate(lines[table_start:]) if line.strip() == "") + table_start

    header = [
        "| Method       | Size     | WikiText-2 PPL |",
        "|--------------|----------|---------------|",
        "| FP16         | 4.80 GB  | 5.91          |",
        "| Q4_K_M       | 2.80 GB  | 6.48          |",
        new_row,
    ]

    new_table = "\n".join(header) + "\n"
    new_readme = "\n".join(lines[:table_start+1]) + "\n" + new_table + "\n".join(lines[table_end:])

    Path("README.md").write_text(new_readme)
    print("README.md table automatically updated!")

if __name__ == "__main__":
    main()
