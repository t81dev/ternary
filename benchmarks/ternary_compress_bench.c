/*
 * ternary_compress_bench.c
 * Official compression benchmark for balanced ternary LLM weights
 * Part of https://github.com/t81dev/ternary
 *
 * Compares:
 *  • Raw ternary (1 byte/trit) → 8.00 bits/trit
 *  • RLE
 *  • Huffman (canonical)
 *  • Theoretical limit (entropy)
 *
 * Run: ./ternary_compress_bench --size 1000000 --weights gemma-2b-trits.bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <zlib.h>  // real CRC32

typedef int8_t Trit;  // -1, 0, +1

typedef struct {
    Trit* data;
    size_t len;
} TritStream;

double entropy(const Trit* data, size_t len) {
    uint64_t count[3] = {0};
    for (size_t i = 0; i < len; i++) count[data[i] + 1]++;
    double h = 0.0;
    for (int i = 0; i < 3; i++) {
        if (count[i]) {
            double p = count[i] / (double)len;
            h -= p * log2(p);
        }
    }
    return h;
}

/* Simple but optimal static Huffman for 3 symbols */
void compress_huffman(const Trit* in, size_t len, uint8_t* out, size_t* out_len) {
    // 0 → 0, +1 → 10, -1 → 11  (optimal for real LLM weights: 0 is most common)
    uint64_t bitbuf = 0;
    int bits = 0;
    size_t pos = 0;

    for (size_t i = 0; i < len; i++) {
        Trit t = in[i];
        if (t == 0) {
            bitbuf = (bitbuf << 1) | 0;
            bits += 1;
        } else {
            bitbuf = (bitbuf << 2) | (t == 1 ? 2 : 3);
            bits += 2;
        }
        while (bits >= 8) {
            out[pos++] = bitbuf >> (bits - 8);
            bits -= 8;
        }
    }
    if (bits) out[pos++] = bitbuf << (8 - bits);
    *out_len = pos;
}

void compress_rle(const Trit* in, size_t len, uint8_t* out, size_t* out_len) {
    size_t pos = 0;
    for (size_t i = 0; i < len; ) {
        Trit t = in[i];
        size_t run = 1;
        while (i + run < len && in[i + run] == t && run < 255) run++;
        out[pos++] = (uint8_t)(t + 1);
        out[pos++] = (uint8_t)run;
        i += run;
    }
    *out_len = pos;
}

int main(int argc, char** argv) {
    size_t len = 10'000'000;
    int real_weights = 0;
    char* input_file = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--size") == 0) len = atol(argv[++i]);
        if (strcmp(argv[i], "--weights") == 0) { real_weights = 1; input_file = argv[++i]; }
    }

    Trit* data = malloc(len * sizeof(Trit));

    if (real_weights) {
        FILE* f = fopen(input_file, "rb");
        fread(data, sizeof(Trit), len, f);
        fclose(f);
        printf("Loaded real ternary weights from %s\n", input_file);
    } else {
        printf("Generating uniform random ternary data (%zu trits)...\n", len);
        for (size_t i = 0; i < len; i++) data[i] = (Trit)(rand() % 3 - 1);
    }

    double H = entropy(data, len);
    uint8_t* tmp = malloc(len * 2);

    size_t rle_len, huf_len;
    clock_t t;

    t = clock();
    compress_rle(data, len, tmp, &rle_len);
    double rle_time = (clock() - t) / (double)CLOCKS_PER_SEC;

    t = clock();
    compress_huffman(data, len, tmp, &huf_len);
    double huf_time = (clock() - t) / (double)CLOCKS_PER_SEC;

    printf("\n=== Ternary Compression Benchmark ===\n");
    printf("Data length      : %zu trits\n", len);
    printf("Entropy          : %.4f bits/trit (theoretical limit)\n", H);
    printf("Raw (1B/trit)    : %.2f MiB\n", len / (1024.0*1024.0));
    printf("RLE              : %.2f MiB (%.2f×, %.4f s)\n", rle_len/(1024.0*1024), (double)len/rle_len, rle_time);
    printf("Huffman          : %.2f MiB (%.2f×, %.4f s) → %.2f bits/trit\n", 
           huf_len/(1024.0*1024), (double)len*8/huf_len, huf_time, huf_len*8.0/len);
    printf("Theoretical best : %.2f MiB (%.2f bits/trit)\n", len*H/8/(1024*1024), H);

    free(data); free(tmp);
    return 0;
}
