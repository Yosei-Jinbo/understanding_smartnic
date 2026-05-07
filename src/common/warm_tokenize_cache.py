#!/usr/bin/env python3
"""
学習を走らせずに wikitext-103 + OPT tokenizer のキャッシュを事前生成する。
初回だけ 1-2 分かかるが、以降の全 run がキャッシュから秒でロードできるようになる。

Usage:
    cd /home/y-jinbo/HasegawaLab/performance_evaluation
    python -m common.warm_tokenize_cache --model opt-1.3b --dataset wikitext-103

Env:
    TOKENIZED_CACHE_DIR=~/.cache/hf_tokenized (デフォルト)
"""
import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="opt-1.3b")
    parser.add_argument("--dataset", default="wikitext-103")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--kind", choices=["causal", "mlm"], default="causal")
    args = parser.parse_args()

    t0 = time.perf_counter()
    if args.kind == "causal":
        from common.text_dataset import get_causal_lm_datasets
        train_ds, test_ds, tok = get_causal_lm_datasets(
            dataset_name=args.dataset, model_name=args.model, seq_len=args.seq_len,
        )
    else:
        from common.text_dataset import get_mlm_datasets
        train_ds, test_ds, tok = get_mlm_datasets(
            dataset_name=args.dataset, model_name=args.model, seq_len=args.seq_len,
        )
    elapsed = time.perf_counter() - t0
    print(f"[warm-up] done: {len(train_ds)} train / {len(test_ds)} test chunks, {elapsed:.1f}s")


if __name__ == "__main__":
    main()
