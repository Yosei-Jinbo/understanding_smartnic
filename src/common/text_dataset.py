import os
import hashlib
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# キャッシュパスは TOKENIZED_CACHE_DIR で上書き可(デフォルト ~/.cache/hf_tokenized)。
_TOKENIZED_CACHE_DIR = Path(
    os.environ.get("TOKENIZED_CACHE_DIR", Path.home() / ".cache" / "hf_tokenized")
)


def _tokenized_cache_path(tokenizer_name: str, dataset_id: str, split: str) -> Path:
    """cache file path: tokenizer + dataset + split を含む一意な名前"""
    key = f"{tokenizer_name}__{dataset_id}__{split}"
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    safe = key.replace("/", "_")
    return _TOKENIZED_CACHE_DIR / f"{safe}__{h}.pt"


def _tokenize_and_concat_cached(
    split_obj,
    tokenizer,
    tokenizer_name: str,
    dataset_id: str,
    split_name: str,
) -> torch.Tensor:
    """split のテキストを全て tokenize → 連結した LongTensor を返す。
    結果は (tokenizer, dataset, split) 単位でキャッシュする。
    """
    cache_path = _tokenized_cache_path(tokenizer_name, dataset_id, split_name)

    if cache_path.exists():
        print(f"[CausalLM cache] hit:  {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    print(f"[CausalLM cache] miss: {cache_path} (初回のみ tokenize 実行)")

    texts = [t for t in split_obj["text"] if t.strip()]

    # 大きなバッチサイズで一気に tokenize(fast tokenizer の multi-threaded 経路に乗る)
    batch_size = 1024
    all_ids: list[int] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(chunk, return_attention_mask=False, add_special_tokens=False)
        for ids in enc["input_ids"]:
            all_ids.extend(ids)

    t = torch.tensor(all_ids, dtype=torch.long)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(t, tmp_path)
    os.replace(tmp_path, cache_path)  # atomic rename
    print(f"[CausalLM cache] saved: {cache_path} ({t.numel()} tokens)")

    return t

class HFTextClassificationTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, tokenizer, text_col="text", label_col="label", max_length=256):
        self.hf_split = hf_split
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_col = label_col
        self.max_length = max_length

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        ex = self.hf_split[idx]
        enc = self.tokenizer(
            ex[self.text_col],
            truncation=True,
            max_length=self.max_length,
            padding=False,          # padding は collate_fn 側で
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),            # [L]
            "attention_mask": enc["attention_mask"].squeeze(0),  # [L]
            "labels": torch.tensor(int(ex[self.label_col]), dtype=torch.long),
        }

def get_text_datasets(
    dataset_name="imdb",
    model_name="distilbert/distilbert-base-uncased",
    max_length=256,
):
    ds = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = HFTextClassificationTorchDataset(ds["train"], tokenizer, max_length=max_length)
    test_dataset  = HFTextClassificationTorchDataset(ds["test"],  tokenizer, max_length=max_length)
    return train_dataset, test_dataset, tokenizer


# Causal LM 用データセット (WikiText-103 等)
class CausalLMDataset(torch.utils.data.Dataset):
    """テキスト全体をトークナイズし、固定長チャンクに分割した Causal LM 用データセット"""

    def __init__(self, token_ids, seq_len):
        n = len(token_ids) // seq_len
        self.data = token_ids[: n * seq_len].view(n, seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return {
            "input_ids": chunk,
            "attention_mask": torch.ones_like(chunk),
            "labels": chunk.clone(),
        }


_CAUSAL_LM_TOKENIZERS = {
    "opt-1.3b": "facebook/opt-1.3b",
    "opt_1.3b": "facebook/opt-1.3b",
    "llama-7b": "meta-llama/Llama-2-7b-hf",
    "llama_7b": "meta-llama/Llama-2-7b-hf",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-3b": "meta-llama/Llama-3.2-3B",
    "llama_3b": "meta-llama/Llama-3.2-3B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
}


def get_causal_lm_datasets(
    dataset_name="wikitext-103",
    model_name="opt-1.3b",
    seq_len=1024,
):
    """WikiText-103 等の Causal LM データセットを返す。"""
    if dataset_name in ("wikitext-103", "wikitext_103"):
        dataset_id = "wikitext-103-raw-v1"
    else:
        dataset_id = dataset_name

    tokenizer_name = _CAUSAL_LM_TOKENIZERS.get(model_name.lower(), model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # pad_token が無いモデル (LLaMA 等) への対処
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # cache hit 時は load_dataset を呼ばずに済ませたい(HF HTTP 待ちも避ける)。
    train_cache = _tokenized_cache_path(tokenizer_name, dataset_id, "train")
    test_cache = _tokenized_cache_path(tokenizer_name, dataset_id, "test")

    if train_cache.exists() and test_cache.exists():
        print(f"[CausalLM cache] both splits cached, skipping load_dataset")
        ds = None
    else:
        if dataset_name in ("wikitext-103", "wikitext_103"):
            ds = load_dataset("wikitext", "wikitext-103-raw-v1")
        else:
            ds = load_dataset(dataset_name)

    train_ids = _tokenize_and_concat_cached(
        ds["train"] if ds is not None else None,
        tokenizer, tokenizer_name, dataset_id, "train",
    )
    test_ids = _tokenize_and_concat_cached(
        ds["test"] if ds is not None else None,
        tokenizer, tokenizer_name, dataset_id, "test",
    )

    train_dataset = CausalLMDataset(train_ids, seq_len)
    test_dataset = CausalLMDataset(test_ids, seq_len)

    print(f"[CausalLM] tokenizer={tokenizer_name} seq_len={seq_len}")
    print(f"[CausalLM] train: {len(train_ids)} tokens → {len(train_dataset)} chunks")
    print(f"[CausalLM] test:  {len(test_ids)} tokens → {len(test_dataset)} chunks")

    return train_dataset, test_dataset, tokenizer


# Masked Language Model (DeBERTa 等)
_MLM_TOKENIZERS = {
    "deberta-xl": "microsoft/deberta-xlarge",
    "deberta_xl": "microsoft/deberta-xlarge",
}


class MLMDataset(torch.utils.data.Dataset):
    """Masked Language Model dataset with dynamic masking (15%)"""

    def __init__(self, token_ids: torch.Tensor, seq_len: int, mask_token_id: int,
                 mlm_probability: float = 0.15):
        n = len(token_ids) // seq_len
        self.data = token_ids[: n * seq_len].view(n, seq_len)
        self.mask_token_id = mask_token_id
        self.mlm_probability = mlm_probability

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        input_ids = self.data[idx].clone()
        labels = input_ids.clone()
        mask = torch.rand(len(input_ids)) < self.mlm_probability
        input_ids[mask] = self.mask_token_id
        labels[~mask] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
        }


def get_mlm_datasets(
    dataset_name="wikitext-103",
    model_name="deberta-xl",
    seq_len=512,
):
    if dataset_name in ("wikitext-103", "wikitext_103"):
        dataset_id = "wikitext-103-raw-v1"
    else:
        dataset_id = dataset_name

    tokenizer_name = _MLM_TOKENIZERS.get(model_name.lower(), model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_cache = _tokenized_cache_path(tokenizer_name, dataset_id, "train")
    test_cache = _tokenized_cache_path(tokenizer_name, dataset_id, "test")

    if train_cache.exists() and test_cache.exists():
        print(f"[MLM cache] both splits cached, skipping load_dataset")
        ds = None
    else:
        if dataset_name in ("wikitext-103", "wikitext_103"):
            ds = load_dataset("wikitext", "wikitext-103-raw-v1")
        else:
            ds = load_dataset(dataset_name)

    train_ids = _tokenize_and_concat_cached(
        ds["train"] if ds is not None else None,
        tokenizer, tokenizer_name, dataset_id, "train",
    )
    test_ids = _tokenize_and_concat_cached(
        ds["test"] if ds is not None else None,
        tokenizer, tokenizer_name, dataset_id, "test",
    )

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")

    train_dataset = MLMDataset(train_ids, seq_len, mask_token_id)
    test_dataset = MLMDataset(test_ids, seq_len, mask_token_id)

    print(f"[MLM] tokenizer={tokenizer_name} seq_len={seq_len}")
    print(f"[MLM] train: {len(train_ids)} tokens → {len(train_dataset)} chunks")
    print(f"[MLM] test:  {len(test_ids)} tokens → {len(test_dataset)} chunks")

    return train_dataset, test_dataset, tokenizer
