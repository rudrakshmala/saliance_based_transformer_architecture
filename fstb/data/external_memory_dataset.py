# fstb/data/external_memory_dataset.py
"""Unified interface for external conversational datasets.

Supported datasets:
- PersonaChat ("persona_chat")
- MultiWOZ ("multi_woz")
- NarrativeQA ("narrative_qa")
- HotpotQA ("hotpot_qa")
- LongMemEval ("long_mem_eval") – falls back to any long‑context dataset

The class downloads via the `datasets` library, applies deterministic
splits, tokenizes with a provided tokenizer, and optionally truncates the
context to a configurable length.

All processed examples are cached on disk to make subsequent runs fast.
"""

from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

import datasets
from torch.utils.data import Dataset


def _hash_dict(d: Dict[str, Any]) -> str:
    """Return a short hash for a dict – used for cache naming."""
    json_repr = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.sha256(json_repr).hexdigest()[:8]


class ExternalMemoryDataset(Dataset):
    """A torch Dataset that yields tokenised conversational examples.

    Parameters
    ----------
    dataset_name: str
        Name of the external dataset (e.g. "persona_chat").
    split: str
        One of "train", "validation", "test".
    tokenizer: Any
        Tokenizer instance with a ``__call__(text, truncation, max_length)``
        signature (compatible with HuggingFace tokenizers).
    max_context_len: int, optional
        Maximum number of tokens for the full conversation context.
    cache_dir: str, optional
        Directory where raw and tokenised caches are stored.
    seed: int, optional
        Seed for deterministic shuffling / splitting.
    """

    SUPPORTED = {
        "persona_chat": "persona_chat",
        "multi_woz": "multi_woz",
        "narrative_qa": "narrativeqa",
        "hotpot_qa": "hotpotqa",
        "long_mem_eval": "long_mem_eval",
    }

    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer: Any,
        max_context_len: int = 1024,
        cache_dir: str = "./data/cache",
        seed: int = 42,
        **tokenizer_kwargs,
    ) -> None:
        assert split in {"train", "validation", "test"}
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = tokenizer
        self.max_context_len = max_context_len
        self.cache_dir = Path(cache_dir)
        self.seed = seed
        self.tokenizer_kwargs = tokenizer_kwargs

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_cache_path = self.cache_dir / f"{dataset_name}_{split}_raw.json"
        self.token_cache_path = self.cache_dir / (
            f"{dataset_name}_{split}_tok_{max_context_len}_{_hash_dict(tokenizer_kwargs)}.json"
        )

        if self.token_cache_path.is_file():
            # Load tokenised cache directly
            with open(self.token_cache_path, "r", encoding="utf-8") as f:
                self.examples = json.load(f)
        else:
            # Load / download raw data then tokenise
            raw = self._load_or_download_raw()
            self.examples = self._tokenise(raw)
            with open(self.token_cache_path, "w", encoding="utf-8") as f:
                json.dump(self.examples, f)

    # ---------------------------------------------------------------------
    # Raw data handling
    # ---------------------------------------------------------------------
    def _load_or_download_raw(self) -> List[Dict[str, Any]]:
        if self.raw_cache_path.is_file():
            with open(self.raw_cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Map our name to the huggingface identifier
        hf_name = self.SUPPORTED.get(self.dataset_name)
        if hf_name is None:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        ds = datasets.load_dataset(hf_name)
        # Some datasets use "validation" vs "dev" – normalise
        split_name = self.split
        if split_name not in ds:
            # try alternative naming conventions
            if split_name == "validation" and "dev" in ds:
                split_name = "dev"
            elif split_name == "test" and "validation" in ds:
                split_name = "validation"
        raw_split = ds[split_name]
        # Convert to a list of unified dialog dicts
        processed = []
        for entry in raw_split:
            processed.append(self._standardise_entry(entry))
        # Deterministic shuffle & split if needed (already split by HF)
        with open(self.raw_cache_path, "w", encoding="utf-8") as f:
            json.dump(processed, f)
        return processed

    def _standardise_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Convert dataset‑specific format into a unified dialog dict.
        Output format:
        {
            "dialog_id": str,
            "turns": List[str],   # alternating user / system utterances
            "metadata": {...}
        }
        """
        # PersonaChat example
        if self.dataset_name == "persona_chat":
            # HF version already gives "dialog" with alternating turns
            return {
                "dialog_id": entry.get("dialog_id", ""),
                "turns": entry["dialog"],
                "metadata": {"persona": entry.get("persona", [])},
            }
        elif self.dataset_name == "multi_woz":
            # MultiWOZ provides "dialogue" field
            return {
                "dialog_id": entry.get("dialogue_id", ""),
                "turns": entry["dialogue"],
                "metadata": {"domain": entry.get("domain", "")},
            }
        elif self.dataset_name == "narrative_qa":
            # Use the story and question/answer as a two‑turn dialog
            story = entry.get("story", "")
            question = entry.get("question", "")
            answer = entry.get("answer", "")
            return {
                "dialog_id": entry.get("id", ""),
                "turns": [story, f"Q: {question}", f"A: {answer}"],
                "metadata": {},
            }
        elif self.dataset_name == "hotpot_qa":
            # Use context + question + answer
            context = entry.get("context", "")
            question = entry.get("question", "")
            answer = entry.get("answer", "")
            return {
                "dialog_id": entry.get("id", ""),
                "turns": [context, f"Q: {question}", f"A: {answer}"],
                "metadata": {},
            }
        elif self.dataset_name == "long_mem_eval":
            # Assume dataset provides a list of utterances under "dialog"
            return {
                "dialog_id": entry.get("id", ""),
                "turns": entry.get("dialog", []),
                "metadata": {},
            }
        else:
            # Fallback – treat any "text" field as a single turn
            return {
                "dialog_id": entry.get("id", ""),
                "turns": [entry.get("text", "")],
                "metadata": {},
            }

    # ---------------------------------------------------------------------
    # Tokenisation
    # ---------------------------------------------------------------------
    def _tokenise(self, raw_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tokenised = []
        for ex in raw_examples:
            # Concatenate turns with a special separator token if tokenizer has one
            sep_token = self.tokenizer.sep_token or "\n"
            full_text = f" {sep_token} ".join(ex["turns"]).strip()
            toks = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_context_len,
                **self.tokenizer_kwargs,
            )
            tokenised.append(
                {
                    "dialog_id": ex["dialog_id"],
                    "input_ids": toks["input_ids"],
                    "attention_mask": toks.get("attention_mask", []),
                    "metadata": ex["metadata"],
                }
            )
        return tokenised

    # ---------------------------------------------------------------------
    # Torch Dataset API
    # ---------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]

# Helper function for deterministic reproducible splits across datasets
def deterministic_split(
    data: List[Any], train_frac: float = 0.8, val_frac: float = 0.1, seed: int = 42
) -> Tuple[List[Any], List[Any], List[Any]]:
    import random

    rng = random.Random(seed)
    shuffled = data[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]
