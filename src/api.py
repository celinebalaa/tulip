# Added by celinebalaa (2026-11), not part of upstream TULIP-TCR; GPL-3.0 like the rest of this repository.
"""
Programmatic API of the tulip_tcr fork: build the model once, then score CDR3b x peptide in memory.

``load`` is the model/tokenizer boilerplate of upstream ``predict.py`` ``main()`` lifted into a function.
``score_reference`` is upstream's own scoring path (``TCRDataset`` + ``get_logscore``) run per peptide, kept
unchanged as the oracle every optimised path is tested against. ``score`` is the function tcrkit calls.

Score convention (as upstream ``predict.py``): ``score = -1 * get_logscore`` = log-likelihood of the peptide
given the TCR. Higher means more model-compatible; comparable only for a fixed peptide.
"""

import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, BertConfig, BertModel

from .multiTrans import BertLastPooler, TCRDataset, Tulip, TulipPetal, get_logscore

MIS = "<MIS>"
UPSTREAM_BATCH_SIZE = 100       # hard-coded in upstream get_logscore's own DataLoader


@dataclass
class TulipHandle:
    """A loaded model with the tokenizers it needs."""
    model: Any
    tokenizer: Any
    mhctok: Any
    device: torch.device
    pad_token_id: int


def resolve_device(device: Union[str, torch.device] = "auto") -> torch.device:
    """``'auto'``: CUDA if available, else Apple MPS, else CPU."""
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _data_dir(name: str) -> str:
    """Tokenizer / config directory: next to this package once installed, else at the repository root."""
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (here, os.path.dirname(here)):
        path = os.path.join(base, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"{name}/ not found next to {here}")


def load_tokenizers():
    """``(aa tokenizer, mhc tokenizer)`` exactly as upstream ``predict.py`` builds them."""
    tokenizer = AutoTokenizer.from_pretrained(_data_dir("aatok"))
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({"sep_token": MIS})
    if tokenizer.cls_token is None:
        tokenizer.add_special_tokens({"cls_token": "<CLS>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<EOS>"})
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<MASK>"})
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single="<CLS> $A <EOS>",
        pair="<CLS> $A <MIS> $B:1 <EOS>:1",
        special_tokens=[("<EOS>", 2), ("<CLS>", 3), ("<MIS>", 4)],
    )
    mhctok = AutoTokenizer.from_pretrained(_data_dir("mhctok"))
    return tokenizer, mhctok


def build_model(modelconfig: dict, tokenizer, mhctok) -> Tulip:
    """Untrained Tulip (three encoders, three decoders) for ``modelconfig`` (num_attn_heads, num_hidden_layers, hidden_size)."""
    vocabsize = len(tokenizer._tokenizer.get_vocab())
    mhcvocabsize = len(mhctok._tokenizer.get_vocab())
    encoder_config = BertConfig(
        vocab_size=vocabsize,
        max_position_embeddings=50,
        num_attention_heads=modelconfig["num_attn_heads"],
        num_hidden_layers=modelconfig["num_hidden_layers"],
        hidden_size=modelconfig["hidden_size"],
        type_vocab_size=1,
        pad_token_id=tokenizer.pad_token_id)
    encoder_config.mhc_vocab_size = mhcvocabsize
    encoderA, encoderB, encoderE = (BertModel(config=encoder_config) for _ in range(3))

    decoder_config = BertConfig(
        vocab_size=vocabsize,
        max_position_embeddings=50,
        num_attention_heads=modelconfig["num_attn_heads"],
        num_hidden_layers=modelconfig["num_hidden_layers"],
        hidden_size=modelconfig["hidden_size"],
        type_vocab_size=1,
        is_decoder=True,
        pad_token_id=tokenizer.pad_token_id)
    decoder_config.add_cross_attention = True
    decoders = []
    for _ in range(3):
        d = TulipPetal(config=decoder_config)
        d.pooler = BertLastPooler(config=decoder_config)
        decoders.append(d)
    return Tulip(encoderA=encoderA, encoderB=encoderB, encoderE=encoderE,
                 decoderA=decoders[0], decoderB=decoders[1], decoderE=decoders[2])


def normalize_config(config: dict) -> dict:
    """
    ``{num_attn_heads, num_hidden_layers, hidden_size}`` from either config format.

    Upstream's ``configs/*.config.json`` already have that form. A HuggingFace ``config.json`` saved with the model
    (``{"encoder": {...}, "decoder": {...}}``) is reduced to it; encoder and decoder must agree, as predict.py builds
    both from one set of numbers.
    """
    if "encoder" not in config:
        missing = [k for k in ("num_attn_heads", "num_hidden_layers", "hidden_size") if k not in config]
        if missing:
            raise ValueError(f"config lacks {missing}")
        return {k: config[k] for k in ("num_attn_heads", "num_hidden_layers", "hidden_size")}
    enc, dec = config["encoder"], config.get("decoder", config["encoder"])
    out = {"num_attn_heads": enc["num_attention_heads"], "num_hidden_layers": enc["num_hidden_layers"],
           "hidden_size": enc["hidden_size"]}
    other = {"num_attn_heads": dec["num_attention_heads"], "num_hidden_layers": dec["num_hidden_layers"],
             "hidden_size": dec["hidden_size"]}
    if out != other:
        raise ValueError(f"encoder {out} and decoder {other} differ: not a TULIP shallow-style model")
    return out


def load(weights: str, config: Union[str, dict], device: Union[str, torch.device] = "auto") -> TulipHandle:
    """
    Build the model for ``config`` (a path to the json, or the dict; upstream's small format or a HuggingFace ``config.json``), load ``weights`` and move it to ``device``.

    ``map_location`` is the device itself, so a checkpoint saved on CUDA loads on CPU or MPS.
    """
    if not isinstance(config, dict):
        with open(config) as fh:
            config = json.load(fh)
    config = normalize_config(config)
    dev = resolve_device(device)
    tokenizer, mhctok = load_tokenizers()
    model = build_model(config, tokenizer, mhctok)
    model.load_state_dict(torch.load(weights, map_location=dev, weights_only=True))
    model.to(dev)
    model.eval()
    return TulipHandle(model=model, tokenizer=tokenizer, mhctok=mhctok, device=dev,
                       pad_token_id=tokenizer.pad_token_id)


def _per_row(value: Union[None, str, Sequence[str]], n: int, name: str) -> list:
    if value is None:
        return [MIS] * n
    if isinstance(value, str):
        return [value] * n
    value = list(value)
    if len(value) != n:
        raise ValueError(f"{name} has {len(value)} entries for {n} CDR3b")
    return value


def make_dataset(handle: TulipHandle, cdr3b: Sequence[str], peptide: str, cdr3a=None, mhc=None) -> TCRDataset:
    """In-memory dataset of ``cdr3b`` x one peptide (upstream ``empty_init`` / ``append``; ``binder=1``)."""
    n = len(cdr3b)
    alphas, mhcs = _per_row(cdr3a, n, "cdr3a"), _per_row(mhc, n, "mhc")
    ds = TCRDataset.empty_init(handle.tokenizer, handle.device, handle.mhctok)
    for a, b, m in zip(alphas, cdr3b, mhcs):
        ds.append(MHC=m, alpha=a, beta=b, peptide=peptide, binder=1)
    return ds


def score_reference(handle: TulipHandle, cdr3b: Sequence[str], peptides: Sequence[str],
                    cdr3a=None, mhc=None) -> np.ndarray:
    """
    Upstream's scoring path, unchanged: one ``get_logscore`` per peptide. Shape ``(len(cdr3b), len(peptides))``.

    Slow by design (batches of 100, a device sync per sequence, all three decoders, the encoders re-run for
    every peptide). Kept as the oracle for ``score``.
    """
    out = np.empty((len(cdr3b), len(peptides)), dtype=np.float64)
    for j, pep in enumerate(peptides):
        ds = make_dataset(handle, cdr3b, pep, cdr3a, mhc)
        out[:, j] = -1 * np.array(get_logscore(ds, handle.model, ignore_index=handle.pad_token_id))
    return out


def score(handle: TulipHandle, cdr3b: Sequence[str], peptides: Sequence[str], cdr3a=None, mhc=None,
          batch_size: Optional[int] = None) -> np.ndarray:
    """
    Log-likelihood of every peptide given every CDR3b: ``-1 * get_logscore``, shape ``(len(cdr3b), len(peptides))``.

    ``cdr3a`` / ``mhc``: ``None`` (the ``<MIS>`` placeholder, as in the historical runs), one string for all rows,
    or one per CDR3b.

    Currently this *is* :func:`score_reference`. ``batch_size`` has no effect yet because upstream
    ``get_logscore`` hard-codes 100; a value other than that warns instead of being silently ignored.
    """
    if batch_size is not None and int(batch_size) != UPSTREAM_BATCH_SIZE:
        warnings.warn(f"batch_size={batch_size} is ignored: upstream get_logscore uses {UPSTREAM_BATCH_SIZE}",
                      stacklevel=2)
    if len(cdr3b) == 0:
        return np.empty((0, len(peptides)), dtype=np.float64)
    return score_reference(handle, cdr3b, peptides, cdr3a, mhc)
