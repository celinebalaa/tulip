# Added by celinebalaa (2026-11), not part of upstream TULIP-TCR; GPL-3.0 like the rest of this repository.
"""
tulip_tcr.api against the public checkpoint shipped in model_weights/ (hidden 128, shallow config).

The oracle is upstream's own path: a CSV read by TCRDataset and scored by get_logscore, as predict.py does.
"""

import os
import warnings

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from src import api  # noqa: E402
from src.multiTrans import TCRDataset, get_logscore  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "model_weights", "pytorch_model.bin")
CONFIG = os.path.join(ROOT, "configs", "shallow.config.json")
CDR3 = ["CASSLAPGATNEKLFF", "CASSPDRGEQYF", "CASSQETQYF", "CSARDLGGTDTQYF", "CASSLGQGNTEAFF",
        "CASSIRSSYEQYF", "CAWSVGDTQYF", "CASSLTGGNYGYTF", "CASSYSGGAEQFF", "CATSDLRGNQPQHF", "CASS"]
PEPTIDES = ["GILGFVFTL", "NLVPMVATV", "VEALYLVCG"]


@pytest.fixture(scope="module")
def handle():
    return api.load(WEIGHTS, CONFIG, device="cpu")


def csv_scores(handle, tmp_path, cdr3b, peptide):
    """Upstream predict.py path: CSV -> TCRDataset(target_peptide) -> get_logscore, times -1."""
    path = tmp_path / f"{peptide}.csv"
    pd.DataFrame({"CDR3a": "<MIS>", "CDR3b": cdr3b, "MHC": "<MIS>", "peptide": peptide, "binder": 1}).to_csv(path)
    ds = TCRDataset(str(path), handle.tokenizer, handle.device, target_peptide=peptide, mhctok=handle.mhctok)
    return -1 * np.array(get_logscore(ds, handle.model, ignore_index=handle.tokenizer.pad_token_id))


def test_tokenizer_and_checkpoint_sizes(handle):
    """The sizes the plan and tcrkit's register_model assert."""
    assert len(handle.tokenizer._tokenizer.get_vocab()) == 30
    assert len(handle.mhctok._tokenizer.get_vocab()) == 55
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    assert tuple(state["mhc_embeddings.weight"].shape) == (55, 128)
    words = [k for k in state if k.endswith("embeddings.word_embeddings.weight")]
    assert words and all(tuple(state[k].shape) == (30, 128) for k in words)


def test_score_shape_finite_and_sign(handle):
    out = api.score(handle, CDR3, PEPTIDES)
    assert out.shape == (len(CDR3), 3) and np.isfinite(out).all()
    assert (out < 0).all()                      # -1 * summed NLL: a log-likelihood


def test_matches_upstream_csv_path(handle, tmp_path):
    out = api.score(handle, CDR3, PEPTIDES)
    for j, pep in enumerate(PEPTIDES):
        np.testing.assert_allclose(out[:, j], csv_scores(handle, tmp_path, CDR3, pep), rtol=0, atol=1e-6)


def test_score_does_not_depend_on_batch_composition(handle):
    """Padding is masked: a sequence scores the same alone as inside a batch (tolerance, not bitwise)."""
    together = api.score(handle, CDR3, PEPTIDES[:1])[:, 0]
    alone = np.array([api.score(handle, [c], PEPTIDES[:1])[0, 0] for c in CDR3])
    np.testing.assert_allclose(together, alone, rtol=0, atol=1e-5)


def test_placeholders(handle):
    base = api.score(handle, CDR3[:4], PEPTIDES[:1])
    np.testing.assert_array_equal(base, api.score(handle, CDR3[:4], PEPTIDES[:1], cdr3a="<MIS>", mhc="<MIS>"))
    np.testing.assert_array_equal(base, api.score(handle, CDR3[:4], PEPTIDES[:1], cdr3a=["<MIS>"] * 4))
    with pytest.raises(ValueError, match="cdr3a"):
        api.score(handle, CDR3[:4], PEPTIDES[:1], cdr3a=["<MIS>"])


def test_empty_input(handle):
    assert api.score(handle, [], PEPTIDES).shape == (0, 3)


def test_batch_size_is_not_silently_ignored(handle):
    with pytest.warns(UserWarning, match="ignored"):
        api.score(handle, CDR3[:2], PEPTIDES[:1], batch_size=512)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        api.score(handle, CDR3[:2], PEPTIDES[:1], batch_size=100)
        api.score(handle, CDR3[:2], PEPTIDES[:1])


def test_dataset_selects_exact_peptide(tmp_path):
    """Patch 3: a shorter peptide must not admit the rows of a longer one that contains it."""
    path = tmp_path / "panel.csv"
    pd.DataFrame({"CDR3a": "<MIS>", "CDR3b": ["CASSA", "CASSB", "CASSC"], "MHC": "<MIS>",
                  "peptide": ["VEALYLVCG", "VEALYLVCGERG", "VEALYLVCGERG"], "binder": 1}).to_csv(path)
    tok, mhctok = api.load_tokenizers()
    short = TCRDataset(str(path), tok, torch.device("cpu"), target_peptide="VEALYLVCG", mhctok=mhctok)
    assert short.beta == ["CASSA"]
    long = TCRDataset(str(path), tok, torch.device("cpu"), target_peptide="VEALYLVCGERG", mhctok=mhctok)
    assert long.beta == ["CASSB", "CASSC"]
    both = TCRDataset(str(path), tok, torch.device("cpu"), target_peptide=["VEALYLVCG", "VEALYLVCGERG"],
                      mhctok=mhctok)
    assert len(both.beta) == 3                  # a collection still means membership


def test_resolve_device():
    assert api.resolve_device("cpu") == torch.device("cpu")
    assert api.resolve_device("auto").type in ("cuda", "mps", "cpu")
    assert api.resolve_device(torch.device("cpu")) == torch.device("cpu")


def test_load_accepts_config_dict(handle):
    import json
    h2 = api.load(WEIGHTS, json.load(open(CONFIG)), device="cpu")
    np.testing.assert_array_equal(api.score(handle, CDR3[:3], PEPTIDES[:1]), api.score(h2, CDR3[:3], PEPTIDES[:1]))


# ------------------------------------------------------------------ cross-version oracle and score direction
FIXTURE = os.path.join(ROOT, "tests", "data", "upstream_scores_transformers_4.32.1.csv")


def test_matches_pristine_upstream(handle):
    """
    Scores written by unmodified upstream predict.py on transformers 4.32.1 (see tests/data/README.md).

    The tolerance is float32 rounding between transformers versions (different attention kernels): observed max
    absolute difference 1.2e-5 on scores up to -16, i.e. about 1e-6 relative, with identical rankings.
    """
    fx = pd.read_csv(FIXTURE)
    for pep, g in fx.groupby("peptide", sort=False):
        got = api.score(handle, g["CDR3b"].tolist(), [pep])[:, 0]
        np.testing.assert_allclose(got, g["score"].to_numpy(), rtol=1e-5, atol=3e-5)
        assert pd.Series(got).corr(g["score"].reset_index(drop=True), method="spearman") > 0.99999


def test_higher_score_means_binder():
    """Direction check on the labelled data shipped with upstream: higher score separates binders (AUC > 0.5)."""
    from sklearn.metrics import roc_auc_score
    h = api.load(WEIGHTS, CONFIG, device="cpu")
    df = pd.read_csv(os.path.join(ROOT, "data", "VDJ_test_2.csv"))
    d = df[df.peptide == "GILGFVFTL"].sample(300, random_state=0)
    scores = api.score(h, d["CDR3b"].tolist(), ["GILGFVFTL"])[:, 0]       # beta-only, <MIS> placeholders (as tcrkit)
    assert roc_auc_score(d["binder"], scores) > 0.8                       # observed 0.908 on 400 rows
    assert roc_auc_score(d["binder"], -scores) < 0.2


def test_huggingface_style_config(handle):
    """The config.json saved next to the weights ({"encoder": ..., "decoder": ...}) loads and scores identically."""
    import json
    hf = os.path.join(ROOT, "model_weights", "config.json")
    assert "encoder" in json.load(open(hf))
    assert api.normalize_config(json.load(open(hf))) == json.load(open(CONFIG))
    h2 = api.load(WEIGHTS, hf, device="cpu")
    np.testing.assert_array_equal(api.score(handle, CDR3[:3], PEPTIDES[:1]), api.score(h2, CDR3[:3], PEPTIDES[:1]))


def test_normalize_config_rejects_mismatch_and_missing():
    with pytest.raises(ValueError, match="differ"):
        api.normalize_config({"encoder": {"num_attention_heads": 1, "num_hidden_layers": 2, "hidden_size": 128},
                              "decoder": {"num_attention_heads": 2, "num_hidden_layers": 2, "hidden_size": 128}})
    with pytest.raises(ValueError, match="lacks"):
        api.normalize_config({"hidden_size": 128})
