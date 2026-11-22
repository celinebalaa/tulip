# Added by celinebalaa (2026-11), not part of upstream TULIP-TCR; GPL-3.0 like the rest of this repository.
"""MHC handling of tulip_tcr.api: the closed vocabulary, strict validation, and agreement with unmodified upstream."""

import os

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from src import api  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "model_weights", "pytorch_model.bin")
CONFIG = os.path.join(ROOT, "configs", "shallow.config.json")
FIXTURE = os.path.join(ROOT, "tests", "data", "upstream_scores_mhc_transformers_4.32.1.csv")
CDR3 = ["CASSLAPGATNEKLFF", "CASSPDRGEQYF", "CASSQETQYF", "CSARDLGGTDTQYF", "CASSLGQGNTEAFF", "CASSIRSSYEQYF"]


@pytest.fixture(scope="module")
def handle():
    return api.load(WEIGHTS, CONFIG, device="cpu")


def test_vocabulary_is_the_closed_55_word_list(handle):
    v = api.mhc_vocabulary(handle)
    assert len(v) == 55 and "<MIS>" in v
    assert {"HLA-A*02:01", "HLA-B*07:02", "HLA-C*07:02", "H-2Kb", "H-2Db", "HLA class I", "HLA class II"} <= v
    assert "HLA-A*02:05" not in v and "H-2Kd" in v


def test_unknown_mhc_raises_with_suggestions(handle):
    with pytest.raises(ValueError, match="HLA-A\\*02:01"):
        api.score(handle, CDR3[:2], ["GILGFVFTL"], mhc="HLA-A*02:05")
    with pytest.raises(ValueError, match="silently become"):
        api.score(handle, CDR3[:2], ["GILGFVFTL"], mhc=["HLA-A*02:01", "typo"])
    with pytest.raises(ValueError, match="Unknown MHC"):
        api.score(handle, [], ["GILGFVFTL"], mhc="nope")                    # checked even for empty input


def test_mhc_changes_the_score_and_placeholder_equals_none(handle):
    base = api.score(handle, CDR3, ["GILGFVFTL"])
    np.testing.assert_array_equal(base, api.score(handle, CDR3, ["GILGFVFTL"], mhc="<MIS>"))
    a2 = api.score(handle, CDR3, ["GILGFVFTL"], mhc="HLA-A*02:01")
    assert np.abs(a2 - base).max() > 0.1                                       # a real effect, not noise
    assert not np.allclose(a2, api.score(handle, CDR3, ["GILGFVFTL"], mhc="HLA-A*03:01"))


def test_per_row_mhc_list_equals_separate_calls(handle):
    mix = ["HLA-A*02:01", "<MIS>", "HLA-A*02:01", "<MIS>", "H-2Kb", "<MIS>"]
    both = api.score(handle, CDR3, ["GILGFVFTL"], mhc=mix)[:, 0]
    one = {m: api.score(handle, CDR3, ["GILGFVFTL"], mhc=m)[:, 0] for m in set(mix)}
    np.testing.assert_allclose(both, [one[m][i] for i, m in enumerate(mix)], rtol=0, atol=1e-5)


def test_matches_pristine_upstream_with_mhc(handle):
    """Scores written by unmodified upstream predict.py (transformers 4.32.1) on rows that carry their own MHC."""
    fx = pd.read_csv(FIXTURE)
    for (pep, mhc), g in fx.groupby(["peptide", "mhc"], sort=False):
        got = api.score(handle, g["CDR3b"].tolist(), [pep], mhc=mhc)[:, 0]
        np.testing.assert_allclose(got, g["score"].to_numpy(), rtol=1e-5, atol=3e-5)
        assert pd.Series(got).corr(g["score"].reset_index(drop=True), method="spearman") > 0.99999
        wrong = api.score(handle, g["CDR3b"].tolist(), [pep])[:, 0]           # without the MHC it must NOT match
        assert np.abs(wrong - g["score"].to_numpy()).max() > 0.05
