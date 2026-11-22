`upstream_scores_transformers_4.32.1.csv`: scores written by **unmodified upstream `predict.py`** (commit 798fab9) run with
`transformers==4.32.1`, `tokenizers==0.13.3`, torch 2.15, CPU, on `model_weights/pytorch_model.bin` +
`configs/shallow.config.json`, input CSV with `CDR3a=<MIS>`, `MHC=<MIS>`, `binder=1`. It is the cross-version oracle for
`tests/test_api.py::test_matches_pristine_upstream`.

`upstream_scores_mhc_transformers_4.32.1.csv`: the same, but each peptide has its own MHC in the input csv (`GILGFVFTL` /
`HLA-A*02:01`, `NLVPMVATV` / `HLA-A*03:01`, `VEALYLVCG` / `H-2Kb`), 60 CDR3b each. Oracle for
`tests/test_mhc.py::test_matches_pristine_upstream_with_mhc`.
