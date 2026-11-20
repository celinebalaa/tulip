`upstream_scores_transformers_4.32.1.csv`: scores written by **unmodified upstream `predict.py`** (commit 798fab9) run with
`transformers==4.32.1`, `tokenizers==0.13.3`, torch 2.15, CPU, on `model_weights/pytorch_model.bin` +
`configs/shallow.config.json`, input CSV with `CDR3a=<MIS>`, `MHC=<MIS>`, `binder=1`. It is the cross-version oracle for
`tests/test_api.py::test_matches_pristine_upstream`.
