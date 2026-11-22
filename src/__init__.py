# Added by celinebalaa (2026-11), not part of upstream TULIP-TCR; GPL-3.0 like the rest of this repository.
"""tulip_tcr: fork of TULIP-TCR (https://github.com/barthelemymp/TULIP-TCR) with an installable, in-memory API.

``import tulip_tcr`` does not import torch; ``load`` / ``score`` do, on first use.
"""

__version__ = "0.1.0"
__commit__ = None           # upstream commit this fork branches from, see README "Changes from upstream"
UPSTREAM_COMMIT = "see README"


def load(weights, config, device="auto"):
    """Build the model and load ``weights`` (see :func:`tulip_tcr.api.load`)."""
    from .api import load as _load
    return _load(weights, config, device)


def score(handle, cdr3b, peptides, cdr3a=None, mhc=None, batch_size=None):
    """Log-likelihood of each peptide given each CDR3b (see :func:`tulip_tcr.api.score`)."""
    from .api import score as _score
    return _score(handle, cdr3b, peptides, cdr3a=cdr3a, mhc=mhc, batch_size=batch_size)


def mhc_vocabulary(handle):
    """The MHC strings the model knows (see :func:`tulip_tcr.api.mhc_vocabulary`)."""
    from .api import mhc_vocabulary as _v
    return _v(handle)


__all__ = ["__version__", "load", "mhc_vocabulary", "score"]
