"""Shared pytest configuration.

Unigram-LM training delegates to HuggingFace's ``tokenizers`` ``UnigramTrainer``,
whose ``maybe_par_*`` E-step is a parallel ``f64`` reduction. Pin parallelism off
so training runs sequentially and reproducibly (the §3.2 determinism amendment);
this also silences the post-fork ``TOKENIZERS_PARALLELISM`` warning. Set before
``smirk`` is imported so the Rust side reads it on first use.
"""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
