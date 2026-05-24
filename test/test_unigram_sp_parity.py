"""Differential parity: the native Unigram-LM trainer vs reference SentencePiece.

SentencePiece (Kudo's reference Unigram-LM) is the oracle. On the identical
Smirk-glyph stream, the native trainer's multi-glyph piece set agrees strongly
with SentencePiece's. Exact equality is not expected: Smirk force-installs its
full base alphabet as single pieces (every glyph, used or not) where
SentencePiece installs only corpus-exercised characters, a by-design budget
difference. We control for it by matching SentencePiece's target to the native
trainer's multi-piece count, then require a high Jaccard. This test is the
regression guard against reintroducing a divergence from SentencePiece (e.g. the
prune bugs the native trainer was written to avoid).
"""

import json
import random
import tempfile
from pathlib import Path

import pytest

spm = pytest.importorskip("sentencepiece")
import smirk  # noqa: E402

PUA_BASE = 0xE000
FRAGMENTS = [
    "c1ccccc1",
    "CCO",
    "CC(=O)O",
    "CCN",
    "c1ccncc1",
    "C(=O)N",
    "OCCO",
    "CCCC",
    "c1ccc(Cl)cc1",
    "C#N",
    "c1ccsc1",
    "NS(=O)(=O)",
]


def _corpus(n=4000, seed=0):
    rng = random.Random(seed)
    return [
        "".join(rng.choice(FRAGMENTS) for _ in range(rng.randint(1, 4)))
        for _ in range(n)
    ]


def _native_multi(path, vocab_size):
    tok = smirk.train_unigram([str(path)], vocab_size=vocab_size, max_piece_length=128)
    with tempfile.TemporaryDirectory() as d:
        tok.save_pretrained(d)
        vocab = json.loads((Path(d) / "tokenizer.json").read_text())["model"]["vocab"]
    return {tuple(e["glyphs"]) for e in vocab if len(e["glyphs"]) > 1}


def _glyph_stream(molecules):
    """Layer-B chunks -> atomic glyphs (brackets dropped) -> PUA chars."""
    atom = smirk.SmirkTokenizerFast()
    special = set(atom.all_special_tokens)
    g2c: dict[str, str] = {}
    lines: list[str] = []
    for smi in molecules:
        for chunk, _off in atom.pretokenize_layer_b(smi):
            ids = atom(chunk, add_special_tokens=False)["input_ids"]
            glyphs = [
                g
                for g in atom.convert_ids_to_tokens(ids)
                if g not in ("[", "]") and g not in special
            ]
            if glyphs:
                lines.append("".join(g2c.setdefault(g, chr(PUA_BASE + len(g2c))) for g in glyphs))
    return lines, {c: g for g, c in g2c.items()}


def _sp_multi(molecules, vocab_size):
    lines, c2g = _glyph_stream(molecules)
    with tempfile.TemporaryDirectory() as d:
        corpus = Path(d) / "g.txt"
        corpus.write_text("\n".join(lines))
        spm.SentencePieceTrainer.train(
            input=str(corpus),
            model_prefix=str(Path(d) / "m"),
            model_type="unigram",
            vocab_size=vocab_size,
            max_sentencepiece_length=128,
            character_coverage=1.0,
            split_by_whitespace=False,
            split_by_unicode_script=False,
            add_dummy_prefix=False,
            normalization_rule_name="identity",
            num_threads=1,
            hard_vocab_limit=False,
            unk_id=0,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
        )
        sp = spm.SentencePieceProcessor(model_file=str(Path(d) / "m.model"))
        out = set()
        for i in range(sp.get_piece_size()):
            p = sp.id_to_piece(i)
            if p.startswith("<") or len(p) <= 1:
                continue
            out.add(tuple(c2g[ch] for ch in p))
    n_glyphs = len({c for c in c2g})
    return out, n_glyphs


def test_native_matches_sentencepiece():
    mols = _corpus()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "corpus.smi"
        path.write_text("\n".join(mols))
        native = _native_multi(path, vocab_size=256)

    # Match SentencePiece's target to the native multi-piece count plus its
    # (corpus-only) base, so both trainers get the same multi-glyph budget.
    _, n_glyphs = _sp_multi(mols, vocab_size=64)
    sp, _ = _sp_multi(mols, vocab_size=len(native) + n_glyphs + 1)

    inter = native & sp
    union = native | sp
    jac = len(inter) / len(union)
    assert jac >= 0.6, (
        f"native vs SentencePiece Jaccard {jac:.3f} too low "
        f"(native={len(native)}, sp={len(sp)}, shared={len(inter)})"
    )
