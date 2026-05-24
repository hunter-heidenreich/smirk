//! Unigram-LM tokenizer arm — a glyph-aware Unigram model and its trainer.
//!
//! Mirrors `src/gpe/` (the BPE arm). [`UnigramModel`] holds a glyph-piece
//! vocabulary and segments by Viterbi argmax; [`UnigramTrainer`] mirrors
//! `GpeTrainer`'s knobs and shared Layer A/B/C front-end and fits the model
//! with a native EM/prune loop faithful to SentencePiece (`vocab-tokenizer-clms`
//! study, §3.2). The EM ([`em`]), lattice ([`lattice`]), and seed extraction
//! ([`seed`]) operate on glyph-id sequences directly.

mod em;
mod lattice;
mod model;
mod seed;
mod trainer;

pub use model::UnigramModel;
pub use trainer::UnigramTrainer;
