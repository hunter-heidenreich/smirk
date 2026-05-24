use std::collections::{HashMap, HashSet};

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use tokenizers::parallelism::MaybeParallelBridge;
use tokenizers::{AddedToken, Result, Trainer};

use super::em::train_pieces;
use super::model::{UnigramModel, UnigramPiece};
use super::seed::make_seed;
use crate::shared::{compute_alphabet, tokenize_words};

// §3.2 Unigram-LM hyperparameters frozen by the preregistration. All four
// (`seed_size`, `max_piece_length`, `n_sub_iterations`, `shrinking_factor`)
// are §3.2-frozen but exposed as `UnigramTrainer` field defaults below so the
// Phase-2 probes (seed-cap spot-check, prune-schedule spot-check,
// max-piece-length contingency) can vary them.

/// Unigram-LM sibling trainer — the Unigram arm's analogue of `GpeTrainer`.
///
/// Mirrors `GpeTrainer`'s public knobs and shares the Layer A/B/C front-end
/// (`compute_alphabet` + `tokenize_words`); it fits the model with a native
/// EM/prune loop ([`crate::unigram::em`]) faithful to SentencePiece's
/// `unigram_model_trainer.cc` (`vocab-tokenizer-clms` study, §3.2).
#[derive(Builder, Debug, Deserialize, Serialize, Clone)]
#[builder(default)]
pub struct UnigramTrainer {
    /// Exposed for `GpeTrainer` API parity only. The native trainer follows
    /// SentencePiece, which has no separate seed-frequency floor (candidate
    /// substrings must merely occur more than once); rare candidates are
    /// excluded by `freq * length` seed scoring and likelihood-loss pruning
    /// (main.tex §9 amendment, 2026-05-17).
    pub min_frequency: u64,
    /// Target vocabulary size.
    pub vocab_size: usize,
    /// Initial alphabet — every glyph here is guaranteed a length-1 piece.
    pub alphabet: HashSet<String>,
    /// Cap on the initial alphabet size (applied by `compute_alphabet`).
    pub limit_alphabet: Option<usize>,
    /// Special tokens, returned for the tokenizer to add as added-tokens.
    pub special_tokens: Vec<AddedToken>,
    /// Layer C: whether `[` / `]` glyphs are kept in the training stream.
    pub merge_brackets: bool,
    /// Seed-pool cap (SentencePiece `seed_sentencepiece_size`). §3.2 freezes
    /// this at the default (1_000_000); exposed for the Phase-2.5 seed-cap
    /// spot-check.
    pub seed_size: usize,
    /// Maximum piece length, in glyphs. §3.2 freezes this at the default
    /// (128); exposed as a knob for the Phase-2 max-piece-length contingency.
    pub max_piece_length: usize,
    /// EM sub-iterations per prune round. §3.2 freezes this at the default (2);
    /// exposed as a knob for the Phase-2.5 prune-schedule spot-check.
    pub n_sub_iterations: u32,
    /// Shrinking factor (fraction of pieces kept per prune round). §3.2 freezes
    /// this at the default (0.75); exposed as a knob for the Phase-2.5
    /// prune-schedule spot-check.
    pub shrinking_factor: f64,
    /// Internal corpus word-count map, populated by `feed`.
    word_counts: HashMap<String, u64>,
}

impl Default for UnigramTrainer {
    fn default() -> Self {
        Self {
            min_frequency: 0,
            vocab_size: 1024,
            alphabet: HashSet::new(),
            limit_alphabet: None,
            special_tokens: Vec::new(),
            merge_brackets: false,
            seed_size: 1_000_000,
            max_piece_length: 128,
            n_sub_iterations: 2,
            shrinking_factor: 0.75,
            word_counts: HashMap::new(),
        }
    }
}

impl UnigramTrainer {
    pub fn builder() -> UnigramTrainerBuilder {
        UnigramTrainerBuilder::default()
    }

    /// Train a [`UnigramModel`] from corpus word counts.
    ///
    /// The Layer A/B/C front-end is identical to the BPE arm; the resulting
    /// glyph-id pre-tokens feed a native EM/prune loop ([`train_pieces`]). The
    /// fixed base alphabet is then installed as length-1 pieces, so both arms
    /// target the same `vocab_size` (`vocab-tokenizer-clms` study, §3.2).
    pub fn do_train(
        &self,
        word_counts: &HashMap<String, u64>,
        model: &mut UnigramModel,
    ) -> Result<Vec<AddedToken>> {
        // Shared Layer A/B/C front-end — the same glyph-id words the BPE arm
        // trains on. `id2w` maps every glyph id back to its glyph string.
        let mut w2id: HashMap<String, u32> = HashMap::new();
        let mut id2w: Vec<String> = Vec::new();
        compute_alphabet(
            &model.tokenize,
            word_counts,
            &self.alphabet,
            self.limit_alphabet,
            &mut w2id,
            &mut id2w,
        );
        let (words, counts) = tokenize_words(
            &model.tokenize,
            word_counts,
            self.merge_brackets,
            &mut w2id,
            &mut id2w,
        );

        // Glyph-id pre-tokens with corpus counts, sorted so EM's float
        // accumulation order is fixed (the trained vocabulary is reproducible).
        let mut sentences: Vec<(Vec<u32>, i64)> = words
            .iter()
            .zip(counts.iter())
            .filter(|(w, _)| !w.glyphs().is_empty())
            .map(|(w, &c)| (w.glyphs().to_vec(), c))
            .collect();
        sentences.sort_by(|a, b| a.0.cmp(&b.0));

        // The final vocabulary is the fixed base alphabet plus multi-glyph
        // pieces plus the unknown token; `vocab_size` (mirrored from
        // `GpeTrainer`) counts all of them. The EM targets the multi-glyph
        // budget plus the corpus-exercised single glyphs (the pieces it can
        // actually fit); finalize trims to exactly that count.
        let base: Vec<String> = {
            let mut b: Vec<String> = self.alphabet.iter().cloned().collect();
            b.sort();
            b
        };
        let required: HashSet<u32> = sentences
            .iter()
            .flat_map(|(g, _)| g.iter().copied())
            .collect();
        let n_multi = self.vocab_size.saturating_sub(base.len() + 1);
        let target = n_multi + required.len();

        let seed = make_seed(&sentences, self.max_piece_length, self.seed_size);
        let trained = train_pieces(
            seed,
            &sentences,
            target,
            self.n_sub_iterations,
            self.shrinking_factor,
            &required,
        );

        // Split the trained pieces: single-glyph scores feed the base alphabet,
        // multi-glyph pieces (glyph ids -> glyph strings) are kept as trained.
        let mut single_scores: HashMap<u32, f64> = HashMap::new();
        let mut multi: Vec<UnigramPiece> = Vec::new();
        for (glyphs, score) in &trained {
            match glyphs.as_slice() {
                [g] => {
                    single_scores.insert(*g, *score);
                }
                _ => {
                    let strs: Vec<String> =
                        glyphs.iter().map(|&g| id2w[g as usize].clone()).collect();
                    multi.push(UnigramPiece {
                        glyphs: strs,
                        score: *score,
                    });
                }
            }
        }

        // §3.2: the fixed base alphabet is installed in every vocabulary as
        // length-1 pieces, regardless of which glyphs the corpus exercised.
        let floor = multi
            .iter()
            .map(|p| p.score)
            .chain(single_scores.values().copied())
            .fold(f64::INFINITY, f64::min);
        let floor = if floor.is_finite() { floor } else { 0.0 };
        let mut pieces: Vec<UnigramPiece> = base
            .into_iter()
            .map(|glyph| {
                let score = w2id
                    .get(&glyph)
                    .and_then(|id| single_scores.get(id).copied())
                    .unwrap_or(floor);
                UnigramPiece {
                    glyphs: vec![glyph],
                    score,
                }
            })
            .collect();
        pieces.extend(multi);

        model.with_pieces(pieces);
        Ok(self.special_tokens.clone())
    }
}

impl Trainer for UnigramTrainer {
    type Model = UnigramModel;

    fn train(&self, model: &mut UnigramModel) -> Result<Vec<AddedToken>> {
        self.do_train(&self.word_counts, model)
    }

    fn should_show_progress(&self) -> bool {
        false
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        let words: Result<HashMap<String, u64>> = iterator
            .maybe_par_bridge()
            .map(|sequence| {
                let words = process(sequence.as_ref())?;
                let mut map = HashMap::new();
                for word in words {
                    map.entry(word).and_modify(|c| *c += 1).or_insert(1);
                }
                Ok(map)
            })
            .reduce(
                || Ok(HashMap::new()),
                |acc, ws| {
                    let mut acc = acc?;
                    for (k, v) in ws? {
                        acc.entry(k).and_modify(|c| *c += v).or_insert(v);
                    }
                    Ok(acc)
                },
            );

        self.word_counts = words?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pre_tokenizers::SmirkPreTokenizer;
    use tokenizers::Model;

    fn word_counts() -> HashMap<String, u64> {
        // A toy corpus with clear repetition so EM has something to fit.
        [("CCO", 8), ("CCS", 6), ("OCC", 5), ("CCC", 4), ("CN", 2)]
            .into_iter()
            .map(|(s, c)| (s.to_string(), c))
            .collect()
    }

    fn train(vocab_size: usize) -> UnigramModel {
        // train_unigram supplies the base alphabet from the tokenizer vocab;
        // mirror that here with the glyphs the toy corpus exercises.
        let alphabet: HashSet<String> =
            ["C", "O", "S", "N"].into_iter().map(String::from).collect();
        let trainer = UnigramTrainer {
            vocab_size,
            alphabet,
            ..Default::default()
        };
        let mut model = UnigramModel::new(
            "[UNK]".to_string(),
            SmirkPreTokenizer::default(),
            Vec::new(),
        );
        trainer.do_train(&word_counts(), &mut model).unwrap();
        model
    }

    #[test]
    fn training_produces_a_usable_model() {
        let model = train(64);
        // Base glyphs appearing in the corpus are length-1 pieces.
        for glyph in ["C", "O", "S", "N"] {
            assert!(
                model.token_to_id(glyph).is_some(),
                "missing base glyph {glyph}"
            );
        }
        // The trained tokenizer round-trips a held-out string.
        let decoded: String = model
            .tokenize("CCO")
            .unwrap()
            .iter()
            .map(|t| t.value.clone())
            .collect();
        assert_eq!(decoded, "CCO");
    }

    #[test]
    fn training_is_piece_set_deterministic() {
        // §3.2 determinism amendment: the Unigram piece *set* is reproducible.
        let mut a: Vec<String> = train(64).get_vocab().into_keys().collect();
        let mut b: Vec<String> = train(64).get_vocab().into_keys().collect();
        a.sort();
        b.sort();
        assert_eq!(a, b);
    }

    /// Build a trainer over the toy corpus' alphabet, overriding one knob.
    fn trainer_with(seed_size: usize, max_piece_length: usize) -> UnigramTrainer {
        let alphabet: HashSet<String> =
            ["C", "O", "S", "N"].into_iter().map(String::from).collect();
        UnigramTrainer {
            vocab_size: 64,
            alphabet,
            seed_size,
            max_piece_length,
            ..Default::default()
        }
    }

    fn empty_model() -> UnigramModel {
        UnigramModel::new(
            "[UNK]".to_string(),
            SmirkPreTokenizer::default(),
            Vec::new(),
        )
    }

    #[test]
    fn max_piece_length_one_yields_only_base_pieces() {
        // A one-glyph cap leaves HuggingFace no multi-glyph pieces to seed, so
        // the trained vocabulary is exactly the base alphabet plus the unk.
        let mut model = empty_model();
        trainer_with(1_000_000, 1)
            .do_train(&word_counts(), &mut model)
            .unwrap();
        assert_eq!(model.get_vocab_size(), 5); // C, O, S, N + [UNK]
    }

    #[test]
    fn a_custom_seed_size_still_trains() {
        // The seed_size knob reaches the native seed extractor and trains cleanly.
        let mut model = empty_model();
        trainer_with(32, 128)
            .do_train(&word_counts(), &mut model)
            .unwrap();
        for glyph in ["C", "O", "S", "N"] {
            assert!(model.token_to_id(glyph).is_some());
        }
    }

    #[test]
    fn a_custom_prune_schedule_still_trains() {
        // The n_sub_iterations / shrinking_factor knobs drive the native EM/prune
        // loop and the trainer produces a usable model with the coarsened §3.2
        // spot-check schedule.
        let alphabet: HashSet<String> =
            ["C", "O", "S", "N"].into_iter().map(String::from).collect();
        let trainer = UnigramTrainer {
            vocab_size: 64,
            alphabet,
            n_sub_iterations: 3,
            shrinking_factor: 0.9,
            ..Default::default()
        };
        let mut model = empty_model();
        trainer.do_train(&word_counts(), &mut model).unwrap();
        for glyph in ["C", "O", "S", "N"] {
            assert!(model.token_to_id(glyph).is_some());
        }
    }
}
