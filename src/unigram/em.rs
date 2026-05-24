//! Native Unigram-LM EM/prune/finalize, faithful to SentencePiece.
//!
//! This replaces the previous delegation to the HuggingFace `tokenizers`
//! `UnigramTrainer`. It mirrors SentencePiece's `unigram_model_trainer.cc`
//! (`vocab-tokenizer-clms` study, §3.2):
//!
//! * **E-step** — forward/backward marginals over each pre-token's lattice
//!   ([`crate::unigram::lattice`]).
//! * **M-step** — the Bayesian/Digamma update with the `0.5` expected-frequency
//!   floor.
//! * **prune** — likelihood-loss pruning. The reference HuggingFace port has
//!   two bugs here that let degenerate homopolymer pieces (`CCCCCCCC…`) survive
//!   pruning SentencePiece would apply, both fixed in this port:
//!     1. The keep/drop test uses `freq == 0 && !always_keep` where
//!        SentencePiece uses `||` — so HuggingFace keeps every *redundant* piece
//!        (one whose own surface re-segments into a better split) that appears
//!        anywhere, instead of dropping it.
//!     2. The alternative-segmentation normalization uses `alternatives.len()`
//!        (the *total* piece count) instead of `alternatives[i].size()` (the
//!        per-piece alternative count), inflating high-frequency pieces' loss.
//! * **finalize** — re-inject required single glyphs, then take the
//!   highest-scoring pieces up to the target size.
//!
//! Pieces are glyph-id sequences (`Vec<u32>`); index 0 of the working vector is
//! a reserved unknown sentinel (empty glyphs, `NaN` score), matching the
//! reference's `<UNK>`-at-0 convention. Training is single-threaded with
//! total-order tie-breaks, so the trained vocabulary is bit-reproducible.

use std::collections::{HashMap, HashSet};

use super::lattice::Lattice;

/// Unknown-glyph Viterbi penalty below the model's minimum score
/// (SentencePiece `kUnkPenalty`).
const K_UNK_PENALTY: f64 = 10.0;

/// Expected-count floor below which a piece is dropped in the M-step
/// (SentencePiece `kExpectedFrequencyThreshold`).
const EXPECTED_FREQUENCY_THRESHOLD: f64 = 0.5;

/// A working piece vector: index 0 is the unknown sentinel (empty, `NaN`).
pub type Pieces = Vec<(Vec<u32>, f64)>;

/// SentencePiece's `Digamma` (the M-step's Bayesian-prior update).
fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 7.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    x -= 0.5;
    let xx = 1.0 / x;
    let xx2 = xx * xx;
    let xx4 = xx2 * xx2;
    result += x.ln() + (1.0 / 24.0) * xx2 - (7.0 / 960.0) * xx4 + (31.0 / 8064.0) * xx4 * xx2
        - (127.0 / 30720.0) * xx4 * xx4;
    result
}

/// A snapshot of the current piece vocabulary, with the lookup structures the
/// E-step / prune lattices need.
pub struct TrainerModel {
    pieces: Pieces,
    by_glyphs: HashMap<Vec<u32>, usize>,
    min_score: f64,
    max_len: usize,
}

impl TrainerModel {
    pub fn new(pieces: Pieces) -> Self {
        let mut by_glyphs = HashMap::with_capacity(pieces.len());
        let mut min_score = f64::INFINITY;
        let mut max_len = 1usize;
        for (id, (glyphs, score)) in pieces.iter().enumerate() {
            if id == 0 {
                continue; // unk sentinel
            }
            by_glyphs.insert(glyphs.clone(), id);
            max_len = max_len.max(glyphs.len());
            if score.is_finite() {
                min_score = min_score.min(*score);
            }
        }
        let min_score = if min_score.is_finite() {
            min_score
        } else {
            0.0
        };
        Self {
            pieces,
            by_glyphs,
            min_score,
            max_len,
        }
    }

    /// Populate a lattice over one pre-token's glyph-id sequence. Every position
    /// with no single-glyph piece gets an unknown node, so the lattice is always
    /// connected.
    fn populate_nodes(&self, word: &[u32], lattice: &mut Lattice) {
        let unk_score = self.min_score - K_UNK_PENALTY;
        let n = word.len();
        for begin in 0..n {
            let mut has_single = false;
            let max_l = self.max_len.min(n - begin);
            for length in 1..=max_l {
                if let Some(&id) = self.by_glyphs.get(&word[begin..begin + length]) {
                    lattice.insert(begin, length, self.pieces[id].1, id);
                    if length == 1 {
                        has_single = true;
                    }
                }
            }
            if !has_single {
                lattice.insert(begin, 1, unk_score, 0);
            }
        }
    }

    /// E-step: returns (objective, num_tokens, expected counts per piece id).
    pub fn run_e_step(&self, sentences: &[(Vec<u32>, i64)]) -> (f64, u64, Vec<f64>) {
        let total: i64 = sentences.iter().map(|(_, c)| *c).sum();
        let mut expected = vec![0.0; self.pieces.len()];
        let mut objective = 0.0;
        let mut num_tokens: u64 = 0;
        for (word, count) in sentences {
            if word.is_empty() {
                continue;
            }
            let mut lattice = Lattice::new(word.len());
            self.populate_nodes(word, &mut lattice);
            let z = lattice.populate_marginal(*count as f64, &mut expected);
            objective -= z / total as f64;
            num_tokens += lattice.viterbi().len() as u64;
        }
        (objective, num_tokens, expected)
    }

    /// M-step: drop sub-threshold pieces, then apply the Digamma update.
    pub fn run_m_step(&self, expected: &[f64]) -> Pieces {
        assert_eq!(self.pieces.len(), expected.len());
        let mut kept: Pieces = Vec::with_capacity(self.pieces.len());
        let mut sum = 0.0;
        for (i, ((glyphs, _), &freq)) in self.pieces.iter().zip(expected).enumerate() {
            if i == 0 {
                kept.push((glyphs.clone(), f64::NAN)); // unk
                continue;
            }
            if freq < EXPECTED_FREQUENCY_THRESHOLD {
                continue;
            }
            kept.push((glyphs.clone(), freq));
            sum += freq;
        }
        let logsum = digamma(sum);
        kept.into_iter()
            .map(|(g, c)| (g, digamma(c) - logsum))
            .collect()
    }

    /// Likelihood-loss pruning toward `vocab_size_target`, keeping the fraction
    /// `shrinking_factor` of pieces each round (SentencePiece `PruneSentencePieces`).
    pub fn prune(
        &self,
        sentences: &[(Vec<u32>, i64)],
        vocab_size_target: usize,
        shrinking_factor: f64,
    ) -> Pieces {
        let np = self.pieces.len();
        let mut always_keep = vec![true; np];
        let mut alternatives: Vec<Vec<usize>> = vec![Vec::new(); np];

        // For each piece, find how its own surface re-segments without it.
        for id in 1..np {
            let surface = self.pieces[id].0.clone();
            let mut lattice = Lattice::new(surface.len());
            self.populate_nodes(&surface, &mut lattice);
            let nbests = lattice.nbest(2);
            if nbests.len() == 1 {
                always_keep[id] = true;
            } else if nbests[0].len() >= 2 {
                always_keep[id] = false;
            } else if nbests[0].len() == 1 {
                always_keep[id] = true;
                alternatives[id] = nbests[1].clone();
            }
        }

        // Viterbi frequencies over the corpus.
        let mut freq = vec![0.0f64; np];
        for (word, count) in sentences {
            if word.is_empty() {
                continue;
            }
            let mut lattice = Lattice::new(word.len());
            self.populate_nodes(word, &mut lattice);
            for id in lattice.viterbi() {
                freq[id] += *count as f64;
            }
        }

        let sum: f64 = freq.iter().sum();
        let logsum = sum.ln();
        let mut new_pieces: Pieces = vec![self.pieces[0].clone()]; // unk
        let mut candidates: Vec<(usize, f64)> = vec![];

        for id in 1..np {
            // SentencePiece drops a piece that is never on a Viterbi path OR is
            // not always-kept (its own surface re-segments into a better split,
            // so it is redundant). The HuggingFace port uses `&&` here, which
            // keeps every redundant piece that happens to appear — that is what
            // lets long homopolymer pieces survive. We use SentencePiece's `||`.
            if freq[id] == 0.0 || !always_keep[id] {
                continue;
            }
            if alternatives[id].is_empty() {
                // Irreducible piece (unique segmentation of its surface). Keep.
                new_pieces.push(self.pieces[id].clone());
                continue;
            }
            let logprob_sp = freq[id].ln() - logsum;
            // SentencePiece uses the per-piece alternative count here. The
            // HuggingFace port's `alternatives.len()` (total pieces) is a second
            // bug that inflates the loss of high-frequency pieces — see the
            // module docs.
            let n_alt = alternatives[id].len() as f64;
            let logsum_alt = (sum + freq[id] * (n_alt - 1.0)).ln();
            let mut logprob_alt = 0.0;
            for &n in &alternatives[id] {
                logprob_alt += (freq[n] + freq[id]).ln() - logsum_alt;
            }
            let f = freq[id] / sum;
            let loss = f * (logprob_sp - logprob_alt);
            candidates.push((id, loss));
        }

        let desired = (vocab_size_target * 11) / 10;
        let pruned_size = desired.max((np as f64 * shrinking_factor) as usize);
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        for (id, _) in candidates {
            if new_pieces.len() >= pruned_size {
                break;
            }
            new_pieces.push(self.pieces[id].clone());
        }
        new_pieces
    }

    /// Trim to exactly `target` pieces: required single glyphs first (model
    /// score, or a floor with an increasing tie-break penalty when absent), then
    /// the highest-scoring remaining pieces. The unk sentinel is dropped.
    pub fn finalize(&self, required: &HashSet<u32>, target: usize) -> Pieces {
        let existing: HashMap<&Vec<u32>, f64> = self
            .pieces
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != 0)
            .map(|(_, (g, s))| (g, *s))
            .collect();

        let mut out: Pieces = Vec::with_capacity(target);
        let mut inserted: HashSet<Vec<u32>> = HashSet::new();
        let mut min_score_penalty = 0.0;
        let min_score_penalty_delta = 0.0001;

        let mut req: Vec<u32> = required.iter().copied().collect();
        req.sort_unstable();
        for g in req {
            let key = vec![g];
            let score = match existing.get(&key) {
                Some(s) if s.is_finite() => *s,
                _ => {
                    let s = self.min_score + min_score_penalty;
                    min_score_penalty += min_score_penalty_delta;
                    s
                }
            };
            inserted.insert(key.clone());
            out.push((key, score));
        }

        let mut rest: Pieces = self
            .pieces
            .iter()
            .enumerate()
            .filter(|(i, (g, _))| *i != 0 && !inserted.contains(g))
            .map(|(_, (g, s))| (g.clone(), if s.is_nan() { 0.0 } else { *s }))
            .collect();
        rest.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        for piece in rest {
            if out.len() >= target {
                break;
            }
            out.push(piece);
        }

        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out
    }
}

/// Run the full EM/prune loop from a seed vocabulary to a `target`-size piece
/// set (single + multi glyph pieces, excluding the unk sentinel and the fixed
/// base alphabet, which the caller force-installs).
///
/// `seed` is the log-prob seed vocabulary (no unk); `sentences` are the
/// glyph-id pre-tokens with corpus counts; `required` are the glyph ids that
/// must survive finalization (the corpus-exercised glyphs).
pub fn train_pieces(
    seed: Pieces,
    sentences: &[(Vec<u32>, i64)],
    target: usize,
    n_sub_iterations: u32,
    shrinking_factor: f64,
    required: &HashSet<u32>,
) -> Pieces {
    let mut pieces: Pieces = Vec::with_capacity(seed.len() + 1);
    pieces.push((Vec::new(), f64::NAN)); // unk at index 0
    pieces.extend(seed);

    let desired = (target * 11) / 10;
    loop {
        for _ in 0..n_sub_iterations {
            let model = TrainerModel::new(std::mem::take(&mut pieces));
            let (_obj, _ntok, expected) = model.run_e_step(sentences);
            pieces = model.run_m_step(&expected);
        }
        if pieces.len() <= desired {
            break;
        }
        let model = TrainerModel::new(std::mem::take(&mut pieces));
        pieces = model.prune(sentences, target, shrinking_factor);
    }

    let model = TrainerModel::new(pieces);
    model.finalize(required, target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digamma_matches_reference_values() {
        // psi(1) = -gamma; psi(2) = 1 - gamma; gamma = 0.5772156649...
        assert!((digamma(1.0) - (-0.577_215_664_9)).abs() < 1e-6);
        assert!((digamma(2.0) - (1.0 - 0.577_215_664_9)).abs() < 1e-6);
        // psi(10) = 2.251752589066721...
        assert!((digamma(10.0) - 2.251_752_589_066_72).abs() < 1e-9);
    }

    #[test]
    fn m_step_drops_subthreshold_pieces_and_keeps_unk() {
        // pieces: unk, [1] (expected 4.0), [2] (expected 0.2 -> dropped), [1,2] (3.0)
        let pieces: Pieces = vec![
            (vec![], f64::NAN),
            (vec![1], 0.0),
            (vec![2], 0.0),
            (vec![1, 2], 0.0),
        ];
        let model = TrainerModel::new(pieces);
        let out = model.run_m_step(&[f64::NAN, 4.0, 0.2, 3.0]);
        // unk + [1] + [1,2]; [2] dropped.
        assert_eq!(out.len(), 3);
        assert!(out[0].1.is_nan());
        assert_eq!(out[1].0, vec![1]);
        assert_eq!(out[2].0, vec![1, 2]);
        // Scores are digamma(freq) - digamma(sum), sum = 7.0.
        let logsum = digamma(7.0);
        assert!((out[1].1 - (digamma(4.0) - logsum)).abs() < 1e-12);
    }

    #[test]
    fn prune_drops_redundant_multi_piece() {
        // Corpus of "1 1" repeated. Pieces: unk, [1] (-1.0), [1,1] (-3.0). The
        // [1,1] piece is redundant: its surface's best segmentation is the split
        // [1][1] (-2.0 > -3.0), so always_keep is false and SentencePiece drops
        // it (freq == 0 || !always_keep). The essential single [1] is kept.
        let sentences = vec![(vec![1u32, 1u32], 100i64)];
        let pieces: Pieces = vec![(vec![], f64::NAN), (vec![1], -1.0), (vec![1, 1], -3.0)];
        let model = TrainerModel::new(pieces);
        let out = model.prune(&sentences, 4, 0.75);
        let surfaces: Vec<&Vec<u32>> = out.iter().skip(1).map(|(g, _)| g).collect();
        assert!(surfaces.contains(&&vec![1u32]), "essential single dropped");
        assert!(
            !surfaces.contains(&&vec![1u32, 1u32]),
            "redundant multi piece survived"
        );
    }
}
