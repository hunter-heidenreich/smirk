//! Seed-vocabulary extraction for the native Unigram-LM trainer.
//!
//! Mirrors SentencePiece's `MakeSeedSentencePieces`: every corpus glyph is a
//! required single, and frequent multi-glyph substrings (occurring more than
//! once, up to `max_piece_length`) are scored `freq * length` and capped at
//! `seed_size`. Substring enumeration uses the same `esaxx` suffix array as the
//! reference. Glyph ids are bridged to Private Use Area chars only to satisfy
//! esaxx's `&str` interface and decoded back immediately, so no PUA codepoint
//! reaches the trainer's working set or any artifact.

use std::collections::HashMap;

use super::em::Pieces;
use crate::pua;

/// Sentence boundary so the suffix array never enumerates a piece spanning two
/// pre-tokens. Outside the bridge PUA block, so it can't collide with a glyph.
const BOUNDARY: char = '\0';

/// Convert raw seed scores to log-probabilities in place (SentencePiece `ToLogProb`).
fn to_log_prob(pieces: &mut Pieces) {
    let sum: f64 = pieces.iter().map(|(_, s)| *s).sum();
    let logsum = sum.ln();
    for (_, s) in pieces.iter_mut() {
        *s = s.ln() - logsum;
    }
}

/// Build the seed vocabulary (log-prob scores) from glyph-id pre-tokens.
pub fn make_seed(
    sentences: &[(Vec<u32>, i64)],
    max_piece_length: usize,
    seed_size: usize,
) -> Pieces {
    let mut flat = String::new();
    let mut all_chars: HashMap<u32, i64> = HashMap::new();
    for (glyphs, count) in sentences {
        if glyphs.is_empty() {
            continue;
        }
        for &g in glyphs {
            flat.push(pua::encode_glyph(g));
            *all_chars.entry(g).or_insert(0) += *count;
        }
        flat.push(BOUNDARY);
    }
    flat.shrink_to_fit();

    let mut seed: Pieces = Vec::new();

    // Required single glyphs, frequency-descending (glyph id descending on ties).
    let mut singles: Vec<(i64, u32)> = all_chars.into_iter().map(|(g, c)| (c, g)).collect();
    singles.sort_by(|a, b| b.cmp(a));
    for (count, g) in singles {
        seed.push((vec![g], count as f64));
    }

    if !flat.is_empty() && max_piece_length > 1 {
        let suffix = esaxx_rs::suffix_rs(&flat).expect("suffix array over glyph stream");
        let mut substrings: Vec<(u64, Vec<u32>)> = Vec::new();
        for (chars, freq) in suffix.iter() {
            let len = chars.len();
            if len <= 1 || len > max_piece_length {
                continue;
            }
            if chars.contains(&BOUNDARY) {
                continue;
            }
            let glyphs: Vec<u32> = chars.iter().map(|&c| pua::decode_glyph(c)).collect();
            let score = freq as u64 * len as u64;
            substrings.push((score, glyphs));
        }
        // SentencePiece order: score desc, length desc, surface asc.
        substrings.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then(b.1.len().cmp(&a.1.len()))
                .then(a.1.cmp(&b.1))
        });
        for (score, glyphs) in substrings {
            seed.push((glyphs, score as f64));
            if seed.len() >= seed_size {
                break;
            }
        }
    }

    to_log_prob(&mut seed);
    seed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_includes_every_corpus_glyph_as_a_single() {
        let sentences = vec![(vec![0u32, 1, 0], 3i64), (vec![1u32, 2], 2)];
        let seed = make_seed(&sentences, 16, 1_000_000);
        let singles: Vec<&Vec<u32>> = seed
            .iter()
            .map(|(g, _)| g)
            .filter(|g| g.len() == 1)
            .collect();
        assert!(singles.contains(&&vec![0u32]));
        assert!(singles.contains(&&vec![1u32]));
        assert!(singles.contains(&&vec![2u32]));
    }

    #[test]
    fn frequent_substring_is_seeded() {
        // "0 1" occurs 5 times across the corpus -> a high freq*len candidate.
        let sentences = vec![(vec![0u32, 1], 5i64), (vec![0u32, 1, 2], 0)];
        let seed = make_seed(&sentences, 16, 1_000_000);
        let multis: Vec<&Vec<u32>> = seed
            .iter()
            .map(|(g, _)| g)
            .filter(|g| g.len() > 1)
            .collect();
        assert!(
            multis.contains(&&vec![0u32, 1u32]),
            "frequent pair not seeded"
        );
    }

    #[test]
    fn max_piece_length_one_seeds_no_multi() {
        let sentences = vec![(vec![0u32, 0, 0, 0], 10i64)];
        let seed = make_seed(&sentences, 1, 1_000_000);
        assert!(seed.iter().all(|(g, _)| g.len() == 1));
    }
}
