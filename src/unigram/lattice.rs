//! Training-time lattice over a glyph-id sequence for the native Unigram-LM EM.
//!
//! A faithful port of SentencePiece's `Lattice` (and HuggingFace `tokenizers`'
//! `models::unigram::lattice`), but indexed by **glyph position** rather than
//! byte offset: a node covers `glyphs[pos..pos + length]`. The forward/backward
//! marginals, the Viterbi backtrace, and the A* `n`-best all mirror the
//! reference so the native trainer matches SentencePiece (`vocab-tokenizer-clms`
//! study, §3.2). It carries no Private Use Area encoding — pieces are glyph-id
//! sequences throughout.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Node id reserved for the begin-of-sentence / end-of-sentence sentinels.
/// They never index an `expected`/`freq` vector (the marginal and Viterbi-freq
/// loops only visit real piece nodes), so any out-of-range value is safe.
const SENTINEL_ID: usize = usize::MAX;

/// Numerically stable log-sum-exp accumulation (SentencePiece `LogSumExp`).
///
/// `init_mode` seeds the accumulator with `y` (the first term); thereafter it
/// folds `x` (running total) with `y` (new term), short-circuiting when the
/// terms differ by more than `k_minus_log_epsilon`.
fn log_sum_exp(x: f64, y: f64, init_mode: bool) -> f64 {
    if init_mode {
        return y;
    }
    let (vmin, vmax) = if x > y { (y, x) } else { (x, y) };
    const K_MINUS_LOG_EPSILON: f64 = 50.0;
    if vmax > vmin + K_MINUS_LOG_EPSILON {
        vmax
    } else {
        vmax + ((vmin - vmax).exp() + 1.0).ln()
    }
}

#[derive(Clone, Debug)]
struct Node {
    /// Piece id (index into the model's piece vector); `SENTINEL_ID` for bos/eos.
    id: usize,
    /// Begin glyph position.
    pos: usize,
    /// Piece log-probability score.
    score: f64,
    /// Best path score reaching the *end* of this node (filled by `viterbi`).
    backtrace_score: f64,
    /// Predecessor node on the best path (filled by `viterbi`).
    prev: Option<usize>,
}

/// A lattice over a single glyph-id sequence.
pub struct Lattice {
    len: usize,
    nodes: Vec<Node>,
    /// Node indices that begin at each position (`0..=len`).
    begin_nodes: Vec<Vec<usize>>,
    /// Node indices that end at each position (`0..=len`).
    end_nodes: Vec<Vec<usize>>,
    eos_node: usize,
}

impl Lattice {
    /// Build an empty lattice for a sequence of `len` glyphs (bos + eos only).
    pub fn new(len: usize) -> Self {
        let mut nodes: Vec<Node> = Vec::with_capacity(len + 2);
        let mut begin_nodes = vec![Vec::new(); len + 1];
        let mut end_nodes = vec![Vec::new(); len + 1];

        // bos at node index 0, eos at node index 1 (mirrors the reference order).
        nodes.push(Node {
            id: SENTINEL_ID,
            pos: 0,
            score: 0.0,
            backtrace_score: 0.0,
            prev: None,
        });
        nodes.push(Node {
            id: SENTINEL_ID,
            pos: len,
            score: 0.0,
            backtrace_score: 0.0,
            prev: None,
        });
        end_nodes[0].push(0);
        begin_nodes[len].push(1);

        Self {
            len,
            nodes,
            begin_nodes,
            end_nodes,
            eos_node: 1,
        }
    }

    /// Insert a piece node covering `glyphs[pos..pos + length]`.
    pub fn insert(&mut self, pos: usize, length: usize, score: f64, id: usize) {
        let node_id = self.nodes.len();
        self.nodes.push(Node {
            id,
            pos,
            score,
            backtrace_score: 0.0,
            prev: None,
        });
        self.begin_nodes[pos].push(node_id);
        self.end_nodes[pos + length].push(node_id);
    }

    /// Best (max sum-of-log-prob) path, as the piece ids on it left-to-right.
    pub fn viterbi(&mut self) -> Vec<usize> {
        let paths = self.viterbi_nodes();
        paths.into_iter().map(|n| self.nodes[n].id).collect()
    }

    /// Best path as node indices (excludes bos/eos), left-to-right.
    fn viterbi_nodes(&mut self) -> Vec<usize> {
        for pos in 0..=self.len {
            if self.begin_nodes[pos].is_empty() {
                return vec![];
            }
            let rnodes = self.begin_nodes[pos].clone();
            let lnodes = self.end_nodes[pos].clone();
            for &rid in &rnodes {
                let rscore = self.nodes[rid].score;
                let mut best_score = 0.0;
                let mut best_node: Option<usize> = None;
                for &lid in &lnodes {
                    let score = self.nodes[lid].backtrace_score + rscore;
                    if best_node.is_none() || score > best_score {
                        best_node = Some(lid);
                        best_score = score;
                    }
                }
                match best_node {
                    Some(b) => {
                        self.nodes[rid].prev = Some(b);
                        self.nodes[rid].backtrace_score = best_score;
                    }
                    None => return vec![],
                }
            }
        }

        let mut results: Vec<usize> = vec![];
        let mut node = match self.nodes[self.eos_node].prev {
            Some(p) => p,
            None => return vec![],
        };
        while self.nodes[node].prev.is_some() {
            results.push(node);
            node = self.nodes[node].prev.unwrap();
        }
        results.reverse();
        results
    }

    /// Forward/backward marginals: accumulate `expected[piece_id]` and return
    /// `freq * log Z` (the sentence's weighted log-partition).
    pub fn populate_marginal(&self, freq: f64, expected: &mut [f64]) -> f64 {
        let n_nodes = self.nodes.len();
        let mut alpha = vec![0.0; n_nodes];
        let mut beta = vec![0.0; n_nodes];

        for pos in 0..=self.len {
            for &rid in &self.begin_nodes[pos] {
                for (k, &lid) in self.end_nodes[pos].iter().enumerate() {
                    alpha[rid] =
                        log_sum_exp(alpha[rid], self.nodes[lid].score + alpha[lid], k == 0);
                }
            }
        }
        for pos in (0..=self.len).rev() {
            for &lid in &self.end_nodes[pos] {
                for (k, &rid) in self.begin_nodes[pos].iter().enumerate() {
                    beta[lid] = log_sum_exp(beta[lid], self.nodes[rid].score + beta[rid], k == 0);
                }
            }
        }

        let z = alpha[self.eos_node];
        for pos in 0..self.len {
            for &node in &self.begin_nodes[pos] {
                let id = self.nodes[node].id;
                let total = alpha[node] + self.nodes[node].score + beta[node] - z;
                expected[id] += freq * total.exp();
            }
        }
        freq * z
    }

    /// The `n` best paths, each as the piece ids on it (left-to-right). Mirrors
    /// the reference A* search; used by pruning's alternative-segmentation probe.
    pub fn nbest(&mut self, n: usize) -> Vec<Vec<usize>> {
        match n {
            0 => vec![],
            1 => vec![self.viterbi()],
            _ => {
                self.viterbi_nodes(); // fill backtrace_score

                // Hypothesis arena: `next` chains toward eos; `fx` is the
                // best achievable full-path score through this node.
                struct Hyp {
                    node: usize,
                    next: Option<usize>,
                    gx: f64,
                }
                let mut arena: Vec<Hyp> = Vec::new();
                let mut agenda: BinaryHeap<HeapItem> = BinaryHeap::new();

                let eos = self.eos_node;
                let eos_score = self.nodes[eos].score;
                arena.push(Hyp {
                    node: eos,
                    next: None,
                    gx: eos_score,
                });
                agenda.push(HeapItem {
                    fx: eos_score,
                    idx: 0,
                });

                let mut results: Vec<Vec<usize>> = vec![];
                while let Some(HeapItem { idx: top_idx, .. }) = agenda.pop() {
                    let node = arena[top_idx].node;
                    if node == 0 {
                        // Reached bos: reconstruct the path between bos and eos.
                        let mut path: Vec<usize> = vec![];
                        let mut cursor = arena[top_idx].next;
                        while let Some(c) = cursor {
                            if arena[c].next.is_none() {
                                break; // eos hypothesis
                            }
                            path.push(self.nodes[arena[c].node].id);
                            cursor = arena[c].next;
                        }
                        results.push(path);
                        if results.len() == n {
                            return results;
                        }
                        continue;
                    }
                    let pos = self.nodes[node].pos;
                    let top_gx = arena[top_idx].gx;
                    for &lid in &self.end_nodes[pos].clone() {
                        let fx = self.nodes[lid].backtrace_score + top_gx;
                        let gx = self.nodes[lid].score + top_gx;
                        let h = arena.len();
                        arena.push(Hyp {
                            node: lid,
                            next: Some(top_idx),
                            gx,
                        });
                        agenda.push(HeapItem { fx, idx: h });
                    }

                    // Bound the agenda for pathological (long, repetitive) inputs.
                    const K_MAX_AGENDA: usize = 100_000;
                    const K_MIN_AGENDA: usize = 512;
                    if agenda.len() > K_MAX_AGENDA {
                        let keep = K_MIN_AGENDA.min(n * 10);
                        let mut trimmed = BinaryHeap::new();
                        for _ in 0..keep {
                            if let Some(item) = agenda.pop() {
                                trimmed.push(item);
                            }
                        }
                        agenda = trimmed;
                    }
                }
                results
            }
        }
    }
}

/// Max-heap item ordered by `fx` (best full-path score first).
struct HeapItem {
    fx: f64,
    idx: usize,
}
impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.fx == other.fx
    }
}
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.fx.total_cmp(&other.fx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hand-checked two-piece lattice: glyphs [10, 10] with pieces
    /// "10" (id 1, score -1.0), "10 10" (id 2, score -1.5), each glyph also a
    /// single (id 1). Viterbi should prefer the single "10 10" piece.
    fn two_glyph_lattice() -> Lattice {
        // pieces: id1 = [10] (-1.0), id2 = [10,10] (-1.5)
        let mut lat = Lattice::new(2);
        lat.insert(0, 1, -1.0, 1); // glyph 0
        lat.insert(1, 1, -1.0, 1); // glyph 1
        lat.insert(0, 2, -1.5, 2); // both glyphs
        lat
    }

    #[test]
    fn viterbi_prefers_the_higher_scoring_piece() {
        let mut lat = two_glyph_lattice();
        // single+single = -2.0; the 2-glyph piece = -1.5 (higher).
        assert_eq!(lat.viterbi(), vec![2]);
    }

    #[test]
    fn marginal_z_matches_hand_computation() {
        let lat = two_glyph_lattice();
        let mut expected = vec![0.0; 3];
        let z = lat.populate_marginal(1.0, &mut expected);
        // Two paths: [-1.0,-1.0] sum -2.0 and [-1.5]; logZ = logsumexp(-2,-1.5).
        let want = ((-2.0f64).exp() + (-1.5f64).exp()).ln();
        assert!((z - want).abs() < 1e-9, "z={z} want={want}");
    }

    #[test]
    fn nbest_returns_both_segmentations_best_first() {
        let mut lat = two_glyph_lattice();
        let paths = lat.nbest(2);
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0], vec![2]); // best: the 2-glyph piece
        assert_eq!(paths[1], vec![1, 1]); // alternative: two singles
    }
}
