//! CPU sampling: temperature, top-k, top-p (nucleus).

use rand::Rng;
use rand::RngExt;

pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: -1,
            top_p: 1.0,
        }
    }
}

impl SamplingParams {
    pub fn is_greedy(&self) -> bool {
        (self.temperature <= 0.0 || self.top_k == 1) && self.top_p >= 1.0
    }
}

/// Sample a token from logits using temperature, top-k, and top-p.
///
/// Pipeline: temperature scale → top-k truncate → softmax → top-p truncate → multinomial.
pub fn sample(logits: &[f32], params: &SamplingParams, rng: &mut impl Rng) -> u32 {
    assert!(!logits.is_empty(), "sample() called with empty logits");

    let temperature = params.temperature;

    // t <= 0 means greedy — return argmax
    if temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
    }

    // (index, scaled_logit) pairs
    let mut candidates: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, l / temperature))
        .collect();

    // Sort descending by logit
    candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    // Top-k: keep only top k candidates
    if params.top_k > 0 && (params.top_k as usize) < candidates.len() {
        candidates.truncate(params.top_k as usize);
    }

    // Softmax
    let max_logit = candidates[0].1;
    let mut probs: Vec<f32> = candidates.iter().map(|(_, l)| (l - max_logit).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }

    // Top-p: cumulative probability cutoff
    if params.top_p < 1.0 {
        let mut cumsum = 0.0f32;
        let mut cutoff = probs.len();
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum > params.top_p {
                cutoff = i + 1;
                break;
            }
        }
        candidates.truncate(cutoff);
        probs.truncate(cutoff);

        // Re-normalize
        let sum: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum;
        }
    }

    // Multinomial sample
    let r: f32 = rng.random();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return candidates[i].0;
        }
    }

    // Fallback for numerical edge case
    candidates.last().unwrap().0
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_greedy_defaults() {
        let params = SamplingParams::default();
        assert!(params.is_greedy());
    }

    #[test]
    fn test_greedy_top_k_1() {
        let params = SamplingParams {
            temperature: 0.7,
            top_k: 1,
            top_p: 1.0,
        };
        assert!(params.is_greedy());
    }

    #[test]
    fn test_not_greedy() {
        let params = SamplingParams {
            temperature: 0.7,
            top_k: -1,
            top_p: 1.0,
        };
        assert!(!params.is_greedy());
    }

    #[test]
    fn test_sample_deterministic_with_seed() {
        let logits = vec![1.0, 2.0, 3.0, 0.5];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: -1,
            top_p: 1.0,
        };
        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(42);
        let t1 = sample(&logits, &params, &mut rng1);
        let t2 = sample(&logits, &params, &mut rng2);
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_sample_top_k_1_picks_argmax() {
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
        };
        let mut rng = StdRng::seed_from_u64(0);
        let token = sample(&logits, &params, &mut rng);
        assert_eq!(token, 1); // index of max logit
    }

    #[test]
    fn test_sample_respects_top_p() {
        // One dominant logit — with very low top_p, should always pick it
        let mut logits = vec![0.0; 100];
        logits[42] = 100.0;
        let params = SamplingParams {
            temperature: 1.0,
            top_k: -1,
            top_p: 0.1,
        };
        let mut rng = StdRng::seed_from_u64(123);
        for _ in 0..10 {
            assert_eq!(sample(&logits, &params, &mut rng), 42);
        }
    }

    #[test]
    fn test_sample_temperature_zero_returns_argmax() {
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let params = SamplingParams {
            temperature: 0.0,
            top_k: -1,
            top_p: 1.0,
        };
        let mut rng = StdRng::seed_from_u64(0);
        assert_eq!(sample(&logits, &params, &mut rng), 1);
    }

    #[test]
    fn test_negative_temperature_is_greedy() {
        let logits = vec![1.0, 5.0, 3.0];
        let params = SamplingParams {
            temperature: -1.0,
            top_k: -1,
            top_p: 1.0,
        };
        let mut rng = StdRng::seed_from_u64(0);
        assert_eq!(sample(&logits, &params, &mut rng), 1);
    }

    #[test]
    fn test_single_logit() {
        let logits = vec![42.0];
        let params = SamplingParams { temperature: 1.0, top_k: -1, top_p: 1.0 };
        let mut rng = StdRng::seed_from_u64(0);
        assert_eq!(sample(&logits, &params, &mut rng), 0);
    }

    #[test]
    #[should_panic(expected = "empty logits")]
    fn test_empty_logits_panics() {
        let logits: Vec<f32> = vec![];
        let params = SamplingParams { temperature: 1.0, top_k: -1, top_p: 1.0 };
        let mut rng = StdRng::seed_from_u64(0);
        sample(&logits, &params, &mut rng);
    }

    #[test]
    fn test_negative_logits_argmax() {
        let logits = vec![-10.0, -5.0, -20.0, -1.0];
        let params = SamplingParams { temperature: 0.0, top_k: -1, top_p: 1.0 };
        let mut rng = StdRng::seed_from_u64(0);
        assert_eq!(sample(&logits, &params, &mut rng), 3); // -1.0 is largest
    }

    #[test]
    fn test_top_k_restricts_candidates() {
        // logits: idx=0 is highest, idx=1 second, idx=2 third, rest are low
        let mut logits = vec![-100.0; 20];
        logits[0] = 5.0;
        logits[1] = 4.0;
        logits[2] = 3.0;
        let params = SamplingParams { temperature: 1.0, top_k: 3, top_p: 1.0 };
        let mut rng = StdRng::seed_from_u64(0);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            seen.insert(sample(&logits, &params, &mut rng));
        }
        // Only tokens 0, 1, 2 should ever appear
        assert!(seen.is_subset(&[0u32, 1, 2].iter().copied().collect()));
        // All three should appear with enough samples
        assert_eq!(seen.len(), 3);
    }

    #[test]
    fn test_low_temperature_concentrates() {
        // Very low temperature → almost always picks argmax
        let logits = vec![1.0, 2.0, 10.0, 3.0];
        let params = SamplingParams { temperature: 0.01, top_k: -1, top_p: 1.0 };
        let mut rng = StdRng::seed_from_u64(0);
        let n = 500;
        let count_argmax = (0..n).filter(|_| sample(&logits, &params, &mut rng) == 2).count();
        assert!(count_argmax == n, "t=0.01: expected all argmax, got {}/{}", count_argmax, n);
    }

    #[test]
    fn test_high_temperature_spreads() {
        // Very high temperature → nearly uniform, all 4 tokens should appear
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let params = SamplingParams { temperature: 100.0, top_k: -1, top_p: 1.0 };
        let mut rng = StdRng::seed_from_u64(0);
        let mut counts = [0u32; 4];
        let n = 2000;
        for _ in 0..n {
            counts[sample(&logits, &params, &mut rng) as usize] += 1;
        }
        // Each token should appear at least 15% of the time (uniform = 25%)
        for (i, &c) in counts.iter().enumerate() {
            assert!(c > (n as u32) * 15 / 100,
                "token {} appeared only {}/{} times, expected ~25%", i, c, n);
        }
    }

    #[test]
    fn test_equal_logits_uniform() {
        let logits = vec![0.0; 5];
        let params = SamplingParams { temperature: 1.0, top_k: -1, top_p: 1.0 };
        let mut rng = StdRng::seed_from_u64(42);
        let mut counts = [0u32; 5];
        let n = 2000;
        for _ in 0..n {
            counts[sample(&logits, &params, &mut rng) as usize] += 1;
        }
        // Each should appear at least 10% (uniform = 20%)
        for (i, &c) in counts.iter().enumerate() {
            assert!(c > (n as u32) * 10 / 100,
                "token {} appeared only {}/{} times, expected ~20%", i, c, n);
        }
    }

    #[test]
    fn test_top_k_and_top_p_combined() {
        // 5 tokens, top_k=3 keeps top 3, then top_p=0.5 further narrows
        // logits: [10, 9, 1, 0, 0] → after top_k=3: [10, 9, 1]
        // softmax(10,9,1) ≈ [0.731, 0.269, 0.0007] → cumsum > 0.5 at idx=0
        // so top_p=0.5 keeps only token 0
        let logits = vec![10.0, 9.0, 1.0, 0.0, 0.0];
        let params = SamplingParams { temperature: 1.0, top_k: 3, top_p: 0.5 };
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..50 {
            let t = sample(&logits, &params, &mut rng);
            assert!(t == 0, "expected only token 0 with top_k=3 + top_p=0.5, got {}", t);
        }
    }

    #[test]
    fn test_top_p_small_picks_top_only() {
        // top_p just above 0 — only the single highest-probability token survives
        let logits = vec![3.0, 1.0, 2.0, 1.0];
        let params = SamplingParams { temperature: 1.0, top_k: -1, top_p: 0.01 };
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..50 {
            assert_eq!(sample(&logits, &params, &mut rng), 0);
        }
    }
}
