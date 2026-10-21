//! Dependency-free nearest-rank timing summary.

pub struct Summary {
    pub mean: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
    pub over_budget: usize,
}

pub fn summarize(samples: &[f64], budget: f64) -> Summary {
    assert!(
        !samples.is_empty(),
        "cannot summarize an empty timing window"
    );
    assert!(samples.iter().all(|v| v.is_finite() && *v >= 0.0));
    assert!(budget.is_finite() && budget >= 0.0);
    let mut sorted = samples.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    let rank = |percent: usize| sorted[(sorted.len() * percent).div_ceil(100) - 1];
    Summary {
        mean: samples.iter().sum::<f64>() / samples.len() as f64,
        p50: rank(50),
        p95: rank(95),
        p99: rank(99),
        max: *sorted.last().unwrap(),
        over_budget: samples.iter().filter(|v| **v > budget).count(),
    }
}
