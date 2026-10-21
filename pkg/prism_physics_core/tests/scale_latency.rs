#[path = "../benches/support/latency.rs"]
mod latency;

#[test]
fn nearest_rank_percentiles_count_tail_and_preserve_input() {
    let values: Vec<_> = (1..=200).rev().map(f64::from).collect();
    let stats = latency::summarize(&values, 190.0);
    assert_eq!(
        (stats.p50, stats.p95, stats.p99, stats.max),
        (100.0, 190.0, 198.0, 200.0)
    );
    assert_eq!(stats.mean, 100.5);
    assert_eq!(stats.over_budget, 10);
    assert_eq!(values[0], 200.0);
}

#[test]
fn single_sample_and_ties_do_not_underflow() {
    let stats = latency::summarize(&[3.0], 3.0);
    assert_eq!(
        (stats.mean, stats.p50, stats.p95, stats.p99, stats.max),
        (3.0, 3.0, 3.0, 3.0, 3.0)
    );
    assert_eq!(stats.over_budget, 0);
}

#[test]
#[should_panic(expected = "empty timing window")]
fn empty_window_is_rejected() {
    latency::summarize(&[], 1.0);
}

#[test]
#[should_panic]
fn non_finite_measurement_is_rejected() {
    latency::summarize(&[f64::NAN], 1.0);
}
