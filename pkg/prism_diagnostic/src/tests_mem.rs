//! §24.3 memory tracking tests: leak reconciliation, per-category budgets, and
//! fragmentation analysis. These exercise the feature-neutral core arithmetic
//! with hand-computed oracles (no global allocator installed).

use crate::mem::budget::{MemBudgetRegistry, MemBudgetStatus};
use crate::mem::fragmentation::{analyze_fragmentation, occupancy_map, Span};
use crate::mem::leak::LeakCheckpoint;

// ---- leak reconciliation ----------------------------------------------------

#[test]
fn leak_balanced_scope_reports_no_residual() {
    let start = LeakCheckpoint::new(1_000, 10);
    let end = LeakCheckpoint::new(1_000, 10);
    let report = start.reconcile(end);
    assert!(report.balanced());
    assert!(!report.leaked());
    assert!(!report.over_freed());
    assert_eq!(report.residual_bytes, 0);
    assert_eq!(report.residual_allocations, 0);
}

#[test]
fn leak_positive_residual_is_a_leak() {
    let start = LeakCheckpoint::new(1_000, 10);
    let end = LeakCheckpoint::new(1_512, 13);
    let report = start.reconcile(end);
    assert!(report.leaked());
    assert!(!report.over_freed());
    assert_eq!(report.residual_bytes, 512);
    assert_eq!(report.residual_allocations, 3);
}

#[test]
fn leak_negative_residual_is_over_free() {
    let start = LeakCheckpoint::new(2_000, 20);
    let end = LeakCheckpoint::new(1_750, 18);
    let report = start.reconcile(end);
    assert!(report.over_freed());
    assert!(!report.leaked());
    assert_eq!(report.residual_bytes, -250);
    assert_eq!(report.residual_allocations, -2);
}

// ---- memory budgets ---------------------------------------------------------

#[test]
fn mem_budget_declare_update_and_lookup() {
    let mut reg = MemBudgetRegistry::new();
    assert!(reg.is_empty());
    reg.declare("assets", 1_000).declare("render", 2_000);
    assert_eq!(reg.len(), 2);
    // Re-declaring updates in place rather than appending.
    reg.declare("assets", 1_500);
    assert_eq!(reg.len(), 2);
    assert_eq!(reg.budget_of("assets"), Some(1_500));
    assert_eq!(reg.budget_of("render"), Some(2_000));
    assert_eq!(reg.budget_of("missing"), None);
}

#[test]
fn mem_budget_single_evaluate_flags_overspend() {
    let mut reg = MemBudgetRegistry::new();
    reg.declare("render", 2_000);
    let within = reg.evaluate("render", 1_800).unwrap();
    assert!(!within.over_budget);
    assert_eq!(within.overspend_bytes, 0);

    let over = reg.evaluate("render", 2_600).unwrap();
    assert!(over.over_budget);
    assert_eq!(over.overspend_bytes, 600);
    assert!((over.utilization() - 1.3).abs() < 1e-9);

    assert!(reg.evaluate("nope", 1).is_none());
}

#[test]
fn mem_budget_evaluate_all_aggregates_and_reports_offenders() {
    let mut reg = MemBudgetRegistry::new();
    reg.declare("assets", 1_000)
        .declare("render", 2_000)
        .declare("gameplay", 500);

    let measured = |cat: &str| -> u64 {
        match cat {
            "assets" => 1_200, // over by 200
            "render" => 1_500, // within
            "gameplay" => 900, // over by 400
            _ => 0,
        }
    };
    let report = reg.evaluate_all(measured);
    assert_eq!(report.statuses.len(), 3);
    assert_eq!(report.total_measured_bytes, 1_200 + 1_500 + 900);
    assert_eq!(report.total_budget_bytes, 1_000 + 2_000 + 500);
    assert!(report.any_over_budget);
    assert_eq!(report.total_overspend_bytes(), 200 + 400);

    let offenders: Vec<&MemBudgetStatus> = report.offenders().collect();
    assert_eq!(offenders.len(), 2);
    assert_eq!(offenders[0].category, "assets");
    assert_eq!(offenders[1].category, "gameplay");

    // Declaration order is preserved for stable diffing/HUD.
    assert_eq!(report.statuses[0].category, "assets");
    assert_eq!(report.statuses[1].category, "render");
    assert_eq!(report.statuses[2].category, "gameplay");
}

#[test]
fn mem_budget_unmeasured_category_treated_as_zero() {
    let mut reg = MemBudgetRegistry::new();
    reg.declare("assets", 1_000);
    let report = reg.evaluate_all(|_| 0);
    assert!(!report.any_over_budget);
    assert_eq!(report.total_measured_bytes, 0);
}

// ---- fragmentation ----------------------------------------------------------

#[test]
fn fragmentation_empty_pool_is_one_big_free_run() {
    let report = analyze_fragmentation(1_024, &[]);
    assert_eq!(report.used_bytes, 0);
    assert_eq!(report.free_bytes, 1_024);
    assert_eq!(report.free_run_count, 1);
    assert_eq!(report.largest_free_run, 1_024);
    assert!((report.fragmentation_ratio() - 0.0).abs() < 1e-9);
    assert!((report.occupancy_ratio() - 0.0).abs() < 1e-9);
    assert!(report.can_fit(1_024));
    assert!(!report.can_fit(1_025));
}

#[test]
fn fragmentation_full_pool_has_no_free_runs() {
    let report = analyze_fragmentation(1_000, &[Span::new(0, 1_000)]);
    assert_eq!(report.used_bytes, 1_000);
    assert_eq!(report.free_bytes, 0);
    assert_eq!(report.free_run_count, 0);
    assert_eq!(report.largest_free_run, 0);
    // No free bytes left to fragment.
    assert!((report.fragmentation_ratio() - 0.0).abs() < 1e-9);
    assert!((report.occupancy_ratio() - 1.0).abs() < 1e-9);
}

#[test]
fn fragmentation_scattered_holes_measured_against_largest_run() {
    // capacity 1000: occupied [0,100) [200,300) [900,1000)
    // free runs: [100,200)=100, [300,900)=600 -> 2 runs, largest=600, free=700
    let report = analyze_fragmentation(
        1_000,
        &[Span::new(0, 100), Span::new(200, 100), Span::new(900, 100)],
    );
    assert_eq!(report.used_bytes, 300);
    assert_eq!(report.free_bytes, 700);
    assert_eq!(report.free_run_count, 2);
    assert_eq!(report.largest_free_run, 600);
    // 1 - 600/700
    assert!((report.fragmentation_ratio() - (1.0 - 600.0 / 700.0)).abs() < 1e-9);
    assert!(report.can_fit(600));
    assert!(!report.can_fit(601));
}

#[test]
fn fragmentation_normalizes_unsorted_overlapping_and_oob_spans() {
    // Unsorted, overlapping, and extending past capacity; must merge/clamp to
    // the same result as a clean [0,300) [400,600) layout within capacity 600.
    let messy = analyze_fragmentation(
        600,
        &[
            Span::new(400, 500), // clamps to [400,600)
            Span::new(0, 200),
            Span::new(100, 200), // overlaps previous -> [0,300)
            Span::new(50, 10),   // fully inside -> no-op
        ],
    );
    let clean = analyze_fragmentation(600, &[Span::new(0, 300), Span::new(400, 600 - 400)]);
    assert_eq!(messy, clean);
    assert_eq!(messy.used_bytes, 300 + 200);
    assert_eq!(messy.free_bytes, 100);
    assert_eq!(messy.free_run_count, 1);
    assert_eq!(messy.largest_free_run, 100);
}

#[test]
fn occupancy_map_buckets_report_coverage_percentage() {
    // capacity 100, occupied [0,50): bucket 0 (0..50) full, bucket 1 (50..100) empty.
    let map = occupancy_map(100, &[Span::new(0, 50)], 2);
    assert_eq!(map, alloc::vec![100, 0]);

    // Half-covered bucket: occupied [0,25) over 4 buckets of width 25 -> first
    // bucket fully covered, rest empty.
    let map = occupancy_map(100, &[Span::new(0, 25)], 4);
    assert_eq!(map, alloc::vec![100, 0, 0, 0]);

    // Partial coverage within a single bucket: occupied [0,10) over 1 bucket of
    // width 100 -> 10%.
    let map = occupancy_map(100, &[Span::new(0, 10)], 1);
    assert_eq!(map, alloc::vec![10]);

    // Degenerate inputs yield an empty map.
    assert!(occupancy_map(0, &[Span::new(0, 10)], 4).is_empty());
    assert!(occupancy_map(100, &[], 0).is_empty());
}
