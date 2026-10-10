//! §24.5 statistical sampling profiler tests: symbol interning, flat-profile
//! self/inclusive folding, top-down / bottom-up call trees, collapsed stacks,
//! per-lane / per-thread faceting, and instrumented + sampled fusion. Every
//! expectation is a hand-computed oracle over a fixed four-sample buffer, so a
//! refactor cannot silently change an aggregation result.

use crate::sampling::{
    call_tree, collapsed_stacks, facet_by_lane, facet_by_thread, fuse, FoldDirection, FusionSource,
    InstrumentedSpan, LaneKind, SamplingProfiler, SymbolTable,
};

/// The 1 ms interval used throughout, so one sample tick is worth `1_000_000` ns.
const INTERVAL: u64 = 1_000_000;

/// Build the canonical fixture buffer. Interning order fixes the frame ids:
/// `main=0`, `update=1`, `physics=2`, `render=3`.
///
/// Samples (root to leaf), all weight 1:
/// - A: main;update;physics  (Main, thread 1)
/// - B: main;update;physics  (Main, thread 1)
/// - C: main;update;render   (Render, thread 2)
/// - D: main;render          (Compute, thread 3)
fn fixture() -> SamplingProfiler {
    let mut profiler = SamplingProfiler::new(INTERVAL);
    profiler.record_named(LaneKind::Main, 1, 10, &["main", "update", "physics"]);
    profiler.record_named(LaneKind::Main, 1, 20, &["main", "update", "physics"]);
    profiler.record_named(LaneKind::Render, 2, 30, &["main", "update", "render"]);
    profiler.record_named(LaneKind::Compute, 3, 40, &["main", "render"]);
    profiler
}

// ---- symbol interning -------------------------------------------------------

#[test]
fn interning_is_first_seen_and_stable() {
    let mut table = SymbolTable::new();
    assert!(table.is_empty());
    let main = table.intern("main");
    let update = table.intern("update");
    assert_eq!(main.0, 0);
    assert_eq!(update.0, 1);
    // Re-interning returns the same id and does not grow the table.
    assert_eq!(table.intern("main"), main);
    assert_eq!(table.len(), 2);
    assert_eq!(table.get("update"), Some(update));
    assert_eq!(table.get("absent"), None);
    assert_eq!(table.resolve(main), Some("main"));
    assert_eq!(table.resolve(update), Some("update"));
}

#[test]
fn fixture_assigns_expected_frame_ids() {
    let profiler = fixture();
    let symbols = profiler.symbols();
    assert_eq!(symbols.get("main").map(|f| f.0), Some(0));
    assert_eq!(symbols.get("update").map(|f| f.0), Some(1));
    assert_eq!(symbols.get("physics").map(|f| f.0), Some(2));
    assert_eq!(symbols.get("render").map(|f| f.0), Some(3));
    assert_eq!(profiler.len(), 4);
    assert_eq!(profiler.total_weight(), 4);
}

// ---- flat profile -----------------------------------------------------------

#[test]
fn flat_profile_self_and_inclusive_counts() {
    let profiler = fixture();
    let flat = profiler.flat_profile();

    assert_eq!(flat.total_samples(), 4);
    assert_eq!(flat.total_nanos(), 4 * INTERVAL);

    // Oracle self counts: physics leaf in A,B => 2; render leaf in C,D => 2;
    // main/update are never leaves => 0.
    // Oracle inclusive counts (deduped per sample): main 4, update 3,
    // physics 2, render 2.
    let main = profiler.symbols().get("main").unwrap();
    let update = profiler.symbols().get("update").unwrap();
    let physics = profiler.symbols().get("physics").unwrap();
    let render = profiler.symbols().get("render").unwrap();

    let main_row = flat.get(main).unwrap();
    assert_eq!(main_row.self_samples, 0);
    assert_eq!(main_row.inclusive_samples, 4);
    assert_eq!(main_row.inclusive_nanos, 4 * INTERVAL);

    assert_eq!(flat.get(update).unwrap().inclusive_samples, 3);
    assert_eq!(flat.get(physics).unwrap().self_samples, 2);
    assert_eq!(flat.get(physics).unwrap().inclusive_samples, 2);
    assert_eq!(flat.get(render).unwrap().self_samples, 2);
    assert_eq!(flat.get(render).unwrap().inclusive_samples, 2);
    assert_eq!(flat.get(physics).unwrap().self_nanos, 2 * INTERVAL);
}

#[test]
fn flat_profile_row_order_self_desc_then_frame_id() {
    let profiler = fixture();
    let flat = profiler.flat_profile();
    // Self: physics(2)=2, render(3)=2, main(0)=0, update(1)=0. Ties by frame id
    // ascending => [physics, render, main, update].
    let order: Vec<u32> = flat.rows().iter().map(|r| r.frame.0).collect();
    assert_eq!(order, alloc::vec![2, 3, 0, 1]);
    assert_eq!(flat.hottest().unwrap().frame.0, 2);
}

#[test]
fn flat_profile_self_fraction() {
    let profiler = fixture();
    let flat = profiler.flat_profile();
    let physics = profiler.symbols().get("physics").unwrap();
    // physics self 2 of 4 total samples => 0.5.
    assert!((flat.self_fraction(physics) - 0.5).abs() < 1e-12);
}

// ---- collapsed stacks -------------------------------------------------------

#[test]
fn collapsed_stacks_fold_and_render() {
    let profiler = fixture();
    let collapsed = collapsed_stacks(profiler.samples());
    // Distinct stacks lexicographically by frame-id path:
    // [0,1,2]=2, [0,1,3]=1, [0,3]=1.
    assert_eq!(collapsed.len(), 3);
    assert_eq!(
        collapsed[0].frames.iter().map(|f| f.0).collect::<Vec<_>>(),
        alloc::vec![0, 1, 2]
    );
    assert_eq!(collapsed[0].samples, 2);
    assert_eq!(collapsed[1].samples, 1);
    assert_eq!(collapsed[2].samples, 1);

    let symbols = profiler.symbols();
    assert_eq!(
        collapsed[0].to_folded_string(symbols),
        "main;update;physics 2"
    );
    assert_eq!(
        collapsed[1].to_folded_string(symbols),
        "main;update;render 1"
    );
    assert_eq!(collapsed[2].to_folded_string(symbols), "main;render 1");
}

#[test]
fn collapsed_stack_unknown_id_placeholder() {
    use crate::sampling::{CollapsedStack, FrameId};
    let empty = SymbolTable::new();
    let stack = CollapsedStack {
        frames: alloc::vec![FrameId(7)],
        samples: 3,
    };
    assert_eq!(stack.to_folded_string(&empty), "?7 3");
}

// ---- call trees -------------------------------------------------------------

#[test]
fn call_tree_top_down_structure() {
    let profiler = fixture();
    let tree = profiler.call_tree(FoldDirection::TopDown);

    // Node insertion order is deterministic:
    // 0 root, 1 main, 2 update, 3 physics, 4 render(under update),
    // 5 render(under main).
    assert_eq!(tree.len(), 6);
    assert_eq!(tree.direction(), FoldDirection::TopDown);
    assert!(!tree.is_empty());

    let root = tree.root();
    assert_eq!(root.frame, None);
    assert_eq!(root.inclusive_samples, 4);
    assert_eq!(root.children, alloc::vec![1]);

    let main = tree.node(1).unwrap();
    assert_eq!(main.inclusive_samples, 4);
    assert_eq!(main.self_samples, 0);
    // main's children sorted by inclusive desc: update(3) before render(1).
    assert_eq!(main.children, alloc::vec![2, 5]);

    let update = tree.node(2).unwrap();
    assert_eq!(update.inclusive_samples, 3);
    // physics(incl 2) before render(incl 1).
    assert_eq!(update.children, alloc::vec![3, 4]);

    assert_eq!(tree.node(3).unwrap().self_samples, 2); // physics
    assert_eq!(tree.node(4).unwrap().self_samples, 1); // render under update
    assert_eq!(tree.node(5).unwrap().self_samples, 1); // render under main

    assert_eq!(tree.node(3).unwrap().self_nanos(INTERVAL), 2 * INTERVAL);
    // Deepest chain root->main->update->physics counts 3 edges.
    assert_eq!(tree.max_depth(), 3);
}

#[test]
fn call_tree_bottom_up_roots_are_leaves() {
    let profiler = fixture();
    let tree = profiler.call_tree(FoldDirection::BottomUp);
    let root = tree.root();
    assert_eq!(root.inclusive_samples, 4);
    // Reversed stacks: leaves physics(incl 2) and render(incl 2) become the two
    // roots under the synthetic root, ordered by frame id on the inclusive tie.
    assert_eq!(root.children.len(), 2);
    let first = tree.node(root.children[0]).unwrap();
    let second = tree.node(root.children[1]).unwrap();
    assert_eq!(first.frame, profiler.symbols().get("physics"));
    assert_eq!(second.frame, profiler.symbols().get("render"));
    assert_eq!(first.inclusive_samples, 2);
    assert_eq!(second.inclusive_samples, 2);
    assert_eq!(tree.max_depth(), 3);
}

// ---- faceting ---------------------------------------------------------------

#[test]
fn facet_by_lane_canonical_order_and_weights() {
    let profiler = fixture();
    let lanes = facet_by_lane(profiler.samples(), INTERVAL);
    // Lanes present: Main(A,B), Render(C), Compute(D), in canonical order.
    assert_eq!(lanes.len(), 3);
    assert_eq!(lanes[0].lane, LaneKind::Main);
    assert_eq!(lanes[0].total_weight, 2);
    assert_eq!(lanes[0].total_nanos, 2 * INTERVAL);
    assert_eq!(lanes[1].lane, LaneKind::Render);
    assert_eq!(lanes[1].total_weight, 1);
    assert_eq!(lanes[2].lane, LaneKind::Compute);
    assert_eq!(lanes[2].total_weight, 1);
    // Main lane's only leaf is physics (A,B) => self 2.
    let physics = profiler.symbols().get("physics").unwrap();
    assert_eq!(lanes[0].flat.get(physics).unwrap().self_samples, 2);
}

#[test]
fn facet_by_thread_ascending_ids_and_lane() {
    let profiler = fixture();
    let threads = facet_by_thread(profiler.samples(), INTERVAL);
    assert_eq!(threads.len(), 3);
    assert_eq!(threads[0].thread_id, 1);
    assert_eq!(threads[0].lane, LaneKind::Main);
    assert_eq!(threads[0].total_weight, 2);
    assert_eq!(threads[1].thread_id, 2);
    assert_eq!(threads[1].lane, LaneKind::Render);
    assert_eq!(threads[2].thread_id, 3);
    assert_eq!(threads[2].lane, LaneKind::Compute);
}

// ---- fusion -----------------------------------------------------------------

#[test]
fn fuse_overlays_instrumented_and_sampled() {
    let profiler = fixture();
    let flat = profiler.flat_profile();
    let spans = alloc::vec![
        // physics: instrumented 2 ms == sampled 2 ms => agreement 1.0.
        InstrumentedSpan::new("physics", 2 * INTERVAL, 2 * INTERVAL, 2),
        // render: instrumented 4 ms vs sampled 2 ms => agreement 0.5.
        InstrumentedSpan::new("render", 4 * INTERVAL, 2 * INTERVAL, 2),
        // audio_mix: never sampled => instrumented only.
        InstrumentedSpan::new("audio_mix", 500_000, 500_000, 1),
    ];
    let fused = fuse(&flat, profiler.symbols(), &spans);

    let physics = fused.get("physics").unwrap();
    assert_eq!(physics.source, FusionSource::Both);
    assert_eq!(physics.agreement_ratio, Some(1.0));
    assert_eq!(physics.call_count, Some(2));

    let render = fused.get("render").unwrap();
    assert_eq!(render.source, FusionSource::Both);
    assert_eq!(render.agreement_ratio, Some(0.5));

    assert_eq!(
        fused.get("audio_mix").unwrap().source,
        FusionSource::InstrumentedOnly
    );
    assert_eq!(fused.get("main").unwrap().source, FusionSource::SampledOnly);
    assert_eq!(
        fused.get("update").unwrap().source,
        FusionSource::SampledOnly
    );

    // Only render disagrees beyond +/-25%.
    let disagreements = fused.disagreements(0.25);
    assert_eq!(disagreements.len(), 1);
    assert_eq!(disagreements[0].name, "render");
}

#[test]
fn fuse_entry_ordering_by_effective_inclusive() {
    let profiler = fixture();
    let flat = profiler.flat_profile();
    let spans = alloc::vec![
        InstrumentedSpan::new("physics", 2 * INTERVAL, 2 * INTERVAL, 2),
        InstrumentedSpan::new("render", 4 * INTERVAL, 2 * INTERVAL, 2),
        InstrumentedSpan::new("audio_mix", 500_000, 500_000, 1),
    ];
    let fused = fuse(&flat, profiler.symbols(), &spans);
    // Effective inclusive: main 4ms, render 4ms, update 3ms, physics 2ms,
    // audio_mix 0.5ms. Ties by name ascending => main before render.
    let names: Vec<&str> = fused.entries().iter().map(|e| e.name.as_str()).collect();
    assert_eq!(
        names,
        alloc::vec!["main", "render", "update", "physics", "audio_mix"]
    );
}

#[test]
fn empty_profiler_folds_to_empty() {
    let profiler = SamplingProfiler::new(0); // interval clamped to 1.
    assert!(profiler.is_empty());
    assert_eq!(profiler.interval_nanos(), 1);
    let flat = profiler.flat_profile();
    assert_eq!(flat.total_samples(), 0);
    assert!(flat.rows().is_empty());
    assert!(collapsed_stacks(profiler.samples()).is_empty());
    // A tree always has the synthetic root.
    let tree = call_tree(profiler.samples(), FoldDirection::TopDown, 1);
    assert_eq!(tree.len(), 1);
    assert_eq!(tree.max_depth(), 0);
}

extern crate alloc;
