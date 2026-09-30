//! Real-device parity for the LOD-selection twin: [`GpuLodSelector`] must
//! reproduce the CPU golden
//! [`select_lod`](prism_render_architecture::virtual_geometry::select_lod) for
//! every cluster, across the coarsen/refine/hold hysteresis branches and the
//! velocity-scaled prefetch.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! A selection is a pair of discrete level indices, not a continuous value. The
//! scenes place every query well away from a budget boundary so the emitted
//! levels are stable under any legal float reassociation and are asserted
//! index-for-index against the golden with no tolerance. The scenes span every
//! display level of the chain, the hold / refine / coarsen hysteresis branches,
//! the probe-absent (`None`) and probe-present previous paths, and a prefetch
//! query that must land no coarser than its display level.
//!
//! Provenance: standard screen-space-error LOD selection with hysteresis and
//! velocity-scaled prefetch; no Unreal Engine source or derived code.

extern crate alloc;

use alloc::collections::BTreeSet;

use prism_render_architecture::virtual_geometry::{
    select_lod, GeometryLodPolicy, LodLevel, LodProjection,
};
use prism_virtual_geometry_gpu::{GpuContext, GpuLodSelector, LodQuery};

/// The four-level chain from the golden's own tests: level `0` finest through
/// level `3` coarsest, geometric error quadrupling each step.
fn chain() -> [LodLevel; 4] {
    [
        LodLevel {
            level: 0,
            geometric_error: 0.01,
        },
        LodLevel {
            level: 1,
            geometric_error: 0.04,
        },
        LodLevel {
            level: 2,
            geometric_error: 0.16,
        },
        LodLevel {
            level: 3,
            geometric_error: 0.64,
        },
    ]
}

/// 1000px viewport, 90 deg vertical fov (`tan(45 deg) = 1`) => focal 500px, the
/// golden's own test projection.
fn projection() -> LodProjection {
    LodProjection::from_half_fov_tan(1000.0, 1.0)
}

/// Asserts the twin matches the golden index-for-index for `clusters` under
/// `policy`, and returns the display levels that appeared.
fn assert_parity(
    ctx: &GpuContext,
    policy: GeometryLodPolicy,
    clusters: &[LodQuery],
) -> BTreeSet<u32> {
    let selector = GpuLodSelector::new(ctx);
    let levels = chain();
    let proj = projection();
    let gpu = selector.select(ctx, proj, policy, &levels, clusters);
    assert_eq!(gpu.len(), clusters.len(), "one selection per cluster");

    let mut seen_display: BTreeSet<u32> = BTreeSet::new();
    for (i, &(dist, closing, prev)) in clusters.iter().enumerate() {
        let expected = select_lod(&levels, proj, dist, closing, policy, prev)
            .expect("non-empty chain always selects");
        assert_eq!(
            gpu[i].level, expected.level,
            "display level mismatch for query {:?}: gpu {}, cpu {}",
            clusters[i], gpu[i].level, expected.level
        );
        assert_eq!(
            gpu[i].prefetch_level, expected.prefetch_level,
            "prefetch level mismatch for query {:?}: gpu {}, cpu {}",
            clusters[i], gpu[i].prefetch_level, expected.prefetch_level
        );
        assert!(
            gpu[i].prefetch_level <= gpu[i].level,
            "prefetch must never be coarser than display for query {:?}",
            clusters[i]
        );
        seen_display.insert(gpu[i].level);
    }
    seen_display
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_lod_select_matches_cpu_golden_across_all_levels() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping lod-select parity: no wgpu adapter on this host");
        return;
    };
    // target 2px, no hysteresis: distances chosen so each of the four levels is
    // the coarsest that fits, every projected error well clear of the 2px
    // boundary (nearest boundary values are 1.6/6.4 at dist 50, etc.).
    let policy = GeometryLodPolicy {
        target_error_pixels: 2.0,
        ..Default::default()
    };
    let clusters: Vec<LodQuery> = vec![
        (4.0, 0.0, None),    // finest 0 fits, L1 (5px) does not
        (12.0, 0.0, None),   // coarsest 1 (1.67px) fits, L2 (6.67px) does not
        (50.0, 0.0, None),   // coarsest 2 (1.6px) fits, L3 (6.4px) does not
        (200.0, 0.0, None),  // coarsest 3 (1.6px) fits
        (50.0, 0.0, Some(3)),// prev too coarse (6.4px > 2px) -> refine to 2
        (12.0, 0.0, Some(0)),// prev too fine, fresh 1 sits under budget -> coarsen to 1
    ];
    let seen = assert_parity(&ctx, policy, &clusters);
    assert_eq!(
        seen,
        BTreeSet::from([0, 1, 2, 3]),
        "scene must exercise every display level, saw {seen:?}"
    );
}

#[test]
fn gpu_lod_select_honours_hysteresis_deadband() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // target 10px, hysteresis 6px: at dist 6 the fresh pick refines to level 1,
    // but a held level 2 (~13.3px) stays inside the relaxed 16px budget, so the
    // selector holds 2 rather than popping finer.
    let policy = GeometryLodPolicy {
        target_error_pixels: 10.0,
        hysteresis_pixels: 6.0,
        ..Default::default()
    };
    let clusters: Vec<LodQuery> = vec![
        (6.0, 0.0, Some(2)), // held at 2 (inside relaxed budget)
        (6.0, 0.0, None),    // fresh pick refines to 1
    ];
    let seen = assert_parity(&ctx, policy, &clusters);
    // Held (2) and fresh (1) must genuinely differ, else the deadband is untested.
    assert_eq!(
        seen,
        BTreeSet::from([1, 2]),
        "hysteresis scene must show both held and fresh levels, saw {seen:?}"
    );
}

#[test]
fn gpu_lod_select_prefetches_no_coarser_than_display() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // A fast-approaching camera: the prefetch distance is pulled well nearer, so
    // the prefetch level is finer than (or equal to) the display level.
    let policy = GeometryLodPolicy {
        target_error_pixels: 2.0,
        prefetch_velocity_scale: 4.0,
        ..Default::default()
    };
    let clusters: Vec<LodQuery> = vec![
        (45.0, 5.0, None),
        (200.0, 20.0, None),
        (100.0, 10.0, Some(2)),
    ];
    // Parity (incl. the prefetch <= display invariant) is asserted inside.
    let _ = assert_parity(&ctx, policy, &clusters);
}

#[test]
fn empty_chain_selects_nothing() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let selector = GpuLodSelector::new(&ctx);
    let clusters: Vec<LodQuery> = vec![(10.0, 0.0, None)];
    let out = selector.select(
        &ctx,
        projection(),
        GeometryLodPolicy::default(),
        &[],
        &clusters,
    );
    assert!(out.is_empty(), "an empty chain yields no selections");
}

#[test]
fn empty_clusters_selects_nothing() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let selector = GpuLodSelector::new(&ctx);
    let levels = chain();
    let out = selector.select(
        &ctx,
        projection(),
        GeometryLodPolicy::default(),
        &levels,
        &[],
    );
    assert!(out.is_empty(), "no clusters yields no selections");
}
