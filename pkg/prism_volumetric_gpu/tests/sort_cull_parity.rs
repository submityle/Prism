//! Real-device parity for the sort/cull decision twin:
//! [`GpuSortCull`](prism_volumetric_gpu::sort_cull::GpuSortCull) must reproduce
//! the `CPU` golden
//! [`sort_cull`](prism_render_architecture::particle::sort_cull) across the depth
//! key, the blend-driven strategy matrix, the `AABB` algebra, the
//! frustum/distance/`HZB` cull classification, the significance blend and the
//! per-frame sleep accumulator — on an empty batch, a set of named fixtures
//! placed clear of every branch boundary, and a large pseudo-random batch
//! compared lane for lane.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of comparisons and
//! arithmetic, so `CPU` and `GPU` evaluate the same closed form in the same
//! associativity. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The comparison therefore allows `abs_diff <= 1e-4`
//! or `rel_diff <= 1e-3` on the continuous values (`signed_distance`, `center`,
//! `half_extents`, `bounding_radius`, `significance`) and asserts an *exact*
//! match on the quantized keys, the discrete classification codes and every
//! boolean. Fixtures and the random batch are placed clear of the quantization
//! half-step, the `+/-radius` tangent and the distance / sleep thresholds so a
//! legal `ULP` perturbation never flips a verdict.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::sort_cull`；
//! standard depth-key quantization, frustum-plane sphere culling and bounds
//! algebra; no third-party engine source or derived code.

use prism_render_architecture::particle::sort_cull::{
    BlendMode, CullReason, Frustum, Plane, SleepParams, SleepState, SortDecision,
};
use prism_render_architecture::particle::{SortStrategy, Vec3};
use prism_volumetric_gpu::sort_cull::{
    cpu_reference, GpuSortCull, GpuSortCullQuery, GpuSortCullResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the continuous values. A `GPU` may fuse a
/// multiply-add the scalar reference leaves separate, perturbing the low
/// mantissa bits by a few units in the last place; `1e-4` admits that legal
/// slack while still failing a genuinely wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Returns whether two [`Vec3`] values agree component-wise within tolerance.
fn close_vec3(a: Vec3, b: Vec3) -> bool {
    close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z)
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// The axis-aligned unit box `[-1, 1]` on every axis as an inward-facing
/// six-plane frustum, the fixture most of the cull cases probe.
fn unit_box_frustum() -> Frustum {
    Frustum::new([
        Plane::new(Vec3::new(1.0, 0.0, 0.0), 1.0),
        Plane::new(Vec3::new(-1.0, 0.0, 0.0), 1.0),
        Plane::new(Vec3::new(0.0, 1.0, 0.0), 1.0),
        Plane::new(Vec3::new(0.0, -1.0, 0.0), 1.0),
        Plane::new(Vec3::new(0.0, 0.0, 1.0), 1.0),
        Plane::new(Vec3::new(0.0, 0.0, -1.0), 1.0),
    ])
}

/// A sensible default query every fixture tweaks: a mid-range front-to-back
/// depth clear of the quantization half-step, an order-independent blend, a
/// valid three-point box, the unit-box frustum, a small visible sphere, a
/// disabled-sleep significance and an awake emitter.
fn base() -> GpuSortCullQuery {
    GpuSortCullQuery {
        depth: 20.0,
        near: 0.0,
        far: 100.0,
        back_to_front: false,
        sort: SortDecision {
            blend: BlendMode::Opaque,
            particle_count: 0,
            radix_min_count: 1024,
            prefer_shared_oit: false,
        },
        points: [
            Vec3::new(-1.0, 0.0, 2.0),
            Vec3::new(3.0, -4.0, 0.0),
            Vec3::new(0.0, 5.0, -6.0),
        ],
        plane_point: Vec3::new(0.25, 0.0, 0.0),
        frustum: unit_box_frustum(),
        sphere_center: Vec3::ZERO,
        sphere_radius: 0.1,
        camera_position: Vec3::new(0.0, 0.0, -10.0),
        max_distance: 100.0,
        occluded: false,
        screen_coverage: 0.5,
        speed: 0.0,
        speed_ref: 10.0,
        sleep_state: SleepState::default(),
        sleep_significance: 0.5,
        sleep_params: SleepParams {
            sleep_below: 0.1,
            frames_to_sleep: 3,
        },
    }
}

/// Runs the `GPU` dispatch and asserts strict lane-for-lane parity against the
/// `CPU` golden: every quantized key, discrete code and boolean matches exactly
/// and every continuous value matches within tolerance. Returns the `GPU`
/// verdicts for extra per-test assertions. Use only for fixtures placed clear of
/// every branch boundary.
fn check(
    ctx: &GpuContext,
    gpu: &GpuSortCull,
    queries: &[GpuSortCullQuery],
) -> Vec<GpuSortCullResult> {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let c = cpu_reference(q);
        assert_eq!(
            g.quantized_depth, c.quantized_depth,
            "lane {lane}: quantized_depth"
        );
        assert_eq!(g.sort_key, c.sort_key, "lane {lane}: sort_key");
        assert_eq!(g.needs_sort, c.needs_sort, "lane {lane}: needs_sort");
        assert_eq!(
            g.sort_strategy, c.sort_strategy,
            "lane {lane}: sort_strategy"
        );
        assert_eq!(
            g.aabb_is_valid, c.aabb_is_valid,
            "lane {lane}: aabb_is_valid"
        );
        assert_eq!(
            g.aabb_empty_valid, c.aabb_empty_valid,
            "lane {lane}: aabb_empty_valid"
        );
        assert_eq!(
            g.frustum_intersects, c.frustum_intersects,
            "lane {lane}: frustum_intersects"
        );
        assert_eq!(g.cull_visible, c.cull_visible, "lane {lane}: cull_visible");
        assert_eq!(g.cull_reason, c.cull_reason, "lane {lane}: cull_reason");
        assert_eq!(g.hzb_visible, c.hzb_visible, "lane {lane}: hzb_visible");
        assert_eq!(g.hzb_reason, c.hzb_reason, "lane {lane}: hzb_reason");
        assert_eq!(
            g.out_idle_frames, c.out_idle_frames,
            "lane {lane}: out_idle_frames"
        );
        assert_eq!(g.out_asleep, c.out_asleep, "lane {lane}: out_asleep");

        assert!(
            close_vec3(g.aabb_center, c.aabb_center),
            "lane {lane}: aabb_center gpu {:?} cpu {:?}",
            g.aabb_center,
            c.aabb_center
        );
        assert!(
            close_vec3(g.aabb_half_extents, c.aabb_half_extents),
            "lane {lane}: aabb_half_extents gpu {:?} cpu {:?}",
            g.aabb_half_extents,
            c.aabb_half_extents
        );
        assert!(
            close(g.aabb_bounding_radius, c.aabb_bounding_radius),
            "lane {lane}: aabb_bounding_radius gpu {} cpu {}",
            g.aabb_bounding_radius,
            c.aabb_bounding_radius
        );
        assert!(
            close_vec3(g.aabb_union_center, c.aabb_union_center),
            "lane {lane}: aabb_union_center gpu {:?} cpu {:?}",
            g.aabb_union_center,
            c.aabb_union_center
        );
        assert!(
            close(g.plane0_signed_distance, c.plane0_signed_distance),
            "lane {lane}: plane0_signed_distance gpu {} cpu {}",
            g.plane0_signed_distance,
            c.plane0_signed_distance
        );
        assert!(
            close(g.significance, c.significance),
            "lane {lane}: significance gpu {} cpu {}",
            g.significance,
            c.significance
        );
    }
    got
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSortCull::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn depth_key_both_orderings() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSortCull::new(&ctx);
    // depth 20 over [0, 100]: t * 65535 = 13107.0 exactly, so its fractional
    // part is 0.0 — far from the 0.5 floor boundary and immune to a ULP flip.
    let mut front = base();
    front.depth = 20.0;
    front.back_to_front = false;
    let mut back = base();
    back.depth = 20.0;
    back.back_to_front = true;
    let got = check(&ctx, &gpu, &[front, back]);
    assert_eq!(got[0].quantized_depth, 13107, "front key");
    assert_eq!(got[0].sort_key, 13107, "front sort key keeps depth");
    assert_eq!(
        got[1].sort_key,
        65535 - 13107,
        "back-to-front inverts the key"
    );
}

#[test]
fn depth_degenerate_range_and_clamp() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSortCull::new(&ctx);
    // Degenerate range short-circuits to 0; out-of-range depth clamps to the far
    // end (65535) without touching the half-step.
    let mut degenerate = base();
    degenerate.depth = 5.0;
    degenerate.near = 10.0;
    degenerate.far = 10.0;
    let mut clamp_high = base();
    clamp_high.depth = 1000.0;
    let got = check(&ctx, &gpu, &[degenerate, clamp_high]);
    assert_eq!(got[0].quantized_depth, 0, "far <= near yields 0");
    assert_eq!(got[1].quantized_depth, 65535, "over-far clamps to the top");
}

#[test]
fn needs_sort_over_every_blend_mode() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSortCull::new(&ctx);
    let modes = [
        BlendMode::Opaque,
        BlendMode::Additive,
        BlendMode::Premultiplied,
        BlendMode::AlphaBlend,
    ];
    let queries: Vec<GpuSortCullQuery> = modes
        .iter()
        .map(|&m| {
            let mut q = base();
            q.sort.blend = m;
            q
        })
        .collect();
    let got = check(&ctx, &gpu, &queries);
    assert!(!got[0].needs_sort, "opaque needs no sort");
    assert!(!got[1].needs_sort, "additive needs no sort");
    assert!(!got[2].needs_sort, "premultiplied needs no sort");
    assert!(got[3].needs_sort, "alpha blend needs a sort");
}

#[test]
fn strategy_matrix_covers_every_branch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSortCull::new(&ctx);
    // Order-independent blend never sorts.
    let mut opaque = base();
    opaque.sort.blend = BlendMode::Opaque;
    opaque.sort.particle_count = 10_000;
    // Alpha blend but a single particle: nothing to order.
    let mut single = base();
    single.sort.blend = BlendMode::AlphaBlend;
    single.sort.particle_count = 1;
    // Alpha blend, many particles, routed to the shared OIT path.
    let mut shared = base();
    shared.sort.blend = BlendMode::AlphaBlend;
    shared.sort.particle_count = 5_000;
    shared.sort.prefer_shared_oit = true;
    // Standalone, at or above the radix threshold.
    let mut radix = base();
    radix.sort.blend = BlendMode::AlphaBlend;
    radix.sort.particle_count = 4_096;
    radix.sort.radix_min_count = 1_024;
    radix.sort.prefer_shared_oit = false;
    // Standalone, below the radix threshold.
    let mut bitonic = base();
    bitonic.sort.blend = BlendMode::AlphaBlend;
    bitonic.sort.particle_count = 64;
    bitonic.sort.radix_min_count = 1_024;
    bitonic.sort.prefer_shared_oit = false;
    let got = check(&ctx, &gpu, &[opaque, single, shared, radix, bitonic]);
    assert_eq!(got[0].sort_strategy, SortStrategy::None);
    assert_eq!(got[1].sort_strategy, SortStrategy::None);
    assert_eq!(got[2].sort_strategy, SortStrategy::SharedOit);
    assert_eq!(got[3].sort_strategy, SortStrategy::ViewDepthRadix);
    assert_eq!(got[4].sort_strategy, SortStrategy::ViewDepthBitonic);
}

#[test]
fn aabb_algebra_over_a_valid_box() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSortCull::new(&ctx);
    let mut q = base();
    q.points = [
        Vec3::new(-1.0, 0.0, 2.0),
        Vec3::new(3.0, -4.0, 0.0),
        Vec3::new(0.0, 5.0, -6.0),
    ];
    let got = check(&ctx, &gpu, &[q]);
    // Bounds are min (-1,-4,-6) .. max (3,5,2); center (1, 0.5, -2).
    assert!(got[0].aabb_is_valid, "a three-point box is valid");
    assert!(!got[0].aabb_empty_valid, "the empty box is never valid");
    assert!(
        close_vec3(got[0].aabb_center, Vec3::new(1.0, 0.5, -2.0)),
        "center {:?}",
        got[0].aabb_center
    );
    assert!(
        close_vec3(got[0].aabb_half_extents, Vec3::new(2.0, 4.5, 4.0)),
        "half-extents {:?}",
        got[0].aabb_half_extents
    );
}

#[test]
fn plane_signed_distance_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSortCull::new(&ctx);
    // plane0 of the unit box is x >= -1 (normal +x, d = 1); a probe at x = 2.5
    // is 3.5 inside, well clear of the plane.
    let mut q = base();
    q.plane_point = Vec3::new(2.5, 0.0, 0.0);
    let got = check(&ctx, &gpu, &[q]);
    assert!(
        close(got[0].plane0_signed_distance, 3.5),
        "signed distance {}",
        got[0].plane0_signed_distance
    );
}

#[test]
fn frustum_keeps_inside_and_straddling_rejects_outside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSortCull::new(&ctx);
    // Centered small sphere: comfortably inside every plane.
    let mut inside = base();
    inside.sphere_center = Vec3::ZERO;
    inside.sphere_radius = 0.1;
    // Center just outside +x but the radius straddles the plane: kept
    // (conservative), with a 0.3 margin away from the tangent.
    let mut straddle = base();
    straddle.sphere_center = Vec3::new(1.2, 0.0, 0.0);
    straddle.sphere_radius = 0.5;
    // Fully outside +x by well over a radius: rejected.
    let mut outside = base();
    outside.sphere_center = Vec3::new(3.0, 0.0, 0.0);
    outside.sphere_radius = 0.5;
    let got = check(&ctx, &gpu, &[inside, straddle, outside]);
    assert!(got[0].frustum_intersects, "inside sphere kept");
    assert!(got[1].frustum_intersects, "straddling sphere kept");
    assert!(!got[2].frustum_intersects, "outside sphere rejected");
}

#[test]
fn cull_reports_the_first_failing_test() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSortCull::new(&ctx);
    // Inside, close: visible.
    let mut ok = base();
    ok.sphere_center = Vec3::ZERO;
    ok.sphere_radius = 0.2;
    ok.max_distance = 100.0;
    // Beyond distance is reported before frustum (camera at z = -10, sphere at
    // z = 90 is 100 away, cull distance 50 with a wide margin).
    let mut far = base();
    far.sphere_center = Vec3::new(0.0, 0.0, 90.0);
    far.sphere_radius = 0.2;
    far.max_distance = 50.0;
    // Outside frustum, within distance.
    let mut out = base();
    out.sphere_center = Vec3::new(5.0, 0.0, 0.0);
    out.sphere_radius = 0.2;
    out.max_distance = 100.0;
    let got = check(&ctx, &gpu, &[ok, far, out]);
    assert!(got[0].cull_visible, "near inside sphere visible");
    assert_eq!(got[0].cull_reason, CullReason::Visible);
    assert!(!got[1].cull_visible);
    assert_eq!(got[1].cull_reason, CullReason::BeyondDistance);
    assert!(!got[2].cull_visible);
    assert_eq!(got[2].cull_reason, CullReason::OutsideFrustum);
}

#[test]
fn hzb_only_culls_visible_particles() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSortCull::new(&ctx);
    // Visible sphere, HZB reports occluded: folds to an HZB cull.
    let mut occluded_visible = base();
    occluded_visible.sphere_center = Vec3::ZERO;
    occluded_visible.sphere_radius = 0.2;
    occluded_visible.occluded = true;
    // Visible sphere, HZB clear: stays visible.
    let mut clear_visible = base();
    clear_visible.sphere_center = Vec3::ZERO;
    clear_visible.sphere_radius = 0.2;
    clear_visible.occluded = false;
    // Already outside the frustum: HZB keeps the earlier reason.
    let mut already_out = base();
    already_out.sphere_center = Vec3::new(5.0, 0.0, 0.0);
    already_out.sphere_radius = 0.2;
    already_out.occluded = true;
    let got = check(&ctx, &gpu, &[occluded_visible, clear_visible, already_out]);
    assert!(!got[0].hzb_visible);
    assert_eq!(got[0].hzb_reason, CullReason::HzbOccluded);
    assert!(got[1].hzb_visible);
    assert_eq!(got[1].hzb_reason, CullReason::Visible);
    assert!(!got[2].hzb_visible);
    assert_eq!(
        got[2].hzb_reason,
        CullReason::OutsideFrustum,
        "an already-culled decision keeps its reason"
    );
}

#[test]
fn significance_takes_the_dominant_term() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSortCull::new(&ctx);
    // Large but still.
    let mut big_still = base();
    big_still.screen_coverage = 0.8;
    big_still.speed = 0.0;
    big_still.speed_ref = 10.0;
    // Small but fast (motion saturates to 1).
    let mut small_fast = base();
    small_fast.screen_coverage = 0.1;
    small_fast.speed = 10.0;
    small_fast.speed_ref = 10.0;
    // Non-positive reference ignores motion.
    let mut no_ref = base();
    no_ref.screen_coverage = 0.3;
    no_ref.speed = 100.0;
    no_ref.speed_ref = 0.0;
    let got = check(&ctx, &gpu, &[big_still, small_fast, no_ref]);
    assert!(close(got[0].significance, 0.8), "coverage dominant");
    assert!(close(got[1].significance, 1.0), "motion saturates");
    assert!(
        close(got[2].significance, 0.3),
        "no reference ignores motion"
    );
}

#[test]
fn sleep_accumulates_then_wakes_instantly() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSortCull::new(&ctx);
    let params = SleepParams {
        sleep_below: 0.1,
        frames_to_sleep: 3,
    };
    // Four frames chained: three idle (sig 0.05 < 0.1, each 0.05 clear of the
    // threshold) accumulate to a sleep at frame 3, then a loud frame wakes it.
    let mut state = SleepState::default();
    let mut queries = Vec::new();
    for &sig in &[0.05, 0.05, 0.05, 0.9] {
        let mut q = base();
        q.sleep_state = state;
        q.sleep_significance = sig;
        q.sleep_params = params;
        queries.push(q);
        // Advance the host-side mirror so the next frame feeds on this verdict.
        let advanced = cpu_reference(queries.last().expect("just pushed"));
        state = SleepState {
            idle_frames: advanced.out_idle_frames,
            asleep: advanced.out_asleep,
        };
    }
    let got = check(&ctx, &gpu, &queries);
    assert_eq!(got[0].out_idle_frames, 1);
    assert!(!got[0].out_asleep);
    assert_eq!(got[2].out_idle_frames, 3);
    assert!(got[2].out_asleep, "three idle frames reach sleep");
    assert_eq!(got[3].out_idle_frames, 0);
    assert!(!got[3].out_asleep, "a loud frame wakes instantly");
}

#[test]
fn random_batch_matches_lane_for_lane() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSortCull::new(&ctx);
    let frustum = unit_box_frustum();
    let blends = [
        BlendMode::Opaque,
        BlendMode::Additive,
        BlendMode::Premultiplied,
        BlendMode::AlphaBlend,
    ];
    let mut state: u64 = 0x5f3d_c0ff_ee15_600d;
    let mut queries: Vec<GpuSortCullQuery> = Vec::new();
    while queries.len() < 4_096 {
        // Depth clear of the quantization half-step: reject when t * 65535 lands
        // within 0.1 of a .5 boundary where a ULP flip could change the floor.
        let depth = lcg(&mut state) * 100.0;
        let scaled = (depth / 100.0) * 65535.0;
        let frac = scaled - scaled.floor();
        if (frac - 0.5).abs() < 0.1 {
            continue;
        }

        // Sphere center in [-3, 3]^3 with a small radius, kept clear of every
        // plane tangent (|signed_distance + radius| >= 0.05) so the frustum
        // verdict is unambiguous.
        let center = Vec3::new(
            lcg(&mut state) * 6.0 - 3.0,
            lcg(&mut state) * 6.0 - 3.0,
            lcg(&mut state) * 6.0 - 3.0,
        );
        let radius = 0.05 + lcg(&mut state) * 0.15;
        let mut tangent = false;
        for plane in frustum.planes {
            let sd = plane.signed_distance(center);
            if (sd + radius).abs() < 0.05 {
                tangent = true;
            }
        }
        if tangent {
            continue;
        }

        // Distance cull: half the queries disable it, the rest use a finite
        // range kept clear of the cull shell (||camera - center| - cull_at| >=
        // margin) so the BeyondDistance verdict never ties.
        let camera = Vec3::new(0.0, 0.0, -10.0);
        let use_distance = lcg(&mut state) < 0.5;
        let max_distance = if use_distance {
            4.0 + lcg(&mut state) * 10.0
        } else {
            0.0
        };
        if use_distance {
            let dist = camera.distance(center);
            let cull_at = max_distance + radius;
            if (dist - cull_at).abs() < 0.1 {
                continue;
            }
        }

        // Sleep significance clear of its threshold so the branch is unambiguous.
        let sleep_below = 0.1 + lcg(&mut state) * 0.4;
        let sleep_sig = lcg(&mut state);
        if (sleep_sig - sleep_below).abs() < 0.05 {
            continue;
        }

        let blend = blends[(lcg(&mut state) * 4.0) as usize % 4];
        let q = GpuSortCullQuery {
            depth,
            near: 0.0,
            far: 100.0,
            back_to_front: lcg(&mut state) < 0.5,
            sort: SortDecision {
                blend,
                particle_count: (lcg(&mut state) * 8_192.0) as u32,
                radix_min_count: 1 + (lcg(&mut state) * 4_096.0) as u32,
                prefer_shared_oit: lcg(&mut state) < 0.5,
            },
            points: [
                Vec3::new(
                    lcg(&mut state) * 20.0 - 10.0,
                    lcg(&mut state) * 20.0 - 10.0,
                    lcg(&mut state) * 20.0 - 10.0,
                ),
                Vec3::new(
                    lcg(&mut state) * 20.0 - 10.0,
                    lcg(&mut state) * 20.0 - 10.0,
                    lcg(&mut state) * 20.0 - 10.0,
                ),
                Vec3::new(
                    lcg(&mut state) * 20.0 - 10.0,
                    lcg(&mut state) * 20.0 - 10.0,
                    lcg(&mut state) * 20.0 - 10.0,
                ),
            ],
            plane_point: Vec3::new(
                lcg(&mut state) * 6.0 - 3.0,
                lcg(&mut state) * 6.0 - 3.0,
                lcg(&mut state) * 6.0 - 3.0,
            ),
            frustum,
            sphere_center: center,
            sphere_radius: radius,
            camera_position: camera,
            max_distance,
            occluded: lcg(&mut state) < 0.5,
            screen_coverage: lcg(&mut state),
            speed: lcg(&mut state) * 20.0,
            // Reference comfortably positive so the motion branch is stable.
            speed_ref: 5.0 + lcg(&mut state) * 10.0,
            sleep_state: SleepState {
                idle_frames: (lcg(&mut state) * 6.0) as u32,
                asleep: lcg(&mut state) < 0.5,
            },
            sleep_significance: sleep_sig,
            sleep_params: SleepParams {
                sleep_below,
                frames_to_sleep: 1 + (lcg(&mut state) * 5.0) as u32,
            },
        };
        queries.push(q);
    }

    let got = check(&ctx, &gpu, &queries);

    // A large random spread must exercise both verdict classes on the branchy
    // outputs, so the test is not trivially passing on a one-sided batch.
    let mut saw_visible = false;
    let mut saw_culled = false;
    let mut saw_needs_sort = false;
    let mut saw_no_sort = false;
    let mut saw_asleep = false;
    let mut saw_awake = false;
    for g in &got {
        saw_visible |= g.cull_visible;
        saw_culled |= !g.cull_visible;
        saw_needs_sort |= g.needs_sort;
        saw_no_sort |= !g.needs_sort;
        saw_asleep |= g.out_asleep;
        saw_awake |= !g.out_asleep;
    }
    assert!(
        saw_visible && saw_culled,
        "random batch should produce both cull verdicts"
    );
    assert!(
        saw_needs_sort && saw_no_sort,
        "random batch should mix sorting and order-independent blends"
    );
    assert!(
        saw_asleep && saw_awake,
        "random batch should mix asleep and awake emitters"
    );
}
