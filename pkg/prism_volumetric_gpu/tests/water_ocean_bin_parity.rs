//! Real-device parity for the ocean `clipmap` bin twin:
//! [`GpuWaterOceanBin`](prism_volumetric_gpu::water_ocean_bin::GpuWaterOceanBin)
//! must reproduce the per-body ring selection and geomorph weighting that the
//! `CPU` golden
//! [`bin_ocean_patches`](prism_render_architecture::water::ocean_lod::bin_ocean_patches)
//! performs via
//! [`resolve_ocean_patch`](prism_render_architecture::water::ocean_lod::resolve_ocean_patch),
//! across hand-chosen configs, boundary distances, mismatched slice lengths,
//! and a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden
//! [`bin_ocean_patches`](prism_render_architecture::water::ocean_lod::bin_ocean_patches)
//! is public and pure, so the expected plan is built in-host by calling it
//! directly; its per-body
//! [`resolve_ocean_patch`](prism_render_architecture::water::ocean_lod::resolve_ocean_patch)
//! pins each resolved body. A `GPU == oracle` pass is therefore directly a
//! `GPU == golden` pass. The host-side slice walk (a body with no matching
//! distance entry is skipped, mirroring the golden's `distances.get(i)`) is
//! reproduced by
//! [`GpuWaterOceanBin::pair_queries`](prism_volumetric_gpu::water_ocean_bin::GpuWaterOceanBin::pair_queries),
//! and the variable-length per-ring bucket assembly of
//! [`OceanClipmapPlan`](prism_render_architecture::water::ocean_lod::OceanClipmapPlan)
//! is reconstructed here on the host from the `GPU` results and compared bucket
//! for bucket.
//!
//! # Parity criterion
//!
//! The carried `body` identifier, the selected `ring`, and the produced count
//! are pure integer classification and are asserted exactly. The `morph` weight
//! threads through a subtract, a divide and a `clamp`, so a `GPU` divide may
//! land a few units in the last place from the scalar reference; it is asserted
//! within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Conditioning
//!
//! Because `ring` is a discrete tier, the random sweep rejects any distance
//! within a margin of a ring's outer radius (where the ring selection could
//! flip on a last-place difference) and within a margin of the selected ring's
//! morph band start (where the morph branch could flip). The degenerate
//! `morph_fraction = 0` and `ring_count = 0` cases are exercised separately with
//! wide-margin fixtures.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::ocean_lod`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::ocean_lod::{
    bin_ocean_patches, resolve_ocean_patch, OceanClipmapConfig,
};
use prism_render_architecture::water::WaterBodyHandle;
use prism_volumetric_gpu::water_ocean_bin::{
    GpuWaterOceanBin, WaterOceanBinQuery, WaterOceanBinResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the geomorph weight. A `GPU` divide may land a few
/// units in the last place from the scalar reference; `1e-4` admits that legal
/// slack while still failing a wrong port.
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

/// One batch: a shared `clipmap` config plus the parallel `bodies`/`distances`
/// slices the golden walks.
struct Batch {
    bodies: Vec<WaterBodyHandle>,
    distances: Vec<f32>,
    cfg: OceanClipmapConfig,
}

impl Batch {
    /// Builds the `GPU` queries for this batch, mirroring the golden's host-side
    /// slice walk via [`GpuWaterOceanBin::pair_queries`].
    fn queries(&self) -> Vec<WaterOceanBinQuery> {
        let ids: Vec<u32> = self.bodies.iter().map(|h| h.0).collect();
        GpuWaterOceanBin::pair_queries(
            &ids,
            &self.distances,
            self.cfg.ring_count,
            self.cfg.inner_radius,
            self.cfg.radius_growth,
            self.cfg.morph_fraction,
        )
    }

    /// Number of valid pairings the golden produces: `min(bodies, distances)`.
    fn valid_count(&self) -> usize {
        self.bodies.len().min(self.distances.len())
    }
}

/// Pins every `GPU` result in a batch against the in-host golden
/// `bin_ocean_patches`, both per-body (order-preserving) and after
/// reconstructing the per-ring buckets.
fn check(ctx: &GpuContext, gpu: &GpuWaterOceanBin, batch: &Batch) {
    let queries = batch.queries();
    let got = gpu.evaluate(ctx, &queries);

    let plan = bin_ocean_patches(&batch.bodies, &batch.distances, batch.cfg);
    let valid = batch.valid_count();

    assert_eq!(
        queries.len(),
        valid,
        "paired query count must equal min(bodies, distances)"
    );
    assert_eq!(
        got.len(),
        valid,
        "gpu result count must equal the paired query count"
    );
    assert_eq!(
        got.len(),
        plan.total(),
        "gpu result count must equal the golden plan's total patch count"
    );

    // Per-body, order-preserving check against the golden `resolve_ocean_patch`.
    for (idx, result) in got.iter().enumerate() {
        let body = batch.bodies[idx];
        let want = resolve_ocean_patch(body, batch.distances[idx], batch.cfg);
        assert_eq!(
            result.body, body.0,
            "body {idx} identifier: gpu {} vs cpu {}",
            result.body, body.0
        );
        assert_eq!(
            result.ring, want.ring,
            "body {idx} ring: gpu {} vs cpu {}",
            result.ring, want.ring
        );
        assert!(
            close(result.morph, want.morph),
            "body {idx} morph: gpu {} vs cpu {}",
            result.morph,
            want.morph
        );
    }

    // Reconstruct the per-ring buckets from the GPU results exactly as the
    // golden `OceanClipmapPlan::push` does (clamp the ring to the last bucket,
    // preserve input order) and compare bucket for bucket.
    let buckets = (batch.cfg.ring_count.max(1)) as usize;
    let mut gpu_rings: Vec<Vec<&WaterOceanBinResult>> = Vec::with_capacity(buckets);
    for _ in 0..buckets {
        gpu_rings.push(Vec::new());
    }
    for result in &got {
        let index = (result.ring as usize).min(buckets - 1);
        gpu_rings[index].push(result);
    }

    for (ring, gpu_bucket) in gpu_rings.iter().enumerate() {
        let want_bucket = plan.bucket(ring as u32);
        assert_eq!(
            gpu_bucket.len(),
            want_bucket.len(),
            "ring {ring} bucket length: gpu {} vs cpu {}",
            gpu_bucket.len(),
            want_bucket.len()
        );
        for (slot, (gpu_patch, want_patch)) in gpu_bucket.iter().zip(want_bucket.iter()).enumerate()
        {
            assert_eq!(
                gpu_patch.body, want_patch.body.0,
                "ring {ring} slot {slot} body: gpu {} vs cpu {}",
                gpu_patch.body, want_patch.body.0
            );
            assert_eq!(
                gpu_patch.ring, want_patch.ring,
                "ring {ring} slot {slot} ring: gpu {} vs cpu {}",
                gpu_patch.ring, want_patch.ring
            );
            assert!(
                close(gpu_patch.morph, want_patch.morph),
                "ring {ring} slot {slot} morph: gpu {} vs cpu {}",
                gpu_patch.morph,
                want_patch.morph
            );
        }
    }
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws a scalar in `[lo, hi]` at milli resolution from `state`.
fn draw(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let span = ((hi - lo) * 1000.0) as u32;
    lo + (lcg(state) % (span + 1)) as f32 / 1000.0
}

/// Whether `distance` keeps a comfortable margin from every ring's outer radius
/// (so the discrete ring selection is stable) and from the selected ring's
/// morph band start (so the morph branch is stable). Uses only arithmetic and
/// comparisons, never an `f32` `==`.
fn well_conditioned(distance: f32, cfg: OceanClipmapConfig) -> bool {
    let margin = 1.0_f32;
    // Keep clear of every ring boundary where ring selection could flip.
    for ring in 0..cfg.ring_count {
        let outer = cfg.ring_outer_radius(ring);
        if (distance - outer).abs() < margin {
            return false;
        }
    }
    // Keep clear of the selected ring's morph band start where the morph branch
    // could flip, when morphing is active.
    let selected = {
        let mut r = 0;
        let last = cfg.last_ring();
        while r < last {
            if distance <= cfg.ring_outer_radius(r) {
                break;
            }
            r += 1;
        }
        r
    };
    let inner = cfg.ring_inner_radius(selected);
    let outer = cfg.ring_outer_radius(selected);
    let band = outer - inner;
    if band > 1.0e-3 && cfg.morph_fraction > 1.0e-3 {
        let morph_start = outer - band * cfg.morph_fraction.min(1.0);
        if (distance - morph_start).abs() < margin {
            return false;
        }
    }
    true
}

/// Draws one well-conditioned random distance for a config, kept clear of every
/// ring boundary and the morph band start.
fn random_distance(state: &mut u64, cfg: OceanClipmapConfig) -> f32 {
    loop {
        let distance = draw(state, 0.0, 2000.0);
        if well_conditioned(distance, cfg) {
            return distance;
        }
    }
}

/// The reference `clipmap` config used in the golden tests: ring outer radii
/// `32, 64, 128, 256` with a `25%` morph band.
const CLIPMAP: OceanClipmapConfig = OceanClipmapConfig {
    ring_count: 4,
    inner_radius: 32.0,
    radius_growth: 2.0,
    morph_fraction: 0.25,
};

/// A batch of hand-chosen bodies spanning distinct rings and branches, each
/// distance kept clear of every discrete tie.
fn fixture_batch() -> Batch {
    // Distances chosen inside distinct rings/branches of CLIPMAP:
    // 10 -> ring 0 pre-morph, 28 -> ring 0 morph band, 40 -> ring 1 pre-morph,
    // 90 -> ring 2 pre-morph, 900 -> clamped last ring (morph saturates).
    let distances = vec![10.0, 28.0, 40.0, 90.0, 900.0];
    let bodies = vec![
        WaterBodyHandle(7),
        WaterBodyHandle(3),
        WaterBodyHandle(11),
        WaterBodyHandle(5),
        WaterBodyHandle(2),
    ];
    Batch {
        bodies,
        distances,
        cfg: CLIPMAP,
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_ocean_bin parity: no wgpu adapter");
        return;
    };
    let gpu = GpuWaterOceanBin::new(&ctx);
    // The host short-circuits an empty batch (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn fixture_batch_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOceanBin::new(&ctx);
    check(&ctx, &gpu, &fixture_batch());
}

#[test]
fn more_bodies_than_distances_skips_tail() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOceanBin::new(&ctx);
    // Five bodies but only three distances: the golden's `distances.get(i)`
    // skips the tail bodies, so only three patches are produced.
    let batch = Batch {
        bodies: vec![
            WaterBodyHandle(1),
            WaterBodyHandle(2),
            WaterBodyHandle(3),
            WaterBodyHandle(4),
            WaterBodyHandle(5),
        ],
        distances: vec![10.0, 90.0, 900.0],
        cfg: CLIPMAP,
    };
    assert_eq!(batch.valid_count(), 3);
    check(&ctx, &gpu, &batch);
}

#[test]
fn more_distances_than_bodies_ignores_extra() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOceanBin::new(&ctx);
    // Three bodies but five distances: the extra distances have no body, so
    // only three patches are produced.
    let batch = Batch {
        bodies: vec![WaterBodyHandle(9), WaterBodyHandle(8), WaterBodyHandle(7)],
        distances: vec![10.0, 40.0, 900.0, 50.0, 20.0],
        cfg: CLIPMAP,
    };
    assert_eq!(batch.valid_count(), 3);
    check(&ctx, &gpu, &batch);
}

#[test]
fn zero_bodies_produces_empty_plan() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOceanBin::new(&ctx);
    // No bodies: the paired batch is empty, so the host short-circuits with no
    // dispatch and the golden plan is empty.
    let batch = Batch {
        bodies: Vec::new(),
        distances: vec![10.0, 40.0, 90.0],
        cfg: CLIPMAP,
    };
    assert_eq!(batch.valid_count(), 0);
    check(&ctx, &gpu, &batch);
}

#[test]
fn morph_disabled_batch_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOceanBin::new(&ctx);
    // morph_fraction = 0: hard ring boundaries, morph is 0 inside a ring.
    let cfg = OceanClipmapConfig {
        morph_fraction: 0.0,
        ..CLIPMAP
    };
    let batch = Batch {
        bodies: vec![WaterBodyHandle(1), WaterBodyHandle(2), WaterBodyHandle(3)],
        distances: vec![10.0, 90.0, 900.0],
        cfg,
    };
    check(&ctx, &gpu, &batch);
}

#[test]
fn zero_ring_count_bins_into_single_bucket() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOceanBin::new(&ctx);
    // ring_count = 0 is a degenerate single-ring config: last_ring saturates to
    // 0, every body resolves to ring 0, and the plan has one bucket.
    let cfg = OceanClipmapConfig {
        ring_count: 0,
        ..CLIPMAP
    };
    let batch = Batch {
        bodies: vec![WaterBodyHandle(4), WaterBodyHandle(5)],
        distances: vec![50.0, 150.0],
        cfg,
    };
    check(&ctx, &gpu, &batch);
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOceanBin::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    // A single large batch of well-conditioned random bodies pins every output
    // across many workgroups; 512 bodies fit the module's soft batching ceiling.
    let mut bodies = Vec::with_capacity(512);
    let mut distances = Vec::with_capacity(512);
    for i in 0..512u32 {
        bodies.push(WaterBodyHandle(i));
        distances.push(random_distance(&mut state, CLIPMAP));
    }
    let batch = Batch {
        bodies,
        distances,
        cfg: CLIPMAP,
    };
    check(&ctx, &gpu, &batch);
}
