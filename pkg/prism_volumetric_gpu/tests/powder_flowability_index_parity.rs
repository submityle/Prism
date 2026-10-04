//! Real-device parity for the bulk-powder flowability twin:
//! [`GpuPowderFlowabilityIndex`](prism_volumetric_gpu::powder_flowability_index::GpuPowderFlowabilityIndex)
//! must reproduce the `CPU` golden `PowderFlowability` of
//! `prism_physics_core::collider::flowability`. Granular flowability is graded
//! from two bulk measurements: the poured bulk density `ρ_b` and the tapped
//! density `ρ_t`. They define the Carr compressibility index
//! `C = 100 · (ρ_t − ρ_b)/ρ_t` and the Hausner ratio `H = ρ_t / ρ_b`, from
//! which a seven-level USP flow character is read off the Carr index.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the finiteness-and-order gate, then the Carr index, Hausner ratio and the
//! `<=`-chain classification — written out directly so the test never imports
//! `prism_render_architecture` or `prism_physics_core`, and uses no vector math
//! library.
//!
//! The fixtures cover one representative of each of the seven USP classes (each
//! kept clear of the `10 / 15 / 20 / 25 / 31 / 37` Carr thresholds so round-off
//! cannot flip the discrete class), every degenerate rejection (`ρ_t < ρ_b`, a
//! non-positive density, a `NaN` density and an infinite density), a batch of
//! two or more elements mixing valid and invalid inputs to validate the
//! `std430` stride, and an empty batch the host short-circuits with no
//! dispatch. A sweep over random finite inputs whose Carr index stays well
//! inside a class window follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The golden and the kernel evaluate the same `f32` closed form but need not
//! be bit-exact (a `GPU` may contract a multiply-add). Each continuous output
//! is compared with `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`); the
//! discrete `flow_character` class and the `valid` flag are compared exactly.
//! The fixtures keep the Carr index away from the classification knees so the
//! two sides agree on the discrete class.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::flowability`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::powder_flowability_index::{
    GpuPowderFlowabilityIndex, PowderFlowabilityIndexQuery, PowderFlowabilityIndexResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Ordered finiteness predicate matching the kernel's `abs(x) < 3.0e38` guard
/// (which rejects both infinities and `NaN`), rather than a bare `x == x`.
fn finite(x: f32) -> bool {
    x.abs() < 3.0e38
}

/// Independent host oracle: reproduces `PowderFlowability::from_densities`
/// followed by `carr_index`, `hausner_ratio` and `flow_character` in the golden
/// operator order and `f32` arithmetic. Returns the Carr index, the Hausner
/// ratio, the USP class and the validity flag.
fn oracle(q: &PowderFlowabilityIndexQuery) -> (f32, f32, u32, bool) {
    let rb = q.bulk_density;
    let rt = q.tapped_density;
    let valid_rb = finite(rb) && rb > 0.0;
    let valid_rt = finite(rt) && rt > 0.0;
    let order_ok = rt >= rb;
    let base_valid = valid_rb && valid_rt && order_ok;
    let safe_rt = if base_valid { rt } else { 1.0 };
    let safe_rb = if base_valid { rb } else { 1.0 };
    let carr = 100.0 * (rt - rb) / safe_rt;
    let hausner = rt / safe_rb;
    let outputs_finite = finite(carr) && finite(hausner);
    let valid = base_valid && outputs_finite;
    let fc = classify(carr);
    if valid {
        (carr, hausner, fc, true)
    } else {
        (0.0, 0.0, 0u32, false)
    }
}

/// The USP flow-character classification: a chain of ordered `<=` compares on
/// the Carr index, matching the kernel exactly.
fn classify(carr: f32) -> u32 {
    if carr <= 10.0 {
        0u32
    } else if carr <= 15.0 {
        1u32
    } else if carr <= 20.0 {
        2u32
    } else if carr <= 25.0 {
        3u32
    } else if carr <= 31.0 {
        4u32
    } else if carr <= 37.0 {
        5u32
    } else {
        6u32
    }
}

/// Mixed absolute-or-relative closeness for a continuous quantity.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= 1.0e-4 {
        return true;
    }
    diff <= 1.0e-3 * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// Asserts a single `GPU` result matches the independent oracle: the discrete
/// `valid` flag and `flow_character` class exactly, and each continuous output
/// to tolerance when valid.
fn assert_parity(gpu: &PowderFlowabilityIndexResult, q: &PowderFlowabilityIndexQuery, label: &str) {
    let (carr, hausner, fc, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid {
        assert_eq!(
            gpu.flow_character, fc,
            "{label}: flow_character mismatch gpu={} oracle={}",
            gpu.flow_character, fc
        );
        assert!(
            close(gpu.carr_index, carr),
            "{label}: carr_index mismatch gpu={} oracle={}",
            gpu.carr_index,
            carr
        );
        assert!(
            close(gpu.hausner_ratio, hausner),
            "{label}: hausner_ratio mismatch gpu={} oracle={}",
            gpu.hausner_ratio,
            hausner
        );
    }
}

/// Builds a query from a target Carr index with `ρ_t = 1.0`, so
/// `ρ_b = 1 − C/100`. Used by the per-class fixtures to land cleanly inside a
/// class window.
fn query_for_carr(carr_percent: f32) -> PowderFlowabilityIndexQuery {
    PowderFlowabilityIndexQuery::new(1.0 - carr_percent / 100.0, 1.0)
}

#[test]
fn class_excellent() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPowderFlowabilityIndex::new(&ctx);
    // C≈5, well below the 10 threshold.
    let q = query_for_carr(5.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid);
    assert_eq!(out[0].flow_character, 0);
    assert_parity(&out[0], &q, "excellent");
}

#[test]
fn class_good() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPowderFlowabilityIndex::new(&ctx);
    // C≈12.5, clear of the 10 and 15 thresholds.
    let q = query_for_carr(12.5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].flow_character, 1);
    assert_parity(&out[0], &q, "good");
}

#[test]
fn class_fair() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPowderFlowabilityIndex::new(&ctx);
    // C≈17.5, clear of the 15 and 20 thresholds.
    let q = query_for_carr(17.5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].flow_character, 2);
    assert_parity(&out[0], &q, "fair");
}

#[test]
fn class_passable() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPowderFlowabilityIndex::new(&ctx);
    // C≈22.5, clear of the 20 and 25 thresholds.
    let q = query_for_carr(22.5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].flow_character, 3);
    assert_parity(&out[0], &q, "passable");
}

#[test]
fn class_poor() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPowderFlowabilityIndex::new(&ctx);
    // C≈28, clear of the 25 and 31 thresholds.
    let q = query_for_carr(28.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].flow_character, 4);
    assert_parity(&out[0], &q, "poor");
}

#[test]
fn class_very_poor() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPowderFlowabilityIndex::new(&ctx);
    // C≈34, clear of the 31 and 37 thresholds.
    let q = query_for_carr(34.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].flow_character, 5);
    assert_parity(&out[0], &q, "very_poor");
}

#[test]
fn class_extremely_poor() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPowderFlowabilityIndex::new(&ctx);
    // C≈45, well above the 37 threshold.
    let q = query_for_carr(45.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].flow_character, 6);
    assert_parity(&out[0], &q, "extremely_poor");
}

#[test]
fn tapped_below_bulk_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPowderFlowabilityIndex::new(&ctx);
    // Tapping can only compact the bed, so ρ_t < ρ_b is rejected.
    let q = PowderFlowabilityIndexQuery::new(1.2, 1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(!out[0].valid);
    assert_eq!(out[0].carr_index, 0.0);
    assert_eq!(out[0].hausner_ratio, 0.0);
    assert_eq!(out[0].flow_character, 0);
    assert_parity(&out[0], &q, "tapped_below_bulk");
}

#[test]
fn non_positive_density_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPowderFlowabilityIndex::new(&ctx);
    let q = PowderFlowabilityIndexQuery::new(0.0, 1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(!out[0].valid);
    assert_eq!(out[0].carr_index, 0.0);
    assert_parity(&out[0], &q, "non_positive");
}

#[test]
fn nan_density_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPowderFlowabilityIndex::new(&ctx);
    let q = PowderFlowabilityIndexQuery::new(f32::NAN, 1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(!out[0].valid);
    assert_parity(&out[0], &q, "nan_density");
}

#[test]
fn infinite_density_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPowderFlowabilityIndex::new(&ctx);
    let q = PowderFlowabilityIndexQuery::new(0.9, f32::INFINITY);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(!out[0].valid);
    assert_parity(&out[0], &q, "inf_density");
}

#[test]
fn batch_mixes_valid_and_invalid_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPowderFlowabilityIndex::new(&ctx);
    let queries = vec![
        query_for_carr(5.0),
        query_for_carr(28.0),
        PowderFlowabilityIndexQuery::new(1.2, 1.0),
        query_for_carr(45.0),
        PowderFlowabilityIndexQuery::new(f32::NAN, 1.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert!(out[0].valid);
    assert!(out[1].valid);
    assert!(!out[2].valid);
    assert!(out[3].valid);
    assert!(!out[4].valid);
    assert_eq!(out[0].flow_character, 0);
    assert_eq!(out[1].flow_character, 4);
    assert_eq!(out[3].flow_character, 6);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPowderFlowabilityIndex::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPowderFlowabilityIndex::new(&ctx);
    let mut lcg = Lcg::new(0x0C1A_7E55);
    // Six safe Carr windows, each kept at least ~1.0 away from the 10/15/20/
    // 25/31/37 classification knees so f32 round-off cannot flip the discrete
    // class. Each entry is (lo, hi, expected class).
    let windows: [(f32, f32, u32); 7] = [
        (2.0, 8.0, 0),
        (11.5, 13.5, 1),
        (16.5, 18.5, 2),
        (21.5, 23.5, 3),
        (26.5, 29.5, 4),
        (32.5, 35.5, 5),
        (40.0, 55.0, 6),
    ];
    let mut queries = Vec::with_capacity(512);
    let mut expected = Vec::with_capacity(512);
    while queries.len() < 512 {
        let w = windows[(lcg.next_u32() % 7) as usize];
        let carr = lcg.next_range(w.0, w.1);
        // ρ_t ∈ [0.5, 2.0], ρ_b = ρ_t·(1 − C/100). Since 0 < C < 100, this
        // keeps ρ_b strictly positive and ρ_t ≥ ρ_b, so every sample is valid.
        let rt = lcg.next_range(0.5, 2.0);
        let rb = rt * (1.0 - carr / 100.0);
        queries.push(PowderFlowabilityIndexQuery::new(rb, rt));
        expected.push(w.2);
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert!(res.valid, "sweep[{i}] should be valid");
        assert_eq!(res.flow_character, expected[i], "sweep[{i}] class mismatch");
        assert_parity(res, q, &format!("sweep[{i}]"));
    }
}

/// A small deterministic linear-congruential generator; the fixture carries no
/// external randomness. Constants are the Numerical Recipes values.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// A `[0, 1)` fraction built from the top bits, keeping the fixture pure
    /// integer host-side with no transcendental call.
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A `[lo, hi)` fraction.
    fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_unit()
    }
}
