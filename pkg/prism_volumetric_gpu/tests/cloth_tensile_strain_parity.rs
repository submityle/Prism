//! Real-device parity for the tensile-strain twin:
//! [`GpuClothTensileStrain`](prism_volumetric_gpu::cloth_tensile_strain::GpuClothTensileStrain)
//! must reproduce the `CPU` golden `tensile_strain` of
//! `prism_physics_core::soft::damage::strain`, the single scalar both the
//! tearing and plasticity damage models key off: the signed edge strain
//! `(length - rest_length) / rest_length`, with a `rest_length <= EPS_REST`
//! degeneracy floor that reports the edge invalid.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the ordered floor compare and the single guarded subtraction and division —
//! written out directly so the test never imports `prism_physics_core` or
//! `prism_render_architecture`. It mirrors the reference branch for branch: a
//! rest length at or below `EPS_REST = 1e-9` yields `valid = 0` with a cleared
//! strain (the golden's `None`), otherwise the signed ratio and `valid = 1`.
//!
//! The fixtures cover the regimes the kernel must honor: a rest length exactly
//! on the floor (degenerate), a rest length just above it, pure tension, pure
//! compression and zero strain, plus a multi-element batch that validates the
//! `std430` array stride end to end. A sweep over random rest lengths and
//! separations away from the floor follows, plus an empty batch the host
//! short-circuits with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The strain threads through a subtraction and a division, so `CPU` and `GPU`
//! evaluate the same closed form but need not be bit-exact. The continuous
//! comparison is `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`);
//! the discrete `valid` flag is compared exactly. The sweep keeps rest lengths
//! comfortably above the floor so parity never sits on the degeneracy knife
//! edge.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::damage::strain::tensile_strain`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::cloth_tensile_strain::{ClothTensileStrainQuery, GpuClothTensileStrain};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;

/// Degeneracy floor on the rest length, matching the golden `EPS_REST`.
const EPS_REST: f32 = 1.0e-9;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Independent host re-implementation of the golden `tensile_strain`, returning
/// the signed strain and the `valid` flag without importing the golden crate.
fn oracle(rest_length: f32, length: f32) -> (f32, u32) {
    if rest_length <= EPS_REST {
        (0.0, 0)
    } else {
        ((length - rest_length) / rest_length, 1)
    }
}

/// Dispatches a single query and asserts its parity against the host oracle.
fn assert_parity(ctx: &GpuContext, gpu: &GpuClothTensileStrain, q: ClothTensileStrainQuery) {
    let results = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(results.len(), 1, "one result per query");
    let r = results[0];
    let (strain, valid) = oracle(q.rest_length, q.length);
    assert_eq!(r.valid, valid, "valid mismatch: query={q:?}");
    if valid == 1 {
        assert!(
            close(r.strain, strain),
            "strain mismatch: gpu={} cpu={strain} query={q:?}",
            r.strain
        );
    } else {
        assert_eq!(
            r.strain, 0.0,
            "degenerate strain must be cleared: query={q:?}"
        );
    }
}

#[test]
fn degenerate_rest_on_floor_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothTensileStrain::new(&ctx);
    // Rest length exactly on the floor: the golden returns None → valid = 0.
    assert_parity(&ctx, &gpu, ClothTensileStrainQuery::new(1.0e-9, 0.5));
}

#[test]
fn degenerate_rest_below_floor_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothTensileStrain::new(&ctx);
    // A sub-floor rest length and a zero rest length are both degenerate.
    assert_parity(&ctx, &gpu, ClothTensileStrainQuery::new(5.0e-10, 0.3));
    assert_parity(&ctx, &gpu, ClothTensileStrainQuery::new(0.0, 0.3));
}

#[test]
fn just_above_floor_is_valid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothTensileStrain::new(&ctx);
    // Rest length just above the floor is valid; strain is huge but finite.
    assert_parity(&ctx, &gpu, ClothTensileStrainQuery::new(2.0e-9, 2.0e-9));
}

#[test]
fn pure_tension() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothTensileStrain::new(&ctx);
    // Stretched edge: length > rest → positive strain (here +0.25).
    assert_parity(&ctx, &gpu, ClothTensileStrainQuery::new(2.0, 2.5));
}

#[test]
fn pure_compression() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothTensileStrain::new(&ctx);
    // Compressed edge: length < rest → negative strain (here -0.25).
    assert_parity(&ctx, &gpu, ClothTensileStrainQuery::new(2.0, 1.5));
}

#[test]
fn zero_strain_at_rest() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothTensileStrain::new(&ctx);
    // Length equals rest → exactly zero strain.
    assert_parity(&ctx, &gpu, ClothTensileStrainQuery::new(1.3, 1.3));
}

#[test]
fn multi_element_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothTensileStrain::new(&ctx);
    // A mixed batch (valid, degenerate, tension, compression) exercises the
    // std430 array stride: every slot must decode at the right byte offset.
    let queries = [
        ClothTensileStrainQuery::new(1.0, 1.1),
        ClothTensileStrainQuery::new(1.0e-9, 7.0),
        ClothTensileStrainQuery::new(4.0, 2.0),
        ClothTensileStrainQuery::new(0.5, 0.5),
        ClothTensileStrainQuery::new(3.0, 9.0),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (strain, valid) = oracle(q.rest_length, q.length);
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        if valid == 1 {
            assert!(
                close(r.strain, strain),
                "batch strain mismatch: gpu={} cpu={strain} query={q:?}",
                r.strain
            );
        } else {
            assert_eq!(r.strain, 0.0, "degenerate strain must be cleared");
        }
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothTensileStrain::new(&ctx);
    assert!(
        gpu.evaluate(&ctx, &[]).is_empty(),
        "empty batch returns an empty vector with no dispatch"
    );
}

/// A small deterministic linear-congruential generator so the sweep needs no
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

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothTensileStrain::new(&ctx);
    let mut rng = Lcg::new(0x51_7A_2C_93);
    let mut queries = Vec::with_capacity(512);
    for _ in 0..512 {
        // Rest length stays comfortably above the floor so parity never sits on
        // the degeneracy knife edge; separation spans slack to stretch.
        let rest_length = rng.next_range(0.1, 5.0);
        let length = rng.next_range(0.0, 10.0);
        queries.push(ClothTensileStrainQuery::new(rest_length, length));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (strain, valid) = oracle(q.rest_length, q.length);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        if valid == 1 {
            assert!(
                close(r.strain, strain),
                "sweep strain mismatch: gpu={} cpu={strain} query={q:?}",
                r.strain
            );
        }
    }
}
