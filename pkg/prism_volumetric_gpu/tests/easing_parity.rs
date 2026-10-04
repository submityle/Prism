//! Real-device parity for the easing twin:
//! [`GpuEasing`](prism_volumetric_gpu::easing::GpuEasing) must reproduce the
//! `CPU` golden `prism_math::curve::easing` family of 14 scalar easing curves.
//! Each query carries a function id (`0..=13`) and the parameter `t`; any id
//! `> 13` is rejected as invalid.
//!
//! The oracle here is an independent re-implementation of all 14 closed forms,
//! written out directly so the test never imports `prism_math`,
//! `prism_render_architecture` or `prism_physics_core`. The transcendental
//! curves use Rust `std` `f32::sin`/`cos`/`powf`, which differ slightly from
//! the golden `libm` backend and the device built-ins; the tolerance absorbs
//! that gap.
//!
//! The fixtures cover hand-verified spot values, an out-of-range id (invalid), a
//! mixed batch of two or more elements that validates the `std430` stride, an
//! empty batch the host short-circuits, and a 512-step sweep over all 14 ids
//! with `t` kept away from the piecewise knees so the branch decision agrees on
//! both sides.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! The eased scalar is compared with `abs <= 1e-4 || rel <= 1e-3`
//! (`REL_FLOOR = 1e-6`); the discrete `valid` flag is compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_math::curve::easing`；无第三方引擎源码或衍生代码。

use core::f32::consts::{FRAC_PI_2, PI};

use prism_volumetric_gpu::easing::{EasingQuery, EasingResult, GpuEasing};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: reproduces every golden easing curve in operator
/// order, returning the eased value and the validity flag. An id `> 13` is
/// invalid, yielding `(0.0, 0)`.
fn oracle(q: &EasingQuery) -> (f32, u32) {
    let t = q.t;
    let eased = match q.func_id {
        0 => {
            let tc = t.clamp(0.0, 1.0);
            tc * tc * (3.0 - 2.0 * tc)
        }
        1 => {
            let tc = t.clamp(0.0, 1.0);
            tc * tc * tc * (tc * (tc * 6.0 - 15.0) + 10.0)
        }
        2 => t * t,
        3 => t * (2.0 - t),
        4 => {
            if t < 0.5 {
                2.0 * t * t
            } else {
                let u = -2.0 * t + 2.0;
                1.0 - u * u * 0.5
            }
        }
        5 => t * t * t,
        6 => {
            let u = 1.0 - t;
            1.0 - u * u * u
        }
        7 => {
            if t < 0.5 {
                4.0 * t * t * t
            } else {
                let u = -2.0 * t + 2.0;
                1.0 - u * u * u * 0.5
            }
        }
        8 => 1.0 - (t * FRAC_PI_2).cos(),
        9 => (t * FRAC_PI_2).sin(),
        10 => -0.5 * ((PI * t).cos() - 1.0),
        11 => {
            if t <= 0.0 {
                0.0
            } else {
                2.0f32.powf(10.0 * (t - 1.0))
            }
        }
        12 => {
            if t >= 1.0 {
                1.0
            } else {
                1.0 - 2.0f32.powf(-10.0 * t)
            }
        }
        13 => {
            if t <= 0.0 {
                0.0
            } else if t >= 1.0 {
                1.0
            } else if t < 0.5 {
                0.5 * 2.0f32.powf(20.0 * t - 10.0)
            } else {
                1.0 - 0.5 * 2.0f32.powf(-20.0 * t + 10.0)
            }
        }
        _ => return (0.0, 0),
    };
    (eased, 1)
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
/// `valid` flag exactly, and the `eased` scalar to tolerance when valid.
fn assert_parity(gpu: &EasingResult, q: &EasingQuery, label: &str) {
    let (eased, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid == 1 {
        assert!(
            close(gpu.eased, eased),
            "{label}: eased mismatch gpu={} oracle={}",
            gpu.eased,
            eased
        );
    }
}

#[test]
fn spot_values_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEasing::new(&ctx);
    let queries = vec![
        EasingQuery::new(0, 0.5), // smoothstep(0.5) = 0.5
        EasingQuery::new(2, 0.5), // quad_in(0.5) = 0.25
        EasingQuery::new(5, 0.5), // cubic_in(0.5) = 0.125
        EasingQuery::new(9, 1.0), // sine_out(1.0) = 1
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert!(close(out[0].eased, 0.5));
    assert!(close(out[1].eased, 0.25));
    assert!(close(out[2].eased, 0.125));
    assert!(close(out[3].eased, 1.0));
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("spot[{i}]"));
    }
}

#[test]
fn out_of_range_id_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEasing::new(&ctx);
    let a = EasingQuery::new(14, 0.5);
    let b = EasingQuery::new(99, 0.3);
    let out = gpu.evaluate(&ctx, &[a, b]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].valid, 0, "id 14 should be invalid");
    assert_eq!(out[1].valid, 0, "id 99 should be invalid");
    assert_eq!(out[0].eased, 0.0);
    assert_eq!(out[1].eased, 0.0);
    assert_parity(&out[0], &a, "oob14");
    assert_parity(&out[1], &b, "oob99");
}

#[test]
fn all_ids_at_a_safe_point_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEasing::new(&ctx);
    // 0.3 is away from every piecewise knee (0.5 for in-out, 0/1 for expo).
    let queries: Vec<EasingQuery> = (0u32..=13u32).map(|id| EasingQuery::new(id, 0.3)).collect();
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1, "id {i} should be valid");
        assert_parity(res, q, &format!("id[{i}]"));
    }
}

#[test]
fn batch_mixes_valid_and_invalid_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEasing::new(&ctx);
    let queries = vec![
        EasingQuery::new(1, 0.4),
        EasingQuery::new(50, 0.2),
        EasingQuery::new(8, 0.7),
        EasingQuery::new(13, 0.6),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[1].valid, 0);
    assert_eq!(out[2].valid, 1);
    assert_eq!(out[3].valid, 1);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEasing::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEasing::new(&ctx);
    let mut lcg = Lcg::new(0xEA51_9000);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let func_id = lcg.next_u32() % 14;
        // Keep t away from the piecewise knees (0.5 for the in-out curves and
        // 0.0/1.0 for the exponentials) so the branch decision cannot flip.
        let lower = lcg.next_u32() % 2 == 0;
        let t = if lower {
            lcg.next_range(0.05, 0.45)
        } else {
            lcg.next_range(0.55, 0.95)
        };
        queries.push(EasingQuery::new(func_id, t));
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
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
