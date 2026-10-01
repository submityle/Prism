//! Real-device **bit-exact** parity for the isolated hair forward-scatter power
//! twin: [`GpuHairForwardScatterPower`] must reproduce the `CPU` golden
//! [`forward_scatter_power`](prism_render_architecture::hair::dual_scatter_sh::forward_scatter_power)
//! (re-exported as
//! [`reference_forward_scatter_power`](prism_hair_gpu::forward_scatter_power::reference_forward_scatter_power))
//! for a batch of `(base, exponent)` queries, raising each forward-scatter
//! factor to its crossed-strand count. The suite drives the `exponent = 0`
//! identity, the `exponent = 1` sanitised base, a sweep of integer powers of a
//! decaying factor, a growing factor above one, the `base = 0` edge
//! (`0^0 = 1`, `0^{n>0} = 0`), a non-finite base sanitising to the golden,
//! the empty no-op, and a large multi-workgroup batch that crosses the 64-wide
//! dispatch boundary with a divergent per-thread trip count.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Like the dither twin this kernel is checked **bit-exact**: the accumulator is
//! a dependent chain of `f32` multiplies with no add to fuse into an fma, and
//! the loop trip count is read from a buffer so the compiler cannot fold it into
//! a closed-form `pow`, so the device must reproduce Rust's rounding to the bit.
//! Every output is therefore compared by its raw bit pattern
//! ([`f32::to_bits`]) rather than an approximate difference. This test file
//! never uses a float `==`/`!=` or `sin`/`cos`; all comparisons go through the
//! bit pattern and all inputs are explicit literals or integer-derived values.
//!
//! Provenance: standard `Zinke` dual-scattering global multiplier plus `wgpu`
//! compute dispatch; no third-party engine source or derived code.

use prism_hair_gpu::forward_scatter_power::{reference_forward_scatter_power, PowerQuery};
use prism_hair_gpu::GpuContext;
use prism_hair_gpu::GpuHairForwardScatterPower;

/// Acquires a headless context, or `None` (with a skip notice) when the host has
/// no `wgpu` adapter so the suite stays green off-device.
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn context_or_skip(label: &str) -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping {label}: no wgpu adapter on this host");
            None
        }
    }
}

/// Dispatches one batch through the device twin.
fn run(ctx: &GpuContext, queries: &[PowerQuery]) -> Vec<f32> {
    GpuHairForwardScatterPower::new(ctx).eval(ctx, queries)
}

/// Asserts a whole batch matches the `CPU` golden **bit-for-bit**.
fn assert_batch_exact(got: &[f32], queries: &[PowerQuery]) {
    assert_eq!(
        got.len(),
        queries.len(),
        "one value per query (got {}, want {})",
        got.len(),
        queries.len()
    );
    for (i, (&g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let want = reference_forward_scatter_power(*q);
        let gb = g.to_bits();
        let wb = want.to_bits();
        assert_eq!(
            gb, wb,
            "query {i} (base bits {:#010x}, exponent {}): device bits {gb:#010x} must equal golden bits {wb:#010x}",
            q.base.to_bits(),
            q.exponent
        );
    }
}

#[test]
fn exponent_zero_is_one() {
    let Some(ctx) = context_or_skip("exponent_zero_is_one") else {
        return;
    };
    // n = 0 returns exactly 1.0 for every base, including non-finite ones.
    let queries = [
        PowerQuery {
            base: 0.0,
            exponent: 0,
        },
        PowerQuery {
            base: 0.5,
            exponent: 0,
        },
        PowerQuery {
            base: 2.0,
            exponent: 0,
        },
        PowerQuery {
            base: f32::NAN,
            exponent: 0,
        },
        PowerQuery {
            base: f32::INFINITY,
            exponent: 0,
        },
    ];
    let got = run(&ctx, &queries);
    assert_batch_exact(&got, &queries);
    let one_bits = 1.0f32.to_bits();
    for (i, &v) in got.iter().enumerate() {
        assert_eq!(
            v.to_bits(),
            one_bits,
            "query {i}: exponent 0 must be exactly 1.0"
        );
    }
}

#[test]
fn exponent_one_is_sanitised_base() {
    let Some(ctx) = context_or_skip("exponent_one_is_sanitised_base") else {
        return;
    };
    // n = 1 returns the sanitised base (1 * base), bit-for-bit.
    let queries = [
        PowerQuery {
            base: 0.0,
            exponent: 1,
        },
        PowerQuery {
            base: 0.25,
            exponent: 1,
        },
        PowerQuery {
            base: 0.875,
            exponent: 1,
        },
        PowerQuery {
            base: 3.5,
            exponent: 1,
        },
    ];
    let got = run(&ctx, &queries);
    assert_batch_exact(&got, &queries);
}

#[test]
fn integer_powers_match_golden() {
    let Some(ctx) = context_or_skip("integer_powers_match_golden") else {
        return;
    };
    // A decaying forward factor raised to a sweep of crossing counts: the
    // device must reproduce the golden's dependent-multiply rounding exactly.
    let base = 0.8_f32;
    let queries: Vec<PowerQuery> = (0u32..16)
        .map(|exponent| PowerQuery { base, exponent })
        .collect();
    let got = run(&ctx, &queries);
    assert_batch_exact(&got, &queries);
}

#[test]
fn base_above_one_grows() {
    let Some(ctx) = context_or_skip("base_above_one_grows") else {
        return;
    };
    // A factor above one is physically out of range for transmittance but is a
    // legal input; the twin must still match the golden product chain.
    let base = 1.125_f32;
    let queries: Vec<PowerQuery> = (0u32..12)
        .map(|exponent| PowerQuery { base, exponent })
        .collect();
    let got = run(&ctx, &queries);
    assert_batch_exact(&got, &queries);
}

#[test]
fn base_zero_powers() {
    let Some(ctx) = context_or_skip("base_zero_powers") else {
        return;
    };
    // 0^0 = 1 (empty product), 0^{n>0} = 0.
    let queries = [
        PowerQuery {
            base: 0.0,
            exponent: 0,
        },
        PowerQuery {
            base: 0.0,
            exponent: 1,
        },
        PowerQuery {
            base: 0.0,
            exponent: 2,
        },
        PowerQuery {
            base: 0.0,
            exponent: 7,
        },
    ];
    let got = run(&ctx, &queries);
    assert_batch_exact(&got, &queries);
    let one_bits = 1.0f32.to_bits();
    let zero_bits = 0.0f32.to_bits();
    assert_eq!(got[0].to_bits(), one_bits, "0^0 must be exactly 1.0");
    for (i, &v) in got.iter().enumerate().skip(1) {
        assert_eq!(
            v.to_bits(),
            zero_bits,
            "query {i}: 0^(n>0) must be exactly 0.0"
        );
    }
}

#[test]
fn non_finite_base_sanitises_to_zero() {
    let Some(ctx) = context_or_skip("non_finite_base_sanitises_to_zero") else {
        return;
    };
    // A non-finite base sanitises to 0 before the power, so n>0 collapses to 0
    // and n==0 stays the empty-product 1 — matching the golden bit-for-bit.
    let queries = [
        PowerQuery {
            base: f32::NAN,
            exponent: 3,
        },
        PowerQuery {
            base: f32::INFINITY,
            exponent: 4,
        },
        PowerQuery {
            base: f32::NEG_INFINITY,
            exponent: 2,
        },
        PowerQuery {
            base: f32::NAN,
            exponent: 0,
        },
    ];
    let got = run(&ctx, &queries);
    assert_batch_exact(&got, &queries);
    let zero_bits = 0.0f32.to_bits();
    let one_bits = 1.0f32.to_bits();
    assert_eq!(got[0].to_bits(), zero_bits, "NaN^3 sanitises to 0^3 = 0");
    assert_eq!(got[1].to_bits(), zero_bits, "inf^4 sanitises to 0^4 = 0");
    assert_eq!(got[2].to_bits(), zero_bits, "-inf^2 sanitises to 0^2 = 0");
    assert_eq!(got[3].to_bits(), one_bits, "NaN^0 sanitises to 0^0 = 1");
}

#[test]
fn empty_batch_is_noop() {
    let Some(ctx) = context_or_skip("empty_batch_is_noop") else {
        return;
    };
    let got = run(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty vector");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 130 queries span three 64-wide workgroups and exercise a divergent
    // per-thread trip count; every 13th base is forced non-finite so the
    // sanitiser path is crossed inside a full dispatch. All inputs are
    // integer-derived so the batch is deterministic.
    let count = 130usize;
    let queries: Vec<PowerQuery> = (0..count)
        .map(|i| {
            let base = if i % 13 == 0 {
                f32::NAN
            } else {
                // A decaying factor in (0, 1] derived from the index.
                (i as f32 + 1.0) / (count as f32 + 1.0)
            };
            let exponent = (i % 9) as u32;
            PowerQuery { base, exponent }
        })
        .collect();
    let got = run(&ctx, &queries);
    assert_batch_exact(&got, &queries);
}
