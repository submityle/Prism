//! Real-device **exact** parity for the isolated hair card-chain mip classifier
//! twin: [`GpuHairMipForFootprint`] must reproduce the `CPU` golden
//! [`mip_for_footprint`](prism_render_architecture::hair::card_bake::mip_for_footprint)
//! (re-exported as
//! [`reference_mip_for_footprint`](prism_hair_gpu::mip_for_footprint::reference_mip_for_footprint))
//! for a batch of `(max_dim, base_texels, max_mip)` queries, counting how many
//! times each footprint's longest side can be doubled before reaching the
//! full-resolution card side. The suite drives the full-resolution mip `0`
//! case, a complete power-of-two chain, the zero-footprint coarsest level, the
//! `max_mip` clamp, the `base_texels = 0` floor, a non-power-of-two footprint,
//! the saturating-double guard near `u32::MAX`, the empty no-op, and a large
//! multi-workgroup batch that crosses the 64-wide dispatch boundary with a
//! divergent per-thread trip count.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`
//! integer arithmetic, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The kernel is pure integer arithmetic with no float anywhere, so the device
//! reproduces the scalar reference identically — there is no rounding or fma to
//! diverge. Every output is therefore compared **exactly** ([`assert_eq!`] on
//! the integer mip level) rather than within a tolerance. This test file never
//! uses a float `==`/`!=` or `sin`/`cos`; every input is an explicit integer
//! literal or integer-derived value.
//!
//! Provenance: standard power-of-two mip-chain classification plus `wgpu`
//! compute dispatch; no third-party engine source or derived code.

use prism_hair_gpu::mip_for_footprint::{reference_mip_for_footprint, MipQuery};
use prism_hair_gpu::GpuContext;
use prism_hair_gpu::GpuHairMipForFootprint;

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
fn run(ctx: &GpuContext, queries: &[MipQuery]) -> Vec<u32> {
    GpuHairMipForFootprint::new(ctx).eval(ctx, queries)
}

/// Asserts a whole batch matches the `CPU` golden exactly.
fn assert_batch_exact(got: &[u32], queries: &[MipQuery]) {
    assert_eq!(
        got.len(),
        queries.len(),
        "one level per query (got {}, want {})",
        got.len(),
        queries.len()
    );
    for (i, (&g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let want = reference_mip_for_footprint(*q);
        assert_eq!(
            g, want,
            "query {i} (max_dim {}, base_texels {}, max_mip {}): device level {g} must equal golden level {want}",
            q.max_dim, q.base_texels, q.max_mip
        );
    }
}

#[test]
fn full_resolution_is_mip_zero() {
    let Some(ctx) = context_or_skip("full_resolution_is_mip_zero") else {
        return;
    };
    // A footprint at (or above half) the full card side cannot be doubled
    // without exceeding base, so it stays at mip 0.
    let queries = [
        MipQuery {
            max_dim: 256,
            base_texels: 256,
            max_mip: 8,
        },
        MipQuery {
            max_dim: 200,
            base_texels: 256,
            max_mip: 8,
        },
        MipQuery {
            max_dim: 129,
            base_texels: 256,
            max_mip: 8,
        },
    ];
    let got = run(&ctx, &queries);
    assert_batch_exact(&got, &queries);
    for (i, &v) in got.iter().enumerate() {
        assert_eq!(v, 0, "query {i}: footprint above half base must be mip 0");
    }
}

#[test]
fn power_of_two_chain_matches_golden() {
    let Some(ctx) = context_or_skip("power_of_two_chain_matches_golden") else {
        return;
    };
    // 256 base: 256->0, 128->1, 64->2, 32->3, 16->4, 8->5, 4->6, 2->7, 1->8.
    let base = 256u32;
    let queries: Vec<MipQuery> = (0u32..=8)
        .map(|k| MipQuery {
            max_dim: base >> k,
            base_texels: base,
            max_mip: 8,
        })
        .collect();
    let got = run(&ctx, &queries);
    assert_batch_exact(&got, &queries);
    for (k, &v) in got.iter().enumerate() {
        assert_eq!(v, k as u32, "256 >> {k} must sit at mip {k}");
    }
}

#[test]
fn zero_footprint_is_coarsest_level() {
    let Some(ctx) = context_or_skip("zero_footprint_is_coarsest_level") else {
        return;
    };
    // A zero footprint reports max_mip (the coarsest level) directly.
    let queries = [
        MipQuery {
            max_dim: 0,
            base_texels: 256,
            max_mip: 8,
        },
        MipQuery {
            max_dim: 0,
            base_texels: 1024,
            max_mip: 3,
        },
        MipQuery {
            max_dim: 0,
            base_texels: 0,
            max_mip: 0,
        },
    ];
    let got = run(&ctx, &queries);
    assert_batch_exact(&got, &queries);
    assert_eq!(got[0], 8, "zero footprint reports max_mip 8");
    assert_eq!(got[1], 3, "zero footprint reports max_mip 3");
    assert_eq!(got[2], 0, "zero footprint with max_mip 0 reports 0");
}

#[test]
fn clamps_to_max_mip() {
    let Some(ctx) = context_or_skip("clamps_to_max_mip") else {
        return;
    };
    // A tiny footprint in a tall chain would reach a deep mip, but max_mip caps
    // it: 1 in a 256 base is naturally mip 8, clamped to 2 / 0.
    let queries = [
        MipQuery {
            max_dim: 1,
            base_texels: 256,
            max_mip: 2,
        },
        MipQuery {
            max_dim: 1,
            base_texels: 256,
            max_mip: 0,
        },
        MipQuery {
            max_dim: 4,
            base_texels: 256,
            max_mip: 3,
        },
    ];
    let got = run(&ctx, &queries);
    assert_batch_exact(&got, &queries);
    assert_eq!(got[0], 2, "clamped to max_mip 2");
    assert_eq!(got[1], 0, "clamped to max_mip 0");
    assert_eq!(got[2], 3, "4 in 256 is naturally mip 6, clamped to 3");
}

#[test]
fn base_texels_floored_to_one() {
    let Some(ctx) = context_or_skip("base_texels_floored_to_one") else {
        return;
    };
    // base_texels is floored to 1, so a footprint of 1 cannot double (1*2 > 1)
    // and stays mip 0; a larger footprint also stays mip 0.
    let queries = [
        MipQuery {
            max_dim: 1,
            base_texels: 0,
            max_mip: 8,
        },
        MipQuery {
            max_dim: 1,
            base_texels: 1,
            max_mip: 8,
        },
        MipQuery {
            max_dim: 7,
            base_texels: 0,
            max_mip: 8,
        },
    ];
    let got = run(&ctx, &queries);
    assert_batch_exact(&got, &queries);
    for (i, &v) in got.iter().enumerate() {
        assert_eq!(
            v, 0,
            "query {i}: base floored to 1 keeps footprint at mip 0"
        );
    }
}

#[test]
fn non_power_of_two_stops_at_fit() {
    let Some(ctx) = context_or_skip("non_power_of_two_stops_at_fit") else {
        return;
    };
    // max_dim 3, base 100: 3->6->12->24->48->96 (<=100), 96*2=192 > 100 stops,
    // so 5 doublings = mip 5. max_dim 10, base 100: 10->20->40->80 (<=100),
    // 80*2=160 > 100 stops = mip 3.
    let queries = [
        MipQuery {
            max_dim: 3,
            base_texels: 100,
            max_mip: 15,
        },
        MipQuery {
            max_dim: 10,
            base_texels: 100,
            max_mip: 15,
        },
    ];
    let got = run(&ctx, &queries);
    assert_batch_exact(&got, &queries);
    assert_eq!(got[0], 5, "3 doubles to 96 within 100 in 5 steps");
    assert_eq!(got[1], 3, "10 doubles to 80 within 100 in 3 steps");
}

#[test]
fn saturating_double_near_u32_max() {
    let Some(ctx) = context_or_skip("saturating_double_near_u32_max") else {
        return;
    };
    // A footprint past half of u32::MAX would overflow a naive double; the
    // saturating double caps it at u32::MAX so it never aliases back under base.
    // With base < u32::MAX these stay at mip 0 (the first double already
    // exceeds base), matching the golden's saturating_mul(2).
    let queries = [
        MipQuery {
            max_dim: 0x8000_0000,
            base_texels: 1000,
            max_mip: 31,
        },
        MipQuery {
            max_dim: u32::MAX,
            base_texels: u32::MAX,
            max_mip: 31,
        },
        MipQuery {
            max_dim: 0xC000_0000,
            base_texels: 0xF000_0000,
            max_mip: 31,
        },
    ];
    let got = run(&ctx, &queries);
    assert_batch_exact(&got, &queries);
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
    // per-thread trip count (each footprint reaches a different depth); every
    // 13th footprint is forced to zero so the coarsest-level path is crossed
    // inside a full dispatch. All inputs are integer-derived so the batch is
    // deterministic.
    let count = 130usize;
    let queries: Vec<MipQuery> = (0..count)
        .map(|i| {
            let max_dim = if i % 13 == 0 { 0 } else { (i as u32) + 1 };
            MipQuery {
                max_dim,
                base_texels: 1024,
                max_mip: 12,
            }
        })
        .collect();
    let got = run(&ctx, &queries);
    assert_batch_exact(&got, &queries);
}
