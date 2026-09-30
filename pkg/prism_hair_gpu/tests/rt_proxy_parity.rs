//! Real-device parity for the ray-traced-reflection role twin:
//! [`GpuHairRtProxy`] must reproduce the `CPU` golden
//! [`resolve_rt_role`](prism_render_architecture::hair::rt_proxy::resolve_rt_role)
//! for a batch of grooms under a shared policy, including the small-coverage
//! exclusion (every tier), the strands-downgrade-to-proxy default, the
//! strands-traced-when-opted-in case, the cards/mesh-stay-proxy case, the
//! exactly-at-gate inclusion, negative coverage exclusion, and a mixed
//! large-batch case that exercises all three roles at once, plus the empty
//! no-op.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The decision is a two-gate integer/compare map with no transcendental and no
//! fused multiply-add, so the `CPU` and `GPU` produce the identical role for
//! every finite-coverage groom. Parity is asserted as exact per-groom role
//! equality — a single mismatch fails the test. The mixed batch also asserts
//! that all three roles actually occur, so a degenerate all-one-role kernel
//! could not pass. Non-finite (`NaN`) coverage is intentionally not exercised
//! on-device (a driver's fast-math mode may make `NaN` handling
//! nondeterministic); the reference's own `NaN` case is covered by its unit
//! tests.
//!
//! Provenance: standard LOD-driven RT reflection proxy/exclusion policy; no
//! Unreal Engine source or derived code.

use prism_hair_gpu::rt_proxy::{reference_rt_role, GpuHairRtProxy};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::rt_proxy::{RtProxyPolicy, RtReflectionRole};
use prism_render_architecture::hair::HairLodTier;

const ALL_TIERS: [HairLodTier; 4] = [
    HairLodTier::Strands,
    HairLodTier::ReducedStrands,
    HairLodTier::Cards,
    HairLodTier::Mesh,
];

/// A policy that opts strands into RT (top-end profile) at the conservative gate.
const PERMISSIVE: RtProxyPolicy = RtProxyPolicy {
    min_coverage_for_proxy: 0.05,
    allow_strands_in_rt: true,
};

/// Asserts every `GPU` role exactly equals its `CPU` golden for the batch.
fn assert_parity(policy: RtProxyPolicy, queries: &[(HairLodTier, f32)], gpu: &[RtReflectionRole]) {
    assert_eq!(gpu.len(), queries.len(), "one role per groom");
    for (i, (&(tier, coverage), &g)) in queries.iter().zip(gpu.iter()).enumerate() {
        let c = reference_rt_role(tier, coverage, policy);
        assert_eq!(
            g, c,
            "groom {i}: gpu {g:?}, cpu {c:?} (tier {tier:?}, coverage {coverage})"
        );
    }
}

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

#[test]
fn small_coverage_is_excluded_regardless_of_tier() {
    let Some(ctx) = context_or_skip("rt_proxy small-coverage parity") else {
        return;
    };
    let proxy = GpuHairRtProxy::new(&ctx);
    let policy = RtProxyPolicy::CONSERVATIVE;
    let queries: Vec<(HairLodTier, f32)> = ALL_TIERS.iter().map(|&t| (t, 0.01)).collect();
    let roles = proxy.eval(&ctx, policy, &queries);
    assert_parity(policy, &queries, &roles);
    assert!(
        roles.iter().all(|&r| r == RtReflectionRole::Excluded),
        "coverage below the gate excludes every tier"
    );
}

#[test]
fn strands_downgrade_to_proxy_by_default() {
    let Some(ctx) = context_or_skip("rt_proxy default-downgrade parity") else {
        return;
    };
    let proxy = GpuHairRtProxy::new(&ctx);
    let policy = RtProxyPolicy::CONSERVATIVE;
    let queries = [
        (HairLodTier::Strands, 0.5),
        (HairLodTier::ReducedStrands, 0.5),
    ];
    let roles = proxy.eval(&ctx, policy, &queries);
    assert_parity(policy, &queries, &roles);
    assert!(
        roles.iter().all(|&r| r == RtReflectionRole::Proxy),
        "strands downgrade to proxy when the policy does not opt them in"
    );
}

#[test]
fn strands_traced_when_opted_in() {
    let Some(ctx) = context_or_skip("rt_proxy opt-in parity") else {
        return;
    };
    let proxy = GpuHairRtProxy::new(&ctx);
    let queries = [
        (HairLodTier::Strands, 0.5),
        (HairLodTier::ReducedStrands, 0.5),
    ];
    let roles = proxy.eval(&ctx, PERMISSIVE, &queries);
    assert_parity(PERMISSIVE, &queries, &roles);
    assert!(
        roles.iter().all(|&r| r == RtReflectionRole::FullStrands),
        "strand-based tiers trace real strands when the policy opts them in"
    );
}

#[test]
fn card_and_mesh_tiers_stay_proxy_even_when_strands_allowed() {
    let Some(ctx) = context_or_skip("rt_proxy card/mesh parity") else {
        return;
    };
    let proxy = GpuHairRtProxy::new(&ctx);
    let queries = [(HairLodTier::Cards, 0.5), (HairLodTier::Mesh, 0.5)];
    let roles = proxy.eval(&ctx, PERMISSIVE, &queries);
    assert_parity(PERMISSIVE, &queries, &roles);
    assert!(
        roles.iter().all(|&r| r == RtReflectionRole::Proxy),
        "cards/mesh are their own proxy and are never promoted to strands"
    );
}

#[test]
fn coverage_exactly_at_gate_participates() {
    let Some(ctx) = context_or_skip("rt_proxy at-gate parity") else {
        return;
    };
    let proxy = GpuHairRtProxy::new(&ctx);
    let policy = RtProxyPolicy::CONSERVATIVE;
    // Coverage == min_coverage_for_proxy is inclusive (the gate is a strict
    // less-than), so it reflects as a proxy.
    let queries = [(HairLodTier::Cards, policy.min_coverage_for_proxy)];
    let roles = proxy.eval(&ctx, policy, &queries);
    assert_parity(policy, &queries, &roles);
    assert_eq!(roles[0], RtReflectionRole::Proxy);
}

#[test]
fn negative_coverage_is_excluded() {
    let Some(ctx) = context_or_skip("rt_proxy negative-coverage parity") else {
        return;
    };
    let proxy = GpuHairRtProxy::new(&ctx);
    let policy = RtProxyPolicy::CONSERVATIVE;
    let queries = [(HairLodTier::Strands, -1.0), (HairLodTier::Cards, -0.25)];
    let roles = proxy.eval(&ctx, policy, &queries);
    assert_parity(policy, &queries, &roles);
    assert!(
        roles.iter().all(|&r| r == RtReflectionRole::Excluded),
        "negative coverage is far below the gate and never reflects"
    );
}

#[test]
fn mixed_batch_exercises_all_three_roles() {
    let Some(ctx) = context_or_skip("rt_proxy mixed-batch parity") else {
        return;
    };
    let proxy = GpuHairRtProxy::new(&ctx);
    // Permissive policy so opted-in strands can reach FullStrands while cards,
    // mesh and sub-gate grooms take Proxy / Excluded — all three roles occur.
    let queries = [
        (HairLodTier::Strands, 0.80),        // FullStrands
        (HairLodTier::ReducedStrands, 0.40), // FullStrands
        (HairLodTier::Cards, 0.60),          // Proxy
        (HairLodTier::Mesh, 0.90),           // Proxy
        (HairLodTier::Strands, 0.02),        // Excluded (below gate)
        (HairLodTier::Cards, 0.049),         // Excluded (just below gate)
        (HairLodTier::ReducedStrands, 0.05), // FullStrands (exactly at gate)
        (HairLodTier::Mesh, 0.10),           // Proxy
    ];
    let roles = proxy.eval(&ctx, PERMISSIVE, &queries);
    assert_parity(PERMISSIVE, &queries, &roles);

    assert!(
        roles.contains(&RtReflectionRole::FullStrands),
        "batch must contain a FullStrands role"
    );
    assert!(
        roles.contains(&RtReflectionRole::Proxy),
        "batch must contain a Proxy role"
    );
    assert!(
        roles.contains(&RtReflectionRole::Excluded),
        "batch must contain an Excluded role"
    );
}

#[test]
fn empty_batch_is_a_no_op() {
    let Some(ctx) = context_or_skip("rt_proxy empty parity") else {
        return;
    };
    let proxy = GpuHairRtProxy::new(&ctx);
    let roles = proxy.eval(&ctx, RtProxyPolicy::CONSERVATIVE, &[]);
    assert!(
        roles.is_empty(),
        "empty batch yields no roles and no dispatch"
    );
}
