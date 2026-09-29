//! Ray-traced reflection role for hair grooms.
//!
//! Tracing individual strands in the ray-traced reflection pass is
//! prohibitively expensive: a groom is millions of sub-pixel curve segments,
//! and a reflection ray that hits hair pays the full traversal cost for a
//! contribution that is usually tiny and out of focus. Production engines
//! therefore rarely put real strands in reflections; they substitute a cheap
//! *proxy* (the LOD card/mesh shell) or *exclude* hair from reflections
//! entirely and let it appear only in the primary view (design §7 "RT 侧",
//! §8 降级矩阵 "RT 反射": 高配 proxy 参与 / 基线 · 兜底 排除).
//!
//! This module owns that policy decision purely and deterministically
//! (design §9): a groom's LOD tier plus its screen coverage plus a
//! [`RtProxyPolicy`] resolve to one [`RtReflectionRole`], with no traversal,
//! no allocation, and no hidden state. The actual BVH build and ray traversal
//! live in the RT backend (the non-portable bucket, design §9); this module
//! only tells that backend *which* representation of the groom to register.

use super::HairLodTier;

/// How a groom participates in the ray-traced reflection pass.
///
/// The three roles trade reflection fidelity against traversal cost, matching
/// the design's RT reflection row (高配 → [`RtReflectionRole::Proxy`],
/// 基线/兜底 → [`RtReflectionRole::Excluded`]). [`RtReflectionRole::FullStrands`]
/// is the opt-in cinematic case where real strand geometry is traced.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RtReflectionRole {
    /// Trace the groom's actual strand geometry in reflections. The most
    /// faithful and by far the most expensive option; reserved for hero
    /// close-ups on top-end hardware, and only ever chosen for strand-based
    /// tiers when the policy explicitly opts in.
    FullStrands,
    /// Register a cheap proxy (the LOD card/mesh shell) in the reflection BVH
    /// instead of strands. Hair still shows up in mirrors and glossy surfaces
    /// but at a fraction of the traversal cost.
    Proxy,
    /// Skip the groom in the reflection pass entirely; it appears only in the
    /// primary (raster) view. The cheapest option and the sensible default for
    /// small, distant, or off-screen grooms.
    Excluded,
}

impl RtReflectionRole {
    /// `true` when the groom is registered in the reflection BVH in any form
    /// (as strands or as a proxy), i.e. it is not [`RtReflectionRole::Excluded`].
    #[must_use]
    pub fn participates_in_rt(self) -> bool {
        !matches!(self, RtReflectionRole::Excluded)
    }

    /// `true` only when actual strand geometry is traced, i.e. the role is
    /// [`RtReflectionRole::FullStrands`]. Callers use this to decide whether to
    /// build a strand BVH (expensive) or reuse the proxy geometry.
    #[must_use]
    pub fn traces_strands(self) -> bool {
        matches!(self, RtReflectionRole::FullStrands)
    }
}

/// Policy knobs that map a groom's tier and coverage to an [`RtReflectionRole`].
///
/// This is the per-quality-level dial behind the design's RT reflection row: a
/// baseline profile sets a high `min_coverage_for_proxy` (or effectively
/// excludes hair by keeping coverage under it) and leaves `allow_strands_in_rt`
/// off, while a top-end profile lowers the coverage gate and may opt strands in
/// for hero shots.
#[derive(Clone, Copy, Debug)]
pub struct RtProxyPolicy {
    /// Minimum screen coverage (same units as [`super::lod`]) at or above which
    /// a groom is worth registering in the reflection BVH. Below this the groom
    /// is [`RtReflectionRole::Excluded`] as too small to matter in reflections.
    pub min_coverage_for_proxy: f32,
    /// When `true`, strand-based tiers are traced as real strands
    /// ([`RtReflectionRole::FullStrands`]) instead of being downgraded to a
    /// proxy. Off by default: strands in RT are a hero-shot luxury.
    pub allow_strands_in_rt: bool,
}

impl RtProxyPolicy {
    /// The conservative default used by the baseline profile: only reasonably
    /// large grooms reflect, always through a proxy, never as real strands.
    pub const CONSERVATIVE: Self = Self {
        min_coverage_for_proxy: 0.05,
        allow_strands_in_rt: false,
    };
}

/// Resolves how a groom participates in ray-traced reflections.
///
/// The decision is a two-step gate:
///
/// 1. **Visibility gate** — a groom whose `coverage` is below
///    `policy.min_coverage_for_proxy` (or non-finite, e.g. off-screen) is
///    [`RtReflectionRole::Excluded`]; it is too small to justify any reflection
///    cost.
/// 2. **Representation gate** — a groom that passes the visibility gate is
///    registered as a proxy by default. A strand-based `tier`
///    ([`HairLodTier::is_strand_based`]) is only ever traced as
///    [`RtReflectionRole::FullStrands`] when `policy.allow_strands_in_rt` is
///    set; otherwise, and for the already-cheap `Cards`/`Mesh` tiers, the role
///    is [`RtReflectionRole::Proxy`].
///
/// The mapping is total and deterministic: the same inputs always yield the
/// same role, and no input panics (non-finite coverage degrades to `Excluded`).
#[must_use]
pub fn resolve_rt_role(
    tier: HairLodTier,
    coverage: f32,
    policy: RtProxyPolicy,
) -> RtReflectionRole {
    // Visibility gate: too small / off-screen never enters the reflection BVH.
    // Small (or negative / -inf) coverage falls below the gate, and a NaN is
    // treated as invisible; both are excluded. Written without a negated
    // partial-ord compare (design bans `!(a >= b)` style) so clippy stays quiet.
    if coverage < policy.min_coverage_for_proxy || coverage.is_nan() {
        return RtReflectionRole::Excluded;
    }

    // Representation gate: strands only when explicitly opted in.
    if tier.is_strand_based() && policy.allow_strands_in_rt {
        RtReflectionRole::FullStrands
    } else {
        RtReflectionRole::Proxy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_coverage_is_excluded_regardless_of_tier() {
        let policy = RtProxyPolicy::CONSERVATIVE;
        for tier in [
            HairLodTier::Strands,
            HairLodTier::ReducedStrands,
            HairLodTier::Cards,
            HairLodTier::Mesh,
        ] {
            let role = resolve_rt_role(tier, 0.01, policy);
            assert_eq!(role, RtReflectionRole::Excluded);
            assert!(!role.participates_in_rt());
        }
    }

    #[test]
    fn strands_downgrade_to_proxy_by_default() {
        let policy = RtProxyPolicy::CONSERVATIVE;
        let role = resolve_rt_role(HairLodTier::Strands, 0.5, policy);
        assert_eq!(role, RtReflectionRole::Proxy);
        assert!(role.participates_in_rt());
        assert!(!role.traces_strands());
    }

    #[test]
    fn strands_traced_when_opted_in() {
        let policy = RtProxyPolicy {
            min_coverage_for_proxy: 0.05,
            allow_strands_in_rt: true,
        };
        let role = resolve_rt_role(HairLodTier::ReducedStrands, 0.5, policy);
        assert_eq!(role, RtReflectionRole::FullStrands);
        assert!(role.traces_strands());
    }

    #[test]
    fn card_and_mesh_tiers_stay_proxy_even_when_strands_allowed() {
        let policy = RtProxyPolicy {
            min_coverage_for_proxy: 0.05,
            allow_strands_in_rt: true,
        };
        // Cards/Mesh are not strand-based, so opting strands in does not
        // promote them; they are already their own cheap proxy.
        assert_eq!(
            resolve_rt_role(HairLodTier::Cards, 0.5, policy),
            RtReflectionRole::Proxy
        );
        assert_eq!(
            resolve_rt_role(HairLodTier::Mesh, 0.5, policy),
            RtReflectionRole::Proxy
        );
    }

    #[test]
    fn coverage_exactly_at_gate_participates() {
        let policy = RtProxyPolicy::CONSERVATIVE;
        // Coverage == min_coverage_for_proxy is inclusive (>=), so it reflects.
        let role = resolve_rt_role(HairLodTier::Cards, policy.min_coverage_for_proxy, policy);
        assert_eq!(role, RtReflectionRole::Proxy);
    }

    #[test]
    fn nan_and_negative_infinity_coverage_are_excluded() {
        let policy = RtProxyPolicy::CONSERVATIVE;
        // NaN (invisible/unknown) and -inf (far below the gate) never reflect.
        assert_eq!(
            resolve_rt_role(HairLodTier::Strands, f32::NAN, policy),
            RtReflectionRole::Excluded
        );
        assert_eq!(
            resolve_rt_role(HairLodTier::Strands, f32::NEG_INFINITY, policy),
            RtReflectionRole::Excluded
        );
    }

    #[test]
    fn negative_coverage_is_excluded() {
        let policy = RtProxyPolicy::CONSERVATIVE;
        assert_eq!(
            resolve_rt_role(HairLodTier::Strands, -1.0, policy),
            RtReflectionRole::Excluded
        );
    }
}
