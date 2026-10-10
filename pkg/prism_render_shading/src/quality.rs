//! Shading quality tiers and the per-frame **expensive-lobe sample budget**.
//!
//! Several physically based lobes integrate a profile or march a ray at
//! shade time: separable subsurface scattering convolves a screen-space blur
//! over a tap kernel, Penner pre-integrated skin runs a curvature quadrature,
//! the Burley translucency profile integrates a radial energy term, and the
//! sky multiple-scattering LUT bakes a Fibonacci-sphere/ray-march estimate.
//! Each of those is parameterized by a *sample count*; this module turns a
//! single [`ShadingQuality`] knob into a coherent [`LobeSampleBudget`] of those
//! counts, so lowering quality scales every expensive lobe together instead of
//! per-lobe magic numbers scattered across the renderer.
//!
//! # Why a budget and not a permutation key (design, §7.2 / §13 P1-e)
//! The quality tier is deliberately **not** part of the pipeline specialization
//! key. It never selects a different shader permutation, never collapses the
//! lobe set, and never branches per pixel: it only scales *how many samples*
//! each already-compiled lobe takes. That keeps the permutation count bounded
//! (one über per material, not one-per-tier) and makes quality a smooth,
//! host-side knob — the counts are uploaded as an ordinary pass uniform and may
//! change every frame without recompilation or visible popping.
//!
//! # Invariants (checked by tests)
//! * Every field is **monotone non-decreasing** from [`ShadingQuality::Low`] to
//!   [`ShadingQuality::Ultra`]: raising quality never lowers a sample count.
//! * The SSS separable tap count is always **odd and `>= 3`**, matching
//!   [`crate::gi::subsurface::separable::SeparableKernel`]'s symmetric-kernel contract.
//! * [`ShadingQuality::High`] reproduces the renderer's historical built-in
//!   counts, so the default path is bit-for-bit unchanged.
//! * [`LobeSampleBudget::with_global_scale`] is a *continuous* perf knob: a
//!   scale of `1.0` is the identity, scales in `(0, 1)` only ever lower counts,
//!   and model-preserving floors (odd `>= 3` taps, etc.) are never breached.
//!
//! Every function here is a deterministic pure function — no RNG, no I/O, no
//! GPU, no `unsafe`.

/// Discrete shading-quality tier driving the expensive-lobe sample budget.
///
/// Ordered ascending by cost; `Ord` compares the intrinsic quality level so
/// `Low < Medium < High < Ultra`. [`Default`] is [`ShadingQuality::High`], the
/// tier whose budget matches the renderer's historical hard-coded counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ShadingQuality {
    /// Cheapest tier: minimal taps/marches, intended for low-end or far LOD.
    Low = 0,
    /// Balanced tier for mid-range hardware.
    Medium = 1,
    /// Default tier; reproduces the renderer's historical built-in counts.
    High = 2,
    /// Reference tier: maximum sample counts for hero assets and captures.
    Ultra = 3,
}

impl Default for ShadingQuality {
    #[inline]
    fn default() -> Self {
        ShadingQuality::High
    }
}

impl ShadingQuality {
    /// Every tier in ascending cost order.
    pub const ALL: [ShadingQuality; 4] = [
        ShadingQuality::Low,
        ShadingQuality::Medium,
        ShadingQuality::High,
        ShadingQuality::Ultra,
    ];

    /// Dense tier index in `0..4` (ascending), suitable for array lookups and
    /// GPU uniform encoding.
    #[inline]
    pub const fn index(self) -> u32 {
        self as u32
    }

    /// Inverse of [`index`](Self::index); out-of-range indices clamp to the
    /// nearest valid tier so a corrupt uniform can never panic.
    #[inline]
    pub const fn from_index(index: u32) -> Self {
        match index {
            0 => ShadingQuality::Low,
            1 => ShadingQuality::Medium,
            2 => ShadingQuality::High,
            _ => ShadingQuality::Ultra,
        }
    }

    /// The expensive-lobe sample budget for this tier.
    #[inline]
    pub const fn budget(self) -> LobeSampleBudget {
        LobeSampleBudget::for_quality(self)
    }
}

/// Per-frame sample counts for the expensive shade-time lobes.
///
/// Uploaded to the GPU as a plain pass uniform (see `shaders/lobe_budget.wesl`
/// for the byte-compatible twin) and consumed on the CPU golden path through
/// the `*_for_budget` adapters on [`crate::gi::subsurface::separable::SeparableKernel`],
/// [`crate::preintegrated_diffuse_for_budget`],
/// [`crate::diffusion_profile_radial_energy_for_budget`], and
/// [`crate::bake_multiscatter_lut_for_budget`].
///
/// All counts are `u32` so the layout is trivially shareable with WESL. The
/// field order is the GPU struct order and must not be reordered without
/// updating the twin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LobeSampleBudget {
    /// Separable SSS screen-space blur taps (odd, `>= 3`).
    pub sss_separable_taps: u32,
    /// Penner pre-integrated skin curvature-quadrature steps.
    pub sss_preintegration_steps: u32,
    /// Burley translucency radial-energy integration samples.
    pub translucency_profile_samples: u32,
    /// Sky multiple-scattering Fibonacci-sphere direction samples.
    pub multiscatter_dir_samples: u32,
    /// Sky multiple-scattering per-direction ray-march steps.
    pub multiscatter_march_samples: u32,
}

impl LobeSampleBudget {
    /// Lowest tap count the symmetric SSS kernel accepts (odd, `>= 3`).
    pub const MIN_SSS_TAPS: u32 = 3;
    /// Lowest pre-integration quadrature step count that keeps the curvature
    /// integral stable.
    pub const MIN_PREINTEGRATION_STEPS: u32 = 16;
    /// Lowest translucency radial-energy sample count.
    pub const MIN_TRANSLUCENCY_SAMPLES: u32 = 8;
    /// Lowest multiple-scattering direction-sample count.
    pub const MIN_MULTISCATTER_DIR: u32 = 4;
    /// Lowest multiple-scattering march-step count.
    pub const MIN_MULTISCATTER_MARCH: u32 = 8;

    /// The canonical budget for a discrete [`ShadingQuality`] tier.
    ///
    /// Values are hand-tuned to be monotone non-decreasing across tiers; the
    /// SSS tap counts are odd by construction. [`ShadingQuality::High`] matches
    /// the renderer's historical built-ins (`160` pre-integration steps in
    /// particular), so the default path is numerically unchanged.
    #[inline]
    pub const fn for_quality(quality: ShadingQuality) -> Self {
        match quality {
            ShadingQuality::Low => Self {
                sss_separable_taps: 5,
                sss_preintegration_steps: 48,
                translucency_profile_samples: 16,
                multiscatter_dir_samples: 8,
                multiscatter_march_samples: 16,
            },
            ShadingQuality::Medium => Self {
                sss_separable_taps: 9,
                sss_preintegration_steps: 96,
                translucency_profile_samples: 32,
                multiscatter_dir_samples: 16,
                multiscatter_march_samples: 24,
            },
            ShadingQuality::High => Self {
                sss_separable_taps: 13,
                sss_preintegration_steps: 160,
                translucency_profile_samples: 48,
                multiscatter_dir_samples: 32,
                multiscatter_march_samples: 32,
            },
            ShadingQuality::Ultra => Self {
                sss_separable_taps: 17,
                sss_preintegration_steps: 256,
                translucency_profile_samples: 64,
                multiscatter_dir_samples: 64,
                multiscatter_march_samples: 48,
            },
        }
    }

    /// Force `n` odd and `>= 3`, matching [`crate::gi::subsurface::separable::SeparableKernel`]'s kernel
    /// contract so a budget can never request an even or sub-minimal tap count.
    #[inline]
    pub const fn force_odd_at_least_three(n: u32) -> u32 {
        let n = if n < Self::MIN_SSS_TAPS {
            Self::MIN_SSS_TAPS
        } else {
            n
        };
        if n % 2 == 0 {
            n + 1
        } else {
            n
        }
    }

    /// Clamp every field to its model-preserving floor, forcing the SSS taps
    /// odd. Idempotent, and a no-op on any budget produced by
    /// [`for_quality`](Self::for_quality).
    #[inline]
    pub const fn clamped(self) -> Self {
        Self {
            sss_separable_taps: Self::force_odd_at_least_three(self.sss_separable_taps),
            sss_preintegration_steps: clamp_min(
                self.sss_preintegration_steps,
                Self::MIN_PREINTEGRATION_STEPS,
            ),
            translucency_profile_samples: clamp_min(
                self.translucency_profile_samples,
                Self::MIN_TRANSLUCENCY_SAMPLES,
            ),
            multiscatter_dir_samples: clamp_min(
                self.multiscatter_dir_samples,
                Self::MIN_MULTISCATTER_DIR,
            ),
            multiscatter_march_samples: clamp_min(
                self.multiscatter_march_samples,
                Self::MIN_MULTISCATTER_MARCH,
            ),
        }
    }

    /// Scale every count by a continuous factor and re-clamp to floors.
    ///
    /// This is the smooth, per-frame performance knob referenced by §7.2: a
    /// dynamic resolution/pacing controller can push `scale` below `1.0` to
    /// shed sample cost without any shader recompilation or lobe collapse, and
    /// `scale == 1.0` is the exact identity. `scale` is clamped to `(0, 1]`
    /// (values `> 1` are treated as `1.0`, so this only ever *lowers* cost
    /// relative to the tier) and non-finite inputs fall back to `1.0`.
    #[inline]
    pub fn with_global_scale(self, scale: f32) -> Self {
        let scale = if scale.is_finite() {
            scale.clamp(f32::MIN_POSITIVE, 1.0)
        } else {
            1.0
        };
        let apply = |n: u32| -> u32 {
            // Round to nearest; `scale <= 1` so the product never overflows a
            // count that already fits in `u32`.
            let scaled = (n as f32 * scale).round();
            if scaled.is_finite() && scaled >= 0.0 {
                scaled as u32
            } else {
                0
            }
        };
        Self {
            sss_separable_taps: apply(self.sss_separable_taps),
            sss_preintegration_steps: apply(self.sss_preintegration_steps),
            translucency_profile_samples: apply(self.translucency_profile_samples),
            multiscatter_dir_samples: apply(self.multiscatter_dir_samples),
            multiscatter_march_samples: apply(self.multiscatter_march_samples),
        }
        .clamped()
    }

    /// Total sample count across all expensive lobes — a cheap scalar proxy for
    /// the budget's relative cost, handy for telemetry and tests.
    #[inline]
    pub const fn total_samples(self) -> u32 {
        self.sss_separable_taps
            .saturating_add(self.sss_preintegration_steps)
            .saturating_add(self.translucency_profile_samples)
            .saturating_add(self.multiscatter_dir_samples)
            .saturating_add(self.multiscatter_march_samples)
    }
}

impl Default for LobeSampleBudget {
    #[inline]
    fn default() -> Self {
        ShadingQuality::High.budget()
    }
}

/// `const`-friendly `max(value, floor)`.
#[inline]
const fn clamp_min(value: u32, floor: u32) -> u32 {
    if value < floor {
        floor
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiers_round_trip_through_index() {
        for (i, q) in ShadingQuality::ALL.iter().copied().enumerate() {
            assert_eq!(q.index(), i as u32);
            assert_eq!(ShadingQuality::from_index(i as u32), q);
        }
        // Out-of-range clamps to Ultra rather than panicking.
        assert_eq!(ShadingQuality::from_index(99), ShadingQuality::Ultra);
    }

    #[test]
    fn tiers_order_ascending() {
        assert!(ShadingQuality::Low < ShadingQuality::Medium);
        assert!(ShadingQuality::Medium < ShadingQuality::High);
        assert!(ShadingQuality::High < ShadingQuality::Ultra);
        assert_eq!(ShadingQuality::default(), ShadingQuality::High);
    }

    #[test]
    fn budget_is_monotone_non_decreasing_in_every_field() {
        for pair in ShadingQuality::ALL.windows(2) {
            let lo = pair[0].budget();
            let hi = pair[1].budget();
            assert!(hi.sss_separable_taps >= lo.sss_separable_taps);
            assert!(hi.sss_preintegration_steps >= lo.sss_preintegration_steps);
            assert!(hi.translucency_profile_samples >= lo.translucency_profile_samples);
            assert!(hi.multiscatter_dir_samples >= lo.multiscatter_dir_samples);
            assert!(hi.multiscatter_march_samples >= lo.multiscatter_march_samples);
            assert!(hi.total_samples() >= lo.total_samples());
        }
    }

    #[test]
    fn sss_taps_are_always_odd_and_at_least_three() {
        for q in ShadingQuality::ALL {
            let taps = q.budget().sss_separable_taps;
            assert!(taps >= 3, "taps {taps} < 3 for {q:?}");
            assert_eq!(taps % 2, 1, "taps {taps} not odd for {q:?}");
        }
    }

    #[test]
    fn high_tier_matches_historical_builtins() {
        // High must reproduce the renderer's hard-coded defaults so the
        // out-of-the-box path is numerically unchanged.
        let b = ShadingQuality::High.budget();
        assert_eq!(b.sss_preintegration_steps, 160);
    }

    #[test]
    fn force_odd_at_least_three_is_correct() {
        assert_eq!(LobeSampleBudget::force_odd_at_least_three(0), 3);
        assert_eq!(LobeSampleBudget::force_odd_at_least_three(1), 3);
        assert_eq!(LobeSampleBudget::force_odd_at_least_three(2), 3);
        assert_eq!(LobeSampleBudget::force_odd_at_least_three(3), 3);
        assert_eq!(LobeSampleBudget::force_odd_at_least_three(4), 5);
        assert_eq!(LobeSampleBudget::force_odd_at_least_three(7), 7);
        assert_eq!(LobeSampleBudget::force_odd_at_least_three(8), 9);
    }

    #[test]
    fn clamped_is_idempotent_and_noop_on_tier_budgets() {
        for q in ShadingQuality::ALL {
            let b = q.budget();
            assert_eq!(b, b.clamped(), "tier budget {q:?} not already clamped");
            assert_eq!(b.clamped(), b.clamped().clamped());
        }
        // A deliberately-degenerate budget is lifted to the floors.
        let bad = LobeSampleBudget {
            sss_separable_taps: 0,
            sss_preintegration_steps: 0,
            translucency_profile_samples: 0,
            multiscatter_dir_samples: 0,
            multiscatter_march_samples: 0,
        }
        .clamped();
        assert_eq!(bad.sss_separable_taps, LobeSampleBudget::MIN_SSS_TAPS);
        assert_eq!(bad.sss_separable_taps % 2, 1);
        assert_eq!(
            bad.sss_preintegration_steps,
            LobeSampleBudget::MIN_PREINTEGRATION_STEPS
        );
        assert_eq!(
            bad.translucency_profile_samples,
            LobeSampleBudget::MIN_TRANSLUCENCY_SAMPLES
        );
        assert_eq!(
            bad.multiscatter_dir_samples,
            LobeSampleBudget::MIN_MULTISCATTER_DIR
        );
        assert_eq!(
            bad.multiscatter_march_samples,
            LobeSampleBudget::MIN_MULTISCATTER_MARCH
        );
    }

    #[test]
    fn global_scale_one_is_identity() {
        for q in ShadingQuality::ALL {
            let b = q.budget();
            assert_eq!(b.with_global_scale(1.0), b);
        }
    }

    #[test]
    fn global_scale_below_one_never_increases_and_respects_floors() {
        for q in ShadingQuality::ALL {
            let b = q.budget();
            for &s in &[0.75f32, 0.5, 0.25, 0.1, 0.01, f32::MIN_POSITIVE] {
                let scaled = b.with_global_scale(s);
                assert!(scaled.sss_separable_taps <= b.sss_separable_taps);
                assert!(scaled.sss_preintegration_steps <= b.sss_preintegration_steps);
                assert!(scaled.translucency_profile_samples <= b.translucency_profile_samples);
                assert!(scaled.multiscatter_dir_samples <= b.multiscatter_dir_samples);
                assert!(scaled.multiscatter_march_samples <= b.multiscatter_march_samples);
                // Floors and the odd-tap invariant survive any downscale.
                assert!(scaled.sss_separable_taps >= LobeSampleBudget::MIN_SSS_TAPS);
                assert_eq!(scaled.sss_separable_taps % 2, 1);
                assert!(
                    scaled.sss_preintegration_steps >= LobeSampleBudget::MIN_PREINTEGRATION_STEPS
                );
                assert!(
                    scaled.translucency_profile_samples
                        >= LobeSampleBudget::MIN_TRANSLUCENCY_SAMPLES
                );
                assert!(scaled.multiscatter_dir_samples >= LobeSampleBudget::MIN_MULTISCATTER_DIR);
                assert!(
                    scaled.multiscatter_march_samples >= LobeSampleBudget::MIN_MULTISCATTER_MARCH
                );
            }
        }
    }

    #[test]
    fn global_scale_above_one_and_nonfinite_clamp_to_identity() {
        let b = ShadingQuality::High.budget();
        assert_eq!(b.with_global_scale(2.0), b);
        assert_eq!(b.with_global_scale(f32::INFINITY), b);
        assert_eq!(b.with_global_scale(f32::NAN), b);
    }

    #[test]
    fn global_scale_half_lowers_cost() {
        let b = ShadingQuality::Ultra.budget();
        let half = b.with_global_scale(0.5);
        assert!(half.total_samples() < b.total_samples());
    }
}
