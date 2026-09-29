//! `PopcornFX`-style platform-profile coverage matrix and the runtime
//! degradation ladder — the complement to [`super::lod`] (design §28).
//!
//! This module is the **complementary layer** to [`super::lod`]: it does *not*
//! redefine the quality/platform primitives that live there. It reuses
//! [`super::lod::ParticleQuality`] (the fidelity rungs, their divisor and
//! degradation staircase) and [`super::lod::PlatformTier`] (the hardware
//! ceilings), and assumes the caller has already resolved a concrete quality
//! for the running platform through [`super::lod::resolve_quality`]. On top of
//! that foundation this module adds the two pieces §28 still needs:
//!
//! 1. **Platform × quality override matrix.** Where [`super::lod`] answers "how
//!    many particles does this quality keep", this matrix answers "what does a
//!    single emitter look like on *this* platform at *this* quality": the
//!    authored spawn rate is scaled, the pool capacity is capped, the update
//!    cadence is stretched (simulate every `N` frames), and the authored
//!    `ShadingModel` is downgraded (a volumetric `PBR` emitter collapses to a
//!    six-way lighting approximation on mid platforms, or to `Unlit` on the
//!    weakest ones). [`PlatformProfileMatrix`] stores the 4×4 grid of
//!    [`EmitterProfileOverride`] cells and [`apply_profile`] folds a cell onto
//!    an [`EmitterBaseline`] to produce the [`EffectiveEmitterParams`] the
//!    runtime actually spawns and draws with.
//! 2. **Runtime degradation ladder.** When a frame runs over its shared budget
//!    the runtime walks a *fixed, ordered* set of [`DegradationAction`]s —
//!    reduce spawn, disable sorting/`OIT`, lower volume resolution, simplify
//!    shading, reduce update rate, cull the renderer, then finally pause the
//!    simulation. [`degradation_level`] turns an over-budget ratio into how far
//!    down the ladder to walk, and [`order_degradation_candidates`] sorts the
//!    live emitters by the `(priority, screen coverage, distance)` key so the
//!    least-important emitters shed detail first.
//!
//! All math is `+ - * /` and comparisons only (no transcendental functions), no
//! bare `f32` equality is used, and every table is an explicit constant, so a
//! future `GPU`-side reproduction of the same profile decisions is bit-exact.

use core::cmp::Ordering;

use super::lod::{ParticleQuality, PlatformTier};
use super::{EmberShadingModel, EmitterHandle, ShadingBasis};

/// Number of [`ParticleQuality`] rungs the matrix indexes (Low..=Ultra).
pub const QUALITY_COUNT: usize = 4;

/// Number of [`PlatformTier`] rows the matrix indexes (Mobile..=`HighEnd`).
pub const PLATFORM_COUNT: usize = 4;

/// Coverage difference below which two screen fractions are treated as equal
/// when ordering degradation candidates.
const COVERAGE_EPS: f32 = 1.0e-4;

/// Distance difference below which two camera distances are treated as equal
/// when ordering degradation candidates.
const DISTANCE_EPS: f32 = 1.0e-3;

/// The most expensive shading a platform profile will let an emitter keep.
///
/// This is the `ShadingModel` half of a [`EmitterProfileOverride`]: it names how
/// far a `PopcornFX`-style profile is allowed to collapse an emitter's authored
/// response, not a shading model in its own right.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ShadingBudget {
    /// Keep the authored `ShadingModel`, including full volumetric `PBR`.
    Full,
    /// Collapse a [`EmberShadingModel::Hybrid`] to its base lobe and approximate
    /// volumetric `PBR` with cheap six-way directional lighting.
    SixWayApprox,
    /// Force every response to [`EmberShadingModel::Unlit`]; the cheapest path
    /// for the weakest platforms.
    Unlit,
}

/// The authored, platform-agnostic parameters of a single emitter.
///
/// This is the baseline the coverage matrix decimates from — the "full detail"
/// numbers the artist authored before any platform or quality is applied.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmitterBaseline {
    /// Emitter this baseline describes.
    pub handle: EmitterHandle,
    /// Authored spawn rate in particles per second at full detail.
    pub spawn_rate: f32,
    /// Authored maximum live particle capacity at full detail.
    pub capacity: u32,
    /// Authored simulation cadence: simulate once every `N` frames (`1` is every
    /// frame). Stored so a profile can only ever make it coarser, never finer.
    pub update_every_n_frames: u32,
    /// Authored `ShadingModel` for the emitter.
    pub shading: EmberShadingModel,
    /// Whether the emitter renders as a lit volume, so a six-way approximation
    /// is meaningful when the profile downgrades its shading.
    pub volumetric: bool,
}

/// A single cell of the platform × quality coverage matrix.
///
/// Applying a cell to an [`EmitterBaseline`] via [`apply_profile`] yields the
/// [`EffectiveEmitterParams`] used this frame. A cell only ever *reduces* an
/// emitter: the spawn scale is `<= 1`, the capacity is an absolute ceiling, and
/// the update cadence is a floor on `N`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmitterProfileOverride {
    /// Multiplier applied to the authored spawn rate (`0..=1` in the defaults).
    pub spawn_rate_scale: f32,
    /// Absolute ceiling on live particle capacity for this profile.
    pub capacity_cap: u32,
    /// Minimum simulation cadence: at least once every `N` frames.
    pub update_every_n_frames: u32,
    /// How far this profile may collapse the emitter's authored `ShadingModel`.
    pub shading_budget: ShadingBudget,
}

/// The concrete emitter parameters after a profile cell has been applied.
///
/// These are what the runtime actually spawns, pools, steps and shades with for
/// the current `(platform, quality)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EffectiveEmitterParams {
    /// Emitter these parameters describe.
    pub handle: EmitterHandle,
    /// Effective spawn rate in particles per second after scaling.
    pub spawn_rate: f32,
    /// Effective live particle capacity after capping.
    pub capacity: u32,
    /// Effective simulation cadence: simulate once every `N` frames.
    pub update_every_n_frames: u32,
    /// Effective `ShadingModel` after any downgrade.
    pub shading: EmberShadingModel,
    /// `true` when a volumetric `PBR` emitter is being approximated by six-way
    /// directional lighting rather than the full volumetric `PBR` path.
    pub six_way_volumetric_approx: bool,
}

/// Maps a [`ShadingBasis`] lobe back to the equivalent standalone
/// [`EmberShadingModel`], used when collapsing a hybrid to its base lobe.
#[must_use]
fn basis_to_model(basis: ShadingBasis) -> EmberShadingModel {
    match basis {
        ShadingBasis::Unlit => EmberShadingModel::Unlit,
        ShadingBasis::Pbr => EmberShadingModel::Pbr,
        ShadingBasis::Npr => EmberShadingModel::Npr,
        ShadingBasis::Custom(id) => EmberShadingModel::Custom(id),
    }
}

/// Downgrades an authored `ShadingModel` to what a [`ShadingBudget`] permits.
///
/// `Full` keeps the model verbatim. `SixWayApprox` drops a
/// [`EmberShadingModel::Hybrid`] overlay lobe (keeping its physical base) while
/// leaving the flat models untouched. `Unlit` forces the cheapest response.
#[must_use]
pub fn downgrade_shading(model: EmberShadingModel, budget: ShadingBudget) -> EmberShadingModel {
    match budget {
        ShadingBudget::Full => model,
        ShadingBudget::SixWayApprox => match model {
            EmberShadingModel::Hybrid { base, .. } => basis_to_model(base),
            other => other,
        },
        ShadingBudget::Unlit => EmberShadingModel::Unlit,
    }
}

/// Folds a profile override onto an emitter baseline.
///
/// Spawn rate is scaled (never negative), capacity is clamped to the profile
/// ceiling, the update cadence takes the coarser of the authored floor and the
/// profile floor (and is at least every frame), and the shading is downgraded
/// per the profile's [`ShadingBudget`]. The six-way approximation flag is set
/// only when a volumetric emitter meets a `SixWayApprox` budget.
#[must_use]
pub fn apply_profile(
    baseline: EmitterBaseline,
    profile: EmitterProfileOverride,
) -> EffectiveEmitterParams {
    let scaled = baseline.spawn_rate * profile.spawn_rate_scale;
    let spawn_rate = if scaled > 0.0 { scaled } else { 0.0 };
    let cadence = baseline
        .update_every_n_frames
        .max(profile.update_every_n_frames)
        .max(1);
    let six_way =
        baseline.volumetric && matches!(profile.shading_budget, ShadingBudget::SixWayApprox);
    EffectiveEmitterParams {
        handle: baseline.handle,
        spawn_rate,
        capacity: baseline.capacity.min(profile.capacity_cap),
        update_every_n_frames: cadence,
        shading: downgrade_shading(baseline.shading, profile.shading_budget),
        six_way_volumetric_approx: six_way,
    }
}

/// Row index into the coverage matrix for a [`PlatformTier`].
#[must_use]
fn platform_row(platform: PlatformTier) -> usize {
    match platform {
        PlatformTier::Mobile => 0,
        PlatformTier::Console => 1,
        PlatformTier::Desktop => 2,
        PlatformTier::HighEnd => 3,
    }
}

/// The `PopcornFX`-style platform × quality emitter coverage matrix (design §28).
///
/// Indexed by [`PlatformTier`] (row) and [`ParticleQuality`] (column, by its
/// rank). The caller is expected to have already clamped the requested quality
/// to the platform through [`super::lod::resolve_quality`]; every cell is still
/// populated so the matrix is queryable and testable for any pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlatformProfileMatrix {
    /// The 4×4 grid of override cells, `cells[platform_row][quality_rank]`.
    cells: [[EmitterProfileOverride; QUALITY_COUNT]; PLATFORM_COUNT],
}

/// Shorthand for building a matrix cell literal.
#[must_use]
const fn cell(
    spawn_rate_scale: f32,
    capacity_cap: u32,
    update_every_n_frames: u32,
    shading_budget: ShadingBudget,
) -> EmitterProfileOverride {
    EmitterProfileOverride {
        spawn_rate_scale,
        capacity_cap,
        update_every_n_frames,
        shading_budget,
    }
}

impl PlatformProfileMatrix {
    /// Returns the default `PopcornFX`-style coverage matrix.
    ///
    /// Values decimate an emitter more aggressively as the platform weakens and
    /// as the quality drops: mobile forces `Unlit`/six-way shading, small pools,
    /// and multi-frame update cadences, while high-end desktop keeps the full
    /// authored detail.
    #[must_use]
    pub fn popcornfx_default() -> Self {
        use ShadingBudget::{Full, SixWayApprox, Unlit};
        Self {
            cells: [
                // Mobile: Low, Medium, High, Ultra.
                [
                    cell(0.25, 2_000, 3, Unlit),
                    cell(0.35, 4_000, 2, SixWayApprox),
                    cell(0.50, 6_000, 2, SixWayApprox),
                    cell(0.60, 8_000, 1, Full),
                ],
                // Console.
                [
                    cell(0.50, 8_000, 2, SixWayApprox),
                    cell(0.65, 16_000, 1, SixWayApprox),
                    cell(0.80, 24_000, 1, Full),
                    cell(0.90, 32_000, 1, Full),
                ],
                // Desktop.
                [
                    cell(0.60, 16_000, 2, SixWayApprox),
                    cell(0.80, 32_000, 1, Full),
                    cell(1.00, 64_000, 1, Full),
                    cell(1.00, 96_000, 1, Full),
                ],
                // HighEnd.
                [
                    cell(0.75, 32_000, 1, Full),
                    cell(0.90, 64_000, 1, Full),
                    cell(1.00, 128_000, 1, Full),
                    cell(1.00, 262_144, 1, Full),
                ],
            ],
        }
    }

    /// Returns the override cell for a `(platform, quality)` pair.
    #[must_use]
    pub fn override_for(
        &self,
        platform: PlatformTier,
        quality: ParticleQuality,
    ) -> EmitterProfileOverride {
        self.cells[platform_row(platform)][quality.rank() as usize]
    }

    /// Resolves an emitter's effective parameters for a `(platform, quality)`.
    ///
    /// Convenience wrapper that looks up the matrix cell and folds it onto the
    /// baseline in one call.
    #[must_use]
    pub fn resolve(
        &self,
        baseline: EmitterBaseline,
        platform: PlatformTier,
        quality: ParticleQuality,
    ) -> EffectiveEmitterParams {
        apply_profile(baseline, self.override_for(platform, quality))
    }
}

impl Default for PlatformProfileMatrix {
    fn default() -> Self {
        Self::popcornfx_default()
    }
}

/// One rung of the ordered runtime degradation ladder (design §28).
///
/// When a frame runs over budget the runtime enables these actions in ascending
/// [`DegradationAction::severity`] order, cheapest-visible-impact first and the
/// most drastic last. [`DEGRADATION_LADDER`] is the canonical order.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DegradationAction {
    /// Reduce the per-frame spawn rate so fewer new particles are born.
    ReduceSpawn,
    /// Disable per-emitter sorting and route transparency through cheap `OIT`.
    DisableSortingOit,
    /// Drop the volumetric grid to a lower resolution.
    LowerVolumeResolution,
    /// Collapse the `ShadingModel` toward a cheaper response.
    SimplifyShading,
    /// Stretch the update cadence: simulate once every `N` frames instead of
    /// every frame.
    ReduceUpdateRate,
    /// Stop drawing the emitter's renderer while still simulating it.
    CullRenderer,
    /// Pause the emitter's simulation entirely (the last resort).
    PauseSimulation,
}

impl DegradationAction {
    /// Position of this action on the ladder (`0` is enabled first).
    #[must_use]
    pub fn severity(self) -> u8 {
        match self {
            DegradationAction::ReduceSpawn => 0,
            DegradationAction::DisableSortingOit => 1,
            DegradationAction::LowerVolumeResolution => 2,
            DegradationAction::SimplifyShading => 3,
            DegradationAction::ReduceUpdateRate => 4,
            DegradationAction::CullRenderer => 5,
            DegradationAction::PauseSimulation => 6,
        }
    }
}

/// The canonical ordered runtime degradation ladder (design §28).
///
/// Enabling degradation to level `k` means enabling exactly the first `k`
/// entries of this array; see [`active_degradation_actions`].
pub const DEGRADATION_LADDER: [DegradationAction; 7] = [
    DegradationAction::ReduceSpawn,
    DegradationAction::DisableSortingOit,
    DegradationAction::LowerVolumeResolution,
    DegradationAction::SimplifyShading,
    DegradationAction::ReduceUpdateRate,
    DegradationAction::CullRenderer,
    DegradationAction::PauseSimulation,
];

/// Over-budget multipliers at which each successive ladder rung switches on.
///
/// `used > budget * THRESHOLD[k]` enables rung `k`; the array is ascending so
/// the count of exceeded thresholds is the degradation level.
const OVERAGE_THRESHOLDS: [f32; 7] = [1.0, 1.10, 1.25, 1.50, 1.75, 2.0, 3.0];

/// Returns how many ladder rungs to enable for a given frame budget usage.
///
/// The result is in `0..=7`: `0` when the frame is within budget, climbing to
/// [`DEGRADATION_LADDER`]`.len()` when usage is far over. A zero budget with any
/// usage saturates to the full ladder; a zero budget with zero usage is level
/// `0`. Uses only multiply and comparison, so the mapping is deterministic.
#[must_use]
pub fn degradation_level(used: u32, budget: u32) -> usize {
    if budget == 0 {
        return if used == 0 {
            0
        } else {
            DEGRADATION_LADDER.len()
        };
    }
    let used_f = used as f32;
    let budget_f = budget as f32;
    OVERAGE_THRESHOLDS
        .iter()
        .take_while(|&&threshold| used_f > budget_f * threshold)
        .count()
}

/// Returns the ordered prefix of [`DEGRADATION_LADDER`] to enable at `level`.
///
/// `level` is clamped to the ladder length, so an over-large level enables the
/// whole ladder rather than panicking.
#[must_use]
pub fn active_degradation_actions(level: usize) -> &'static [DegradationAction] {
    &DEGRADATION_LADDER[..level.min(DEGRADATION_LADDER.len())]
}

/// Resolves the runtime degradation actions to enable for a budget usage.
///
/// Combines [`degradation_level`] and [`active_degradation_actions`] into the
/// one call the frame loop makes each frame.
#[must_use]
pub fn resolve_runtime_degradation(used: u32, budget: u32) -> &'static [DegradationAction] {
    active_degradation_actions(degradation_level(used, budget))
}

/// A live emitter considered for runtime degradation.
///
/// The degradation order is keyed on `(priority, screen coverage, distance)`:
/// the lowest-priority, smallest-coverage, farthest emitters shed detail first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DegradationCandidate {
    /// Emitter this candidate describes.
    pub handle: EmitterHandle,
    /// Importance priority; higher values are kept at full detail longer.
    pub priority: u32,
    /// Fraction of the screen the emitter covers, in `0..=1`.
    pub screen_coverage: f32,
    /// Distance from the camera; farther emitters are degraded sooner.
    pub distance: f32,
}

/// Compares two `f32`s within a tolerance, returning [`Ordering::Equal`] when
/// they are within `eps`. Avoids bare `f32` equality and never yields a
/// non-total result for the finite values this module deals in.
#[must_use]
fn cmp_f32(a: f32, b: f32, eps: f32) -> Ordering {
    if (a - b).abs() <= eps {
        Ordering::Equal
    } else if a < b {
        Ordering::Less
    } else {
        Ordering::Greater
    }
}

/// Orders degradation candidates so the first entry should be degraded first.
///
/// The sort key is `(priority ascending, screen coverage ascending, distance
/// descending)`: least-important, least-on-screen, farthest emitters come
/// first. The sort is stable, so candidates equal on every key keep their input
/// order for a deterministic result.
pub fn order_degradation_candidates(candidates: &mut [DegradationCandidate]) {
    candidates.sort_by(|a, b| {
        a.priority
            .cmp(&b.priority)
            .then_with(|| cmp_f32(a.screen_coverage, b.screen_coverage, COVERAGE_EPS))
            .then_with(|| cmp_f32(b.distance, a.distance, DISTANCE_EPS))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for comparing expected `f32`s in assertions.
    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1.0e-6
    }

    fn baseline(handle: u32) -> EmitterBaseline {
        EmitterBaseline {
            handle: EmitterHandle(handle),
            spawn_rate: 1_000.0,
            capacity: 100_000,
            update_every_n_frames: 1,
            shading: EmberShadingModel::Pbr,
            volumetric: true,
        }
    }

    fn candidate(handle: u32, priority: u32, coverage: f32, distance: f32) -> DegradationCandidate {
        DegradationCandidate {
            handle: EmitterHandle(handle),
            priority,
            screen_coverage: coverage,
            distance,
        }
    }

    #[test]
    fn matrix_is_dense_for_every_pair() {
        let matrix = PlatformProfileMatrix::popcornfx_default();
        let platforms = [
            PlatformTier::Mobile,
            PlatformTier::Console,
            PlatformTier::Desktop,
            PlatformTier::HighEnd,
        ];
        let qualities = [
            ParticleQuality::Low,
            ParticleQuality::Medium,
            ParticleQuality::High,
            ParticleQuality::Ultra,
        ];
        for platform in platforms {
            for quality in qualities {
                let over = matrix.override_for(platform, quality);
                assert!(over.spawn_rate_scale > 0.0);
                assert!(over.capacity_cap > 0);
                assert!(over.update_every_n_frames >= 1);
            }
        }
    }

    #[test]
    fn capacity_cap_rises_with_quality_on_a_platform() {
        let matrix = PlatformProfileMatrix::popcornfx_default();
        let low = matrix.override_for(PlatformTier::Desktop, ParticleQuality::Low);
        let high = matrix.override_for(PlatformTier::Desktop, ParticleQuality::High);
        assert!(high.capacity_cap > low.capacity_cap);
    }

    #[test]
    fn stronger_platform_never_caps_lower_at_same_quality() {
        let matrix = PlatformProfileMatrix::popcornfx_default();
        let mobile = matrix.override_for(PlatformTier::Mobile, ParticleQuality::Medium);
        let high_end = matrix.override_for(PlatformTier::HighEnd, ParticleQuality::Medium);
        assert!(high_end.capacity_cap >= mobile.capacity_cap);
        assert!(high_end.spawn_rate_scale >= mobile.spawn_rate_scale);
    }

    #[test]
    fn apply_scales_spawn_and_caps_capacity() {
        let over = EmitterProfileOverride {
            spawn_rate_scale: 0.5,
            capacity_cap: 8_000,
            update_every_n_frames: 1,
            shading_budget: ShadingBudget::Full,
        };
        let params = apply_profile(baseline(3), over);
        assert!(approx(params.spawn_rate, 500.0));
        assert_eq!(params.capacity, 8_000);
        assert_eq!(params.handle, EmitterHandle(3));
    }

    #[test]
    fn capacity_cap_never_raises_a_small_baseline() {
        let mut base = baseline(0);
        base.capacity = 1_000;
        let over = EmitterProfileOverride {
            spawn_rate_scale: 1.0,
            capacity_cap: 8_000,
            update_every_n_frames: 1,
            shading_budget: ShadingBudget::Full,
        };
        assert_eq!(apply_profile(base, over).capacity, 1_000);
    }

    #[test]
    fn update_cadence_takes_the_coarser_of_baseline_and_profile() {
        let mut base = baseline(0);
        base.update_every_n_frames = 4;
        let over = EmitterProfileOverride {
            spawn_rate_scale: 1.0,
            capacity_cap: 100_000,
            update_every_n_frames: 2,
            shading_budget: ShadingBudget::Full,
        };
        // Baseline floor of 4 wins over the profile floor of 2.
        assert_eq!(apply_profile(base, over).update_every_n_frames, 4);
    }

    #[test]
    fn full_budget_keeps_the_authored_model() {
        assert_eq!(
            downgrade_shading(EmberShadingModel::Pbr, ShadingBudget::Full),
            EmberShadingModel::Pbr
        );
    }

    #[test]
    fn six_way_budget_collapses_hybrid_to_base_lobe() {
        let hybrid = EmberShadingModel::Hybrid {
            base: ShadingBasis::Pbr,
            overlay: ShadingBasis::Npr,
            weight: 0.5,
        };
        assert_eq!(
            downgrade_shading(hybrid, ShadingBudget::SixWayApprox),
            EmberShadingModel::Pbr
        );
        // A flat model is untouched by the six-way budget.
        assert_eq!(
            downgrade_shading(EmberShadingModel::Npr, ShadingBudget::SixWayApprox),
            EmberShadingModel::Npr
        );
    }

    #[test]
    fn unlit_budget_forces_unlit() {
        let hybrid = EmberShadingModel::Hybrid {
            base: ShadingBasis::Pbr,
            overlay: ShadingBasis::Npr,
            weight: 0.25,
        };
        assert_eq!(
            downgrade_shading(hybrid, ShadingBudget::Unlit),
            EmberShadingModel::Unlit
        );
        assert_eq!(
            downgrade_shading(EmberShadingModel::Custom(9), ShadingBudget::Unlit),
            EmberShadingModel::Unlit
        );
    }

    #[test]
    fn six_way_flag_requires_volumetric_and_approx_budget() {
        let over = EmitterProfileOverride {
            spawn_rate_scale: 1.0,
            capacity_cap: 100_000,
            update_every_n_frames: 1,
            shading_budget: ShadingBudget::SixWayApprox,
        };
        // Volumetric emitter under a six-way budget is approximated.
        assert!(apply_profile(baseline(0), over).six_way_volumetric_approx);

        // Non-volumetric emitter is not.
        let mut flat = baseline(0);
        flat.volumetric = false;
        assert!(!apply_profile(flat, over).six_way_volumetric_approx);

        // Volumetric emitter at full budget keeps full volumetric PBR.
        let full = EmitterProfileOverride {
            shading_budget: ShadingBudget::Full,
            ..over
        };
        assert!(!apply_profile(baseline(0), full).six_way_volumetric_approx);
    }

    #[test]
    fn mobile_low_forces_unlit_and_multi_frame_cadence() {
        let matrix = PlatformProfileMatrix::popcornfx_default();
        let params = matrix.resolve(baseline(1), PlatformTier::Mobile, ParticleQuality::Low);
        assert_eq!(params.shading, EmberShadingModel::Unlit);
        assert!(params.update_every_n_frames >= 2);
        assert_eq!(params.capacity, 2_000);
    }

    #[test]
    fn degradation_level_is_zero_within_budget() {
        assert_eq!(degradation_level(900, 1_000), 0);
        assert_eq!(degradation_level(1_000, 1_000), 0);
    }

    #[test]
    fn degradation_level_climbs_with_overage() {
        assert_eq!(degradation_level(1_050, 1_000), 1);
        assert_eq!(degradation_level(1_200, 1_000), 2);
        assert_eq!(degradation_level(1_400, 1_000), 3);
        assert_eq!(degradation_level(1_600, 1_000), 4);
        assert_eq!(degradation_level(1_900, 1_000), 5);
        assert_eq!(degradation_level(2_500, 1_000), 6);
        assert_eq!(degradation_level(5_000, 1_000), 7);
    }

    #[test]
    fn degradation_level_handles_zero_budget() {
        assert_eq!(degradation_level(0, 0), 0);
        assert_eq!(degradation_level(1, 0), DEGRADATION_LADDER.len());
    }

    #[test]
    fn active_actions_are_an_ordered_prefix() {
        assert!(active_degradation_actions(0).is_empty());
        assert_eq!(
            active_degradation_actions(2),
            &[
                DegradationAction::ReduceSpawn,
                DegradationAction::DisableSortingOit
            ]
        );
        // Over-large levels clamp to the whole ladder.
        assert_eq!(active_degradation_actions(99), &DEGRADATION_LADDER);
    }

    #[test]
    fn resolve_runtime_degradation_end_to_end() {
        let actions = resolve_runtime_degradation(1_200, 1_000);
        assert_eq!(
            actions,
            &[
                DegradationAction::ReduceSpawn,
                DegradationAction::DisableSortingOit
            ]
        );
        assert!(resolve_runtime_degradation(500, 1_000).is_empty());
    }

    #[test]
    fn ladder_is_ordered_by_severity() {
        for pair in DEGRADATION_LADDER.windows(2) {
            assert!(pair[0].severity() < pair[1].severity());
        }
    }

    #[test]
    fn order_degrades_lowest_priority_first() {
        let mut cands = [
            candidate(0, 5, 0.5, 10.0),
            candidate(1, 1, 0.5, 10.0),
            candidate(2, 3, 0.5, 10.0),
        ];
        order_degradation_candidates(&mut cands);
        assert_eq!(cands[0].handle, EmitterHandle(1));
        assert_eq!(cands[1].handle, EmitterHandle(2));
        assert_eq!(cands[2].handle, EmitterHandle(0));
    }

    #[test]
    fn order_tiebreaks_by_coverage_then_distance() {
        let mut cands = [
            // Same priority; smallest coverage should sort first.
            candidate(0, 2, 0.80, 5.0),
            candidate(1, 2, 0.10, 5.0),
            // Same priority and coverage as #1; farther distance sorts first.
            candidate(2, 2, 0.10, 50.0),
        ];
        order_degradation_candidates(&mut cands);
        assert_eq!(cands[0].handle, EmitterHandle(2));
        assert_eq!(cands[1].handle, EmitterHandle(1));
        assert_eq!(cands[2].handle, EmitterHandle(0));
    }
}
