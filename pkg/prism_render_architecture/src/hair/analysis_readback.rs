//! Host-side reduction bridge from the GPU analysis outputs to the CPU
//! decisions they feed.
//!
//! The four hair analysis `WESL` twins ([`analysis_dispatch`]) each emit a
//! *per-element* buffer ([`analysis_buffers`]): `hair_guide_metrics` writes a
//! per-guide `out_metrics` triple, `hair_binding_metrics` writes a
//! per-render-strand `out_blend` triple, `hair_importance` writes a
//! per-render-strand `out_importance` scalar, and `hair_motion_energy` writes a
//! per-particle `out_energy` scalar. Before the `CPU` golden decisions can run,
//! three of those buffers need a small *groom-global* reduction the kernels
//! deliberately leave host-side (a reduction is not embarrassingly parallel and
//! is cheap once per groom). This module is the missing contract for that
//! bridge: it names each reduction, points at the exact source pass/binding and
//! `vec4` lane it folds, the operator it applies, and the `CPU` decision it
//! feeds, plus a device-free reference reduction that reproduces the host fold
//! bit-for-bit so tests can pin the whole `GPU` → host → `CPU` chain against the
//! existing golden functions.
//!
//! The reductions mirror the `CPU` golden references exactly:
//! - [`HairAnalysisReduction::GuideRootRadiusMax`] takes the max over the
//!   `out_metrics.z` (root radius) lane, the divisor
//!   [`density_lod::guide_metrics`](super::density_lod::guide_metrics) uses to
//!   normalize authored thickness (`max_radius`, `0` when the groom has none).
//! - [`HairAnalysisReduction::BlendLengthMax`] and
//!   [`HairAnalysisReduction::BlendCurvatureMax`] take the max over the
//!   `out_blend.x` (length) and `out_blend.y` (curvature) lanes, the `max_len`
//!   and `max_curv` normalizers
//!   [`decimation::compute_importance`](super::decimation::compute_importance)
//!   applies.
//! - [`HairAnalysisReduction::MotionEnergySum`] sums the per-particle
//!   `out_energy` scalars into the groom motion energy
//!   [`sleep::groom_motion_energy`](super::sleep::groom_motion_energy) drives
//!   the hysteretic sleep gate with.
//!
//! `hair_importance`'s `out_importance` needs no reduction: the `CPU` ranking
//! sort [`decimation::build_decimation_order`](super::decimation::build_decimation_order)
//! consumes the per-strand array directly, so it has no entry here.
//!
//! This is a device-free contract, matching the rest of the hair `GPU` ABI
//! layer: it owns no buffers and issues no dispatches. A `GPU` is now available,
//! so the source kernels are compile- and type-checked through the same
//! render-world `ShaderCache` / `wesl` pipeline the runtime uses; the on-device
//! reduction wiring and perf calibration still await real hardware, and the
//! reference fold here lets the host path be validated meanwhile.

use super::analysis_dispatch::{HairAnalysisKind, HairAnalysisPass};
use super::gpu_dispatch::HairGpuCounts;

/// The fold applied when reducing an analysis output lane to a single
/// groom-global scalar. The identity is `0`, matching the `CPU` golden
/// references (all reduced quantities — arc length, curvature, radius, squared
/// speed — are non-negative, and the golden maxima also start at `0`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairReductionOp {
    /// Groom-global maximum, used to normalize authored radius, length and
    /// curvature. Starts from `0` and keeps the larger operand.
    Max,
    /// Groom-global sum, used for the motion energy that gates sleep. Starts
    /// from `0` and accumulates left-to-right.
    Sum,
}

impl HairReductionOp {
    /// The reduction identity, `0` for both operators.
    #[must_use]
    pub fn identity() -> f32 {
        0.0
    }

    /// Folds one element `x` into the running accumulator `acc`. For `Max` the
    /// larger operand wins (ties keep `acc`, so a fresh larger value must be
    /// strictly greater — matching the golden `if r > max_radius` update); for
    /// `Sum` the value is added.
    #[must_use]
    pub fn apply(self, acc: f32, x: f32) -> f32 {
        match self {
            Self::Max => {
                if x > acc {
                    x
                } else {
                    acc
                }
            }
            Self::Sum => acc + x,
        }
    }
}

/// Which component of a `vec4` analysis output a reduction folds. Scalar output
/// buffers (`out_importance`, `out_energy`) are treated as occupying lane
/// [`X`](HairMetricLane::X).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairMetricLane {
    /// `.x` — `out_metrics` arc length, `out_blend` length, or a scalar buffer.
    X,
    /// `.y` — `out_metrics` / `out_blend` curvature.
    Y,
    /// `.z` — `out_metrics` root radius / `out_blend` authored thickness.
    Z,
    /// `.w` — the padding lane (unused by the current reductions).
    W,
}

impl HairMetricLane {
    /// The array index this lane selects from a `[f32; 4]` element.
    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
            Self::W => 3,
        }
    }
}

/// A groom-global reduction of one analysis output that a `CPU` decision needs.
/// Each variant fixes the source pass, the output buffer binding, the lane, the
/// operator and the decision it feeds, so the host can wire the readback and
/// reduction from this contract alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairAnalysisReduction {
    /// Max over `hair_guide_metrics` `out_metrics.z` (root radius): the
    /// authored-thickness normalizer for
    /// [`density_lod::guide_metrics`](super::density_lod::guide_metrics).
    GuideRootRadiusMax,
    /// Max over `hair_binding_metrics` `out_blend.x` (length): the `max_len`
    /// normalizer for
    /// [`decimation::compute_importance`](super::decimation::compute_importance).
    BlendLengthMax,
    /// Max over `hair_binding_metrics` `out_blend.y` (curvature): the
    /// `max_curv` normalizer for
    /// [`decimation::compute_importance`](super::decimation::compute_importance).
    BlendCurvatureMax,
    /// Sum over `hair_motion_energy` `out_energy` (squared speed): the groom
    /// motion energy for
    /// [`sleep::groom_motion_energy`](super::sleep::groom_motion_energy).
    MotionEnergySum,
}

impl HairAnalysisReduction {
    /// Every host reduction, in the canonical analysis order: the three
    /// density-LOD normalizers (guide radius max, blend length max, blend
    /// curvature max) followed by the sleep-gate motion-energy sum.
    pub const ALL: [HairAnalysisReduction; 4] = [
        Self::GuideRootRadiusMax,
        Self::BlendLengthMax,
        Self::BlendCurvatureMax,
        Self::MotionEnergySum,
    ];

    /// The analysis pass whose output buffer this reduction folds.
    #[must_use]
    pub fn pass(self) -> HairAnalysisPass {
        match self {
            Self::GuideRootRadiusMax => HairAnalysisPass::GuideMetrics,
            Self::BlendLengthMax | Self::BlendCurvatureMax => HairAnalysisPass::BindingMetrics,
            Self::MotionEnergySum => HairAnalysisPass::MotionEnergy,
        }
    }

    /// The `@group(0)` binding index of the source output buffer this reduction
    /// reads back: `out_metrics` at `3`, `out_blend` at `2`, `out_energy` at
    /// `2`. Kept in lock-step with [`analysis_buffers`](super::analysis_buffers).
    #[must_use]
    pub fn source_binding(self) -> u32 {
        match self {
            Self::GuideRootRadiusMax => 3,
            Self::BlendLengthMax | Self::BlendCurvatureMax | Self::MotionEnergySum => 2,
        }
    }

    /// The `vec4` lane folded. `out_metrics.z` for the radius max, `out_blend.x`
    /// / `out_blend.y` for the length / curvature maxima, and lane `X` for the
    /// scalar `out_energy` sum.
    #[must_use]
    pub fn lane(self) -> HairMetricLane {
        match self {
            Self::GuideRootRadiusMax => HairMetricLane::Z,
            Self::BlendLengthMax | Self::MotionEnergySum => HairMetricLane::X,
            Self::BlendCurvatureMax => HairMetricLane::Y,
        }
    }

    /// The fold operator: `Max` for the three normalizers, `Sum` for the motion
    /// energy.
    #[must_use]
    pub fn op(self) -> HairReductionOp {
        match self {
            Self::GuideRootRadiusMax | Self::BlendLengthMax | Self::BlendCurvatureMax => {
                HairReductionOp::Max
            }
            Self::MotionEnergySum => HairReductionOp::Sum,
        }
    }

    /// The host decision this reduction feeds: the density-LOD chain for the
    /// three normalizers, the sleep gate for the motion-energy sum. Mirrors
    /// [`HairAnalysisPass::kind`] on the source pass.
    #[must_use]
    pub fn kind(self) -> HairAnalysisKind {
        self.pass().kind()
    }

    /// Number of source elements this reduction folds for a groom with `counts`
    /// domain totals: the element count of the source pass's dispatch domain
    /// (guide strands for the radius max, render strands for the blend maxima,
    /// guide particles for the energy sum).
    #[must_use]
    pub fn element_count(self, counts: &HairGpuCounts) -> u32 {
        counts.domain_count(self.pass().domain())
    }

    /// A short human-readable label for the `CPU` value this reduction feeds,
    /// for diagnostics and design cross-referencing.
    #[must_use]
    pub fn feeds(self) -> &'static str {
        match self {
            Self::GuideRootRadiusMax => {
                "density_lod::guide_metrics authored-radius normalizer (max_radius)"
            }
            Self::BlendLengthMax => "decimation::compute_importance length normalizer (max_len)",
            Self::BlendCurvatureMax => {
                "decimation::compute_importance curvature normalizer (max_curv)"
            }
            Self::MotionEnergySum => "sleep::groom_motion_energy for the hysteretic sleep gate",
        }
    }
}

/// Device-free reference reduction of a `vec4` output buffer: folds `lane` of
/// every element in `elements` with `op`, starting from the operator identity.
/// An empty slice yields the identity (`0`) rather than panicking, matching the
/// empty-groom behavior of the golden references. Scalar output buffers are
/// reduced by packing each value into lane [`X`](HairMetricLane::X) of a
/// `[f32; 4]` (with the other lanes `0`).
#[must_use]
pub fn reduce_lane(elements: &[[f32; 4]], lane: HairMetricLane, op: HairReductionOp) -> f32 {
    let index = lane.index();
    let mut acc = HairReductionOp::identity();
    for element in elements {
        acc = op.apply(acc, element[index]);
    }
    acc
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::hair::dynamics::{StrandParticle, Vec3};
    use crate::hair::gpu_dispatch::HairGpuCounts;
    use crate::hair::sleep::groom_motion_energy;

    #[test]
    fn all_reductions_are_listed_once() {
        assert_eq!(HairAnalysisReduction::ALL.len(), 4);
        for (i, a) in HairAnalysisReduction::ALL.iter().enumerate() {
            for b in &HairAnalysisReduction::ALL[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn empty_input_yields_identity() {
        assert_eq!(
            reduce_lane(&[], HairMetricLane::X, HairReductionOp::Max),
            0.0
        );
        assert_eq!(
            reduce_lane(&[], HairMetricLane::Z, HairReductionOp::Sum),
            0.0
        );
    }

    #[test]
    fn max_folds_the_selected_lane() {
        let elements = [
            [1.0, 9.0, 3.0, 0.0],
            [4.0, 2.0, 7.0, 0.0],
            [2.0, 5.0, 6.0, 0.0],
        ];
        assert_eq!(
            reduce_lane(&elements, HairMetricLane::X, HairReductionOp::Max),
            4.0
        );
        assert_eq!(
            reduce_lane(&elements, HairMetricLane::Y, HairReductionOp::Max),
            9.0
        );
        assert_eq!(
            reduce_lane(&elements, HairMetricLane::Z, HairReductionOp::Max),
            7.0
        );
    }

    #[test]
    fn sum_folds_the_selected_lane() {
        let elements = [
            [1.0, 0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
        ];
        assert_eq!(
            reduce_lane(&elements, HairMetricLane::X, HairReductionOp::Sum),
            4.0
        );
    }

    #[test]
    fn lane_indices_are_dense() {
        assert_eq!(HairMetricLane::X.index(), 0);
        assert_eq!(HairMetricLane::Y.index(), 1);
        assert_eq!(HairMetricLane::Z.index(), 2);
        assert_eq!(HairMetricLane::W.index(), 3);
    }

    #[test]
    fn each_reduction_targets_its_source_output_binding() {
        // Must match the output buffers in `analysis_buffers`:
        // GuideMetrics.OutMetrics = 3, BindingMetrics.OutBlend = 2,
        // MotionEnergy.OutEnergy = 2.
        assert_eq!(
            HairAnalysisReduction::GuideRootRadiusMax.pass(),
            HairAnalysisPass::GuideMetrics
        );
        assert_eq!(
            HairAnalysisReduction::GuideRootRadiusMax.source_binding(),
            3
        );
        assert_eq!(
            HairAnalysisReduction::BlendLengthMax.pass(),
            HairAnalysisPass::BindingMetrics
        );
        assert_eq!(HairAnalysisReduction::BlendLengthMax.source_binding(), 2);
        assert_eq!(
            HairAnalysisReduction::BlendCurvatureMax.pass(),
            HairAnalysisPass::BindingMetrics
        );
        assert_eq!(HairAnalysisReduction::BlendCurvatureMax.source_binding(), 2);
        assert_eq!(
            HairAnalysisReduction::MotionEnergySum.pass(),
            HairAnalysisPass::MotionEnergy
        );
        assert_eq!(HairAnalysisReduction::MotionEnergySum.source_binding(), 2);
    }

    #[test]
    fn lanes_and_ops_match_the_golden_normalizers() {
        assert_eq!(
            HairAnalysisReduction::GuideRootRadiusMax.lane(),
            HairMetricLane::Z
        );
        assert_eq!(
            HairAnalysisReduction::GuideRootRadiusMax.op(),
            HairReductionOp::Max
        );
        assert_eq!(
            HairAnalysisReduction::BlendLengthMax.lane(),
            HairMetricLane::X
        );
        assert_eq!(
            HairAnalysisReduction::BlendLengthMax.op(),
            HairReductionOp::Max
        );
        assert_eq!(
            HairAnalysisReduction::BlendCurvatureMax.lane(),
            HairMetricLane::Y
        );
        assert_eq!(
            HairAnalysisReduction::BlendCurvatureMax.op(),
            HairReductionOp::Max
        );
        assert_eq!(
            HairAnalysisReduction::MotionEnergySum.lane(),
            HairMetricLane::X
        );
        assert_eq!(
            HairAnalysisReduction::MotionEnergySum.op(),
            HairReductionOp::Sum
        );
    }

    #[test]
    fn kinds_group_the_two_decision_chains() {
        assert_eq!(
            HairAnalysisReduction::GuideRootRadiusMax.kind(),
            HairAnalysisKind::DensityLod
        );
        assert_eq!(
            HairAnalysisReduction::BlendLengthMax.kind(),
            HairAnalysisKind::DensityLod
        );
        assert_eq!(
            HairAnalysisReduction::BlendCurvatureMax.kind(),
            HairAnalysisKind::DensityLod
        );
        assert_eq!(
            HairAnalysisReduction::MotionEnergySum.kind(),
            HairAnalysisKind::Sleep
        );
    }

    #[test]
    fn element_counts_follow_the_source_pass_domains() {
        let counts = HairGpuCounts {
            roots: 3,
            guide_strands: 5,
            guide_particles: 40,
            render_strands: 200,
            light_texels: 0,
        };
        assert_eq!(
            HairAnalysisReduction::GuideRootRadiusMax.element_count(&counts),
            5
        );
        assert_eq!(
            HairAnalysisReduction::BlendLengthMax.element_count(&counts),
            200
        );
        assert_eq!(
            HairAnalysisReduction::BlendCurvatureMax.element_count(&counts),
            200
        );
        assert_eq!(
            HairAnalysisReduction::MotionEnergySum.element_count(&counts),
            40
        );

        let empty = HairGpuCounts::default();
        for reduction in HairAnalysisReduction::ALL {
            assert_eq!(reduction.element_count(&empty), 0);
        }
    }

    #[test]
    fn motion_energy_sum_matches_the_cpu_golden() {
        // Build a mixed groom of moving and pinned particles, mirror the
        // `hair_motion_energy` map (out_energy = dot(v, v)) into the reduction
        // input, and confirm the host `Sum` equals `sleep::groom_motion_energy`.
        let particles = [
            StrandParticle {
                position: Vec3::new(1.0, 2.0, 3.0),
                prev_position: Vec3::new(0.0, 0.0, 0.0),
                inverse_mass: 1.0,
            },
            StrandParticle::pinned(Vec3::new(5.0, 5.0, 5.0)),
            StrandParticle {
                position: Vec3::new(-2.0, 0.5, 4.0),
                prev_position: Vec3::new(-1.0, 0.0, 1.0),
                inverse_mass: 1.0,
            },
        ];

        let energies: Vec<[f32; 4]> = particles
            .iter()
            .map(|p| {
                let v = p.position.sub(p.prev_position);
                [v.length_squared(), 0.0, 0.0, 0.0]
            })
            .collect();

        let reduced = reduce_lane(
            &energies,
            HairAnalysisReduction::MotionEnergySum.lane(),
            HairAnalysisReduction::MotionEnergySum.op(),
        );
        let golden = groom_motion_energy(&particles);
        assert_eq!(reduced, golden);
    }

    #[test]
    fn guide_root_radius_max_matches_the_golden_divisor() {
        // Mirror `hair_guide_metrics` out_metrics = (arc, curv, root_radius, 0)
        // and confirm the `Max` over the radius lane equals the divisor
        // `density_lod::guide_metrics` normalizes authored thickness by.
        let radii = [0.4_f32, 1.25, 0.9, 0.0];
        let out_metrics: Vec<[f32; 4]> = radii.iter().map(|&r| [3.0, 0.5, r, 0.0]).collect();
        let reduced = reduce_lane(
            &out_metrics,
            HairAnalysisReduction::GuideRootRadiusMax.lane(),
            HairAnalysisReduction::GuideRootRadiusMax.op(),
        );
        let golden_max = radii
            .iter()
            .fold(0.0_f32, |m, &r| if r > m { r } else { m });
        assert_eq!(reduced, golden_max);
        assert_eq!(reduced, 1.25);
    }

    #[test]
    fn feeds_labels_are_nonempty() {
        for reduction in HairAnalysisReduction::ALL {
            assert!(!reduction.feeds().is_empty());
        }
    }
}
