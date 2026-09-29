//! Device-free byte-layout contract for the hair *analysis* pass `GPU` buffers.
//!
//! [`analysis_dispatch`](super::analysis_dispatch) publishes *how many*
//! workgroups each analysis pass dispatches; this module publishes *what those
//! passes bind* — the authoritative element stride, access mode, element count
//! and total byte size of every storage buffer in the four analysis kernels'
//! `@group(0)`. Exactly as [`interp_buffers`](super::interp_buffers) and its
//! five sibling contracts do for the ten main
//! [`HairComputePass`](super::gpu_dispatch::HairComputePass) kernels, the sizing
//! lives once here in
//! the zero-dependency crate so the render graph binds against a stable ABI
//! instead of hand-computing strides next to the pipeline.
//!
//! The four analysis passes fall in the two host-decision chains
//! [`analysis_dispatch`](super::analysis_dispatch) fixes:
//!
//! - Density-LOD metric chain: `hair_guide_metrics` measures each guide, writing
//!   `out_metrics`; the host normalizes authored thickness and feeds the triple
//!   as the `hair_binding_metrics` input `guide_metrics`, which blends it onto
//!   every render strand into `out_blend`; `hair_importance` then folds
//!   `blend` into a single `[0, 1]` `out_importance` the `CPU`
//!   `decimation::build_decimation_order` ranking sort consumes.
//! - Sleep gate: `hair_motion_energy` reads the shared sim `positions` /
//!   `prev_positions` and writes the per-particle squared velocity `out_energy`
//!   the host sums for the hysteretic sleep gate.
//!
//! The two motion-energy inputs *alias* the persistent guide-`XPBD` state owned
//! by [`gpu_buffers`](super::gpu_buffers) (`HairSimBuffer::Positions` /
//! `PrevPositions`), not fresh allocations — the layout matches so no repacking
//! is needed between the sim and this proxy, mirroring how
//! [`sim_pass_buffers`](super::sim_pass_buffers) aliases the persistent
//! positions for its wind / `SDF` passes.
//!
//! Everything is pure integer arithmetic: byte sizes are clamped up to one
//! element so an empty groom still yields a valid non-empty `WebGPU` storage
//! binding, and nothing panics or divides by zero.

use crate::hair::gpu_buffers::HairBufferAccess;
use crate::hair::gpu_dispatch::HairGpuCounts;

/// Byte stride of a `vec4<f32>` / `vec4<u32>` storage element: four 4-byte
/// scalars, the natural 16-byte stride.
const VEC4_STRIDE: usize = 16;

/// Byte stride of a scalar `f32` / `u32` storage element.
const SCALAR_STRIDE: usize = 4;

/// `std430` array stride of `HairMetricStrandRange` (`start: u32`, `len: u32`):
/// two tightly packed 4-byte scalars, 8 bytes.
const METRIC_RANGE_STRIDE: usize = 8;

/// `std430` array stride of `HairBindingInfluence`: `guides: vec4<u32>` (16) +
/// `weights: vec4<f32>` (16) = 32 bytes, already a multiple of the 16-byte
/// struct alignment so no tail padding is added.
const BINDING_INFLUENCE_STRIDE: usize = 32;

/// One storage buffer bound by the per-guide density-LOD metric kernel
/// (`hair_guide_metrics.wesl` `@group(0)`), in binding order `0..4`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairGuideMetricsBuffer {
    /// `@binding(0)` shared flat pool of resampled guide control points
    /// `array<vec4<f32>>` (`xyz` = position).
    Points,
    /// `@binding(1)` per-guide compacted slice descriptor
    /// `array<HairMetricStrandRange>` (`start`, `len` as `u32`).
    StrandRanges,
    /// `@binding(2)` per-guide authored root radius `array<f32>` (raw; the host
    /// clamps and normalizes).
    RootRadii,
    /// `@binding(3)` per-guide raw metric triple `array<vec4<f32>>`
    /// (`arc_length`, `curvature`, `root_radius`, `0`), written by this pass.
    OutMetrics,
}

impl HairGuideMetricsBuffer {
    /// Every guide-metrics buffer in `@binding` order. Its length matches
    /// [`GuideMetrics`](super::analysis_dispatch::HairAnalysisPass)'s binding
    /// count, keeping this layout in lock-step with the dispatch ABI.
    pub const ALL: [HairGuideMetricsBuffer; 4] = [
        Self::Points,
        Self::StrandRanges,
        Self::RootRadii,
        Self::OutMetrics,
    ];

    /// The `@group(0)` binding index this buffer occupies.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::Points => 0,
            Self::StrandRanges => 1,
            Self::RootRadii => 2,
            Self::OutMetrics => 3,
        }
    }

    /// Byte stride of one element, matching the `WESL` struct / scalar layout.
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            Self::Points | Self::OutMetrics => VEC4_STRIDE,
            Self::StrandRanges => METRIC_RANGE_STRIDE,
            Self::RootRadii => SCALAR_STRIDE,
        }
    }

    /// Whether the kernel reads or read-writes this buffer. Only `out_metrics`
    /// is written; the points pool, ranges and radii are read-only inputs.
    #[must_use]
    pub fn access(self) -> HairBufferAccess {
        match self {
            Self::Points | Self::StrandRanges | Self::RootRadii => HairBufferAccess::Read,
            Self::OutMetrics => HairBufferAccess::ReadWrite,
        }
    }

    /// Whether this pass writes the buffer rather than only reading it.
    #[must_use]
    pub fn is_output(self) -> bool {
        matches!(self.access(), HairBufferAccess::ReadWrite)
    }

    /// Number of elements this buffer holds for a groom with `counts` domain
    /// totals. The flat points pool spans every guide particle; the ranges,
    /// radii and output metrics are one per guide strand.
    #[must_use]
    pub fn element_count(self, counts: &HairGpuCounts) -> u32 {
        match self {
            Self::Points => counts.guide_particles,
            Self::StrandRanges | Self::RootRadii | Self::OutMetrics => counts.guide_strands,
        }
    }

    /// Total byte size of this buffer, clamped up to one element so an empty
    /// groom still yields a valid non-empty `WebGPU` storage binding.
    #[must_use]
    pub fn byte_size(self, counts: &HairGpuCounts) -> usize {
        (self.element_count(counts).max(1) as usize) * self.stride()
    }
}

/// One storage buffer bound by the per-render-strand density-LOD blend kernel
/// (`hair_binding_metrics.wesl` `@group(0)`), in binding order `0..3`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairBindingMetricsBuffer {
    /// `@binding(0)` per-guide normalized metric triple `array<vec4<f32>>`
    /// (the `hair_guide_metrics` output after the host max-radius normalization).
    GuideMetrics,
    /// `@binding(1)` per-render-strand guide influences
    /// `array<HairBindingInfluence>` (`guides: vec4<u32>`, `weights: vec4<f32>`).
    Bindings,
    /// `@binding(2)` per-render-strand blended triple `array<vec4<f32>>`
    /// (`length`, `curvature`, `authored`, `0`), written by this pass.
    OutBlend,
}

impl HairBindingMetricsBuffer {
    /// Every binding-metrics buffer in `@binding` order. Its length matches
    /// [`BindingMetrics`](super::analysis_dispatch::HairAnalysisPass)'s binding
    /// count.
    pub const ALL: [HairBindingMetricsBuffer; 3] =
        [Self::GuideMetrics, Self::Bindings, Self::OutBlend];

    /// The `@group(0)` binding index this buffer occupies.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::GuideMetrics => 0,
            Self::Bindings => 1,
            Self::OutBlend => 2,
        }
    }

    /// Byte stride of one element, matching the `WESL` struct / scalar layout.
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            Self::GuideMetrics | Self::OutBlend => VEC4_STRIDE,
            Self::Bindings => BINDING_INFLUENCE_STRIDE,
        }
    }

    /// Whether the kernel reads or read-writes this buffer. Only `out_blend` is
    /// written; the guide metrics and binding table are read-only inputs.
    #[must_use]
    pub fn access(self) -> HairBufferAccess {
        match self {
            Self::GuideMetrics | Self::Bindings => HairBufferAccess::Read,
            Self::OutBlend => HairBufferAccess::ReadWrite,
        }
    }

    /// Whether this pass writes the buffer rather than only reading it.
    #[must_use]
    pub fn is_output(self) -> bool {
        matches!(self.access(), HairBufferAccess::ReadWrite)
    }

    /// Number of elements this buffer holds for a groom with `counts` domain
    /// totals. The guide-metric input is one per guide strand; the binding
    /// table and blended output are one per render strand.
    #[must_use]
    pub fn element_count(self, counts: &HairGpuCounts) -> u32 {
        match self {
            Self::GuideMetrics => counts.guide_strands,
            Self::Bindings | Self::OutBlend => counts.render_strands,
        }
    }

    /// Total byte size of this buffer, clamped up to one element so an empty
    /// groom still yields a valid non-empty `WebGPU` storage binding.
    #[must_use]
    pub fn byte_size(self, counts: &HairGpuCounts) -> usize {
        (self.element_count(counts).max(1) as usize) * self.stride()
    }
}

/// One storage buffer bound by the per-render-strand importance-fold kernel
/// (`hair_importance.wesl` `@group(0)`), in binding order `0..2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairImportanceBuffer {
    /// `@binding(0)` per-render-strand blended metric triple `array<vec4<f32>>`
    /// (the `hair_binding_metrics` output).
    Blend,
    /// `@binding(1)` per-render-strand normalized importance `array<f32>` in
    /// `[0, 1]`, written by this pass and consumed by the `CPU` ranking sort.
    OutImportance,
}

impl HairImportanceBuffer {
    /// Every importance buffer in `@binding` order. Its length matches
    /// [`Importance`](super::analysis_dispatch::HairAnalysisPass)'s binding
    /// count.
    pub const ALL: [HairImportanceBuffer; 2] = [Self::Blend, Self::OutImportance];

    /// The `@group(0)` binding index this buffer occupies.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::Blend => 0,
            Self::OutImportance => 1,
        }
    }

    /// Byte stride of one element, matching the `WESL` scalar layout.
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            Self::Blend => VEC4_STRIDE,
            Self::OutImportance => SCALAR_STRIDE,
        }
    }

    /// Whether the kernel reads or read-writes this buffer. Only `out_importance`
    /// is written; the blended triple is a read-only input.
    #[must_use]
    pub fn access(self) -> HairBufferAccess {
        match self {
            Self::Blend => HairBufferAccess::Read,
            Self::OutImportance => HairBufferAccess::ReadWrite,
        }
    }

    /// Whether this pass writes the buffer rather than only reading it.
    #[must_use]
    pub fn is_output(self) -> bool {
        matches!(self.access(), HairBufferAccess::ReadWrite)
    }

    /// Number of elements this buffer holds for a groom with `counts` domain
    /// totals. Both the blended input and the importance output are one per
    /// render strand.
    #[must_use]
    pub fn element_count(self, counts: &HairGpuCounts) -> u32 {
        match self {
            Self::Blend | Self::OutImportance => counts.render_strands,
        }
    }

    /// Total byte size of this buffer, clamped up to one element so an empty
    /// groom still yields a valid non-empty `WebGPU` storage binding.
    #[must_use]
    pub fn byte_size(self, counts: &HairGpuCounts) -> usize {
        (self.element_count(counts).max(1) as usize) * self.stride()
    }
}

/// One storage buffer bound by the per-particle sleep motion-energy kernel
/// (`hair_motion_energy.wesl` `@group(0)`), in binding order `0..3`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairMotionEnergyBuffer {
    /// `@binding(0)` current particle state `array<vec4<f32>>` (`xyz` =
    /// position, `w` = inverse mass); aliases the persistent
    /// `HairSimBuffer::Positions`, only `xyz` is read.
    Positions,
    /// `@binding(1)` previous-substep positions `array<vec4<f32>>` (`xyz` used);
    /// aliases the persistent `HairSimBuffer::PrevPositions`.
    PrevPositions,
    /// `@binding(2)` per-particle squared implicit velocity `array<f32>`,
    /// written by this pass and summed host-side into the groom motion energy.
    OutEnergy,
}

impl HairMotionEnergyBuffer {
    /// Every motion-energy buffer in `@binding` order. Its length matches
    /// [`MotionEnergy`](super::analysis_dispatch::HairAnalysisPass)'s binding
    /// count.
    pub const ALL: [HairMotionEnergyBuffer; 3] =
        [Self::Positions, Self::PrevPositions, Self::OutEnergy];

    /// The `@group(0)` binding index this buffer occupies.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::Positions => 0,
            Self::PrevPositions => 1,
            Self::OutEnergy => 2,
        }
    }

    /// Byte stride of one element, matching the `WESL` scalar layout.
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            Self::Positions | Self::PrevPositions => VEC4_STRIDE,
            Self::OutEnergy => SCALAR_STRIDE,
        }
    }

    /// Whether the kernel reads or read-writes this buffer. Only `out_energy` is
    /// written; the shared positions are read-only here.
    #[must_use]
    pub fn access(self) -> HairBufferAccess {
        match self {
            Self::Positions | Self::PrevPositions => HairBufferAccess::Read,
            Self::OutEnergy => HairBufferAccess::ReadWrite,
        }
    }

    /// Whether this pass writes the buffer rather than only reading it.
    #[must_use]
    pub fn is_output(self) -> bool {
        matches!(self.access(), HairBufferAccess::ReadWrite)
    }

    /// Whether this binding aliases the persistent guide-`XPBD` sim state owned
    /// by [`gpu_buffers`](super::gpu_buffers) rather than a fresh allocation.
    /// The two position histories are shared with the sim; only `out_energy` is
    /// a new per-particle scratch buffer.
    #[must_use]
    pub fn aliases_persistent_sim_state(self) -> bool {
        match self {
            Self::Positions | Self::PrevPositions => true,
            Self::OutEnergy => false,
        }
    }

    /// Number of elements this buffer holds for a groom with `counts` domain
    /// totals. Every motion-energy buffer is one per guide particle.
    #[must_use]
    pub fn element_count(self, counts: &HairGpuCounts) -> u32 {
        match self {
            Self::Positions | Self::PrevPositions | Self::OutEnergy => counts.guide_particles,
        }
    }

    /// Total byte size of this buffer, clamped up to one element so an empty
    /// groom still yields a valid non-empty `WebGPU` storage binding.
    #[must_use]
    pub fn byte_size(self, counts: &HairGpuCounts) -> usize {
        (self.element_count(counts).max(1) as usize) * self.stride()
    }
}

/// Total bytes the density-LOD metric chain newly allocates for a groom with
/// `counts` domain totals: the guide-metric output, the per-binding blend and
/// the per-strand importance (the read-only points pool, ranges, radii and
/// binding table are inputs the import / sim passes already own, so they are not
/// counted here).
#[must_use]
pub fn density_lod_scratch_bytes(counts: &HairGpuCounts) -> usize {
    HairGuideMetricsBuffer::OutMetrics.byte_size(counts)
        + HairBindingMetricsBuffer::OutBlend.byte_size(counts)
        + HairImportanceBuffer::OutImportance.byte_size(counts)
}

/// Total bytes the sleep gate newly allocates for a groom with `counts` domain
/// totals: the per-particle `out_energy` scratch. The two position histories
/// alias the persistent sim state and are not counted.
#[must_use]
pub fn sleep_scratch_bytes(counts: &HairGpuCounts) -> usize {
    HairMotionEnergyBuffer::OutEnergy.byte_size(counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::analysis_dispatch::HairAnalysisPass;

    fn sample_counts() -> HairGpuCounts {
        HairGpuCounts {
            roots: 10,
            guide_strands: 100,
            guide_particles: 3200,
            render_strands: 50_000,
            light_texels: 0,
        }
    }

    #[test]
    fn bindings_are_dense_and_ordered() {
        for (index, buffer) in HairGuideMetricsBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
        }
        for (index, buffer) in HairBindingMetricsBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
        }
        for (index, buffer) in HairImportanceBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
        }
        for (index, buffer) in HairMotionEnergyBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
        }
    }

    #[test]
    fn buffer_sets_match_the_dispatch_binding_counts() {
        assert_eq!(
            HairGuideMetricsBuffer::ALL.len() as u32,
            HairAnalysisPass::GuideMetrics.binding_count()
        );
        assert_eq!(
            HairBindingMetricsBuffer::ALL.len() as u32,
            HairAnalysisPass::BindingMetrics.binding_count()
        );
        assert_eq!(
            HairImportanceBuffer::ALL.len() as u32,
            HairAnalysisPass::Importance.binding_count()
        );
        assert_eq!(
            HairMotionEnergyBuffer::ALL.len() as u32,
            HairAnalysisPass::MotionEnergy.binding_count()
        );
    }

    #[test]
    fn strides_match_the_wesl_struct_layout() {
        assert_eq!(HairGuideMetricsBuffer::Points.stride(), 16);
        assert_eq!(HairGuideMetricsBuffer::StrandRanges.stride(), 8);
        assert_eq!(HairGuideMetricsBuffer::RootRadii.stride(), 4);
        assert_eq!(HairGuideMetricsBuffer::OutMetrics.stride(), 16);
        assert_eq!(HairBindingMetricsBuffer::GuideMetrics.stride(), 16);
        assert_eq!(HairBindingMetricsBuffer::Bindings.stride(), 32);
        assert_eq!(HairBindingMetricsBuffer::OutBlend.stride(), 16);
        assert_eq!(HairImportanceBuffer::Blend.stride(), 16);
        assert_eq!(HairImportanceBuffer::OutImportance.stride(), 4);
        assert_eq!(HairMotionEnergyBuffer::Positions.stride(), 16);
        assert_eq!(HairMotionEnergyBuffer::PrevPositions.stride(), 16);
        assert_eq!(HairMotionEnergyBuffer::OutEnergy.stride(), 4);
    }

    #[test]
    fn only_the_last_binding_of_each_pass_is_written() {
        assert!(!HairGuideMetricsBuffer::Points.is_output());
        assert!(!HairGuideMetricsBuffer::StrandRanges.is_output());
        assert!(!HairGuideMetricsBuffer::RootRadii.is_output());
        assert!(HairGuideMetricsBuffer::OutMetrics.is_output());
        assert!(!HairBindingMetricsBuffer::GuideMetrics.is_output());
        assert!(!HairBindingMetricsBuffer::Bindings.is_output());
        assert!(HairBindingMetricsBuffer::OutBlend.is_output());
        assert!(!HairImportanceBuffer::Blend.is_output());
        assert!(HairImportanceBuffer::OutImportance.is_output());
        assert!(!HairMotionEnergyBuffer::Positions.is_output());
        assert!(!HairMotionEnergyBuffer::PrevPositions.is_output());
        assert!(HairMotionEnergyBuffer::OutEnergy.is_output());
    }

    #[test]
    fn motion_energy_positions_alias_the_persistent_sim_state() {
        assert!(HairMotionEnergyBuffer::Positions.aliases_persistent_sim_state());
        assert!(HairMotionEnergyBuffer::PrevPositions.aliases_persistent_sim_state());
        assert!(!HairMotionEnergyBuffer::OutEnergy.aliases_persistent_sim_state());
    }

    #[test]
    fn element_counts_follow_the_groom_domains() {
        let counts = sample_counts();
        assert_eq!(HairGuideMetricsBuffer::Points.element_count(&counts), 3200);
        assert_eq!(
            HairGuideMetricsBuffer::StrandRanges.element_count(&counts),
            100
        );
        assert_eq!(
            HairGuideMetricsBuffer::RootRadii.element_count(&counts),
            100
        );
        assert_eq!(
            HairGuideMetricsBuffer::OutMetrics.element_count(&counts),
            100
        );
        assert_eq!(
            HairBindingMetricsBuffer::GuideMetrics.element_count(&counts),
            100
        );
        assert_eq!(
            HairBindingMetricsBuffer::Bindings.element_count(&counts),
            50_000
        );
        assert_eq!(
            HairBindingMetricsBuffer::OutBlend.element_count(&counts),
            50_000
        );
        assert_eq!(HairImportanceBuffer::Blend.element_count(&counts), 50_000);
        assert_eq!(
            HairImportanceBuffer::OutImportance.element_count(&counts),
            50_000
        );
        assert_eq!(
            HairMotionEnergyBuffer::Positions.element_count(&counts),
            3200
        );
        assert_eq!(
            HairMotionEnergyBuffer::OutEnergy.element_count(&counts),
            3200
        );
    }

    #[test]
    fn byte_sizes_multiply_count_by_stride() {
        let counts = sample_counts();
        assert_eq!(
            HairGuideMetricsBuffer::OutMetrics.byte_size(&counts),
            100 * 16
        );
        assert_eq!(
            HairBindingMetricsBuffer::Bindings.byte_size(&counts),
            50_000 * 32
        );
        assert_eq!(
            HairImportanceBuffer::OutImportance.byte_size(&counts),
            50_000 * 4
        );
        assert_eq!(
            HairMotionEnergyBuffer::OutEnergy.byte_size(&counts),
            3200 * 4
        );
    }

    #[test]
    fn empty_groom_clamps_every_buffer_to_one_element() {
        let counts = HairGpuCounts::default();
        for buffer in HairGuideMetricsBuffer::ALL {
            assert_eq!(buffer.byte_size(&counts), buffer.stride());
        }
        for buffer in HairBindingMetricsBuffer::ALL {
            assert_eq!(buffer.byte_size(&counts), buffer.stride());
        }
        for buffer in HairImportanceBuffer::ALL {
            assert_eq!(buffer.byte_size(&counts), buffer.stride());
        }
        for buffer in HairMotionEnergyBuffer::ALL {
            assert_eq!(buffer.byte_size(&counts), buffer.stride());
        }
    }

    #[test]
    fn scratch_byte_helpers_sum_the_newly_allocated_outputs() {
        let counts = sample_counts();
        // Density-LOD scratch = out_metrics (100*16) + out_blend (50_000*16)
        // + out_importance (50_000*4).
        assert_eq!(
            density_lod_scratch_bytes(&counts),
            100 * 16 + 50_000 * 16 + 50_000 * 4
        );
        // Sleep scratch = out_energy only (positions alias the sim state).
        assert_eq!(sleep_scratch_bytes(&counts), 3200 * 4);
    }
}
