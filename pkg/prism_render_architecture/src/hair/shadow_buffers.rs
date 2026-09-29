//! Device-free byte-layout contract for the hair self-shadow (`Shadow`-stage)
//! `GPU` buffers.
//!
//! [`gpu_dispatch`](super::gpu_dispatch) publishes *how many* workgroups the
//! [`Transmittance`](super::gpu_dispatch::HairComputePass::Transmittance) and
//! [`DeepOpacity`](super::gpu_dispatch::HairComputePass::DeepOpacity) passes
//! dispatch; this module publishes *what those two passes bind* — the
//! authoritative element stride, access mode, element count and total byte size
//! of every storage buffer in `hair_transmittance.wesl`'s and
//! `hair_deep_opacity.wesl`'s `@group(0)`. Exactly as
//! [`gpu_buffers`](super::gpu_buffers) does for the guide-`XPBD` sim and
//! [`interp_buffers`](super::interp_buffers) does for the resolve pass (and as
//! water's device-free `WaterBufferPlan` does for its kernels), the sizing lives
//! once here in the zero-dependency crate so the render graph binds against a
//! stable ABI instead of hand-computing strides next to the pipeline.
//!
//! Both passes read a flat, host-binned `samples` pool (`x` = light-space depth,
//! `y` = opacity) plus per-light-texel slice ranges, and write per-texel
//! transmittance curves — the self-shadow half of §3 阶段 6 / §8. The two
//! kernels quantize the light-space slab differently (voxel scatter vs
//! pre-sorted layer sweep) but share the same buffer dimensioning: a flat sample
//! count and a per-texel depth-slice count. Those two extents are not dispatch
//! domains, so they travel together in [`HairShadowExtent`], mirroring how
//! [`gpu_buffers`](super::gpu_buffers) takes `collider_count` and
//! [`interp_buffers`](super::interp_buffers) takes `render_points`.
//!
//! Everything is pure integer arithmetic: byte sizes are clamped up to one
//! element so an empty groom still yields a valid non-empty `WebGPU` storage
//! binding, the depth-slice count is clamped to at least one (matching both
//! kernels' `layer`/`voxel` clamp), products saturate so degenerate inputs never
//! overflow, and nothing panics or divides by zero.

use crate::hair::gpu_buffers::HairBufferAccess;
use crate::hair::gpu_dispatch::HairGpuCounts;

/// Byte stride of a `vec2<f32>` storage element (also `HairTexelRange` =
/// 2 * `u32`): two 4-byte scalars, 8 bytes.
const VEC2_STRIDE: usize = 8;

/// Byte stride of a scalar `f32` storage element.
const F32_STRIDE: usize = 4;

/// The two non-domain extents both self-shadow passes size their buffers
/// against: the flat sample-pool length and the per-texel depth-slice count
/// (`voxel_count` for transmittance, `layer_count` for deep opacity — the same
/// dimension under two kernel-specific names).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct HairShadowExtent {
    /// Total strand samples flattened across every light texel's slice.
    pub sample_count: u32,
    /// Depth slices per texel in the output transmittance curve.
    pub layer_count: u32,
}

impl HairShadowExtent {
    /// The effective depth-slice count, clamped to at least `1` to match both
    /// kernels clamping `voxel_count` / `layer_count` up to one.
    #[must_use]
    pub fn layers(self) -> u32 {
        self.layer_count.max(1)
    }
}

/// One storage buffer bound by the voxel transmittance kernel
/// (`hair_transmittance.wesl` `@group(0)`), in binding order `0..3`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairTransmittanceBuffer {
    /// `@binding(0)` per-texel slice descriptor `array<HairTexelRange>`
    /// (`offset`, `count` as `u32`) into `samples`.
    Ranges,
    /// `@binding(1)` flat strand samples `array<vec2<f32>>` (`x` = depth,
    /// `y` = opacity).
    Samples,
    /// `@binding(2)` output transmittance curve `array<f32>`, `texel *
    /// voxel_count + voxel` layout, written by this pass.
    OutTransmittance,
}

impl HairTransmittanceBuffer {
    /// Every transmittance buffer in `@binding` order. Its length matches
    /// [`HairComputePass::Transmittance`](super::gpu_dispatch::HairComputePass)'s
    /// binding count.
    pub const ALL: [HairTransmittanceBuffer; 3] =
        [Self::Ranges, Self::Samples, Self::OutTransmittance];

    /// The `@group(0)` binding index in `hair_transmittance.wesl`.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::Ranges => 0,
            Self::Samples => 1,
            Self::OutTransmittance => 2,
        }
    }

    /// Byte stride of one element, matching the `WESL` struct / scalar layout.
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            Self::Ranges | Self::Samples => VEC2_STRIDE,
            Self::OutTransmittance => F32_STRIDE,
        }
    }

    /// Whether the kernel reads or read-writes this buffer.
    #[must_use]
    pub fn access(self) -> HairBufferAccess {
        match self {
            Self::Ranges | Self::Samples => HairBufferAccess::Read,
            Self::OutTransmittance => HairBufferAccess::ReadWrite,
        }
    }

    /// Whether this pass writes the buffer (the self-shadow output) rather than
    /// only reading it.
    #[must_use]
    pub fn is_output(self) -> bool {
        matches!(self.access(), HairBufferAccess::ReadWrite)
    }

    /// Element count for a groom with `counts` domain totals and `extent`
    /// shadow extents: `ranges` is one per light texel, `samples` the flat pool,
    /// and `out_transmittance` is `light_texels * layers` (saturating).
    #[must_use]
    pub fn element_count(self, counts: &HairGpuCounts, extent: HairShadowExtent) -> u32 {
        match self {
            Self::Ranges => counts.light_texels,
            Self::Samples => extent.sample_count,
            Self::OutTransmittance => counts.light_texels.saturating_mul(extent.layers()),
        }
    }

    /// Total byte size, clamped up to one element so an empty groom still yields
    /// a valid non-empty `WebGPU` storage binding.
    #[must_use]
    pub fn byte_size(self, counts: &HairGpuCounts, extent: HairShadowExtent) -> usize {
        let elements = self.element_count(counts, extent).max(1) as usize;
        elements * self.stride()
    }
}

/// One storage buffer bound by the deep-opacity packing kernel
/// (`hair_deep_opacity.wesl` `@group(0)`), in binding order `0..5`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairDeepOpacityBuffer {
    /// `@binding(0)` flat pre-sorted strand samples `array<vec2<f32>>`
    /// (`x` = depth ascending per texel, `y` = opacity).
    Samples,
    /// `@binding(1)` per-texel compacted slice descriptor `array<HairTexelRange>`
    /// (`start`, `count` as `u32`) into `samples`.
    TexelRanges,
    /// `@binding(2)` per-texel front depth of the first layer `array<f32>`,
    /// written by this pass.
    NearDepth,
    /// `@binding(3)` per-texel uniform layer width `array<f32>`, written by this
    /// pass.
    LayerStep,
    /// `@binding(4)` flat texel-major transmittance slab `array<f32>`,
    /// `texel * layer_count + layer` layout, written by this pass.
    Transmittance,
}

impl HairDeepOpacityBuffer {
    /// Every deep-opacity buffer in `@binding` order. Its length matches
    /// [`HairComputePass::DeepOpacity`](super::gpu_dispatch::HairComputePass)'s
    /// binding count.
    pub const ALL: [HairDeepOpacityBuffer; 5] = [
        Self::Samples,
        Self::TexelRanges,
        Self::NearDepth,
        Self::LayerStep,
        Self::Transmittance,
    ];

    /// The `@group(0)` binding index in `hair_deep_opacity.wesl`.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::Samples => 0,
            Self::TexelRanges => 1,
            Self::NearDepth => 2,
            Self::LayerStep => 3,
            Self::Transmittance => 4,
        }
    }

    /// Byte stride of one element, matching the `WESL` struct / scalar layout.
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            Self::Samples | Self::TexelRanges => VEC2_STRIDE,
            Self::NearDepth | Self::LayerStep | Self::Transmittance => F32_STRIDE,
        }
    }

    /// Whether the kernel reads or read-writes this buffer. The sample pool and
    /// texel ranges are read-only inputs; the near depth, layer step and
    /// transmittance slab are written.
    #[must_use]
    pub fn access(self) -> HairBufferAccess {
        match self {
            Self::Samples | Self::TexelRanges => HairBufferAccess::Read,
            Self::NearDepth | Self::LayerStep | Self::Transmittance => HairBufferAccess::ReadWrite,
        }
    }

    /// Whether this pass writes the buffer rather than only reading it.
    #[must_use]
    pub fn is_output(self) -> bool {
        matches!(self.access(), HairBufferAccess::ReadWrite)
    }

    /// Element count for a groom with `counts` domain totals and `extent`
    /// shadow extents: `samples` is the flat pool, `texel_ranges`/`near_depth`/
    /// `layer_step` are one per light texel, and `transmittance` is
    /// `light_texels * layers` (saturating).
    #[must_use]
    pub fn element_count(self, counts: &HairGpuCounts, extent: HairShadowExtent) -> u32 {
        match self {
            Self::Samples => extent.sample_count,
            Self::TexelRanges | Self::NearDepth | Self::LayerStep => counts.light_texels,
            Self::Transmittance => counts.light_texels.saturating_mul(extent.layers()),
        }
    }

    /// Total byte size, clamped up to one element so an empty groom still yields
    /// a valid non-empty `WebGPU` storage binding.
    #[must_use]
    pub fn byte_size(self, counts: &HairGpuCounts, extent: HairShadowExtent) -> usize {
        let elements = self.element_count(counts, extent).max(1) as usize;
        elements * self.stride()
    }
}

/// Total bytes both self-shadow passes write per frame (transmittance curves
/// plus deep-opacity near/step/slab), for capacity planning of the shared
/// self-shadow output allocation.
#[must_use]
pub fn shadow_output_bytes(counts: &HairGpuCounts, extent: HairShadowExtent) -> usize {
    let transmittance: usize = HairTransmittanceBuffer::ALL
        .into_iter()
        .filter(|buffer| buffer.is_output())
        .map(|buffer| buffer.byte_size(counts, extent))
        .sum();
    let deep: usize = HairDeepOpacityBuffer::ALL
        .into_iter()
        .filter(|buffer| buffer.is_output())
        .map(|buffer| buffer.byte_size(counts, extent))
        .sum();
    transmittance + deep
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::gpu_dispatch::HairComputePass;

    fn sample_counts() -> HairGpuCounts {
        HairGpuCounts {
            roots: 10,
            guide_strands: 100,
            guide_particles: 3200,
            render_strands: 50_000,
            light_texels: 4096,
        }
    }

    fn sample_extent() -> HairShadowExtent {
        HairShadowExtent {
            sample_count: 200_000,
            layer_count: 8,
        }
    }

    #[test]
    fn transmittance_bindings_are_dense_and_ordered() {
        for (index, buffer) in HairTransmittanceBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
        }
    }

    #[test]
    fn deep_opacity_bindings_are_dense_and_ordered() {
        for (index, buffer) in HairDeepOpacityBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
        }
    }

    #[test]
    fn buffer_sets_match_the_dispatch_binding_counts() {
        assert_eq!(
            HairTransmittanceBuffer::ALL.len() as u32,
            HairComputePass::Transmittance.binding_count()
        );
        assert_eq!(
            HairDeepOpacityBuffer::ALL.len() as u32,
            HairComputePass::DeepOpacity.binding_count()
        );
    }

    #[test]
    fn strides_match_the_wesl_struct_layout() {
        assert_eq!(HairTransmittanceBuffer::Ranges.stride(), 8);
        assert_eq!(HairTransmittanceBuffer::Samples.stride(), 8);
        assert_eq!(HairTransmittanceBuffer::OutTransmittance.stride(), 4);
        assert_eq!(HairDeepOpacityBuffer::Samples.stride(), 8);
        assert_eq!(HairDeepOpacityBuffer::TexelRanges.stride(), 8);
        assert_eq!(HairDeepOpacityBuffer::NearDepth.stride(), 4);
        assert_eq!(HairDeepOpacityBuffer::LayerStep.stride(), 4);
        assert_eq!(HairDeepOpacityBuffer::Transmittance.stride(), 4);
    }

    #[test]
    fn access_modes_match_the_kernels() {
        assert_eq!(
            HairTransmittanceBuffer::Ranges.access(),
            HairBufferAccess::Read
        );
        assert_eq!(
            HairTransmittanceBuffer::Samples.access(),
            HairBufferAccess::Read
        );
        assert_eq!(
            HairTransmittanceBuffer::OutTransmittance.access(),
            HairBufferAccess::ReadWrite
        );
        assert_eq!(
            HairDeepOpacityBuffer::Samples.access(),
            HairBufferAccess::Read
        );
        assert_eq!(
            HairDeepOpacityBuffer::TexelRanges.access(),
            HairBufferAccess::Read
        );
        assert_eq!(
            HairDeepOpacityBuffer::NearDepth.access(),
            HairBufferAccess::ReadWrite
        );
        assert_eq!(
            HairDeepOpacityBuffer::LayerStep.access(),
            HairBufferAccess::ReadWrite
        );
        assert_eq!(
            HairDeepOpacityBuffer::Transmittance.access(),
            HairBufferAccess::ReadWrite
        );
    }

    #[test]
    fn outputs_are_the_written_curves() {
        assert!(!HairTransmittanceBuffer::Ranges.is_output());
        assert!(!HairTransmittanceBuffer::Samples.is_output());
        assert!(HairTransmittanceBuffer::OutTransmittance.is_output());
        assert!(!HairDeepOpacityBuffer::Samples.is_output());
        assert!(!HairDeepOpacityBuffer::TexelRanges.is_output());
        assert!(HairDeepOpacityBuffer::NearDepth.is_output());
        assert!(HairDeepOpacityBuffer::LayerStep.is_output());
        assert!(HairDeepOpacityBuffer::Transmittance.is_output());
    }

    #[test]
    fn transmittance_element_counts_follow_domains_and_extent() {
        let counts = sample_counts();
        let extent = sample_extent();
        assert_eq!(
            HairTransmittanceBuffer::Ranges.element_count(&counts, extent),
            4096
        );
        assert_eq!(
            HairTransmittanceBuffer::Samples.element_count(&counts, extent),
            200_000
        );
        assert_eq!(
            HairTransmittanceBuffer::OutTransmittance.element_count(&counts, extent),
            4096 * 8
        );
    }

    #[test]
    fn deep_opacity_element_counts_follow_domains_and_extent() {
        let counts = sample_counts();
        let extent = sample_extent();
        assert_eq!(
            HairDeepOpacityBuffer::Samples.element_count(&counts, extent),
            200_000
        );
        assert_eq!(
            HairDeepOpacityBuffer::TexelRanges.element_count(&counts, extent),
            4096
        );
        assert_eq!(
            HairDeepOpacityBuffer::NearDepth.element_count(&counts, extent),
            4096
        );
        assert_eq!(
            HairDeepOpacityBuffer::LayerStep.element_count(&counts, extent),
            4096
        );
        assert_eq!(
            HairDeepOpacityBuffer::Transmittance.element_count(&counts, extent),
            4096 * 8
        );
    }

    #[test]
    fn layer_count_clamps_to_at_least_one() {
        let counts = sample_counts();
        let extent = HairShadowExtent {
            sample_count: 0,
            layer_count: 0,
        };
        assert_eq!(extent.layers(), 1);
        // A zero layer_count still produces one slice per texel, not zero.
        assert_eq!(
            HairTransmittanceBuffer::OutTransmittance.element_count(&counts, extent),
            4096
        );
        assert_eq!(
            HairDeepOpacityBuffer::Transmittance.element_count(&counts, extent),
            4096
        );
    }

    #[test]
    fn byte_sizes_multiply_count_by_stride() {
        let counts = sample_counts();
        let extent = sample_extent();
        assert_eq!(
            HairTransmittanceBuffer::OutTransmittance.byte_size(&counts, extent),
            4096 * 8 * 4
        );
        assert_eq!(
            HairDeepOpacityBuffer::Samples.byte_size(&counts, extent),
            200_000 * 8
        );
        assert_eq!(
            HairDeepOpacityBuffer::Transmittance.byte_size(&counts, extent),
            4096 * 8 * 4
        );
    }

    #[test]
    fn empty_groom_clamps_every_buffer_to_one_element() {
        let counts = HairGpuCounts::default();
        let extent = HairShadowExtent::default();
        for buffer in HairTransmittanceBuffer::ALL {
            assert_eq!(buffer.byte_size(&counts, extent), buffer.stride());
        }
        for buffer in HairDeepOpacityBuffer::ALL {
            assert_eq!(buffer.byte_size(&counts, extent), buffer.stride());
        }
    }

    #[test]
    fn shadow_output_bytes_sums_the_written_curves() {
        let counts = sample_counts();
        let extent = sample_extent();
        // transmittance: out_transmittance; deep: near + step + slab.
        let expected = (4096 * 8 * 4) + (4096 * 4) + (4096 * 4) + (4096 * 8 * 4);
        assert_eq!(shadow_output_bytes(&counts, extent), expected);
    }
}
