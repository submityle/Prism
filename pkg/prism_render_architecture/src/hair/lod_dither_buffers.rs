//! Device-free byte-layout contract for the hair continuous-`LOD` dither
//! (`LodDither`) `GPU` pass.
//!
//! [`gpu_dispatch`](super::gpu_dispatch) publishes *how many* workgroups the
//! [`LodDither`](super::gpu_dispatch::HairComputePass::LodDither) pass dispatches
//! over the render-strand domain; this module publishes *what it binds* — the
//! authoritative element stride, access mode, element count and total byte size
//! of the single storage buffer in `hair_lod_dither.wesl`'s `@group(0)`. As with
//! [`gpu_buffers`](super::gpu_buffers) (sim),
//! [`interp_buffers`](super::interp_buffers) (resolve),
//! [`shadow_buffers`](super::shadow_buffers) (self-shadow),
//! [`import_buffers`](super::import_buffers) (import) and
//! [`sim_pass_buffers`](super::sim_pass_buffers) (per-frame simulate), the sizing
//! lives once here in the zero-dependency crate so the render graph binds
//! against a stable ABI instead of hand-computing strides next to the pipeline.
//!
//! This is the `Resolve`-stage companion to
//! [`interp_buffers`](super::interp_buffers) (design §3 阶段 7 / §7 连续 LOD):
//! `LodDither` stochastically decides, per render strand, whether to keep the
//! finer `LOD` tier this frame, writing a per-strand `u32` keep mask that the
//! rasterizer honours to fade strand counts across tier boundaries without
//! popping. With this pass the import→sim→resolve→shadow `GPU` buffer contract
//! chain is complete.
//!
//! Everything is pure integer arithmetic: the byte size is clamped up to one
//! element so an empty groom still yields a valid non-empty `WebGPU` storage
//! binding, and nothing panics or divides by zero.

use crate::hair::gpu_buffers::HairBufferAccess;
use crate::hair::gpu_dispatch::HairGpuCounts;

/// Byte stride of a scalar `u32` storage element (per-strand keep mask).
const U32_STRIDE: usize = 4;

/// The single storage buffer bound by the `LOD`-dither kernel
/// (`hair_lod_dither.wesl` `@group(0)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairLodDitherBuffer {
    /// `@binding(0)` per-render-strand keep mask `array<u32>` (`1` keeps the
    /// finer tier this frame, `0` drops it), written by this pass.
    OutKeep,
}

impl HairLodDitherBuffer {
    /// Every `LOD`-dither buffer in `@binding` order. Its length matches
    /// [`HairComputePass::LodDither`](super::gpu_dispatch::HairComputePass)'s
    /// binding count.
    pub const ALL: [HairLodDitherBuffer; 1] = [Self::OutKeep];

    /// The `@group(0)` binding index in `hair_lod_dither.wesl`.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::OutKeep => 0,
        }
    }

    /// Byte stride of one element (scalar `u32`).
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            Self::OutKeep => U32_STRIDE,
        }
    }

    /// Whether the kernel reads or read-writes this buffer.
    #[must_use]
    pub fn access(self) -> HairBufferAccess {
        match self {
            Self::OutKeep => HairBufferAccess::ReadWrite,
        }
    }

    /// Whether this pass writes the buffer rather than only reading it.
    #[must_use]
    pub fn is_output(self) -> bool {
        matches!(self.access(), HairBufferAccess::ReadWrite)
    }

    /// Element count: one keep flag per render strand (the dispatch domain).
    #[must_use]
    pub fn element_count(self, counts: &HairGpuCounts) -> u32 {
        match self {
            Self::OutKeep => counts.render_strands,
        }
    }

    /// Total byte size, clamped up to one element so an empty groom still yields
    /// a valid non-empty `WebGPU` storage binding.
    #[must_use]
    pub fn byte_size(self, counts: &HairGpuCounts) -> usize {
        let elements = self.element_count(counts).max(1) as usize;
        elements * self.stride()
    }
}

/// Bytes the `LodDither` pass writes each frame: the per-render-strand keep
/// mask, the only buffer this pass owns.
#[must_use]
pub fn lod_dither_output_bytes(counts: &HairGpuCounts) -> usize {
    HairLodDitherBuffer::ALL
        .into_iter()
        .filter(|buffer| buffer.is_output())
        .map(|buffer| buffer.byte_size(counts))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::gpu_dispatch::HairComputePass;

    fn sample_counts() -> HairGpuCounts {
        HairGpuCounts {
            roots: 100,
            guide_strands: 100,
            guide_particles: 3200,
            render_strands: 50_000,
            light_texels: 0,
        }
    }

    #[test]
    fn binding_is_dense_and_ordered() {
        for (index, buffer) in HairLodDitherBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
        }
    }

    #[test]
    fn buffer_set_matches_the_dispatch_binding_count() {
        assert_eq!(
            HairLodDitherBuffer::ALL.len() as u32,
            HairComputePass::LodDither.binding_count()
        );
    }

    #[test]
    fn stride_matches_the_wesl_scalar_layout() {
        assert_eq!(HairLodDitherBuffer::OutKeep.stride(), 4);
    }

    #[test]
    fn access_and_output_match_the_kernel() {
        assert_eq!(
            HairLodDitherBuffer::OutKeep.access(),
            HairBufferAccess::ReadWrite
        );
        assert!(HairLodDitherBuffer::OutKeep.is_output());
    }

    #[test]
    fn element_count_follows_the_render_strand_domain() {
        let counts = sample_counts();
        assert_eq!(HairLodDitherBuffer::OutKeep.element_count(&counts), 50_000);
    }

    #[test]
    fn byte_size_multiplies_count_by_stride() {
        let counts = sample_counts();
        assert_eq!(HairLodDitherBuffer::OutKeep.byte_size(&counts), 50_000 * 4);
    }

    #[test]
    fn empty_groom_clamps_to_one_element() {
        let counts = HairGpuCounts::default();
        for buffer in HairLodDitherBuffer::ALL {
            assert_eq!(buffer.byte_size(&counts), buffer.stride());
        }
    }

    #[test]
    fn lod_dither_output_bytes_sums_the_keep_mask() {
        let counts = sample_counts();
        assert_eq!(lod_dither_output_bytes(&counts), 50_000 * 4);
    }
}
