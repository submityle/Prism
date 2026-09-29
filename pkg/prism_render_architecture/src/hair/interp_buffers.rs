//! Device-free byte-layout contract for the hair guide-to-render interpolation
//! `GPU` buffers.
//!
//! [`gpu_dispatch`](super::gpu_dispatch) publishes *how many* workgroups the
//! [`Interpolate`](super::gpu_dispatch::HairComputePass::Interpolate) pass
//! dispatches; this module publishes *what that pass binds* — the authoritative
//! element stride, access mode, element count and total byte size of every
//! storage buffer in `hair_interp.wesl`'s `@group(0)`. Exactly as
//! [`gpu_buffers`](super::gpu_buffers) does for the guide-`XPBD` sim (and as
//! water's device-free `WaterBufferPlan` does for its kernels), the sizing lives
//! once here in the zero-dependency crate so the render graph binds against a
//! stable ABI instead of hand-computing strides next to the pipeline.
//!
//! This is the *resolve* half of the §8 GPU-driven pipeline: the interpolation
//! pass reads the freshly solved guide control points plus the per-render-strand
//! binding table and writes the flat render-strand control points that feed
//! `gpu_scene`. `guide_points`, `guide_ranges` and `bindings` are read-only
//! inputs (guides produced by the sim pass, the binding table by import); only
//! `out_points` is written, and the render graph double-buffers *it* (not any
//! input) when it hands the previous frame's render geometry downstream while
//! the next resolve runs.
//!
//! Everything is pure integer arithmetic: byte sizes are clamped up to one
//! element so an empty groom still yields a valid non-empty `WebGPU` storage
//! binding, and nothing panics or divides by zero.

use crate::hair::gpu_buffers::HairBufferAccess;
use crate::hair::gpu_dispatch::HairGpuCounts;

/// Byte stride of a `vec4<f32>` storage element (also `vec4<u32>`): four 4-byte
/// scalars, the natural 16-byte stride.
const VEC4_STRIDE: usize = 16;

/// `std430` array stride of `HairGuideRange` (`offset: u32`, `count: u32`): two
/// tightly packed 4-byte scalars, 8 bytes.
const GUIDE_RANGE_STRIDE: usize = 8;

/// `std430` array stride of `HairRenderBinding`: `guides: vec4<u32>` (16) +
/// `weights: vec4<f32>` (16) + `root_uv: vec2<f32>` (8) + `seed: u32` (4) +
/// `out_offset: u32` (4) = 48 bytes, already a multiple of the 16-byte struct
/// alignment so no tail padding is added.
const RENDER_BINDING_STRIDE: usize = 48;

/// One storage buffer bound by the guide-to-render interpolation kernel
/// (`hair_interp.wesl` `@group(0)`), in binding order `0..4`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HairInterpBuffer {
    /// `@binding(0)` flat pool of every guide control point `array<vec4<f32>>`
    /// (`xyz` used, root first).
    GuidePoints,
    /// `@binding(1)` per-guide slice descriptor `array<HairGuideRange>`
    /// (`offset`, `count` as `u32`).
    GuideRanges,
    /// `@binding(2)` per-render-strand binding table `array<HairRenderBinding>`
    /// (guide indices + weights + root UV + seed + output offset).
    Bindings,
    /// `@binding(3)` flat render-strand control points `array<vec4<f32>>`
    /// (`xyz` = position, `w` = `1.0`), written by this pass.
    OutPoints,
}

impl HairInterpBuffer {
    /// Every interpolation buffer in `@binding` order. Its length matches
    /// [`HairComputePass::Interpolate`](super::gpu_dispatch::HairComputePass)'s
    /// binding count, keeping this layout in lock-step with the dispatch ABI.
    pub const ALL: [HairInterpBuffer; 4] = [
        Self::GuidePoints,
        Self::GuideRanges,
        Self::Bindings,
        Self::OutPoints,
    ];

    /// The `@group(0)` binding index this buffer occupies in `hair_interp.wesl`.
    #[must_use]
    pub fn binding(self) -> u32 {
        match self {
            Self::GuidePoints => 0,
            Self::GuideRanges => 1,
            Self::Bindings => 2,
            Self::OutPoints => 3,
        }
    }

    /// Byte stride of one element, matching the `WESL` struct / scalar layout.
    #[must_use]
    pub fn stride(self) -> usize {
        match self {
            Self::GuidePoints | Self::OutPoints => VEC4_STRIDE,
            Self::GuideRanges => GUIDE_RANGE_STRIDE,
            Self::Bindings => RENDER_BINDING_STRIDE,
        }
    }

    /// Whether the kernel reads or read-writes this buffer. Only `out_points` is
    /// written; the guide points, ranges and binding table are read-only inputs.
    #[must_use]
    pub fn access(self) -> HairBufferAccess {
        match self {
            Self::GuidePoints | Self::GuideRanges | Self::Bindings => HairBufferAccess::Read,
            Self::OutPoints => HairBufferAccess::ReadWrite,
        }
    }

    /// Whether this pass writes the buffer (the resolve output the render graph
    /// double-buffers) rather than only reading it.
    #[must_use]
    pub fn is_output(self) -> bool {
        matches!(self.access(), HairBufferAccess::ReadWrite)
    }

    /// Number of elements this buffer holds for a groom with `counts` domain
    /// totals and `render_points` total render-strand control points.
    ///
    /// `guide_points` is one entry per guide particle, `guide_ranges` one per
    /// guide strand, `bindings` one per render strand, and `out_points` one per
    /// render-strand control point (`render_points`, which is not itself a
    /// dispatch domain, so it is supplied by the caller — mirroring how
    /// [`gpu_buffers`](super::gpu_buffers) takes `collider_count`).
    #[must_use]
    pub fn element_count(self, counts: &HairGpuCounts, render_points: u32) -> u32 {
        match self {
            Self::GuidePoints => counts.guide_particles,
            Self::GuideRanges => counts.guide_strands,
            Self::Bindings => counts.render_strands,
            Self::OutPoints => render_points,
        }
    }

    /// Total byte size of this buffer, clamped up to one element so an empty
    /// groom still yields a valid non-empty `WebGPU` storage binding.
    #[must_use]
    pub fn byte_size(self, counts: &HairGpuCounts, render_points: u32) -> usize {
        let elements = self.element_count(counts, render_points).max(1) as usize;
        elements * self.stride()
    }
}

/// Total bytes of the read-only inputs the interpolation pass consumes
/// (`guide_points`, `guide_ranges`, `bindings`), refreshed by the sim / import
/// passes each frame.
#[must_use]
pub fn input_bytes(counts: &HairGpuCounts, render_points: u32) -> usize {
    HairInterpBuffer::ALL
        .into_iter()
        .filter(|buffer| !buffer.is_output())
        .map(|buffer| buffer.byte_size(counts, render_points))
        .sum()
}

/// Total bytes of the render-geometry output the pass writes (`out_points`),
/// which the render graph double-buffers into `gpu_scene`.
#[must_use]
pub fn output_bytes(counts: &HairGpuCounts, render_points: u32) -> usize {
    HairInterpBuffer::ALL
        .into_iter()
        .filter(|buffer| buffer.is_output())
        .map(|buffer| buffer.byte_size(counts, render_points))
        .sum()
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
            light_texels: 0,
        }
    }

    #[test]
    fn bindings_are_dense_and_ordered() {
        for (index, buffer) in HairInterpBuffer::ALL.into_iter().enumerate() {
            assert_eq!(buffer.binding() as usize, index);
        }
    }

    #[test]
    fn buffer_set_matches_the_dispatch_binding_count() {
        assert_eq!(
            HairInterpBuffer::ALL.len() as u32,
            HairComputePass::Interpolate.binding_count()
        );
    }

    #[test]
    fn strides_match_the_wesl_struct_layout() {
        assert_eq!(HairInterpBuffer::GuidePoints.stride(), 16);
        assert_eq!(HairInterpBuffer::GuideRanges.stride(), 8);
        assert_eq!(HairInterpBuffer::Bindings.stride(), 48);
        assert_eq!(HairInterpBuffer::OutPoints.stride(), 16);
    }

    #[test]
    fn access_modes_match_the_kernel() {
        assert_eq!(
            HairInterpBuffer::GuidePoints.access(),
            HairBufferAccess::Read
        );
        assert_eq!(
            HairInterpBuffer::GuideRanges.access(),
            HairBufferAccess::Read
        );
        assert_eq!(HairInterpBuffer::Bindings.access(), HairBufferAccess::Read);
        assert_eq!(
            HairInterpBuffer::OutPoints.access(),
            HairBufferAccess::ReadWrite
        );
    }

    #[test]
    fn only_out_points_is_written() {
        assert!(!HairInterpBuffer::GuidePoints.is_output());
        assert!(!HairInterpBuffer::GuideRanges.is_output());
        assert!(!HairInterpBuffer::Bindings.is_output());
        assert!(HairInterpBuffer::OutPoints.is_output());
    }

    #[test]
    fn element_counts_follow_the_groom_domains() {
        let counts = sample_counts();
        assert_eq!(
            HairInterpBuffer::GuidePoints.element_count(&counts, 600_000),
            3200
        );
        assert_eq!(
            HairInterpBuffer::GuideRanges.element_count(&counts, 600_000),
            100
        );
        assert_eq!(
            HairInterpBuffer::Bindings.element_count(&counts, 600_000),
            50_000
        );
        assert_eq!(
            HairInterpBuffer::OutPoints.element_count(&counts, 600_000),
            600_000
        );
    }

    #[test]
    fn byte_sizes_multiply_count_by_stride() {
        let counts = sample_counts();
        assert_eq!(
            HairInterpBuffer::GuidePoints.byte_size(&counts, 600_000),
            3200 * 16
        );
        assert_eq!(
            HairInterpBuffer::GuideRanges.byte_size(&counts, 600_000),
            100 * 8
        );
        assert_eq!(
            HairInterpBuffer::Bindings.byte_size(&counts, 600_000),
            50_000 * 48
        );
        assert_eq!(
            HairInterpBuffer::OutPoints.byte_size(&counts, 600_000),
            600_000 * 16
        );
    }

    #[test]
    fn empty_groom_clamps_every_buffer_to_one_element() {
        let counts = HairGpuCounts::default();
        for buffer in HairInterpBuffer::ALL {
            assert_eq!(buffer.byte_size(&counts, 0), buffer.stride());
        }
    }

    #[test]
    fn input_and_output_bytes_partition_the_buffers() {
        let counts = sample_counts();
        let expected_input = 3200 * 16 + 100 * 8 + 50_000 * 48;
        assert_eq!(input_bytes(&counts, 600_000), expected_input);
        assert_eq!(output_bytes(&counts, 600_000), 600_000 * 16);
    }
}
