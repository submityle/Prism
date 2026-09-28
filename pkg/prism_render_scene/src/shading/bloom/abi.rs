//! ABI shared between the bloom compute passes and `shaders/bloom.wesl`.
//!
//! Bloom runs as a chain of compute passes over the resolved, pre-exposed HDR
//! `scene_color`: a full-res copy into a base target, a Karis prefilter +
//! partial-Karis first downsample, a stack of energy-preserving COD 13-tap
//! downsamples building the mip pyramid, a progressive 3x3 tent upsample
//! accumulating the halo back up, and a final combine that blends the
//! accumulated bloom over the scene by the artist intensity.
//!
//! Every pass consumes the single [`GpuBloomParams`] immediate (push-constant)
//! block below, mirroring the shader's `BloomParams` byte-for-byte so machines
//! with and without a GPU agree with the CPU golden in
//! [`prism_render_shading::bloom`].

use bevy_math::UVec2;
use bytemuck::{Pod, Zeroable};
use prism_render_shading::BloomParams;

/// Compute workgroup edge every bloom pass dispatches in, matching the
/// `@workgroup_size(8, 8)` in `bloom.wesl`. One invocation per destination
/// texel, one workgroup per 8x8 tile.
pub(crate) const BLOOM_WORKGROUP_SIZE: u32 = 8;

/// Maximum bloom pyramid depth. Six mips reach a wide, stable halo on 4K while
/// keeping the pass count (and the smallest mip) bounded; the actual depth is
/// clamped to what the view extent supports (a mip stops before any dimension
/// would collapse below one texel).
pub(crate) const BLOOM_MAX_MIPS: u32 = 6;

/// Immediate block consumed by every entry point in `bloom.wesl`.
///
/// Mirrors the shader's `BloomParams`: one over the *source* texture extent
/// (driving the sampling offsets), the artist threshold/knee/intensity/radius,
/// then the *destination* extent for the per-pixel coverage guard. Six `f32`s
/// plus two `u32`s fill exactly 32 bytes, a multiple of the 16-byte immediate
/// alignment with no implicit padding.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuBloomParams {
    /// One over the source texture width (sampling-offset scale).
    pub inv_src_width: f32,
    /// One over the source texture height (sampling-offset scale).
    pub inv_src_height: f32,
    /// Luminance/brightness above which pixels begin to bloom.
    pub threshold: f32,
    /// Soft-threshold knee width (`>= 0`; `0` is a hard threshold).
    pub knee: f32,
    /// Blend weight of the accumulated bloom over the scene in the combine.
    pub intensity: f32,
    /// Tent-upsample spread radius (`0..=1`).
    pub radius: f32,
    /// Destination width in texels (coverage guard).
    pub dst_width: u32,
    /// Destination height in texels (coverage guard).
    pub dst_height: u32,
}

impl GpuBloomParams {
    /// Builds the immediate for one pass from the source/destination extents
    /// and the artist bloom controls. `src_size` scales the sampling offsets;
    /// `dst_size` bounds the coverage guard.
    pub(crate) fn new(src_size: UVec2, dst_size: UVec2, bloom: BloomParams) -> Self {
        Self {
            inv_src_width: if src_size.x == 0 { 0.0 } else { 1.0 / src_size.x as f32 },
            inv_src_height: if src_size.y == 0 { 0.0 } else { 1.0 / src_size.y as f32 },
            threshold: bloom.threshold,
            knee: bloom.knee,
            intensity: bloom.intensity,
            radius: bloom.radius,
            dst_width: dst_size.x,
            dst_height: dst_size.y,
        }
    }
}

/// Bloom pyramid mip extents for one view: the full-res base, then the
/// half-res-and-down mip chain. The chain is at most [`BLOOM_MAX_MIPS`] deep and
/// stops before any dimension collapses below one texel.
///
/// Returned by [`bloom_mip_sizes`] and consumed by both the texture allocation
/// and the pass node (which derives each dispatch's source/destination extent
/// and workgroup count from it).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct BloomMipChain {
    /// Full framebuffer extent (the base copy + combine destination).
    pub full: UVec2,
    /// Half-res-and-down mip extents, `mips[0]` being half of `full`.
    pub mips: Vec<UVec2>,
}

impl BloomMipChain {
    /// Number of mip levels in the pyramid (excluding the full-res base).
    pub(crate) fn level_count(&self) -> usize {
        self.mips.len()
    }
}

/// Computes the bloom mip chain for a framebuffer extent: half-res `mips[0]`,
/// each subsequent level halved (rounding up so a full row/column is never
/// dropped), stopping at [`BLOOM_MAX_MIPS`] levels or before any dimension
/// would fall below one texel. Returns an empty chain for a degenerate extent.
pub(crate) fn bloom_mip_sizes(full: UVec2) -> BloomMipChain {
    let mut mips = Vec::new();
    if full.x == 0 || full.y == 0 {
        return BloomMipChain { full, mips };
    }
    let mut size = full;
    for _ in 0..BLOOM_MAX_MIPS {
        let next = UVec2::new((size.x / 2).max(1), (size.y / 2).max(1));
        // Stop once halving no longer shrinks either axis (a 1xN / Nx1 tail),
        // so the pyramid never stalls on a same-size mip.
        if next == size {
            break;
        }
        mips.push(next);
        size = next;
        if size.x == 1 || size.y == 1 {
            break;
        }
    }
    BloomMipChain { full, mips }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn params_match_the_shader_immediate_layout() {
        // Six f32 + two u32 fill exactly the 32-byte immediate block.
        assert_eq!(size_of::<GpuBloomParams>(), 32);
        assert_eq!(align_of::<GpuBloomParams>(), 4);
    }

    #[test]
    fn params_fold_in_extents_and_controls() {
        let bloom = BloomParams {
            threshold: 1.0,
            knee: 0.5,
            intensity: 0.04,
            radius: 1.0,
        };
        let params = GpuBloomParams::new(UVec2::new(1920, 1080), UVec2::new(960, 540), bloom);
        assert!((params.inv_src_width - 1.0 / 1920.0).abs() < 1.0e-9);
        assert!((params.inv_src_height - 1.0 / 1080.0).abs() < 1.0e-9);
        assert_eq!(params.threshold, 1.0);
        assert_eq!(params.knee, 0.5);
        assert_eq!(params.intensity, 0.04);
        assert_eq!(params.radius, 1.0);
        assert_eq!(params.dst_width, 960);
        assert_eq!(params.dst_height, 540);
    }

    #[test]
    fn zero_extent_params_are_finite() {
        let params = GpuBloomParams::new(UVec2::ZERO, UVec2::ZERO, BloomParams::default());
        assert_eq!(params.inv_src_width, 0.0);
        assert_eq!(params.inv_src_height, 0.0);
    }

    #[test]
    fn mip_chain_halves_until_the_cap() {
        let chain = bloom_mip_sizes(UVec2::new(1920, 1080));
        assert_eq!(chain.full, UVec2::new(1920, 1080));
        assert_eq!(chain.level_count(), BLOOM_MAX_MIPS as usize);
        assert_eq!(chain.mips[0], UVec2::new(960, 540));
        assert_eq!(chain.mips[1], UVec2::new(480, 270));
    }

    #[test]
    fn mip_chain_stops_before_collapsing_a_dimension() {
        // A short axis collapses to 1 quickly; the chain must not stall or emit
        // a zero-sized mip.
        let chain = bloom_mip_sizes(UVec2::new(64, 4));
        assert!(chain.mips.iter().all(|m| m.x >= 1 && m.y >= 1));
        assert!(chain.level_count() <= BLOOM_MAX_MIPS as usize);
        let last = chain.mips.last().copied().unwrap();
        assert!(last.x == 1 || last.y == 1);
    }

    #[test]
    fn degenerate_extent_yields_no_mips() {
        assert_eq!(bloom_mip_sizes(UVec2::ZERO).level_count(), 0);
        assert_eq!(bloom_mip_sizes(UVec2::new(0, 720)).level_count(), 0);
    }
}
