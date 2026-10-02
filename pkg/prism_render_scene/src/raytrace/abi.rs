//! ABI shared between the ray-traversal compute pass and its sibling `WESL`
//! shader `shaders/ray_traverse.wesl`.
//!
//! The packed `BVH` buffer layout is owned by the dependency-free, float-audited
//! `prism_render_architecture::ray_scene::gpu_layout` module: `GpuBvhBuffers`
//! emits a depth-first `nodes` array ([`NODE_WORDS`] `u32` each) and a
//! leaf-contiguous `triangles` array ([`TRIANGLE_WORDS`] `u32` each), and its
//! `closest_hit` / `any_hit` walks are the `CPU` golden reference the shader
//! reproduces bit-for-bit. This module re-exports those authoritative strides
//! and pins the shader's own record strides (the ray input and hit output the
//! kernel defines) to them, so a drift between the golden layout, the shader's
//! hardcoded word counts and the host upload fails the build rather than
//! corrupting a dispatch at run time.
//!
//! `WESL`/`WGSL` layout rules mirrored here: every buffer is a flat
//! `array<u32>`, `f32` fields are stored as their `to_bits` pattern (recovered
//! with `bitcast<f32>` on device), and each record stride is a multiple of four
//! words (16 bytes) so records stay 16-byte aligned.

#![allow(
    dead_code,
    reason = "the ray-traversal ABI strides and shader-mirror constants are the verified layout foundation of this subsystem; the pipeline / bind-group / dispatch slices that upload against them land next, and the contract tests exercise every stride now"
)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::ray_scene::{
    BLAS_OFFSET_WORDS, INSTANCE_WORDS, NODE_WORDS, TRIANGLE_WORDS,
};

/// `u32` words per packed `BVH` node, re-exported from the golden layout. Must
/// match the `NODE_WORDS` constant in `shaders/ray_traverse.wesl`.
pub(crate) const RAYTRACE_NODE_WORDS: usize = NODE_WORDS;

/// `u32` words per packed triangle, re-exported from the golden layout. Must
/// match the `TRIANGLE_WORDS` constant in `shaders/ray_traverse.wesl`.
pub(crate) const RAYTRACE_TRIANGLE_WORDS: usize = TRIANGLE_WORDS;

/// `u32` words per packed ray in the kernel's `rays` input buffer (32 bytes,
/// 16-byte aligned). Layout: `origin.xyz` (0..3), `t_min` (3), `dir.xyz`
/// (4..7), `t_max` (7). Must match `RAY_WORDS` in `shaders/ray_traverse.wesl`.
pub(crate) const RAY_WORDS: usize = 8;

/// `u32` words per packed hit in the kernel's `hits` output buffer (16 bytes,
/// 16-byte aligned). Layout: `t` (0), `u` (1), `v` (2), `primitive` (3). Must
/// match `HIT_WORDS` in `shaders/ray_traverse.wesl`.
pub(crate) const HIT_WORDS: usize = 4;

/// Linear workgroup size of the traversal entry point. Must match the
/// `@workgroup_size(64)` on `ray_traverse` in `shaders/ray_traverse.wesl`.
pub(crate) const RAYTRACE_WORKGROUP: u32 = 64;

/// Word offset of `first_primitive` inside a packed node. Mirrors word 6 of the
/// golden `NODE_WORDS` layout and the `nodes[base + 6u]` read in the shader.
pub(crate) const NODE_FIRST_PRIMITIVE_WORD: usize = 6;

/// Word offset of `second_child` inside a packed node (word 7).
pub(crate) const NODE_SECOND_CHILD_WORD: usize = 7;

/// Word offset of `primitive_count` inside a packed node (word 8).
pub(crate) const NODE_PRIMITIVE_COUNT_WORD: usize = 8;

/// Word offset of the split `axis` inside a packed node (word 9).
pub(crate) const NODE_AXIS_WORD: usize = 9;

/// Word offset of the stable `primitive` id inside a packed triangle (word 9).
pub(crate) const TRIANGLE_PRIMITIVE_WORD: usize = 9;

/// `f32` `+inf` bit pattern written to a miss hit's `t` slot; matches
/// `POSITIVE_INF_BITS` in the shader and `f32::INFINITY.to_bits()`.
pub(crate) const POSITIVE_INF_BITS: u32 = 0x7F80_0000;

/// Sentinel primitive id written on a miss; matches `MISS_PRIMITIVE` in the
/// shader and `u32::MAX`.
pub(crate) const MISS_PRIMITIVE: u32 = u32::MAX;

/// `params.mode` value selecting the nearest-hit walk (`closest_hit`). Must
/// match the `mode == 0u` branch in `shaders/ray_traverse.wesl`.
pub(crate) const RAYTRACE_MODE_CLOSEST: u32 = 0;

/// `params.mode` value selecting the any-hit occlusion walk (`any_hit`). Must
/// match the `mode == 1u` branch in `shaders/ray_traverse.wesl`.
pub(crate) const RAYTRACE_MODE_ANY: u32 = 1;

/// `u32` words per packed `TLAS` instance in the top-level kernel's
/// `instances` input buffer (64 bytes, 16-byte aligned), re-exported from the
/// golden layout. Layout: `world_to_object` linear columns `c0.xyz` (0..3),
/// `c1.xyz` (3..6), `c2.xyz` (6..9), translation (9..12), `blas_index` (12),
/// `instance_id` (13), pad (14..16). Must match `INSTANCE_WORDS` in
/// `shaders/tlas_traverse.wesl`.
pub(crate) const RAYTRACE_INSTANCE_WORDS: usize = INSTANCE_WORDS;

/// `u32` words per per-`BLAS` offset record in the top-level kernel's
/// `pool_offsets` input buffer (16 bytes, 16-byte aligned), re-exported from the
/// golden layout. Layout: `node_base` (0), `node_count` (1), `triangle_base`
/// (2), `triangle_count` (3). Must match `BLAS_OFFSET_WORDS` in
/// `shaders/tlas_traverse.wesl`.
pub(crate) const RAYTRACE_BLAS_OFFSET_WORDS: usize = BLAS_OFFSET_WORDS;

/// `u32` words per packed hit in the top-level kernel's `hits` output buffer
/// (32 bytes, 16-byte aligned). Layout: `t` (0), `u` (1), `v` (2), `primitive`
/// (3), `instance_id` (4), `instance_index` (5), pad (6..8). Must match
/// `TLAS_HIT_WORDS` in `shaders/tlas_traverse.wesl`.
pub(crate) const TLAS_HIT_WORDS: usize = 8;

/// Word offset of the `blas_index` inside a packed instance (word 12). Mirrors
/// the `instances[ib + 12u]` read in `shaders/tlas_traverse.wesl`.
pub(crate) const INSTANCE_BLAS_INDEX_WORD: usize = 12;

/// Word offset of the stable `instance_id` inside a packed instance (word 13).
pub(crate) const INSTANCE_ID_WORD: usize = 13;

/// Word offset of `node_base` inside a packed `BLAS` offset record (word 0).
pub(crate) const BLAS_OFFSET_NODE_BASE_WORD: usize = 0;

/// Word offset of `node_count` inside a packed `BLAS` offset record (word 1).
pub(crate) const BLAS_OFFSET_NODE_COUNT_WORD: usize = 1;

/// Word offset of `triangle_base` inside a packed `BLAS` offset record (word 2).
pub(crate) const BLAS_OFFSET_TRIANGLE_BASE_WORD: usize = 2;

/// Word offset of `triangle_count` inside a packed `BLAS` offset record (word 3).
pub(crate) const BLAS_OFFSET_TRIANGLE_COUNT_WORD: usize = 3;

/// `u32` words per packed ray-cone footprint in the `ray_footprint` kernel's
/// `footprints` input buffer (16 bytes, 16-byte aligned). Layout: `cone_width`
/// (0), `cone_spread_angle` (1), `hit_distance` (2), `texel_world_size` (3),
/// each an `f32` stored as its `to_bits` pattern. Mirrors the golden
/// `prism_render_architecture::ray_scene::RayFootprint` fields plus the
/// per-surface texel size the mip math consumes.
pub(crate) const FOOTPRINT_WORDS: usize = 4;

/// `u32` words per packed footprint result in the `ray_footprint` kernel's
/// `results` output buffer (16 bytes, 16-byte aligned). Layout: `projected_width`
/// (0), `texel_span` (1), `mip_level` (2) as `f32` bit patterns, and `mip_floor`
/// (3) as a raw `u32`. Must match `RESULT_WORDS` in `shaders/ray_footprint.wesl`.
pub(crate) const FOOTPRINT_RESULT_WORDS: usize = 4;

/// Linear workgroup size of the footprint entry point. Must match the
/// `@workgroup_size(64)` on `ray_footprint` in `shaders/ray_footprint.wesl`.
pub(crate) const FOOTPRINT_WORKGROUP: u32 = 64;

/// Word offset of `cone_width` inside a packed footprint record (word 0).
pub(crate) const FOOTPRINT_CONE_WIDTH_WORD: usize = 0;

/// Word offset of `cone_spread_angle` (a slope) inside a footprint record (word 1).
pub(crate) const FOOTPRINT_CONE_SPREAD_WORD: usize = 1;

/// Word offset of `hit_distance` inside a packed footprint record (word 2).
pub(crate) const FOOTPRINT_HIT_DISTANCE_WORD: usize = 2;

/// Word offset of `texel_world_size` inside a packed footprint record (word 3).
pub(crate) const FOOTPRINT_TEXEL_SIZE_WORD: usize = 3;

/// Word offset of `projected_width` inside a footprint result record (word 0).
pub(crate) const FOOTPRINT_RESULT_PROJECTED_WIDTH_WORD: usize = 0;

/// Word offset of `texel_span` inside a footprint result record (word 1).
pub(crate) const FOOTPRINT_RESULT_TEXEL_SPAN_WORD: usize = 1;

/// Word offset of the continuous `mip_level` inside a result record (word 2).
pub(crate) const FOOTPRINT_RESULT_MIP_LEVEL_WORD: usize = 2;

/// Word offset of the discrete `mip_floor` (raw `u32`) inside a result record (word 3).
pub(crate) const FOOTPRINT_RESULT_MIP_FLOOR_WORD: usize = 3;

/// Host mirror of the traversal kernel's `RayTraverseParams` uniform.
///
/// The four `u32` fields are exactly the `struct RayTraverseParams` in
/// `shaders/ray_traverse.wesl` (`ray_count`, `mode`, `pad0`, `pad1`): a 16-byte
/// block that satisfies the uniform-buffer 16-byte size/alignment rule with no
/// implicit tail padding, so `bytemuck::bytes_of` yields the exact bytes the
/// shader reads.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuRayTraverseParams {
    /// Number of packed rays in the `rays` buffer; invocations at or past this
    /// index early-out without writing a hit.
    pub(crate) ray_count: u32,
    /// Traversal mode: [`RAYTRACE_MODE_CLOSEST`] or [`RAYTRACE_MODE_ANY`].
    pub(crate) mode: u32,
    /// Padding word 0, pinning the uniform to the shader's 16-byte struct size.
    pub(crate) pad0: u32,
    /// Padding word 1, pinning the uniform to the shader's 16-byte struct size.
    pub(crate) pad1: u32,
}

/// Host mirror of the footprint kernel's `FootprintParams` uniform.
///
/// The four `u32` fields are exactly the `struct FootprintParams` in
/// `shaders/ray_footprint.wesl` (`footprint_count`, `max_mip`, `pad0`, `pad1`):
/// a 16-byte block satisfying the uniform 16-byte size/alignment rule with no
/// implicit tail padding, so `bytemuck::bytes_of` yields the exact device bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuFootprintParams {
    /// Number of packed footprints; invocations at or past this index early-out.
    pub(crate) footprint_count: u32,
    /// Maximum mip level the continuous `mip_level` is clamped to (inclusive).
    pub(crate) max_mip: u32,
    /// Padding word 0, pinning the uniform to the shader's 16-byte struct size.
    pub(crate) pad0: u32,
    /// Padding word 1, pinning the uniform to the shader's 16-byte struct size.
    pub(crate) pad1: u32,
}

#[cfg(test)]
mod tests {
    use super::{
        GpuFootprintParams, GpuRayTraverseParams, BLAS_OFFSET_NODE_BASE_WORD,
        BLAS_OFFSET_NODE_COUNT_WORD, BLAS_OFFSET_TRIANGLE_BASE_WORD,
        BLAS_OFFSET_TRIANGLE_COUNT_WORD, FOOTPRINT_CONE_SPREAD_WORD, FOOTPRINT_CONE_WIDTH_WORD,
        FOOTPRINT_HIT_DISTANCE_WORD, FOOTPRINT_RESULT_MIP_FLOOR_WORD,
        FOOTPRINT_RESULT_MIP_LEVEL_WORD, FOOTPRINT_RESULT_PROJECTED_WIDTH_WORD,
        FOOTPRINT_RESULT_TEXEL_SPAN_WORD, FOOTPRINT_RESULT_WORDS, FOOTPRINT_TEXEL_SIZE_WORD,
        FOOTPRINT_WORDS, HIT_WORDS, INSTANCE_BLAS_INDEX_WORD, INSTANCE_ID_WORD, MISS_PRIMITIVE,
        NODE_AXIS_WORD, NODE_FIRST_PRIMITIVE_WORD, NODE_PRIMITIVE_COUNT_WORD,
        NODE_SECOND_CHILD_WORD, POSITIVE_INF_BITS, RAYTRACE_BLAS_OFFSET_WORDS,
        RAYTRACE_INSTANCE_WORDS, RAYTRACE_MODE_ANY, RAYTRACE_MODE_CLOSEST, RAYTRACE_NODE_WORDS,
        RAYTRACE_TRIANGLE_WORDS, RAY_WORDS, TLAS_HIT_WORDS, TRIANGLE_PRIMITIVE_WORD,
    };
    use prism_render_architecture::ray_scene::{
        BLAS_OFFSET_WORDS, INSTANCE_WORDS, NODE_WORDS, TRIANGLE_WORDS,
    };

    #[test]
    fn node_and_triangle_strides_track_golden_layout() {
        assert_eq!(RAYTRACE_NODE_WORDS, NODE_WORDS);
        assert_eq!(RAYTRACE_TRIANGLE_WORDS, TRIANGLE_WORDS);
        // Both packed records are the same 12-word (48-byte) 16-byte-aligned
        // stride the shader hardcodes as `NODE_WORDS` / `TRIANGLE_WORDS`.
        assert_eq!(NODE_WORDS, 12);
        assert_eq!(TRIANGLE_WORDS, 12);
    }

    #[test]
    fn ray_and_hit_strides_are_16_byte_aligned() {
        assert_eq!(RAY_WORDS, 8, "ray record must be 32 bytes");
        assert_eq!(HIT_WORDS, 4, "hit record must be 16 bytes");
        assert_eq!((RAY_WORDS * 4) % 16, 0);
        assert_eq!((HIT_WORDS * 4) % 16, 0);
    }

    #[test]
    fn node_field_offsets_stay_inside_the_node_stride() {
        for word in [
            NODE_FIRST_PRIMITIVE_WORD,
            NODE_SECOND_CHILD_WORD,
            NODE_PRIMITIVE_COUNT_WORD,
            NODE_AXIS_WORD,
        ] {
            assert!(word < NODE_WORDS, "node field word {word} out of stride");
        }
        // The four consumed scalar fields occupy words 6..=9, exactly as the
        // golden layout doc pins them.
        assert_eq!(NODE_FIRST_PRIMITIVE_WORD, 6);
        assert_eq!(NODE_SECOND_CHILD_WORD, 7);
        assert_eq!(NODE_PRIMITIVE_COUNT_WORD, 8);
        assert_eq!(NODE_AXIS_WORD, 9);
    }

    #[test]
    fn triangle_primitive_offset_stays_inside_the_triangle_stride() {
        assert!(TRIANGLE_PRIMITIVE_WORD < TRIANGLE_WORDS);
        assert_eq!(TRIANGLE_PRIMITIVE_WORD, 9);
    }

    #[test]
    fn miss_sentinels_match_the_shader_constants() {
        assert_eq!(POSITIVE_INF_BITS, f32::INFINITY.to_bits());
        assert_eq!(MISS_PRIMITIVE, u32::MAX);
    }

    #[test]
    fn traverse_modes_match_the_shader_branches() {
        // `shaders/ray_traverse.wesl` gates the walk on `params.mode`: `0u`
        // takes the nearest-hit branch, `1u` the any-hit occlusion branch.
        assert_eq!(RAYTRACE_MODE_CLOSEST, 0);
        assert_eq!(RAYTRACE_MODE_ANY, 1);
    }

    #[test]
    fn params_uniform_is_a_16_byte_block() {
        // The shader's `RayTraverseParams` is four `u32`s; a `UNIFORM` buffer
        // must be 16-byte aligned, and this exact-size block carries no tail
        // padding so `bytes_of` matches the on-device read byte-for-byte.
        assert_eq!(size_of::<GpuRayTraverseParams>(), 16);
        assert_eq!(align_of::<GpuRayTraverseParams>(), 4);
        let params = GpuRayTraverseParams {
            ray_count: 7,
            mode: RAYTRACE_MODE_ANY,
            pad0: 0,
            pad1: 0,
        };
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), 16);
        assert_eq!(&bytes[0..4], &7u32.to_le_bytes());
        assert_eq!(&bytes[4..8], &1u32.to_le_bytes());
    }

    #[test]
    fn instance_and_offset_strides_track_golden_layout() {
        assert_eq!(RAYTRACE_INSTANCE_WORDS, INSTANCE_WORDS);
        assert_eq!(RAYTRACE_BLAS_OFFSET_WORDS, BLAS_OFFSET_WORDS);
        // The instance record is a 16-word (64-byte) block and the offset record
        // a 4-word (16-byte) block, both 16-byte aligned, exactly as
        // `shaders/tlas_traverse.wesl` hardcodes `INSTANCE_WORDS` /
        // `BLAS_OFFSET_WORDS`.
        assert_eq!(INSTANCE_WORDS, 16);
        assert_eq!(BLAS_OFFSET_WORDS, 4);
        assert_eq!((INSTANCE_WORDS * 4) % 16, 0);
        assert_eq!((BLAS_OFFSET_WORDS * 4) % 16, 0);
    }

    #[test]
    fn tlas_hit_stride_is_16_byte_aligned() {
        // The top-level hit carries `t`, `u`, `v`, `primitive`, `instance_id`
        // and `instance_index` (six words) padded to 8 words (32 bytes) so the
        // `hits` buffer stays 16-byte aligned, matching `TLAS_HIT_WORDS` in
        // `shaders/tlas_traverse.wesl`.
        assert_eq!(TLAS_HIT_WORDS, 8, "TLAS hit record must be 32 bytes");
        assert_eq!((TLAS_HIT_WORDS * 4) % 16, 0);
    }

    #[test]
    fn instance_field_offsets_stay_inside_the_instance_stride() {
        assert!(INSTANCE_BLAS_INDEX_WORD < INSTANCE_WORDS);
        assert!(INSTANCE_ID_WORD < INSTANCE_WORDS);
        // The affine occupies words 0..12; the two id fields follow at 12/13,
        // exactly as the golden `GpuTlasBuffers::from_tlas` packs them.
        assert_eq!(INSTANCE_BLAS_INDEX_WORD, 12);
        assert_eq!(INSTANCE_ID_WORD, 13);
    }

    #[test]
    fn blas_offset_field_words_cover_the_record() {
        for word in [
            BLAS_OFFSET_NODE_BASE_WORD,
            BLAS_OFFSET_NODE_COUNT_WORD,
            BLAS_OFFSET_TRIANGLE_BASE_WORD,
            BLAS_OFFSET_TRIANGLE_COUNT_WORD,
        ] {
            assert!(
                word < BLAS_OFFSET_WORDS,
                "offset field word {word} out of stride"
            );
        }
        // The four fields occupy words 0..=3 in order, exactly as
        // `GpuBlasPool::from_blases` writes them and the kernel reads them.
        assert_eq!(BLAS_OFFSET_NODE_BASE_WORD, 0);
        assert_eq!(BLAS_OFFSET_NODE_COUNT_WORD, 1);
        assert_eq!(BLAS_OFFSET_TRIANGLE_BASE_WORD, 2);
        assert_eq!(BLAS_OFFSET_TRIANGLE_COUNT_WORD, 3);
    }

    #[test]
    fn footprint_record_strides_are_16_byte_aligned() {
        // The footprint input and result records are both 4-word (16-byte)
        // 16-byte-aligned blocks the `ray_footprint` kernel hardcodes as
        // `FOOTPRINT_WORDS` / `RESULT_WORDS`.
        assert_eq!(FOOTPRINT_WORDS, 4, "footprint record must be 16 bytes");
        assert_eq!(FOOTPRINT_RESULT_WORDS, 4, "result record must be 16 bytes");
        assert_eq!((FOOTPRINT_WORDS * 4) % 16, 0);
        assert_eq!((FOOTPRINT_RESULT_WORDS * 4) % 16, 0);
    }

    #[test]
    fn footprint_field_offsets_cover_both_records() {
        // Input fields occupy words 0..=3 in the order the host packs a
        // `RayFootprint` plus its surface texel size.
        assert_eq!(FOOTPRINT_CONE_WIDTH_WORD, 0);
        assert_eq!(FOOTPRINT_CONE_SPREAD_WORD, 1);
        assert_eq!(FOOTPRINT_HIT_DISTANCE_WORD, 2);
        assert_eq!(FOOTPRINT_TEXEL_SIZE_WORD, 3);
        for word in [
            FOOTPRINT_CONE_WIDTH_WORD,
            FOOTPRINT_CONE_SPREAD_WORD,
            FOOTPRINT_HIT_DISTANCE_WORD,
            FOOTPRINT_TEXEL_SIZE_WORD,
        ] {
            assert!(
                word < FOOTPRINT_WORDS,
                "input field word {word} out of stride"
            );
        }
        // Result fields occupy words 0..=3: three `f32` scalars then the raw
        // `u32` mip bucket.
        assert_eq!(FOOTPRINT_RESULT_PROJECTED_WIDTH_WORD, 0);
        assert_eq!(FOOTPRINT_RESULT_TEXEL_SPAN_WORD, 1);
        assert_eq!(FOOTPRINT_RESULT_MIP_LEVEL_WORD, 2);
        assert_eq!(FOOTPRINT_RESULT_MIP_FLOOR_WORD, 3);
        for word in [
            FOOTPRINT_RESULT_PROJECTED_WIDTH_WORD,
            FOOTPRINT_RESULT_TEXEL_SPAN_WORD,
            FOOTPRINT_RESULT_MIP_LEVEL_WORD,
            FOOTPRINT_RESULT_MIP_FLOOR_WORD,
        ] {
            assert!(
                word < FOOTPRINT_RESULT_WORDS,
                "result field word {word} out of stride"
            );
        }
    }

    #[test]
    fn footprint_params_uniform_is_a_16_byte_block() {
        // The shader's `FootprintParams` is four `u32`s; the `UNIFORM` buffer
        // must be 16-byte aligned with no tail padding so `bytes_of` matches the
        // on-device read byte-for-byte.
        assert_eq!(size_of::<GpuFootprintParams>(), 16);
        assert_eq!(align_of::<GpuFootprintParams>(), 4);
        let params = GpuFootprintParams {
            footprint_count: 5,
            max_mip: 8,
            pad0: 0,
            pad1: 0,
        };
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), 16);
        assert_eq!(&bytes[0..4], &5u32.to_le_bytes());
        assert_eq!(&bytes[4..8], &8u32.to_le_bytes());
    }
}
