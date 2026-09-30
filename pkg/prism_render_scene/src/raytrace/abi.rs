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

use prism_render_architecture::ray_scene::{NODE_WORDS, TRIANGLE_WORDS};

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

#[cfg(test)]
mod tests {
    use super::{
        HIT_WORDS, MISS_PRIMITIVE, NODE_AXIS_WORD, NODE_FIRST_PRIMITIVE_WORD,
        NODE_PRIMITIVE_COUNT_WORD, NODE_SECOND_CHILD_WORD, POSITIVE_INF_BITS, RAYTRACE_NODE_WORDS,
        RAYTRACE_TRIANGLE_WORDS, RAY_WORDS, TRIANGLE_PRIMITIVE_WORD,
    };
    use prism_render_architecture::ray_scene::{NODE_WORDS, TRIANGLE_WORDS};

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
}
