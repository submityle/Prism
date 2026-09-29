//! Centralised `var<immediate>` push-constant block sizes for the fixed hair
//! main-spine compute passes ([`HairComputePass`]).
//!
//! Every main-spine kernel binds a small `var<immediate>` params struct at
//! pipeline-issue time (the `WGSL`/`WESL` push-constant block). The buffer ABI
//! modules ([`gpu_buffers`](super::gpu_buffers),
//! [`import_buffers`](super::import_buffers),
//! [`sim_pass_buffers`](super::sim_pass_buffers),
//! [`interp_buffers`](super::interp_buffers),
//! [`lod_dither_buffers`](super::lod_dither_buffers),
//! [`shadow_buffers`](super::shadow_buffers)) own the `@group(0)` storage strides,
//! and [`pass_layout`](super::pass_layout) joins those into the dense binding
//! table. What was still scattered is the *immediate* block size — the render
//! graph needs it to size the push-constant range before it can create the
//! compute pipeline for a pass. The two optional passes already publish their
//! immediate sizes
//! ([`vbd_pass_buffers::PARAMS_IMMEDIATE_BYTES`](super::vbd_pass_buffers::PARAMS_IMMEDIATE_BYTES)
//! and
//! [`self_collision_pass_buffers::PARAMS_IMMEDIATE_BYTES`](super::self_collision_pass_buffers::PARAMS_IMMEDIATE_BYTES));
//! this module is the matching single source of truth for the ten main-spine
//! passes.
//!
//! Each size is the std430 `SizeOf` of the params struct the corresponding
//! kernel declares as `var<immediate>` in its `WESL` twin, computed by hand:
//! `AlignOf` is the max member alignment, offsets round each member up to its
//! own alignment, and `SizeOf` rounds the past-the-end offset up to `AlignOf`.
//! A `vec3<f32>` has alignment 16 and size 12, so any block that opens with a
//! `vec3` (wind, guide `XPBD`) is a multiple of 16; the scalar-only blocks are
//! multiples of 4. The per-constant docs below spell out each field offset so
//! the numbers can be re-derived from the shader source without guessing.
//!
//! Everything is pure and integer: [`params_immediate_bytes`] is a total
//! function over [`HairComputePass`] and never panics.

use crate::hair::gpu_dispatch::HairComputePass;

/// `HairRootParams` for `hair_root_bind.wesl`: three tightly packed `u32`
/// (`root_count`, `triangle_count`, `vertex_count`). Offsets 0/4/8, end 12,
/// alignment 4 → 12 bytes.
pub const ROOT_BIND_PARAMS_BYTES: usize = 12;

/// `HairResampleParams` for `hair_resample.wesl`: two `u32` (`strand_count`,
/// `points_per_strand`). Offsets 0/4, end 8, alignment 4 → 8 bytes.
pub const RESAMPLE_PARAMS_BYTES: usize = 8;

/// `HairRootParams` for `hair_root_skinning.wesl`: byte-identical to the
/// `hair_root_bind.wesl` block (three `u32`) → 12 bytes.
pub const ROOT_SKINNING_PARAMS_BYTES: usize = 12;

/// `HairWindParams` for `hair_wind.wesl`: `direction: vec3<f32>` (offset 0,
/// size 12) then seven scalars `speed`/`gust_amplitude`/`gust_frequency`/
/// `turbulence`/`time`/`dt`/`particle_count` at offsets 12/16/20/24/28/32/36.
/// Past-the-end 40, alignment 16 (the `vec3`) → 48 bytes.
pub const WIND_PARAMS_BYTES: usize = 48;

/// `HairXpbdParams` for `hair_sim.wesl` (the guide `XPBD` solver): `gravity:
/// vec3<f32>` (offset 0, size 12) then eleven scalars
/// `dt`/`edge_compliance`/`local_stiffness`/`global_stiffness`/`lra_stiffness`/
/// `damping`/`substeps`/`iterations`/`strand_count`/`collider_count` at offsets
/// 12..48. Past-the-end 52, alignment 16 → 64 bytes.
pub const GUIDE_SIM_PARAMS_BYTES: usize = 64;

/// `HairSdfParams` for `hair_sdf_collision.wesl`: three `u32` (`iterations`,
/// `primitive_count`, `particle_count`) → 12 bytes.
pub const SDF_COLLISION_PARAMS_BYTES: usize = 12;

/// `HairInterpParams` for `hair_interp.wesl`: seven align-4 scalars
/// `clump_count`/`clump_strength`/`curl_frequency`/`curl_amplitude`/
/// `position_jitter`/`length_jitter`/`strand_count`. Offsets 0..24, end 28,
/// alignment 4 → 28 bytes.
pub const INTERPOLATE_PARAMS_BYTES: usize = 28;

/// `HairDitherParams` for `hair_lod_dither.wesl`: three scalars (`seed`,
/// `strand_count`, `blend`) → 12 bytes.
pub const LOD_DITHER_PARAMS_BYTES: usize = 12;

/// `HairTransmittanceParams` for `hair_transmittance.wesl`: four scalars
/// (`slab_start`, `slab_end`, `voxel_count`, `texel_count`) → 16 bytes.
pub const TRANSMITTANCE_PARAMS_BYTES: usize = 16;

/// `HairDeepOpacityParams` for `hair_deep_opacity.wesl`: three scalars
/// (`texel_count`, `layer_count`, `start_offset`) → 12 bytes.
pub const DEEP_OPACITY_PARAMS_BYTES: usize = 12;

/// std430 byte size of the `var<immediate>` push-constant params block bound by
/// `pass` on the fixed hair main spine.
///
/// This is the size the render graph reserves for the push-constant range when
/// it creates the compute pipeline for a main-spine pass. It is intentionally
/// disjoint from the optional-pass sizes, which live in
/// [`vbd_pass_buffers`](super::vbd_pass_buffers) and
/// [`self_collision_pass_buffers`](super::self_collision_pass_buffers); the
/// guide `VBD` alternative solver declares its own `HairVbdParams` block
/// (48 bytes) rather than reusing the `GuideSim` `HairXpbdParams` block
/// (64 bytes) it can replace.
#[must_use]
pub fn params_immediate_bytes(pass: HairComputePass) -> usize {
    match pass {
        HairComputePass::RootBind => ROOT_BIND_PARAMS_BYTES,
        HairComputePass::Resample => RESAMPLE_PARAMS_BYTES,
        HairComputePass::RootSkinning => ROOT_SKINNING_PARAMS_BYTES,
        HairComputePass::Wind => WIND_PARAMS_BYTES,
        HairComputePass::GuideSim => GUIDE_SIM_PARAMS_BYTES,
        HairComputePass::SdfCollision => SDF_COLLISION_PARAMS_BYTES,
        HairComputePass::Interpolate => INTERPOLATE_PARAMS_BYTES,
        HairComputePass::LodDither => LOD_DITHER_PARAMS_BYTES,
        HairComputePass::Transmittance => TRANSMITTANCE_PARAMS_BYTES,
        HairComputePass::DeepOpacity => DEEP_OPACITY_PARAMS_BYTES,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::optional_pass_dispatch::HairOptionalPass;
    use crate::hair::optional_pass_layout::optional_params_immediate_bytes;
    use crate::hair::self_collision_pass_buffers;
    use crate::hair::vbd_pass_buffers;

    #[test]
    fn every_main_pass_matches_its_constant() {
        assert_eq!(params_immediate_bytes(HairComputePass::RootBind), 12);
        assert_eq!(params_immediate_bytes(HairComputePass::Resample), 8);
        assert_eq!(params_immediate_bytes(HairComputePass::RootSkinning), 12);
        assert_eq!(params_immediate_bytes(HairComputePass::Wind), 48);
        assert_eq!(params_immediate_bytes(HairComputePass::GuideSim), 64);
        assert_eq!(params_immediate_bytes(HairComputePass::SdfCollision), 12);
        assert_eq!(params_immediate_bytes(HairComputePass::Interpolate), 28);
        assert_eq!(params_immediate_bytes(HairComputePass::LodDither), 12);
        assert_eq!(params_immediate_bytes(HairComputePass::Transmittance), 16);
        assert_eq!(params_immediate_bytes(HairComputePass::DeepOpacity), 12);
    }

    #[test]
    fn every_main_pass_has_a_nonzero_multiple_of_four_block() {
        for pass in HairComputePass::ALL {
            let bytes = params_immediate_bytes(pass);
            assert!(bytes > 0, "{pass:?} immediate block must be non-empty");
            assert_eq!(bytes % 4, 0, "{pass:?} immediate block must be 4-aligned");
        }
    }

    #[test]
    fn vec3_leading_blocks_are_sixteen_aligned() {
        // Wind and GuideSim open with a `vec3<f32>` (alignment 16), so their
        // std430 `SizeOf` rounds up to a multiple of 16.
        assert_eq!(params_immediate_bytes(HairComputePass::Wind) % 16, 0);
        assert_eq!(params_immediate_bytes(HairComputePass::GuideSim) % 16, 0);
    }

    #[test]
    fn scalar_only_blocks_stay_below_the_vec3_ones() {
        // Every scalar-only block is smaller than the two vec3-leading blocks;
        // guards against a stray vec3 sneaking into a scalar params struct.
        let vec3_min = params_immediate_bytes(HairComputePass::Wind)
            .min(params_immediate_bytes(HairComputePass::GuideSim));
        for pass in [
            HairComputePass::RootBind,
            HairComputePass::Resample,
            HairComputePass::RootSkinning,
            HairComputePass::SdfCollision,
            HairComputePass::Interpolate,
            HairComputePass::LodDither,
            HairComputePass::Transmittance,
            HairComputePass::DeepOpacity,
        ] {
            assert!(
                params_immediate_bytes(pass) < vec3_min,
                "{pass:?} scalar block unexpectedly as large as a vec3 block",
            );
        }
    }

    #[test]
    fn root_bind_and_skinning_share_the_same_params_block() {
        // Both consume `HairRootParams`, so their immediate sizes must agree.
        assert_eq!(
            params_immediate_bytes(HairComputePass::RootBind),
            params_immediate_bytes(HairComputePass::RootSkinning),
        );
    }

    #[test]
    fn guide_sim_differs_from_its_vbd_alternative() {
        // The optional guide-VBD solver replaces GuideSim but binds its own,
        // smaller params block; the two sizes must not be conflated.
        assert_eq!(vbd_pass_buffers::PARAMS_IMMEDIATE_BYTES, 48);
        assert_eq!(
            optional_params_immediate_bytes(HairOptionalPass::VbdSolve),
            vbd_pass_buffers::PARAMS_IMMEDIATE_BYTES,
        );
        assert_ne!(
            params_immediate_bytes(HairComputePass::GuideSim),
            optional_params_immediate_bytes(HairOptionalPass::VbdSolve),
        );
    }

    #[test]
    fn self_collision_optional_block_is_distinct_from_every_main_block() {
        // The self-collision accumulate/apply pair binds a 20-byte block that
        // is not (coincidentally) equal to any main-spine block, so a frame
        // scheduler cannot alias them by size alone.
        let self_collision = self_collision_pass_buffers::PARAMS_IMMEDIATE_BYTES;
        assert_eq!(self_collision, 20);
        assert_eq!(
            optional_params_immediate_bytes(HairOptionalPass::SelfCollisionAccumulate),
            self_collision,
        );
        assert_eq!(
            optional_params_immediate_bytes(HairOptionalPass::SelfCollisionApply),
            self_collision,
        );
        for pass in HairComputePass::ALL {
            assert_ne!(params_immediate_bytes(pass), self_collision);
        }
    }
}
