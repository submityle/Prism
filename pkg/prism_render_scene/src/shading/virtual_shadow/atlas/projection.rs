//! Pure, CPU-testable geometry for the virtual-shadow-map caster depth pass: the
//! per-page orthographic light projection and the atlas tile rectangle a page's
//! rendered depth is copied into.
//!
//! Every resident clipmap page owns a square world-space footprint on the
//! light's clipmap plane (`world_origin` = its lower-left corner, `world_size` =
//! its edge). This module builds the `world -> light-clip` matrix that maps that
//! footprint onto the wgpu NDC cube so the caster-depth raster fills exactly the
//! texels [`shaders/vsm_sample.wesl`] later reads that page back from, and it
//! computes the symmetric depth half-extent the projection's `z` range covers.
//!
//! # Frame agreement with the sampler
//! The sampler ([`shaders/vsm_sample.wesl`]) projects a world point onto the
//! light plane as `light_xy = (world . light_right, world . light_up)`, snaps it
//! to a page whose lower-left corner is `page_origin`, forms the in-page
//! fraction `frac = clamp((light_xy - page_origin) / page_world_size, 0, 1)`, and
//! reads atlas texel `(tile + frac) / edge` -- where texture UV `(0, 0)` is the
//! tile's **top-left**. To land the same world point on the same texel this
//! projection must therefore map:
//!
//!   * `frac.x = 0` (min `light_right`) -> `NDC.x = -1`; `frac.x = 1` -> `NDC.x = +1`
//!   * `frac.y = 0` (min `light_up`)    -> `NDC.y = +1` (top row); `frac.y = 1` -> `NDC.y = -1`
//!
//! i.e. `NDC.x = 2 (world.light_right - origin.x) / size - 1` and a **y-flipped**
//! `NDC.y = 1 - 2 (world.light_up - origin.y) / size`. The basis
//! `(light_right, light_up)` is taken from
//! [`prism_render_shading::ReceiverProjection::from_light_direction`] -- the same
//! constructor the sampler's CPU golden uses -- so the two sides can never pick a
//! different frame for one light direction.
//!
//! # Depth
//! `NDC.z = (world . light_forward + D) / (2 D)`, so a caster `D` in front of the
//! light plane maps to the near plane (`z = 0`) and one `D` behind maps to the
//! far plane (`z = 1`); the plane through the world origin sits at `z = 0.5`.
//! This is the wgpu `[0, 1]` depth range `vsm_sample.wesl` compares against with
//! `reference_depth <= stored => lit`. `D` ([`caster_depth_half_extent`]) is
//! sized to the coarsest clipmap window so every in-window caster stays inside
//! the range; a tighter scene-AABB fit is left to a later slice (documented in
//! the module report).

use bevy_math::{Mat4, UVec2, Vec2, Vec3, Vec4};
use prism_render_shading::{ClipmapConfig, ReceiverProjection};

/// Floor on a page's world edge, guarding the `2 / size` scale against a
/// degenerate zero-area page (which would divide by zero and emit NaNs).
const MIN_PAGE_WORLD_SIZE: f32 = 1.0e-6;

/// Floor on the depth half-extent, guarding the `1 / (2 D)` depth scale against a
/// zero-thickness range and keeping every stored depth finite.
const MIN_DEPTH_HALF_EXTENT: f32 = 1.0e-3;

/// Builds the column-major `world -> light-clip` matrix that rasterizes a single
/// resident clipmap page's shadow casters into its atlas tile.
///
/// `world_origin` is the page's lower-left corner **in light-plane coordinates**
/// (`(world . light_right, world . light_up)`), `world_size` its world-space
/// edge, `light_direction` the direction the light's rays travel (need not be
/// normalized), and `depth_half_extent` the symmetric world distance in front of
/// and behind the light plane the `[0, 1]` depth range spans.
///
/// The resulting clip space is wgpu's (`z` in `[0, 1]`) with a y-flip so the
/// page's lower-left world corner lands on the atlas tile's top-left texel,
/// matching `shaders/vsm_sample.wesl` (see the module docs for the exact corner
/// mapping). Casters are transformed by `clip = M * vec4(world, 1)` in
/// `shaders/vsm_caster_depth.wesl`.
pub(crate) fn page_light_projection(
    world_origin: Vec2,
    world_size: f32,
    light_direction: Vec3,
    depth_half_extent: f32,
) -> Mat4 {
    // Reuse the sampler's basis constructor so both sides span the light plane
    // with the identical (light_right, light_up) frame for a given direction.
    let projection = ReceiverProjection::from_light_direction(light_direction, Vec3::ZERO, 0.0);
    let light_right = projection.light_right;
    let light_up = projection.light_up;
    // The forward (depth) axis, derived exactly as `from_light_direction` does
    // internally, so it completes the orthonormal (light_right, light_up) frame.
    let light_forward = {
        let dir = light_direction.normalize_or_zero();
        if dir == Vec3::ZERO {
            Vec3::NEG_Y
        } else {
            dir
        }
    };

    let size = world_size.max(MIN_PAGE_WORLD_SIZE);
    let depth = depth_half_extent.max(MIN_DEPTH_HALF_EXTENT);
    // `xy_scale = 2 / size` maps a full page edge onto the 2-wide NDC span;
    // `depth_scale = 1 / (2 D)` maps the `2 D`-thick depth slab onto `[0, 1]`.
    let xy_scale = 2.0 / size;
    let depth_scale = 1.0 / (2.0 * depth);
    let origin_x = world_origin.x;
    let origin_y = world_origin.y;

    // Columns are the per-world-axis coefficients (glam / wgpu are column-major):
    // clip.x =  xy_scale * (world . light_right) - (xy_scale * origin_x + 1)
    // clip.y = -xy_scale * (world . light_up)    + (xy_scale * origin_y + 1)   (y-flip)
    // clip.z =  depth_scale * (world . light_forward) + 0.5
    // clip.w =  1
    Mat4::from_cols(
        Vec4::new(
            xy_scale * light_right.x,
            -xy_scale * light_up.x,
            depth_scale * light_forward.x,
            0.0,
        ),
        Vec4::new(
            xy_scale * light_right.y,
            -xy_scale * light_up.y,
            depth_scale * light_forward.y,
            0.0,
        ),
        Vec4::new(
            xy_scale * light_right.z,
            -xy_scale * light_up.z,
            depth_scale * light_forward.z,
            0.0,
        ),
        Vec4::new(
            -(xy_scale * origin_x + 1.0),
            xy_scale * origin_y + 1.0,
            0.5,
            1.0,
        ),
    )
}

/// Symmetric world-space depth half-extent the per-page projection's `[0, 1]`
/// depth range spans, in front of and behind the light plane through the world
/// origin.
///
/// Sized to the coarsest clipmap level's full resident window
/// (`page_world_size(coarsest) * pages_per_level_edge`) so every caster inside
/// the largest clip window stays within the depth slab and is not clipped away.
/// It is a pure function of the clipmap so a future resolve / receiver path can
/// recompute the identical range. Fitting a tighter slab to the scene's caster
/// AABB is deferred to a later slice.
pub(crate) fn caster_depth_half_extent(clipmap: &ClipmapConfig) -> f32 {
    let coarsest = clipmap.level_count().saturating_sub(1);
    let window = clipmap.page_world_size(coarsest) * f32::from(clipmap.pages_per_level_edge.max(1));
    window.max(MIN_DEPTH_HALF_EXTENT)
}

/// The atlas tile rectangle a resident page is rasterized into, as
/// `(x, y, width, height)` in texels.
///
/// The origin is the page's top-left texel in the atlas (from
/// [`super::atlas_tile_origin`]) and the extent is one `page_size`-square tile.
/// The caster-depth pass sets this rectangle as the render pass viewport *and*
/// scissor so a page's draws write only its own tile of the shared atlas; it is
/// kept as a pure function and unit-tested here.
pub(crate) fn page_viewport_rect(tile_origin: UVec2, page_size: u32) -> (u32, u32, u32, u32) {
    let size = page_size.max(1);
    (tile_origin.x, tile_origin.y, size, size)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Projects `world` through `matrix` and returns clip space divided by `w`
    /// (the NDC point). `w` is always `1` here (the projection is affine), but
    /// dividing keeps the test honest about the clip-space contract.
    fn project_to_ndc(matrix: &Mat4, world: Vec3) -> Vec3 {
        let clip = *matrix * Vec4::new(world.x, world.y, world.z, 1.0);
        clip.truncate() / clip.w
    }

    #[test]
    fn page_corners_map_to_ndc_corners_with_y_flip() {
        // dir = -Z yields the axis-aligned basis light_right = +X, light_up = +Y,
        // light_forward = -Z, so light-plane xy is just world xy and the corner
        // mapping is easy to read off.
        let origin = Vec2::new(2.0, 3.0);
        let size = 4.0;
        let depth = 10.0;
        let matrix = page_light_projection(origin, size, Vec3::NEG_Z, depth);

        // Lower-left world corner -> NDC top-left (-1, +1).
        let lower_left = project_to_ndc(&matrix, Vec3::new(2.0, 3.0, 0.0));
        assert!((lower_left.x - -1.0).abs() < 1.0e-5, "ll.x = {}", lower_left.x);
        assert!((lower_left.y - 1.0).abs() < 1.0e-5, "ll.y = {}", lower_left.y);

        // Upper-right world corner -> NDC bottom-right (+1, -1).
        let upper_right = project_to_ndc(&matrix, Vec3::new(6.0, 7.0, 0.0));
        assert!((upper_right.x - 1.0).abs() < 1.0e-5, "ur.x = {}", upper_right.x);
        assert!((upper_right.y - -1.0).abs() < 1.0e-5, "ur.y = {}", upper_right.y);

        // Page centre -> NDC origin.
        let centre = project_to_ndc(&matrix, Vec3::new(4.0, 5.0, 0.0));
        assert!(centre.x.abs() < 1.0e-5, "c.x = {}", centre.x);
        assert!(centre.y.abs() < 1.0e-5, "c.y = {}", centre.y);
    }

    #[test]
    fn depth_maps_light_slab_to_unit_range() {
        // Same -Z basis: light_forward = -Z, so world_forward = world . (-Z) = -z.
        let matrix = page_light_projection(Vec2::new(2.0, 3.0), 4.0, Vec3::NEG_Z, 10.0);

        // `D` in front of the plane (light_forward = -D, i.e. world z = +D) -> near (0).
        let near = project_to_ndc(&matrix, Vec3::new(4.0, 5.0, 10.0));
        assert!(near.z.abs() < 1.0e-5, "near.z = {}", near.z);
        // On the plane -> mid (0.5).
        let mid = project_to_ndc(&matrix, Vec3::new(4.0, 5.0, 0.0));
        assert!((mid.z - 0.5).abs() < 1.0e-5, "mid.z = {}", mid.z);
        // `D` behind the plane (world z = -D) -> far (1).
        let far = project_to_ndc(&matrix, Vec3::new(4.0, 5.0, -10.0));
        assert!((far.z - 1.0).abs() < 1.0e-5, "far.z = {}", far.z);
    }

    #[test]
    fn xy_is_independent_of_distance_along_the_light() {
        // For an arbitrary slanted light, moving a world point purely along the
        // light-forward axis must not change its NDC xy (the projection is
        // orthographic and light_right/light_up are perpendicular to forward).
        let light = Vec3::new(0.3, -0.8, 0.5);
        let matrix = page_light_projection(Vec2::new(-1.5, 4.25), 7.0, light, 12.0);
        let forward = {
            let dir = light.normalize_or_zero();
            if dir == Vec3::ZERO {
                Vec3::NEG_Y
            } else {
                dir
            }
        };

        let base = Vec3::new(1.0, 2.0, -3.0);
        let shifted = base + forward * 5.0;
        let a = project_to_ndc(&matrix, base);
        let b = project_to_ndc(&matrix, shifted);
        assert!((a.x - b.x).abs() < 1.0e-4, "x drift {} vs {}", a.x, b.x);
        assert!((a.y - b.y).abs() < 1.0e-4, "y drift {} vs {}", a.y, b.y);
        // ...while the depth must advance monotonically along the light.
        assert!(b.z > a.z, "depth should increase away from the light");
    }

    #[test]
    fn degenerate_page_size_stays_finite() {
        // A zero-edge page must not divide by zero: the clamp keeps every lane
        // finite so a bad page cannot poison the whole uniform buffer.
        let matrix = page_light_projection(Vec2::ZERO, 0.0, Vec3::NEG_Y, 0.0);
        assert!(matrix.to_cols_array().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn viewport_rect_is_the_page_sized_tile_at_the_origin() {
        assert_eq!(page_viewport_rect(UVec2::ZERO, 128), (0, 0, 128, 128));
        assert_eq!(page_viewport_rect(UVec2::new(256, 384), 128), (256, 384, 128, 128));
        // A zero page size is clamped to one texel so the copy extent is valid.
        assert_eq!(page_viewport_rect(UVec2::new(10, 20), 0), (10, 20, 1, 1));
    }

    #[test]
    fn depth_half_extent_tracks_the_coarsest_window() {
        let clipmap = ClipmapConfig {
            levels: 4,
            page_size: 128,
            pages_per_level_edge: 8,
            level0_texel_world_size: 0.1,
            level0_max_distance: 10.0,
            page_coord_bias: 32_768,
        };
        let coarsest = clipmap.level_count() - 1;
        let expected =
            clipmap.page_world_size(coarsest) * f32::from(clipmap.pages_per_level_edge);
        let extent = caster_depth_half_extent(&clipmap);
        assert!((extent - expected).abs() < 1.0e-3, "extent {} vs {}", extent, expected);
        assert!(extent >= MIN_DEPTH_HALF_EXTENT);
    }
}
