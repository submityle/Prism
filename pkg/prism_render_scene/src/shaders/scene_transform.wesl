//! Shared helpers for lifting a *local-space* reconstructed surface into world
//! space with a per-instance `world_from_local` affine.
//!
//! The visibility-buffer pipeline stores shading vertices in model/local space
//! (they are the raw mesh attributes, see `geometry::shading::build_shading_geometry`),
//! so every consumer that rebuilds a surface from the visibility buffer must
//! apply the owning instance's transform before the surface can be shaded,
//! reflected or reprojected. The shading resolve, the SSR geometry prepass and
//! the SSR roughness/normal repack all need the identical math, so it lives here
//! once and mirrors the CPU golden `resolve_pixel`:
//!
//! * the position by the full affine `world_from_local`,
//! * the normal by the inverse-transpose of the linear part (correct under a
//!   non-uniform scale),
//! * the tangent/bitangent by the linear part,
//! * the UV untouched (it is transform-invariant).

import prism_render_scene::shaders::gpu_scene::GpuSceneTransform;
import prism_render_scene::shaders::surface::{
    SurfaceSampleGeometry,
    surface_normalize_or,
};

// Byte-faithful copy of Bevy's `bevy_render::maths::affine3_to_square`: expands
// the transposed 3x4 affine into a full column-major 4x4 with the implicit
// `(0, 0, 0, 1)` bottom row. Inlined (rather than imported from
// `bevy_render::maths`) so the module compiles standalone in the off-GPU
// `ShaderCache` tests, matching `shadow_depth.wesl`.
fn scene_affine3_to_square(affine: mat3x4<f32>) -> mat4x4<f32> {
    return transpose(mat4x4<f32>(
        affine[0],
        affine[1],
        affine[2],
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    ));
}

// Inverse-transpose of a linear 3x3, carrying normals correctly under a
// non-uniform scale. Copied verbatim from `shading_resolve.wesl` /
// `opaque.wesl` (cofactor columns over the determinant); returns identity for a
// singular matrix so a degenerate transform is a no-op. Mirrors the CPU golden
// `resolve::normal_matrix`.
fn scene_inverse_transpose_3x3(value: mat3x3<f32>) -> mat3x3<f32> {
    let x = cross(value[1], value[2]);
    let y = cross(value[2], value[0]);
    let z = cross(value[0], value[1]);
    let determinant = dot(value[2], z);
    if abs(determinant) <= 1.0e-8 {
        return mat3x3<f32>(1.0);
    }
    return mat3x3<f32>(x, y, z) / determinant;
}

// Lifts a local-space `SurfaceSampleGeometry` into world space with `transform`,
// returning the transformed copy. Position rides the full affine, the normal
// its inverse-transpose linear part, the tangent/bitangent the linear part; the
// UV is left untouched. Every rotated vector renormalizes with a fallback to
// its pre-transform value so a degenerate transform is a no-op, matching the
// resolve stage line-for-line.
fn scene_apply_current_transform(
    transform: GpuSceneTransform,
    geometry: SurfaceSampleGeometry,
) -> SurfaceSampleGeometry {
    var result = geometry;
    let world_from_local = scene_affine3_to_square(transform.world_from_local);
    let linear = mat3x3<f32>(
        world_from_local[0].xyz,
        world_from_local[1].xyz,
        world_from_local[2].xyz,
    );
    result.position = (world_from_local * vec4<f32>(geometry.position, 1.0)).xyz;
    result.normal =
        surface_normalize_or(scene_inverse_transpose_3x3(linear) * geometry.normal, geometry.normal);
    result.tangent = surface_normalize_or(linear * geometry.tangent, geometry.tangent);
    result.bitangent = surface_normalize_or(linear * geometry.bitangent, geometry.bitangent);
    return result;
}
