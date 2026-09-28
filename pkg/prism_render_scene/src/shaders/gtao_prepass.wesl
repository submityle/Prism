//! GTAO geometry prepass compute entry point.
//!
//! GTAO runs on view-space geometry, but the Prism renderer is a *visibility
//! buffer* deferred pipeline: there is no depth or normal G-buffer to sample.
//! This prepass rebuilds exactly the two inputs `gtao.wesl` (and its CPU golden
//! `prism_render_shading::ao`) consume, one texel per framebuffer pixel:
//!
//! * `linear_depth` — positive view-space distance (`-view_pos.z`); `<= 0`
//!   marks background, matching `GtaoBuffers::sample_depth`'s `None` sentinel.
//! * `view_normal`  — the unit view-space surface normal.
//!
//! The per-pixel reconstruction mirrors `shading_resolve.wesl` branch for
//! branch — decode the visibility ids/metadata, generation-validate the scene
//! instance / geometry header / primitive / vertices, then `interpolate_surface`
//! the barycentric sample — but stops at the *geometry* (no material fetch, no
//! lighting), because GTAO only needs position + normal. Reusing the same
//! `surface.wesl` helpers guarantees the depth/normal the kernel occludes
//! against are the identical surface the resolve stage later shades.

import prism_render_scene::shaders::surface::{
    RenderShadingGeometryHeader,
    GpuShadingVertex,
    GpuShadingPrimitive,
    SurfaceSampleGeometry,
    SHADING_GEOMETRY_FLAG_ACTIVE,
    surface_decode_barycentrics,
    interpolate_surface,
};
import prism_render_scene::shaders::gpu_scene::{
    GpuSceneInstance,
};

// Sentinel packed into the visibility targets for uncovered pixels. Matches
// `VisibilityPixel::is_valid` on the CPU side and `shading_resolve.wesl`.
const VISIBILITY_BACKGROUND: u32 = 0xffffffffu;

// Immediate (push-constant) block. Mirrors `GpuGtaoPrepassParams` in
// `shading/ao/abi.rs` (80 bytes: a 16-byte-aligned mat4x4 followed by the
// framebuffer dimensions and two padding scalars).
struct PrepassParams {
    // Row-major-free `mat4x4<f32>`; columns as uploaded by `Mat4`.
    view_from_world: mat4x4<f32>,
    width: u32,
    height: u32,
    pad0: u32,
    pad1: u32,
}

// --- group 0: visibility inputs + view-space geometry outputs ---
@group(0) @binding(0) var visibility_ids: texture_2d<u32>;
@group(0) @binding(1) var visibility_metadata: texture_2d<u32>;
// Positive linear view depth; `0` = background.
@group(0) @binding(2) var linear_depth_out: texture_storage_2d<r32float, write>;
// Unit view-space normal in `.xyz`.
@group(0) @binding(3) var view_normal_out: texture_storage_2d<rgba16float, write>;

// --- group 1: scene + geometry tables (a subset of the resolve group 2) ---
@group(1) @binding(0) var<storage, read> scene_instances: array<GpuSceneInstance>;
@group(1) @binding(1) var<storage, read> geometry_headers: array<RenderShadingGeometryHeader>;
@group(1) @binding(2) var<storage, read> geometry_vertices: array<GpuShadingVertex>;
@group(1) @binding(3) var<storage, read> geometry_primitives: array<GpuShadingPrimitive>;

var<immediate> params: PrepassParams;

// Reconstructs the world-space surface for one covered pixel, returning `false`
// (and leaving `geometry` untouched) whenever any generation/bounds guard
// rejects the sample, exactly like the resolve stage. Split out so the entry
// point stays a thin orchestrator.
fn reconstruct_surface(coord: vec2<i32>, geometry: ptr<function, SurfaceSampleGeometry>) -> bool {
    let ids = textureLoad(visibility_ids, coord, 0);
    let metadata = textureLoad(visibility_metadata, coord, 0);
    if ids.x == VISIBILITY_BACKGROUND || metadata.x == VISIBILITY_BACKGROUND {
        return false;
    }

    // ids = [scene_index, scene_generation, primitive_index, geometry_lod].
    let scene_index = ids.x;
    if scene_index >= arrayLength(&scene_instances) {
        return false;
    }
    let instance = scene_instances[scene_index];
    if instance.is_active == 0u || instance.generation != ids.y {
        return false;
    }

    let geometry_index = instance.geometry_index;
    if geometry_index >= arrayLength(&geometry_headers) {
        return false;
    }
    let header = geometry_headers[geometry_index];
    if (header.flags & SHADING_GEOMETRY_FLAG_ACTIVE) == 0u {
        return false;
    }
    if header.generation != instance.geometry_generation {
        return false;
    }

    let primitive_index = ids.z;
    if primitive_index >= header.primitive_count {
        return false;
    }
    let primitive_row = header.primitive_offset + primitive_index;
    if primitive_row >= arrayLength(&geometry_primitives) {
        return false;
    }
    let primitive = geometry_primitives[primitive_row];

    let index_a = primitive.indices[0];
    let index_b = primitive.indices[1];
    let index_c = primitive.indices[2];
    if index_a >= header.vertex_count
        || index_b >= header.vertex_count
        || index_c >= header.vertex_count {
        return false;
    }
    let row_a = header.vertex_offset + index_a;
    let row_b = header.vertex_offset + index_b;
    let row_c = header.vertex_offset + index_c;
    if row_a >= arrayLength(&geometry_vertices)
        || row_b >= arrayLength(&geometry_vertices)
        || row_c >= arrayLength(&geometry_vertices) {
        return false;
    }
    let vertex_a = geometry_vertices[row_a];
    let vertex_b = geometry_vertices[row_b];
    let vertex_c = geometry_vertices[row_c];

    let weights = surface_decode_barycentrics(metadata.w);
    *geometry = interpolate_surface(vertex_a, vertex_b, vertex_c, primitive.flags, weights);
    return true;
}

@compute @workgroup_size(8, 8, 1)
fn gtao_prepass(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let coord = vec2<i32>(i32(id.x), i32(id.y));

    var geometry: SurfaceSampleGeometry;
    if !reconstruct_surface(coord, &geometry) {
        // Background: `0` view depth is the kernel's "no sample" sentinel.
        textureStore(linear_depth_out, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(view_normal_out, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // World -> view. Vertices are stored world-space (the resolve stage feeds
    // `geometry.position` straight into world-space lighting), so a single
    // `view_from_world` multiply lands us in the RH, camera-at-origin,
    // looking-down-`-Z` frame `reconstruct()` assumes.
    let view_pos = params.view_from_world * vec4<f32>(geometry.position, 1.0);
    // The view transform is rigid (orthonormal 3x3), so the same matrix rotates
    // normals without an inverse-transpose; `w = 0` drops the translation.
    let view_normal_raw = (params.view_from_world * vec4<f32>(geometry.normal, 0.0)).xyz;
    let length_sq = dot(view_normal_raw, view_normal_raw);
    var view_normal = vec3<f32>(0.0, 0.0, 1.0);
    if length_sq > 1.0e-12 && length_sq < 3.4e38 {
        view_normal = view_normal_raw * inverseSqrt(length_sq);
    }

    // `-view_pos.z` is the positive distance in front of the camera; clamp to a
    // hair above zero so a surface exactly on the near plane is not misread as
    // background by the kernel's `depth > 0` test.
    let linear_depth = max(-view_pos.z, 1.0e-6);
    textureStore(linear_depth_out, coord, vec4<f32>(linear_depth, 0.0, 0.0, 0.0));
    textureStore(view_normal_out, coord, vec4<f32>(view_normal, 0.0));
}
