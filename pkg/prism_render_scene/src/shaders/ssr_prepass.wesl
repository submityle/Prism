//! SSR geometry prepass compute entry point.
//!
//! Screen-space reflection traces against device depth and reflects off the
//! view-space surface normal, but the Prism renderer is a *visibility buffer*
//! deferred pipeline: there is no depth or normal G-buffer to sample. This
//! prepass rebuilds exactly those two inputs the trace (`ssr.wesl`, and its CPU
//! golden `prism_render_shading::screen_space`) consumes, one texel per
//! framebuffer pixel:
//!
//! * `scene_depth`  — reverse-Z device depth (`clip.z / clip.w`); `0` marks
//!   background, matching the trace's `device_depth <= 0` early-out and the HZB
//!   "nearest depth" max-reduction convention (near = 1, far = 0).
//! * `view_normal`  — the unit view-space surface normal, later repacked with
//!   the material roughness into the trace's `normal_roughness` input.
//!
//! The per-pixel reconstruction mirrors `gtao_prepass.wesl` and
//! `shading_resolve.wesl` branch for branch — decode the visibility
//! ids/metadata, generation-validate the scene instance / geometry header /
//! primitive / vertices, then `interpolate_surface` the barycentric sample —
//! but stops at the *geometry* (no material fetch, no lighting), reusing the
//! same `surface.wesl` helpers so the depth/normal SSR reflects against are the
//! identical surface the resolve stage shades. It differs from the GTAO prepass
//! only in the depth it emits: SSR needs the *device* depth its screen-space
//! march and HZB compare against, not GTAO's positive linear view distance.

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

// Immediate (push-constant) block. Mirrors `GpuSsrPrepassParams` in
// `shading/ssr/abi.rs` (144 bytes: two 16-byte-aligned mat4x4 followed by the
// framebuffer dimensions and two padding scalars).
struct PrepassParams {
    // World -> view (rigid, camera-at-origin, looking down -Z).
    view_from_world: mat4x4<f32>,
    // View -> clip (reverse-Z perspective), used to derive device depth.
    clip_from_view: mat4x4<f32>,
    width: u32,
    height: u32,
    pad0: u32,
    pad1: u32,
}

// --- group 0: visibility inputs + view-space geometry outputs ---
@group(0) @binding(0) var visibility_ids: texture_2d<u32>;
@group(0) @binding(1) var visibility_metadata: texture_2d<u32>;
// Reverse-Z device depth; `0` = background / far plane.
@group(0) @binding(2) var scene_depth_out: texture_storage_2d<r32float, write>;
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
fn ssr_prepass(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let coord = vec2<i32>(i32(id.x), i32(id.y));

    var geometry: SurfaceSampleGeometry;
    if !reconstruct_surface(coord, &geometry) {
        // Background: `0` device depth is the trace's far-plane / "no surface"
        // sentinel and the identity element of the HZB max-reduction.
        textureStore(scene_depth_out, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(view_normal_out, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // World -> view. Vertices are stored world-space (the resolve stage feeds
    // `geometry.position` straight into world-space lighting), so a single
    // `view_from_world` multiply lands us in the RH, camera-at-origin,
    // looking-down-`-Z` frame the trace reconstructs against.
    let view_pos = params.view_from_world * vec4<f32>(geometry.position, 1.0);
    // The view transform is rigid (orthonormal 3x3), so the same matrix rotates
    // normals without an inverse-transpose; `w = 0` drops the translation.
    let view_normal_raw = (params.view_from_world * vec4<f32>(geometry.normal, 0.0)).xyz;
    let length_sq = dot(view_normal_raw, view_normal_raw);
    var view_normal = vec3<f32>(0.0, 0.0, 1.0);
    if length_sq > 1.0e-12 && length_sq < 3.4e38 {
        view_normal = view_normal_raw * inverseSqrt(length_sq);
    }

    // Project the view position through the reverse-Z projection and take the
    // post-divide device depth `clip.z / clip.w`. A surface at/behind the
    // camera (`clip.w <= 0`) cannot be seen; store `0` so the trace treats it
    // as background.
    let clip = params.clip_from_view * view_pos;
    var device_depth = 0.0;
    if clip.w > 1.0e-6 {
        // Nudge above zero so a surface exactly on the far plane is not misread
        // as background by the trace's `device_depth > 0` test.
        device_depth = max(clip.z / clip.w, 1.0e-7);
    }
    textureStore(scene_depth_out, coord, vec4<f32>(device_depth, 0.0, 0.0, 0.0));
    textureStore(view_normal_out, coord, vec4<f32>(view_normal, 0.0));
}
