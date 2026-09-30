// project_vertex twin: projects world-space vertices through a clip matrix into
// y-down pixel-space `ScreenVertex`es on-device, mirroring the CPU golden
// `project_vertex` (in `prism_render_architecture::virtual_geometry`).
//
// The shipping meshlet software-raster vertex path transforms each cluster
// vertex world -> clip -> ndc -> viewport pixels once, before the screen-space
// fill (`rasterize_triangle`) and gradient setup (`TriangleGradients::new`)
// consume the resulting `ScreenVertex`es. The CPU golden `project_vertex` pins
// that transform:
//
//   clip  = clip_from_world * vec4(world, 1)   // column-major, WGSL `m * v`
//   ndc   = clip.xyz / clip.w                  // perspective divide
//   uv    = ndc.xy * vec2(0.5, -0.5) + 0.5     // ndc_to_uv (y flipped, y-down)
//   pos   = uv * viewport                      // viewport = (width, height)
//   depth = ndc.z                              // reversed-Z, 1.0 nearest
//
// This kernel is the on-device twin that runs one thread per vertex and writes
// the same `pos`/`depth` the reference derives, so the vis-buffer path's
// per-vertex projection stage is validated bit-for-bit against the reference
// before feeding the already-landed screen-space fill and gradient twins.
//
// Culling convention matches the reference exactly: a vertex on or behind the
// camera plane (`clip.w <= 0`, where the perspective divide is undefined) is
// the case the reference returns `None` for; the twin marks it invalid
// (`valid = 0u`) and zeroes the payload. The shipping shader relies on upstream
// near-plane culling to exclude these; the twin makes that precondition
// explicit rather than emitting a garbage projection.
//
// `clip_from_world` is a column-major `mat4x4<f32>`, matching the CPU
// reference's column-major storage: `m[c]` is column `c`, so
// `clip[r] = sum over c of m[c][r] * world_h[c]`. The scalar product below is
// summed left-to-right (`c = 0,1,2,3`) to match the reference association.
//
// For world positions and matrix entries on integer / dyadic coordinates each
// product and partial sum is exactly representable, so the clip value is
// identical on CPU and GPU regardless of fused-multiply-add contraction. When
// `clip.w` is an exact power of two the reciprocal is exact, the `ndc` divide is
// exact, and the affine `ndc_to_uv` plus `uv * viewport` scaling stay exact for
// power-of-two viewports, so the parity test asserts field-for-field equality
// rather than a tolerance. It uses only the portable core-WGSL subset - no
// optional feature - so this twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard world -> clip -> ndc -> viewport vertex projection for
// software rasterization; no Unreal Engine source or derived code.

struct Params {
    // Column-major world-to-clip matrix, matching the reference storage.
    clip_from_world: mat4x4<f32>,
    // Render-target size in pixels, `(width, height)`.
    viewport: vec2<f32>,
    // Number of vertices in `world_positions` / outputs in `screen`.
    vertex_count: u32,
    pad0: u32,
}

// One world-space input vertex. `12`-byte stride matching the `bytemuck`
// upload struct: (x, y, z).
struct Vertex {
    x: f32,
    y: f32,
    z: f32,
}

// One projected screen vertex. `16`-byte stride.
struct ScreenVertex {
    // `1u` in front of the camera (projection valid), `0u` on/behind the plane.
    valid: u32,
    // Pixel-space position, y-down.
    x: f32,
    y: f32,
    // Reversed-Z depth in `[0, 1]`, `1.0` nearest.
    depth: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> world_positions: array<Vertex>;
@group(0) @binding(2) var<storage, read_write> screen: array<ScreenVertex>;

fn zeroed() -> ScreenVertex {
    return ScreenVertex(0u, 0.0, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.vertex_count) {
        return;
    }

    let v = world_positions[i];
    let m = params.clip_from_world;

    // Column-major matrix-vector product, summed left-to-right (c = 0,1,2,3)
    // to match the reference association. `w = 1` for the homogeneous point.
    let clip_x = m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z + m[3][0];
    let clip_y = m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z + m[3][1];
    let clip_z = m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z + m[3][2];
    let clip_w = m[0][3] * v.x + m[1][3] * v.y + m[2][3] * v.z + m[3][3];

    if (clip_w <= 0.0) {
        // On or behind the camera plane: the reference returns `None` here.
        screen[i] = zeroed();
        return;
    }

    let inv_w = 1.0 / clip_w;
    let ndc_x = clip_x * inv_w;
    let ndc_y = clip_y * inv_w;
    let ndc_z = clip_z * inv_w;

    // `ndc_to_uv`: flips y so uv is y-down, then scales into the viewport.
    let u = ndc_x * 0.5 + 0.5;
    let vv = ndc_y * -0.5 + 0.5;

    screen[i] = ScreenVertex(
        1u,
        u * params.viewport.x,
        vv * params.viewport.y,
        ndc_z,
    );
}
