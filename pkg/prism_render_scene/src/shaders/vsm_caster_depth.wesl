//! Virtual-shadow-map caster depth rasterization (Slice A: physical-atlas fill).
//!
//! This is the GPU pass that *fills* the virtual-shadow-map physical page atlas
//! that `vsm_sample.wesl` later reads a resident page's stored depth back from.
//! For every resident clipmap page the allocator made resident this frame, the
//! raster pass renders the scene's shadow casters through that page's
//! orthographic light projection and stores, per fragment, the wgpu NDC depth
//! `z in [0, 1]` into the atlas tile's `.r` channel.
//!
//! The storage convention is exactly the one `vsm_sample.wesl` reads back:
//! `reference_depth <= stored` means lit (the near = 0, far = 1 wgpu depth
//! range), and the per-page projection is built (on the CPU, in
//! `atlas/projection.rs`) so that the sampler's in-page fraction addresses the
//! same world point this pass rasterizes to that texel:
//!
//!   * NDC.x =  2 * (world.light_right - page_origin.x) / page_world_size - 1
//!   * NDC.y =  1 - 2 * (world.light_up    - page_origin.y) / page_world_size
//!     (the y flip maps the page's lower-left world corner to the texture's top
//!     row, matching `vsm_sample.wesl`'s `atlas_uv = (tile + tap_frac) / edge`).
//!
//! Like `shadow_depth.wesl` this pass is deliberately self-contained (no
//! `import`s): its only inputs are one per-page uniform (the world -> light-clip
//! matrix) and the shared GPU-scene instance / transform buffers, so it compiles
//! and type-checks standalone. One draw list is replayed per resident page; the
//! bound `VsmCasterDepthView` uniform (selected by dynamic offset) carries that
//! page's projection matrix.

// Per-page constants for the resident page currently being rasterized.
struct VsmCasterDepthView {
    // Column-major world -> light-clip matrix for this page's orthographic light
    // projection (wgpu clip: z in [0, 1]). Built by `atlas/projection.rs`.
    view_projection: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> caster_view: VsmCasterDepthView;

// Mirror of the GPU-scene layout owned by `prism_render_architecture::gpu_scene`
// (kept structurally identical to `shadow_depth.wesl` / `visibility_raster.wesl`
// so the same `@group(1)` bind group drives every geometry pass).
struct GpuSceneInstance {
    generation: u32,
    flags: u32,
    geometry_index: u32,
    geometry_generation: u32,
    material_index: u32,
    material_generation: u32,
    render_layers: u32,
    is_active: u32,
}
struct GpuSceneTransform {
    world_from_local: mat3x4<f32>,
}

@group(1) @binding(0) var<storage, read> scene_instances: array<GpuSceneInstance>;
@group(1) @binding(1) var<storage, read> scene_transforms: array<GpuSceneTransform>;

// Expands the packed affine transform to a full 4x4 (matches
// `bevy_render::maths::affine3_to_square`, inlined to keep this shader
// import-free).
fn affine3_to_square(affine: mat3x4<f32>) -> mat4x4<f32> {
    return transpose(mat4x4<f32>(
        affine[0],
        affine[1],
        affine[2],
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    ));
}

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) valid: u32,
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let in_bounds = vertex.instance_index < arrayLength(&scene_instances);
    let index = select(0u, vertex.instance_index, in_bounds);
    let instance = scene_instances[index];
    let valid = in_bounds && instance.is_active != 0u && instance.generation != 0u;

    let world_from_local = affine3_to_square(scene_transforms[index].world_from_local);
    let world_position = (world_from_local * vec4<f32>(vertex.position, 1.0)).xyz;

    var out: VertexOutput;
    // Invalid instances are pushed past the far plane so they rasterize nothing.
    out.clip_position = select(
        vec4<f32>(2.0, 2.0, 2.0, 1.0),
        caster_view.view_projection * vec4<f32>(world_position, 1.0),
        valid,
    );
    out.valid = select(0u, 1u, valid);
    return out;
}

struct FragmentOutput {
    // Single-channel atlas store; `.r` holds the wgpu NDC depth.
    @location(0) stored: vec4<f32>,
}

@fragment
fn fragment(mesh: VertexOutput) -> FragmentOutput {
    // The framebuffer builtin position already carries the wgpu NDC depth in
    // [0, 1]; store it so `vsm_sample.wesl` can compare `reference_depth <=
    // stored` for its lit test.
    let stored = clamp(mesh.clip_position.z, 0.0, 1.0);

    var out: FragmentOutput;
    out.stored = vec4<f32>(stored, stored, stored, 1.0);
    return out;
}
