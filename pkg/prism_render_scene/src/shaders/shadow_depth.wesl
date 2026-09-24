//! Shadow-map depth rasterization pass.
//!
//! This is the GPU pass that *fills* the shadow atlas that `shadow.wesl` later
//! samples.  It renders scene geometry from a shadow view — a directional
//! cascade (matrix from `prism_render_shading::shadow::csm`), a spot-light
//! frustum, or one point-light cube face — and stores, per fragment, either:
//!
//! * the wgpu NDC depth `z in [0, 1]` (directional / spot cascades), or
//! * the light-range-normalized linear distance `dist / range in [0, 1]`
//!   (point-light cube faces),
//!
//! into the atlas layer's `.r` channel, exactly matching the storage convention
//! that `shadow.wesl` reads back (`reference_depth <= stored_depth` ⇒ lit;
//! point lights compare normalized distances).
//!
//! The pass is deliberately self-contained (no `import`s): its only inputs are
//! one per-view uniform and the GPU-scene instance / transform buffers, so it
//! compiles and type-checks standalone.  One draw is issued per atlas layer
//! (cascade or cube face); the `ShadowDepthView` uniform carries that layer's
//! world → light-clip matrix and, for point lights, the light position and
//! inverse range used to normalize distance.

// Storage modes for the atlas `.r` channel.
const SHADOW_DEPTH_MODE_NDC: u32 = 0u; // store NDC depth (directional / spot)
const SHADOW_DEPTH_MODE_DISTANCE: u32 = 1u; // store dist / range (point cube face)

// Per-shadow-view constants for the layer currently being rendered.
struct ShadowDepthView {
    // Column-major world -> light-clip matrix (wgpu clip: z in [0, 1]).
    view_projection: mat4x4<f32>,
    // xyz: light world position; w: 1 / range (only read in distance mode).
    light_position: vec4<f32>,
    // x: storage mode (SHADOW_DEPTH_MODE_*); y, z, w: reserved (0).
    params: vec4<u32>,
}
@group(0) @binding(0) var<uniform> shadow_view: ShadowDepthView;

// Mirror of the GPU-scene layout owned by `prism_render_architecture::gpu_scene`
// (kept structurally identical to `visibility_raster.wesl`).
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
    @location(0) world_position: vec3<f32>,
    @location(1) @interpolate(flat) valid: u32,
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
        shadow_view.view_projection * vec4<f32>(world_position, 1.0),
        valid,
    );
    out.world_position = world_position;
    out.valid = select(0u, 1u, valid);
    return out;
}

struct FragmentOutput {
    // Single-channel atlas store; `.r` holds the depth / distance.
    @location(0) stored: vec4<f32>,
}

@fragment
fn fragment(mesh: VertexOutput) -> FragmentOutput {
    var stored: f32;
    if shadow_view.params.x == SHADOW_DEPTH_MODE_DISTANCE {
        // Point light: normalized linear distance, matching shadow.wesl's cube
        // faces (stores dist / range, not NDC depth).
        let to_fragment = mesh.world_position - shadow_view.light_position.xyz;
        stored = clamp(length(to_fragment) * shadow_view.light_position.w, 0.0, 1.0);
    } else {
        // Directional / spot: the framebuffer builtin position already carries
        // the wgpu NDC depth in [0, 1].
        stored = clamp(mesh.clip_position.z, 0.0, 1.0);
    }

    var out: FragmentOutput;
    out.stored = vec4<f32>(stored, stored, stored, 1.0);
    return out;
}
