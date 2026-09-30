// Fused per-cluster decision twin: composes cull -> LOD -> raster-path on-device
// into the single verdict a GPU-driven cluster loop consumes, mirroring the CPU
// golden `ViewCullContext::decide`.
//
// A GPU-driven virtual-geometry pipeline decides every candidate cluster once
// per frame: cull first, and only a surviving cluster derives a streaming
// priority, selects a LOD (display + prefetch) and picks a raster path. A culled
// cluster carries no LOD, no raster path and a zero priority. The CPU golden
// `ViewCullContext::decide` (in `prism_render_architecture::virtual_geometry`)
// owns that composition; this kernel is the on-device twin that runs one thread
// per cluster and writes the same decision the reference returns (minus the
// host-side residency bookkeeping `decide` performs on the page table, which is
// mutable state that stays on the host).
//
// The three constituent decisions reuse the exact term order of their standalone
// twins (`cluster_cull.wesl`, `lod_select.wesl`, and the `raster_path`
// classifier): the signed-distance/projected-extent frustum test, the
// `geometric_error.max(0) * focal / max(distance, EPS)` projected error, the
// hysteresis level pick and velocity-scaled prefetch, and the
// `triangle_count == 0` / `max_triangle_pixels <= threshold` / capability cascade.
// Every branch is an integer compare or a float compare on identical operands,
// so away from a razor-thin budget boundary the discrete decision is identical
// to the reference regardless of fused-multiply-add contraction. Priority is a
// single multiply-then-divide of correctly-rounded IEEE operations. The kernel
// uses only the portable core-WGSL subset - no optional feature - so this twin
// runs unmodified on Metal, Vulkan and DX12.
//
// All clusters in a dispatch share one LOD chain (the common case: every cluster
// of a mesh shares its level chain and differs only by view distance, closing
// speed and previously displayed level), exactly like `lod_select.wesl`.
//
// Verdict indices match `CullVerdict as u32`: 0=Visible, 1=FrustumCulled,
// 2=OcclusionCulled. Raster-path indices match `GeometryRasterPath as u32`:
// 0=MeshShader, 1=ComputeSoftware, 2=IndirectHardware, 3=FallbackMesh.
//
// Provenance: standard frustum/Hi-Z culling, screen-space-error LOD selection
// and software/hardware raster classification; no Unreal Engine source or
// derived code.

// `f32::EPSILON`, matching the reference distance clamp.
const F32_EPSILON: f32 = 1.1920929e-7;

const VERDICT_VISIBLE: u32 = 0u;
const VERDICT_FRUSTUM_CULLED: u32 = 1u;
const VERDICT_OCCLUSION_CULLED: u32 = 2u;

const RASTER_MESH_SHADER: u32 = 0u;
const RASTER_COMPUTE_SOFTWARE: u32 = 1u;
const RASTER_INDIRECT_HARDWARE: u32 = 2u;
const RASTER_FALLBACK_MESH: u32 = 3u;

struct Params {
    // Six inward-facing frustum planes packed as (normal.xyz, distance).
    planes: array<vec4<f32>, 6>,
    // Half viewport height / tan(0.5 * vertical fov).
    focal_length_pixels: f32,
    // Projected geometric-error budget in pixels.
    target_error_pixels: f32,
    // Dead-band in pixels a level must cross before the selector switches.
    hysteresis_pixels: f32,
    // World-space look-ahead applied per unit of closing speed.
    prefetch_velocity_scale: f32,
    // Screen area below which a cluster prefers the software rasterizer.
    software_pixel_threshold: f32,
    // `1` when the mesh/amplification shader path is available.
    mesh_shader: u32,
    // `1` when hardware indirect (multi-)draw is available.
    hardware_indirect: u32,
    // Number of valid entries in the shared `chain`.
    level_count: u32,
    // Number of clusters in `clusters` / outputs in `decisions`.
    cluster_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One LOD chain entry; `8`-byte stride.
struct ChainLevel {
    level: u32,
    geometric_error: f32,
}

// One cluster's decision inputs; `64`-byte stride matching the upload struct.
struct Cluster {
    // AABB centre in world space.
    center: vec3<f32>,
    // Bounding-sphere radius used for the streaming priority.
    radius: f32,
    // AABB half-extents (support radius per axis).
    half_extents: vec3<f32>,
    // Depth of the cluster's closest point (smaller is nearer).
    closest_depth: f32,
    // Conservative nearest occluder depth over the cluster footprint.
    occluder_depth: f32,
    // `1` when the occlusion probe is present, `0` when it should be skipped.
    has_occlusion: u32,
    // View-space distance to the cluster.
    view_distance: f32,
    // Component of relative velocity reducing that distance (positive =
    // approaching), world units per selection step.
    closing_speed: f32,
    // Largest projected triangle area in the cluster, in pixels squared.
    max_triangle_pixels: f32,
    // Number of triangles in the cluster.
    triangle_count: u32,
    // `1` when `previous_level` holds a valid prior display level.
    has_previous: u32,
    // Display level chosen on the prior step (used for hysteresis).
    previous_level: u32,
}

// One cluster's fused decision; `32`-byte stride.
struct Decision {
    // Cull verdict index.
    verdict: u32,
    // `1` when a LOD selection is present (visible with a non-empty chain).
    has_lod: u32,
    // Display level (valid only when `has_lod == 1`).
    level: u32,
    // Prefetch level (valid only when `has_lod == 1`).
    prefetch_level: u32,
    // Raster-path index (valid only when visible).
    raster_path: u32,
    // Streaming priority in pixels (0 when not visible).
    priority: f32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> chain: array<ChainLevel>;
@group(0) @binding(2) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(3) var<storage, read_write> decisions: array<Decision>;

// Mirrors `LodProjection::projected_error_pixels`.
fn projected_error(geometric_error: f32, view_distance: f32) -> f32 {
    let distance = max(view_distance, F32_EPSILON);
    return max(geometric_error, 0.0) * params.focal_length_pixels / distance;
}

// Sentinel for "no chain level matched".
const NO_LEVEL: u32 = 0xffffffffu;

// Coarsest chain level within `budget` at `distance`; finest when none fit.
fn coarsest_within(view_distance: f32, budget: f32) -> u32 {
    var finest: u32 = 0u;
    var have_finest = false;
    var best: u32 = NO_LEVEL;
    var have_best = false;
    for (var i = 0u; i < params.level_count; i = i + 1u) {
        let c = chain[i];
        if (!have_finest || c.level < finest) {
            finest = c.level;
            have_finest = true;
        }
        if (projected_error(c.geometric_error, view_distance) <= budget) {
            if (!have_best || c.level > best) {
                best = c.level;
                have_best = true;
            }
        }
    }
    if (have_best) {
        return best;
    }
    return finest;
}

// Projected error of the chain entry whose stored level equals `level`.
fn error_of(view_distance: f32, level: u32, out_error: ptr<function, f32>) -> bool {
    for (var i = 0u; i < params.level_count; i = i + 1u) {
        let c = chain[i];
        if (c.level == level) {
            *out_error = projected_error(c.geometric_error, view_distance);
            return true;
        }
    }
    return false;
}

// Chooses one level with hysteresis against an optional `previous`.
fn pick_level(view_distance: f32, has_previous: bool, previous: u32) -> u32 {
    let target_px = max(params.target_error_pixels, 0.0);
    let hysteresis = max(params.hysteresis_pixels, 0.0);
    let desired = coarsest_within(view_distance, target_px);

    if (!has_previous) {
        return desired;
    }
    var previous_error: f32 = 0.0;
    if (!error_of(view_distance, previous, &previous_error)) {
        return desired;
    }
    if (desired == previous) {
        return previous;
    }
    if (desired < previous) {
        if (previous_error > target_px + hysteresis) {
            return desired;
        }
        return previous;
    }
    var desired_error: f32 = 0.0;
    if (!error_of(view_distance, desired, &desired_error)) {
        return previous;
    }
    if (desired_error <= max(target_px - hysteresis, 0.0)) {
        return desired;
    }
    return previous;
}

// Mirrors `select_raster_path`.
fn select_raster_path(max_triangle_pixels: f32, triangle_count: u32) -> u32 {
    if (triangle_count == 0u) {
        return RASTER_FALLBACK_MESH;
    }
    if (max_triangle_pixels <= max(params.software_pixel_threshold, 0.0)) {
        return RASTER_COMPUTE_SOFTWARE;
    }
    if (params.mesh_shader == 1u) {
        return RASTER_MESH_SHADER;
    }
    if (params.hardware_indirect == 1u) {
        return RASTER_INDIRECT_HARDWARE;
    }
    return RASTER_FALLBACK_MESH;
}

// One thread per cluster.
@compute @workgroup_size(64, 1, 1)
fn decide(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.cluster_count) {
        return;
    }
    let c = clusters[i];

    // Cull first: reject on the first plane that fully excludes the AABB.
    for (var p = 0u; p < 6u; p = p + 1u) {
        let plane = params.planes[p];
        let signed_distance =
            plane.x * c.center.x + plane.y * c.center.y + plane.z * c.center.z + plane.w;
        let projected_extent =
            abs(plane.x) * c.half_extents.x
            + abs(plane.y) * c.half_extents.y
            + abs(plane.z) * c.half_extents.z;
        if (signed_distance < -projected_extent) {
            decisions[i] = Decision(VERDICT_FRUSTUM_CULLED, 0u, 0u, 0u, 0u, 0.0, 0u, 0u);
            return;
        }
    }
    if (c.has_occlusion == 1u && c.closest_depth > c.occluder_depth) {
        decisions[i] = Decision(VERDICT_OCCLUSION_CULLED, 0u, 0u, 0u, 0u, 0.0, 0u, 0u);
        return;
    }

    // Visible: streaming priority is the sphere-radius projected error.
    let priority = max(c.radius, 0.0) * params.focal_length_pixels / max(c.view_distance, F32_EPSILON);

    // LOD: only when the shared chain is non-empty (else the golden `select_lod`
    // returns `None`).
    var has_lod = 0u;
    var level = 0u;
    var prefetch_level = 0u;
    if (params.level_count > 0u) {
        has_lod = 1u;
        let has_prev = c.has_previous == 1u;
        let display = pick_level(c.view_distance, has_prev, c.previous_level);
        let look_ahead = max(c.closing_speed, 0.0) * params.prefetch_velocity_scale;
        let prefetch_distance = max(c.view_distance - look_ahead, 0.0);
        let prefetch = pick_level(prefetch_distance, true, display);
        level = display;
        prefetch_level = min(prefetch, display);
    }

    let raster_path = select_raster_path(c.max_triangle_pixels, c.triangle_count);

    decisions[i] = Decision(
        VERDICT_VISIBLE, has_lod, level, prefetch_level, raster_path, priority, 0u, 0u);
}
