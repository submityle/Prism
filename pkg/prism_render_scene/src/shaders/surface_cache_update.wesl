// Surface-cache update pass: confidence-weighted temporal integration.
//
// One invocation per surfel. The invocation reads the frame's freshly sampled
// surfel (`scratch_current`, from the alloc pass) and the persistent surfel
// (`surfels`, carrying the across-frame `EMA` state) and blends them with the
// golden confidence-weighted exponential moving average, resetting the history
// on a disocclusion (anchor jump or normal flip). The persistent buffer is
// updated in place.
//
// Blend weight `alpha = 1 / min(sample_count + 1, max_samples)` (golden
// `confidence_alpha`): `1` on the first sample, falling as confidence accrues,
// floored at `1 / max_samples` so the cache keeps adapting. Mirrors
// `prism_render_shading::gi::surface_cache::integrate_radiance` /
// `is_disoccluded` verbatim.

const MIN_RADIUS: f32 = 1.0e-6;
const F32_FINITE_MAX: f32 = 3.40282347e38;

struct Surfel {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
    radius: f32,
    radiance_x: f32,
    radiance_y: f32,
    radiance_z: f32,
    sample_count: u32,
    valid: u32,
}

// Immediate block mirroring `GpuSurfaceCacheUpdateParams` (16 bytes).
struct UpdateParams {
    count: u32,
    max_samples: u32,
    position_tolerance: f32,
    normal_tolerance: f32,
}

var<immediate> params: UpdateParams;

@group(0) @binding(0) var scratch_current: array<Surfel>;
@group(0) @binding(1) var surfels: array<Surfel>;

fn is_finite(x: f32) -> bool {
    return x == x && abs(x) <= F32_FINITE_MAX;
}

fn finite_or_zero(x: f32) -> f32 {
    if (is_finite(x)) {
        return x;
    }
    return 0.0;
}

fn sanitize_rgb(c: vec3<f32>) -> vec3<f32> {
    let s = vec3<f32>(finite_or_zero(c.x), finite_or_zero(c.y), finite_or_zero(c.z));
    return max(s, vec3<f32>(0.0, 0.0, 0.0));
}

fn surfel_position(s: Surfel) -> vec3<f32> {
    return vec3<f32>(s.pos_x, s.pos_y, s.pos_z);
}

fn surfel_normal(s: Surfel) -> vec3<f32> {
    return vec3<f32>(s.normal_x, s.normal_y, s.normal_z);
}

fn surfel_radiance(s: Surfel) -> vec3<f32> {
    return vec3<f32>(s.radiance_x, s.radiance_y, s.radiance_z);
}

// Golden `is_disoccluded`: history is invalid when the anchor jumps beyond
// `position_tolerance * radius` or the normals disagree beyond
// `normal_tolerance`. Degenerate inputs are treated conservatively.
fn is_disoccluded(prev: Surfel, curr: Surfel) -> bool {
    let delta = surfel_position(curr) - surfel_position(prev);
    let offset = sqrt(dot(delta, delta));
    if (!is_finite(offset)) {
        return true;
    }
    let radius = max(prev.radius, curr.radius);
    let max_offset = radius * max(params.position_tolerance, 0.0);
    if (offset > max_offset) {
        return true;
    }
    let cos_n = dot(surfel_normal(prev), surfel_normal(curr));
    if (!is_finite(cos_n) || cos_n < params.normal_tolerance) {
        return true;
    }
    return false;
}

@compute @workgroup_size(64, 1, 1)
fn surface_cache_update_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.count) {
        return;
    }

    let prev = surfels[idx];
    let curr = scratch_current[idx];
    let has_history = prev.valid == 1u;

    // No fresh sample this frame: drop the slot.
    if (curr.valid == 0u) {
        surfels[idx].pos_x = 0.0;
        surfels[idx].pos_y = 0.0;
        surfels[idx].pos_z = 0.0;
        surfels[idx].normal_x = 0.0;
        surfels[idx].normal_y = 0.0;
        surfels[idx].normal_z = 1.0;
        surfels[idx].radius = MIN_RADIUS;
        surfels[idx].radiance_x = 0.0;
        surfels[idx].radiance_y = 0.0;
        surfels[idx].radiance_z = 0.0;
        surfels[idx].sample_count = 0u;
        surfels[idx].valid = 0u;
        return;
    }

    let sample = sanitize_rgb(surfel_radiance(curr));
    var radiance: vec3<f32>;
    var sample_count: u32;
    if (!has_history || is_disoccluded(prev, curr)) {
        // Reseed: snap to the fresh sample at confidence 1.
        radiance = sample;
        sample_count = 1u;
    } else {
        // Golden `confidence_alpha` + lerp: blend the history toward the
        // sample and advance confidence toward `max_samples`.
        let cap = max(params.max_samples, 1u);
        let n = min(prev.sample_count + 1u, cap);
        let alpha = 1.0 / f32(n);
        radiance = sanitize_rgb(mix(sanitize_rgb(surfel_radiance(prev)), sample, vec3<f32>(alpha, alpha, alpha)));
        sample_count = n;
    }

    // Geometry always from the fresh sample; radiance is the integrated value.
    surfels[idx].pos_x = curr.pos_x;
    surfels[idx].pos_y = curr.pos_y;
    surfels[idx].pos_z = curr.pos_z;
    surfels[idx].normal_x = curr.normal_x;
    surfels[idx].normal_y = curr.normal_y;
    surfels[idx].normal_z = curr.normal_z;
    surfels[idx].radius = curr.radius;
    surfels[idx].radiance_x = radiance.x;
    surfels[idx].radiance_y = radiance.y;
    surfels[idx].radiance_z = radiance.z;
    surfels[idx].sample_count = sample_count;
    surfels[idx].valid = 1u;
}
