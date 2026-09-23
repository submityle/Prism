//! Conservative reverse-Z HZB classification kernel. It writes one stage
//! mask per scene slot; command compaction consumes these masks separately.

struct HzbCullInput {
    uv_min: vec2<f32>,
    uv_max: vec2<f32>,
    nearest_depth: f32,
    projected_velocity: f32,
    is_active: u32,
    padding: u32,
}

struct HzbDispatch {
    candidate_count: u32,
    mip_count: u32,
    output_start: u32,
    phase: u32,
    history_valid: u32,
    camera_cut: u32,
    depth_bias: f32,
    fast_motion_threshold: f32,
}

@group(0) @binding(0) var<storage, read> candidates: array<HzbCullInput>;
@group(0) @binding(1) var hzb: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> stages: array<u32>;

var<immediate> dispatch: HzbDispatch;

fn conservative_occluder_depth(candidate: HzbCullInput, mip: u32) -> f32 {
    let size = vec2<u32>(textureDimensions(hzb, i32(mip)));
    let max_texel = max(size, vec2<u32>(1u)) - vec2<u32>(1u);
    let top_left = min(vec2<u32>(clamp(candidate.uv_min, vec2(0.0), vec2(1.0)) * vec2<f32>(size)), max_texel);
    let bottom_right = min(vec2<u32>(clamp(candidate.uv_max, vec2(0.0), vec2(1.0)) * vec2<f32>(size)), max_texel);
    let a = textureLoad(hzb, top_left, i32(mip)).x;
    let b = textureLoad(hzb, vec2(bottom_right.x, top_left.y), i32(mip)).x;
    let c = textureLoad(hzb, vec2(top_left.x, bottom_right.y), i32(mip)).x;
    let d = textureLoad(hzb, bottom_right, i32(mip)).x;
    return min(min(a, b), min(c, d));
}

@compute @workgroup_size(64)
fn classify_hzb(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= dispatch.candidate_count {
        return;
    }
    let output = dispatch.output_start + index;
    if output >= arrayLength(&candidates) || output >= arrayLength(&stages) {
        return;
    }
    let candidate = candidates[output];
    if candidate.is_active == 0u {
        stages[output] = 0u;
        return;
    }
    let pixel_size = max(
        abs(candidate.uv_max.x - candidate.uv_min.x) * f32(textureDimensions(hzb, 0).x),
        abs(candidate.uv_max.y - candidate.uv_min.y) * f32(textureDimensions(hzb, 0).y),
    );
    let mip = min(u32(max(0.0, ceil(log2(max(pixel_size, 1.0))))), dispatch.mip_count - 1u);
    let occluder_depth = conservative_occluder_depth(candidate, mip);
    let previous_phase = dispatch.phase == 0u;
    let history_unsafe = previous_phase && (dispatch.history_valid == 0u || dispatch.camera_cut != 0u);
    let fast_motion = abs(candidate.projected_velocity) > dispatch.fast_motion_threshold;
    let motion_bias = select(
        abs(candidate.projected_velocity) * 0.25,
        abs(candidate.projected_velocity),
        previous_phase,
    );
    let occluded = !history_unsafe
        && !fast_motion
        && candidate.nearest_depth + abs(dispatch.depth_bias) + motion_bias < occluder_depth;

    // EARLY=1, LATE_RETEST=2, LATE_VISIBLE=4.
    if previous_phase {
        stages[output] = select(1u, 2u, occluded);
    } else {
        let prior = stages[output];
        stages[output] = select(prior | 4u, prior, occluded && (prior & 2u) != 0u);
    }
}
