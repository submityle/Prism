struct VisibilityParams {
    camera: vec4<f32>,
    projection_scale: f32,
    max_distance: f32,
    proxy_count: u32,
    reserved: u32,
    frustum: array<vec4<f32>, 6>,
};

struct VisibilityRecord {
    visible: u32,
    lod: u32,
    error_quantized: u32,
    proxy_id: u32,
    screen_error: f32,
};

@group(0) @binding(0)
var<uniform> params: VisibilityParams;

@group(0) @binding(1)
var<storage, read> positions: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read> errors: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read> metadata: array<vec4<u32>>;

@group(0) @binding(4)
var<storage, read_write> output: array<VisibilityRecord>;

fn lod_for_error(screen_error: f32) -> u32 {
    if screen_error >= 64.0 {
        return 0u;
    }
    if screen_error >= 16.0 {
        return 1u;
    }
    if screen_error >= 4.0 {
        return 2u;
    }
    return 3u;
}

fn quantise_error(screen_error: f32) -> u32 {
    return u32(clamp(floor(screen_error * 256.0 + 0.5), 0.0, 65535.0));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index >= params.proxy_count {
        return;
    }

    let position_radius = positions[index];
    let error_range = errors[index];
    let proxy_id = metadata[index].x;
    var visible = 1u;

    for (var plane_index = 0u; plane_index < 6u; plane_index = plane_index + 1u) {
        let plane = params.frustum[plane_index];
        let signed_distance = dot(plane.xyz, position_radius.xyz) + plane.w;
        if signed_distance < -position_radius.w {
            visible = 0u;
        }
    }

    let delta = position_radius.xyz - params.camera.xyz;
    let distance_squared = dot(delta, delta);
    let global_limit_squared = params.max_distance * params.max_distance;
    let proxy_limit_squared = error_range.y * error_range.y;
    if distance_squared > global_limit_squared || distance_squared > proxy_limit_squared {
        visible = 0u;
    }

    var lod = 0u;
    var error_quantized = 0u;
    var screen_error = 0.0;
    if visible == 1u {
        let distance = sqrt(max(distance_squared, 1.0e-8));
        screen_error = error_range.x * params.projection_scale / distance;
        lod = lod_for_error(screen_error);
        error_quantized = quantise_error(screen_error);
    }

    output[index] = VisibilityRecord(visible, lod, error_quantized, proxy_id, screen_error);
}
