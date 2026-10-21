struct Params {
    body_count: u32,
    field_count: u32,
    _pad: vec2<u32>,
}

struct PackedField {
    // x = shape, y = effect, z = falloff, w = target mask.
    tags: vec4<u32>,
    shape_a: vec4<f32>,
    shape_b: vec4<f32>,
    effect_a: vec4<f32>,
    effect_b: vec4<f32>,
    attenuation: vec4<f32>,
}

struct PackedBody {
    // xyz = position, w = mass.
    position_mass: vec4<f32>,
    velocity: vec4<f32>,
    // x = target mask.
    mask: vec4<u32>,
}

struct PackedSample {
    force: vec4<f32>,
    torque: vec4<f32>,
}

struct Weight {
    valid: bool,
    value: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> fields: array<PackedField>;
@group(0) @binding(2) var<storage, read> bodies: array<PackedBody>;
@group(0) @binding(3) var<storage, read_write> output: array<PackedSample>;

const EPS: f32 = 1.0e-6;
const HALF_PI: f32 = 1.5707963267948966;
const MAX_FINITE: f32 = 3.4028234663852886e38;

fn finite_scalar(value: f32) -> bool {
    return value == value && abs(value) <= MAX_FINITE;
}

fn finite3(value: vec3<f32>) -> bool {
    return finite_scalar(value.x) && finite_scalar(value.y) && finite_scalar(value.z);
}

fn safe_normalize(value: vec3<f32>) -> vec3<f32> {
    let squared = dot(value, value);
    if (finite3(value) && finite_scalar(squared) && squared > EPS * EPS) {
        return value / sqrt(squared);
    }
    return vec3<f32>(0.0);
}

fn falloff_weight(normalized: f32, field: PackedField) -> f32 {
    let x = clamp(normalized, 0.0, 1.0);
    if (field.tags.z == 1u) {
        return 1.0 - x;
    }
    if (field.tags.z == 2u) {
        return 1.0 / (1.0 + x * x / max(field.attenuation.x, EPS));
    }
    return 1.0;
}

fn invalid_weight() -> Weight {
    return Weight(false, 0.0);
}

fn make_weight(value: f32) -> Weight {
    if (!finite_scalar(value)) {
        return invalid_weight();
    }
    return Weight(true, clamp(value, 0.0, 1.0));
}

fn shape_weight(field: PackedField, position: vec3<f32>) -> Weight {
    if (field.tags.z == 2u && (!finite_scalar(field.attenuation.x) || field.attenuation.x < 0.0)) {
        return invalid_weight();
    }

    if (field.tags.x == 0u) {
        return make_weight(1.0);
    }
    if (field.tags.x == 1u) {
        let center = field.shape_a.xyz;
        let radius = field.shape_a.w;
        if (!finite3(center) || !finite_scalar(radius) || radius <= EPS) {
            return invalid_weight();
        }
        let distance = length(position - center);
        if (distance > radius) {
            return invalid_weight();
        }
        return make_weight(falloff_weight(distance / radius, field));
    }
    if (field.tags.x == 2u) {
        let min_bound = field.shape_a.xyz;
        let max_bound = field.shape_b.xyz;
        if (!finite3(min_bound) || !finite3(max_bound) || !all(min_bound < max_bound)) {
            return invalid_weight();
        }
        if (any(position < min_bound) || any(position > max_bound)) {
            return invalid_weight();
        }
        let extent = max(max_bound - min_bound, vec3<f32>(EPS));
        let fraction = (position - min_bound) / extent;
        let edges = min(fraction, vec3<f32>(1.0) - fraction);
        let edge = clamp(min(min(edges.x, edges.y), edges.z), 0.0, 0.5) * 2.0;
        return make_weight(falloff_weight(1.0 - edge, field));
    }
    if (field.tags.x == 3u) {
        let apex = field.shape_a.xyz;
        let length_limit = field.shape_a.w;
        let axis = field.shape_b.xyz;
        let angle = field.shape_b.w;
        let unit_axis = safe_normalize(axis);
        if (!finite3(apex) || !finite3(axis) || all(unit_axis == vec3<f32>(0.0))
            || !finite_scalar(length_limit) || length_limit <= EPS
            || !(angle >= 0.0 && angle < HALF_PI)) {
            return invalid_weight();
        }
        let delta = position - apex;
        let axial = dot(delta, unit_axis);
        if (axial < 0.0 || axial > length_limit) {
            return invalid_weight();
        }
        let radial = length(delta - unit_axis * axial);
        let radius = axial * tan(angle);
        if (radial > radius + EPS) {
            return invalid_weight();
        }
        return make_weight(falloff_weight(axial / length_limit, field));
    }
    return invalid_weight();
}

fn field_force(field: PackedField, position: vec3<f32>, velocity: vec3<f32>, mass: f32) -> vec3<f32> {
    if (field.tags.y == 1u) {
        return safe_normalize(position - field.effect_a.xyz) * field.effect_a.w;
    }
    if (field.tags.y == 2u) {
        return safe_normalize(cross(safe_normalize(field.effect_b.xyz), position - field.effect_a.xyz))
            * field.effect_a.w;
    }
    if (field.tags.y == 3u) {
        return -velocity * max(field.effect_a.w, 0.0) * max(mass, 0.0);
    }
    if (field.tags.y == 4u) {
        return -field.effect_a.xyz * max(field.effect_a.w, 0.0) * max(field.effect_b.w, 0.0);
    }
    return field.effect_a.xyz;
}

@compute @workgroup_size(64)
fn sample_fields(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if (index >= params.body_count) {
        return;
    }

    let body = bodies[index];
    let position = body.position_mass.xyz;
    let velocity = body.velocity.xyz;
    let mass = body.position_mass.w;
    var accumulated = vec3<f32>(0.0);

    if (finite3(position) && finite3(velocity) && finite_scalar(mass)) {
        for (var field_index = 0u; field_index < params.field_count; field_index = field_index + 1u) {
            let field = fields[field_index];
            if ((field.tags.w & body.mask.x) == 0u) {
                continue;
            }
            let weight = shape_weight(field, position);
            if (!weight.valid) {
                continue;
            }
            let weighted_force = field_force(field, position, velocity, mass) * weight.value;
            if (finite3(weighted_force)) {
                accumulated = accumulated + weighted_force;
            }
        }
    }

    output[index] = PackedSample(vec4<f32>(accumulated, 0.0), vec4<f32>(0.0));
}
