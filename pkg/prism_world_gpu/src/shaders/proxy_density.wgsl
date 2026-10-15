// Opt-in coverage stage between geometric visibility and compaction. Keep the
// shared camera/record ABI unchanged; no history keyed by mutable dense slots.
struct VisibilityParams {
    camera: vec4<f32>,
    projection_scale: f32,
    max_distance: f32,
    proxy_count: u32,
    reserved: u32,
    frustum: array<vec4<f32>, 6>,
};

struct DensityParams {
    density: f32,
    transition: f32,
    seed: u32,
    distance_fade: f32,
};

@group(0) @binding(0) var<uniform> camera: VisibilityParams;
@group(0) @binding(1) var<uniform> policy: DensityParams;
@group(0) @binding(2) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> metadata: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> coverage: array<f32>;
// Per-proxy range shares its source column with geometric visibility.
@group(0) @binding(5) var<storage, read> errors: array<vec4<f32>>;

fn density_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn smooth01(value: f32) -> f32 {
    let c = clamp(value, 0.0, 1.0);
    var result = 0.0;
    if c < bitcast<f32>(95u << 23u) {
        if c < bitcast<f32>(63u << 23u) {
            return 0.0;
        }
        // A bit-level exponent shift cannot be algebraically reassociated
        // back into the original underflowing c*c by fast-math compilation.
        let scaled = bitcast<f32>(bitcast<u32>(c) + (32u << 23u));
        result = scaled * scaled * (3.0 - 2.0 * c) * bitcast<f32>(63u << 23u);
    } else {
        result = c * c * (3.0 - 2.0 * c);
    }
    return select(result, 0.0, result < 1.17549435e-38);
}

fn boost_small(value: f32) -> f32 {
    let bits = bitcast<u32>(value);
    if (bits & 0x7f800000u) != 0u {
        return value * 18446744073709551616.0;
    }
    let magnitude = f32(bits & 0x007fffffu) * bitcast<f32>(42u << 23u);
    return select(magnitude, -magnitude, (bits >> 31u) != 0u);
}

fn boosted_delta(a: f32, b: f32) -> f32 {
    if max(abs(a), abs(b)) > 1.0 {
        return (a - b) * 18446744073709551616.0;
    }
    return boost_small(a) - boost_small(b);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= camera.proxy_count {
        return;
    }
    let identity = metadata[index];
    let hash = density_hash(density_hash(identity.x ^ policy.seed) ^ identity.y);
    let rank = f32(hash >> 8u) / 16777216.0;
    // Avoid cancellation of a rounded product when transition is narrow.
    var q = policy.density;
    if policy.transition != 1.0 {
        q = rank + (policy.density - rank) / policy.transition;
    }
    var value = smooth01(q);
    if policy.distance_fade > 0.0 {
        let limit = min(camera.max_distance, errors[index].y);
        let width = min(policy.distance_fade, limit);
        if width <= 1.17549435e-38 {
            value = 0.0;
        } else {
            var delta = positions[index].xyz - camera.camera.xyz;
            let small = limit < 1.0;
            let boost = select(1.0, 18446744073709551616.0, small);
            if small {
                let p = positions[index].xyz;
                let c = camera.camera.xyz;
                delta = vec3<f32>(boosted_delta(p.x, c.x), boosted_delta(p.y, c.y), boosted_delta(p.z, c.z));
            }
            let scaled_limit = limit * boost;
            let scaled_width = width * boost;
            let scale = max(max(abs(delta.x), abs(delta.y)), max(abs(delta.z), 1.17549435e-38 * boost));
            if scale > scaled_limit {
                value = 0.0;
            } else {
                let normalized = delta / scale;
                let distance = sqrt(dot(normalized, normalized)) * scale;
                value *= smooth01((scaled_limit - distance) / scaled_width);
            }
        }
    }
    coverage[index] = select(value, 0.0, value < 1.17549435e-38);
}
