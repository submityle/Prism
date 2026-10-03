// Water caustic-projection compute kernel
// (entry point `water_caustics_project`).
//
// When sunlight refracts through a wavy surface the refracted rays converge
// and diverge, concentrating irradiance into the bright dancing patterns
// called caustics. The cheap real-time route projects, per receiver texel, the
// inverse area distortion (the Jacobian determinant) of the refracted-ray
// mapping: where the surface focuses light the patch shrinks, the Jacobian is
// small, and the caustic gain is large; where it defocuses, the gain fades.
// This pass runs that light-space Jacobian projection across the screen and
// adds the resulting achromatic caustic irradiance onto the lit scene colour.
//
// It mirrors the CPU golden `caustics::jacobian_caustic_gain` and
// `caustics::project_caustic_intensity` byte-for-byte. The route *selection*
// (`caustics::select_caustics`) and the offline photon-density estimate
// (`caustics::photon_splat_density`) stay on the CPU; this kernel is the
// real-time Jacobian route only, the one every shipping engine rasterizes.
//
// ABI (matches `WaterKernel::CausticsProject.descriptor()`: 1 storage buffer,
// 1 uniform, 1 storage texture, 1 sampled texture, 8x8 tile, Screen domain):
//   @binding(0) params      : CausticsParams uniform
//   @binding(1) receiver_in : array<vec4<f32>>, per-texel light-space
//                             (incident_irradiance, jacobian, _, _)
//   @binding(2) scene_in    : texture_2d<f32>, the lit scene colour
//   @binding(3) caustic_out : texture_storage_2d<rgba32float, write>, the lit
//                             scene with the caustic irradiance added (written)
//
// Only `+ - * /`, comparisons, `abs`, `min` and `max` appear — no float
// intrinsic the workspace determinism policy forbids. `EPS` matches the crate's
// `water::EPS` (1e-6) exactly so the near-singular Jacobian guard fires on the
// same inputs as the CPU twin.

const EPS: f32 = 1e-6;

struct CausticsParams {
    max_gain: f32,
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<uniform> params: CausticsParams;
@group(0) @binding(1) var<storage, read> receiver_in: array<vec4<f32>>;
@group(0) @binding(2) var scene_in: texture_2d<f32>;
@group(0) @binding(3) var caustic_out: texture_storage_2d<rgba32float, write>;

// Caustic gain 1 / |jacobian| clamped to `max_gain`; a near-singular focus
// saturates at the cap instead of dividing by zero.
fn jacobian_caustic_gain(jacobian: f32, max_gain: f32) -> f32 {
    let cap = max(max_gain, 0.0);
    let mag = abs(jacobian);
    if (mag <= EPS) {
        return cap;
    }
    return min(1.0 / mag, cap);
}

// Caustic irradiance at a receiver: incident light times the Jacobian gain.
fn project_caustic_intensity(incident: f32, jacobian: f32, max_gain: f32) -> f32 {
    return max(incident, 0.0) * jacobian_caustic_gain(jacobian, max_gain);
}

@compute @workgroup_size(8, 8, 1)
fn water_caustics_project(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let idx = gid.y * params.width + gid.x;
    let rec = receiver_in[idx];
    let incident = rec.x;
    let jacobian = rec.y;

    let scene = textureLoad(scene_in, coord, 0);
    let intensity = project_caustic_intensity(incident, jacobian, params.max_gain);

    // The caustic is achromatic added irradiance; the scene alpha is preserved.
    textureStore(
        caustic_out,
        coord,
        vec4<f32>(scene.x + intensity, scene.y + intensity, scene.z + intensity, scene.w),
    );
}
