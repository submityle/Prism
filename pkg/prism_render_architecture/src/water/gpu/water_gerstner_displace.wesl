// Analytic Gerstner displacement compute kernel (entry `water_gerstner_displace`).
//
// Superposes `wave_count` trochoidal Gerstner wave trains into a per-texel
// world-space displacement field and the closed-form analytic surface normal,
// the near-field / interactive ocean tier that complements the spectral
// inverse-FFT path. Classical trochoid sum (GPU Gems "Effective Water
// Simulation from Physical Models"): for each wave with unit direction `D`,
// wave number `k = 2*PI / L`, angular frequency `omega = speed * sqrt(g*k)`,
// steepness `Q` in [0,1], crest amplitude `A`, and constant phase offset, at
// rest-plane position `p = (x, z)` and `theta = k (D.p) - omega t + phase`:
//
//   horizontal += Q * A * D * cos(theta)   (trochoidal crest bunching)
//   vertical   += A * sin(theta)
//
// The analytic normal accumulates from the flat upright surface (0,1,0) with
// `WA = k * A`: `n.xz -= D * WA * cos(theta)`, `n.y -= Q * WA * sin(theta)`,
// then is normalized (falling back to (0,1,0) for a degenerate accumulator).
// Deterministic: a fixed wave set and time produce the same texel every
// dispatch. Pure classical numerics — no AI/ML.
//
// This is the standalone contract twin of
// `prism_render_architecture::water::gpu::gerstner_displace_kernel`
// (the bit-exact CPU reference). The entry-point name is byte-identical to the
// `WaterKernel::GerstnerDisplace::wesl_entry_point` contract so the pipeline
// cache key resolves. ABI matches `WaterKernel::GerstnerDisplace::descriptor`:
// one storage buffer (the wave trains), one uniform param block, two
// `rgba32float` storage-texture outputs (displacement, normal), an 8x8 tile,
// and the `Grid2d` dispatch domain.
//
// The sandbox has no GPU, so `prism_render_architecture` (a zero-dependency
// crate) ships this source and covers it with the structural ABI test only;
// the standalone naga/wesl compile check runs out of tree, and the CPU twin is
// the texel-for-texel correctness proof.

// Squared-length / wave-number floor below which a direction is treated as
// degenerate and a `1/len` division is skipped; mirrors the CPU
// `prism_render_architecture::water::EPS_LEN_SQ`.
const WATER_EPS_LEN_SQ: f32 = 1.0e-12;

// Deep-water gravitational acceleration (m/s^2); mirrors the CPU
// `prism_render_architecture::water::GRAVITY`.
const WATER_GRAVITY: f32 = 9.81;

// 2*PI, used to build the wave number `k = 2*PI / L`; mirrors the CPU
// `prism_render_architecture::water::TWO_PI`.
const WATER_TWO_PI: f32 = 6.28318530717958647692;

// One Gerstner wave train. 8 f32 lanes = 32 bytes, a whole number of 16-byte
// rows, so a packed `array<GerstnerWave>` needs no inter-element padding.
struct GerstnerWave {
    // Wave-propagation direction (xz plane), x component. Normalized in-shader
    // so a non-unit stored direction still behaves.
    dir_x: f32,
    // Wave-propagation direction (xz plane), z component.
    dir_z: f32,
    // Crest amplitude `A` (m).
    amplitude: f32,
    // Wavelength `L` (m); the wave number is `k = 2*PI / L`.
    wavelength: f32,
    // Steepness `Q` in [0,1]: 0 = round sine wave, 1 = sharp cusp. The host
    // clamps the sum so crests never self-intersect.
    steepness: f32,
    // Phase-speed multiplier on the deep-water `omega = sqrt(g*k)` (1 = physical).
    speed: f32,
    // Constant phase offset (rad) so multiple trains de-correlate.
    phase: f32,
    // Padding lane for a 16-byte-aligned stride; ignored.
    pad: f32,
}

// Per-dispatch Gerstner scalars.
struct GerstnerParams {
    // Grid resolution `N`; output textures are `N*N` texels.
    grid_size: u32,
    // Number of active `GerstnerWave` entries to sum.
    wave_count: u32,
    // Physical patch size `L` (m) mapping texel `(px, py)` to world xz.
    patch_size: f32,
    // Simulation time `t` (s) advancing every wave's phase.
    time: f32,
    // Rest water level (world y, m) added to the summed vertical displacement.
    base_level: f32,
}

@group(0) @binding(0) var<uniform> params: GerstnerParams;
@group(0) @binding(1) var<storage, read> waves: array<GerstnerWave>;
@group(0) @binding(2) var displacement_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var normal_out: texture_storage_2d<rgba32float, write>;

// Deep-water dispersion `omega(k) = sqrt(g*k)`; non-positive `k` yields 0.
fn water_dispersion(k: f32) -> f32 {
    if k <= 0.0 {
        return 0.0;
    }
    return sqrt(WATER_GRAVITY * k);
}

@compute @workgroup_size(8, 8, 1)
fn water_gerstner_displace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.grid_size;
    if gid.x >= n || gid.y >= n {
        return;
    }

    let inv_n = 1.0 / max(f32(n), 1.0);
    let cell = params.patch_size * inv_n;
    let rest_x = f32(gid.x) * cell;
    let rest_z = f32(gid.y) * cell;

    var disp = vec3<f32>(0.0, 0.0, 0.0);
    var nrm = vec3<f32>(0.0, 1.0, 0.0);

    let count = params.wave_count;
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let w = waves[i];
        if w.wavelength <= 0.0 {
            continue;
        }
        let dir_len_sq = w.dir_x * w.dir_x + w.dir_z * w.dir_z;
        if dir_len_sq <= WATER_EPS_LEN_SQ {
            continue;
        }
        let inv_dir_len = 1.0 / sqrt(dir_len_sq);
        let dx = w.dir_x * inv_dir_len;
        let dz = w.dir_z * inv_dir_len;

        let k = WATER_TWO_PI / w.wavelength;
        let omega = w.speed * water_dispersion(k);
        let theta = k * (dx * rest_x + dz * rest_z) - omega * params.time + w.phase;
        let c = cos(theta);
        let s = sin(theta);

        let qa = w.steepness * w.amplitude;
        disp.x = disp.x + qa * dx * c;
        disp.z = disp.z + qa * dz * c;
        disp.y = disp.y + w.amplitude * s;

        let wa = k * w.amplitude;
        nrm.x = nrm.x - dx * wa * c;
        nrm.z = nrm.z - dz * wa * c;
        nrm.y = nrm.y - w.steepness * wa * s;
    }

    var normal = vec3<f32>(0.0, 1.0, 0.0);
    if dot(nrm, nrm) > WATER_EPS_LEN_SQ {
        normal = normalize(nrm);
    }

    let out_pos = vec3<f32>(disp.x, params.base_level + disp.y, disp.z);
    let texel = vec2<i32>(i32(gid.x), i32(gid.y));
    textureStore(displacement_out, texel, vec4<f32>(out_pos, 0.0));
    textureStore(normal_out, texel, vec4<f32>(normal, 0.0));
}
