// Divergence-free curl-noise wind-field twin for groom motion.
//
// Device mirror of the `CPU` golden
// `prism_render_architecture::hair::wind_field::curl_wind_map`. A believable
// groom under wind must gust and swirl without the strands appearing to inflate
// or collapse, i.e. the velocity field must be **divergence-free**
// (`div w = 0`, incompressible). Rather than solve a fluid, film and realtime
// grooms borrow `Bridson` 2007 `curl-noise`: take the curl of a smooth vector
// potential `P`, `w = curl P`. Because `div(curl P) = 0` identically, the field
// is divergence-free *by construction* with no solve and no grid — just sample
// the potential's derivatives at a point. This is the approach used by grooms
// in `UE5` and offline pipelines.
//
// The potential `P = (Px, Py, Pz)` is a deterministic integer **value noise**:
// a `u32` hash of the integer lattice cell seeds a pseudo-random value at each
// corner, and a cubic `smoothstep` (`t*t*(3 - 2t)`) fades a trilinear
// interpolation. Three independent channels (distinct seed offsets) give the
// three potential components; the wind velocity is `amplitude * curl(P)`, with
// the curl approximated by central **finite differences** of step `EPS`. Time is
// injected as a coordinate offset so the field evolves smoothly between frames.
//
// One invocation owns one position (flattened `3*i` layout to avoid `vec3`
// storage alignment, matching `self_collision_grid`): it reads the position,
// evaluates the curl, and writes the three velocity components to its disjoint
// output slice. Invocations past the position count early-return.
//
// Bit-faithfulness vs tolerance: the lattice hash is pure `u32` integer math
// (`bitcast<u32>` reproduces Rust's two's-complement `as u32`, and `u32 *`/`+`
// wrap exactly like `wrapping_mul`/`wrapping_add`), so the hashed corner values
// are exact. The subsequent trilinear blend and the finite-difference curl are
// large multiply-add chains a `GPU` may fuse — and the difference is scaled by
// `inv_two_eps = 500`, amplifying low `ULP` noise — so the final velocity is
// checked within `abs_diff < 1e-4` or `rel_diff < 1e-3`, not bit-for-bit. The
// `frequency` / `amplitude` sanitizers mirror the golden's `sanitized()`
// exactly, so negative / non-finite params collapse to the same safe field the
// reference emits and the kernel never produces a non-finite result.
//
// The kernel uses only `floor`, `bitcast`, dynamic indexing and multiply/add in
// the portable core-`WGSL` subset — no `exp`, `pow`, `sin` or optional device
// feature — so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: `Bridson` 2007 curl-noise and `wgpu` compute dispatch; no Unreal
// Engine source or derived code.

struct Params {
    // Raw authored field controls; sanitized in-shader to mirror the golden.
    frequency: f32,
    amplitude: f32,
    seed: u32,
    time: f32,
    position_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Query positions, flattened three `f32` per point (`3*i`, `3*i+1`, `3*i+2`).
@group(0) @binding(1) var<storage, read> positions: array<f32>;
// Per query: the three wind-velocity components, same flattened layout.
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

// Finite-difference step for the curl; mirrors the golden `EPS`.
const EPS: f32 = 1.0e-3;

// Largest finite f32; `v <= MAX_FINITE_F32` rejects `+inf` (and `v == v` rejects
// NaN), so the guards reproduce `f32::is_finite()` exactly.
const MAX_FINITE_F32: f32 = 3.40282347e+38;

// `u32::MAX as f32` rounds to this value on both the CPU cast and `f32(u32)`
// here, so the normalisation divisor matches the golden bit-for-bit.
const U32_MAX_AS_F32: f32 = 4294967296.0;

// Mirrors the golden `WindFieldParams::sanitized` frequency repair: a finite,
// strictly positive value passes through; everything else falls back to 1.0 (a
// zero frequency would collapse the field to a constant).
fn sanitize_frequency(f: f32) -> f32 {
    if f == f && f > 0.0 && f <= MAX_FINITE_F32 {
        return f;
    }
    return 1.0;
}

// Mirrors the golden amplitude repair: a finite non-negative value passes
// through; everything else (negative, NaN, +/-inf) falls back to 0.0 (no wind).
fn sanitize_amplitude(a: f32) -> f32 {
    if a == a && a >= 0.0 && a <= MAX_FINITE_F32 {
        return a;
    }
    return 0.0;
}

// `xorshift`-style integer finaliser; the only randomness source. `u32`
// multiply wraps exactly like the golden `wrapping_mul`.
fn hash_u32(x_in: u32) -> u32 {
    var x = x_in;
    x ^= x >> 16u;
    x = x * 0x7feb352du;
    x ^= x >> 15u;
    x = x * 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

// Hashes an integer lattice cell plus a seed into a pseudo-random value in
// `[-1, 1]`. `bitcast<u32>` reproduces Rust's two's-complement `ix as u32` for
// negative lattice coordinates; the `u32` adds/multiplies wrap like the golden.
fn lattice_value(seed: u32, ix: i32, iy: i32, iz: i32) -> f32 {
    var h = seed;
    h = h + bitcast<u32>(ix) * 0x9e3779b1u;
    h = hash_u32(h);
    h = h + bitcast<u32>(iy) * 0x85212bffu;
    h = hash_u32(h);
    h = h + bitcast<u32>(iz) * 0x6b43a9b5u;
    h = hash_u32(h);
    let unit = f32(h) / U32_MAX_AS_F32;
    return unit * 2.0 - 1.0;
}

// Cubic `smoothstep` fade on an already-`0..=1` fraction.
fn smooth_fade(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

// Linear interpolation `a + (b - a) * t`.
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

// Deterministic trilinear value noise at `p`, in `[-1, 1]`. Splits `p` into a
// lattice cell and fractional offset, hashes the eight corners, and blends them
// with the cubic fade of the fraction.
fn value_noise(seed: u32, p: vec3<f32>) -> f32 {
    let xi = floor(p.x);
    let yi = floor(p.y);
    let zi = floor(p.z);

    let ix = i32(xi);
    let iy = i32(yi);
    let iz = i32(zi);

    let fx = p.x - xi;
    let fy = p.y - yi;
    let fz = p.z - zi;

    let ux = smooth_fade(fx);
    let uy = smooth_fade(fy);
    let uz = smooth_fade(fz);

    let c000 = lattice_value(seed, ix, iy, iz);
    let c100 = lattice_value(seed, ix + 1, iy, iz);
    let c010 = lattice_value(seed, ix, iy + 1, iz);
    let c110 = lattice_value(seed, ix + 1, iy + 1, iz);
    let c001 = lattice_value(seed, ix, iy, iz + 1);
    let c101 = lattice_value(seed, ix + 1, iy, iz + 1);
    let c011 = lattice_value(seed, ix, iy + 1, iz + 1);
    let c111 = lattice_value(seed, ix + 1, iy + 1, iz + 1);

    let x00 = lerp(c000, c100, ux);
    let x10 = lerp(c010, c110, ux);
    let x01 = lerp(c001, c101, ux);
    let x11 = lerp(c011, c111, ux);

    let y0 = lerp(x00, x10, uy);
    let y1 = lerp(x01, x11, uy);

    return lerp(y0, y1, uz);
}

// Scalar potential channel `channel` (`0`, `1`, `2`) at a noise-space point.
// Distinct per-channel seed offset decorrelates the three components; `time` is
// folded into each coordinate so the field animates smoothly.
fn potential_channel(seed: u32, channel: u32, p: vec3<f32>, time: f32) -> f32 {
    let channel_seed = seed + channel * 0x10000001u;
    let t = time + f32(channel) * 13.0;
    return value_noise(channel_seed, vec3<f32>(p.x + t, p.y - t, p.z + t));
}

// Divergence-free wind velocity at a world position and time. Scales `pos` by
// `frequency` into noise space, central-differences each potential component,
// and returns `amplitude * curl(P)`. Mirrors the golden `curl_wind`.
fn curl_wind(frequency: f32, amplitude: f32, seed: u32, pos: vec3<f32>, time: f32) -> vec3<f32> {
    let freq = sanitize_frequency(frequency);
    let amp = sanitize_amplitude(amplitude);

    let p = vec3<f32>(pos.x * freq, pos.y * freq, pos.z * freq);
    let inv_two_eps = 1.0 / (2.0 * EPS);

    let px_py_hi = potential_channel(seed, 0u, vec3<f32>(p.x, p.y + EPS, p.z), time);
    let px_py_lo = potential_channel(seed, 0u, vec3<f32>(p.x, p.y - EPS, p.z), time);
    let px_pz_hi = potential_channel(seed, 0u, vec3<f32>(p.x, p.y, p.z + EPS), time);
    let px_pz_lo = potential_channel(seed, 0u, vec3<f32>(p.x, p.y, p.z - EPS), time);

    let py_px_hi = potential_channel(seed, 1u, vec3<f32>(p.x + EPS, p.y, p.z), time);
    let py_px_lo = potential_channel(seed, 1u, vec3<f32>(p.x - EPS, p.y, p.z), time);
    let py_pz_hi = potential_channel(seed, 1u, vec3<f32>(p.x, p.y, p.z + EPS), time);
    let py_pz_lo = potential_channel(seed, 1u, vec3<f32>(p.x, p.y, p.z - EPS), time);

    let pz_px_hi = potential_channel(seed, 2u, vec3<f32>(p.x + EPS, p.y, p.z), time);
    let pz_px_lo = potential_channel(seed, 2u, vec3<f32>(p.x - EPS, p.y, p.z), time);
    let pz_py_hi = potential_channel(seed, 2u, vec3<f32>(p.x, p.y + EPS, p.z), time);
    let pz_py_lo = potential_channel(seed, 2u, vec3<f32>(p.x, p.y - EPS, p.z), time);

    let dpz_dy = (pz_py_hi - pz_py_lo) * inv_two_eps;
    let dpy_dz = (py_pz_hi - py_pz_lo) * inv_two_eps;
    let dpx_dz = (px_pz_hi - px_pz_lo) * inv_two_eps;
    let dpz_dx = (pz_px_hi - pz_px_lo) * inv_two_eps;
    let dpy_dx = (py_px_hi - py_px_lo) * inv_two_eps;
    let dpx_dy = (px_py_hi - px_py_lo) * inv_two_eps;

    let wx = dpz_dy - dpy_dz;
    let wy = dpx_dz - dpz_dx;
    let wz = dpy_dx - dpx_dy;

    return vec3<f32>(wx * amp, wy * amp, wz * amp);
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if t >= params.position_count {
        return;
    }
    let base = 3u * t;
    let pos = vec3<f32>(positions[base], positions[base + 1u], positions[base + 2u]);
    let w = curl_wind(params.frequency, params.amplitude, params.seed, pos, params.time);
    out[base] = w.x;
    out[base + 1u] = w.y;
    out[base + 2u] = w.z;
}
