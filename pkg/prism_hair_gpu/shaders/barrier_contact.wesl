//! `wgpu` compute twin of Prism's barrier-based frictional hair contact
//! (`prism_render_architecture::hair::barrier_contact::resolve_contact`).
//!
//! `resolve_contact` is the per-contact core of the real-time simplified
//! Codimensional `IPC` (`C-IPC`, Li 2021) contact tier: a C¹ rational soft
//! barrier pushes two endpoints apart along the contact normal and a
//! semi-implicit `Coulomb` friction impulse cancels the tangential slide,
//! clamped to the friction cone. This kernel is that primitive's isolated twin:
//! one thread per contact reads the pair `(a, b, normal, distance,
//! rel_velocity)` from its *original* state, evaluates the resolution and the
//! corrected positions/velocities, and writes them into its own disjoint output
//! slot — a `Jacobi`-style evaluation with no cross-contact accumulation (the
//! batch `resolve_contacts` wrapper's shared-endpoint `Gauss-Seidel` ordering is
//! deliberately *not* twinned here, since it is inherently sequential).
//!
//! # Barrier (no `ln`)
//!
//! Prism's workspace bans `ln`/`exp`/`pow` so the textbook `C-IPC` log barrier
//! is off-limits. The reference uses a rational soft barrier evaluated on the
//! clamped distance `d_e = clamp(d, d_floor, dhat)`:
//!
//! ```text
//!     force = k * (dhat - d_e) * [ 2 (1/d_e - 1/dhat) + (dhat - d_e)/d_e^2 ]
//! ```
//!
//! which vanishes at `d = dhat` (C¹) and stays large-but-finite at the floor.
//! The host uploads already-sanitized params, so this kernel skips the
//! `sanitized()`/non-finite guards and only reproduces the `d >= dhat -> 0`
//! short-circuit and the clamped evaluation.
//!
//! # Guards
//!
//! A degenerate (zero-length) normal, two pinned endpoints (`inv_mass` sum
//! `<= 0`), or a gap at/beyond `dhat` each make the contact a deterministic
//! no-op: the endpoints pass through unchanged and both impulses are `0`.
//!
//! # Bit-faithfulness
//!
//! Core math is closed-form (one `sqrt`, divides, mul-add), but a `GPU` may fuse
//! multiply-adds the scalar reference leaves separate, so positions, velocities
//! and impulses can differ by a few low-mantissa `ULP`; the parity test asserts
//! each within `abs_diff < 1e-4` or `rel_diff < 1e-3`. The `slipping` flag is a
//! branch pick, so the test cases sit well clear of the cone threshold and the
//! flag is asserted exactly.
//!
//! # Portability
//!
//! The kernel uses only `sqrt`/`dot`/`min`/`max`/`clamp`/multiply/divide in the
//! portable core-`WGSL` subset — no `exp`/`pow`/trig, atomics or optional device
//! feature — so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! Provenance: real-time simplified `C-IPC` (Li 2021) barrier + semi-implicit
//! `Coulomb` friction, plus `wgpu` compute dispatch; no Unreal Engine source or
//! derived code.

// Vectors whose squared length is below this are treated as zero-length
// (mirrors `barrier_contact::EPS_LEN_SQ`).
const EPS_LEN_SQ: f32 = 1.0e-24;
// Denominators below this are treated as degenerate (mirrors `EPS_DENOM`).
const EPS_DENOM: f32 = 1.0e-12;

/// Already-sanitized barrier/friction params plus the contact count, padded to
/// `32` bytes.
struct Params {
    dhat: f32,
    stiffness: f32,
    d_floor: f32,
    friction_mu: f32,
    contact_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One contact in the upload layout. Positions carry inverse mass in `w` and
/// the normal carries the signed gap in `w`, so every field is a `vec4` with no
/// `vec3` alignment hazard.
struct Contact {
    a_pos: vec4<f32>,       // xyz position, w inv_mass
    a_vel: vec4<f32>,       // xyz velocity, w unused
    b_pos: vec4<f32>,       // xyz position, w inv_mass
    b_vel: vec4<f32>,       // xyz velocity, w unused
    normal: vec4<f32>,      // xyz normal (b -> a), w distance
    rel_velocity: vec4<f32>, // xyz velocity of a relative to b, w unused
}

/// One resolved contact: corrected endpoint state plus the scalar outcome.
struct Resolved {
    a_pos: vec4<f32>,  // xyz, w unused
    a_vel: vec4<f32>,  // xyz, w unused
    b_pos: vec4<f32>,  // xyz, w unused
    b_vel: vec4<f32>,  // xyz, w unused
    scalars: vec4<f32>, // normal_impulse, friction_impulse, slipping(0/1), unused
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> contacts: array<Contact>;
@group(0) @binding(2) var<storage, read_write> out_results: array<Resolved>;

// Unit vector along `v`, or zero when `v` is (numerically) zero-length; mirrors
// `Vec3::normalize_or_zero` exactly (reciprocal of `sqrt(len_sq)`).
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if len_sq > EPS_LEN_SQ {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Repulsive barrier force magnitude for a contact gap `d`, with params assumed
// already sanitized by the host. Bit-faithful port of
// `barrier_force_magnitude`: `0` for `d >= dhat`, otherwise the rational soft
// barrier evaluated on the floored distance.
fn barrier_force_magnitude(d: f32) -> f32 {
    if d >= params.dhat {
        return 0.0;
    }
    let de = clamp(d, params.d_floor, params.dhat);
    let gap = params.dhat - de;
    let recip = 1.0 / de - 1.0 / params.dhat;
    let inv_de = 1.0 / de;
    let bracket = 2.0 * recip + gap * (inv_de * inv_de);
    let force = params.stiffness * gap * bracket;
    return max(force, 0.0);
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.contact_count {
        return;
    }
    let c = contacts[i];

    let a_inv = c.a_pos.w;
    let b_inv = c.b_pos.w;

    // The no-op default: endpoints pass through unchanged, resolution NONE.
    var a_pos = c.a_pos.xyz;
    var a_vel = c.a_vel.xyz;
    var b_pos = c.b_pos.xyz;
    var b_vel = c.b_vel.xyz;
    var normal_impulse = 0.0;
    var friction_impulse = 0.0;
    var slipping = 0.0;

    let n = normalize_or_zero(c.normal.xyz);
    let w = a_inv + b_inv;
    let dist = c.normal.w;

    // Guards mirroring `resolve_contact`: degenerate normal, both pinned, or a
    // gap at/beyond `dhat` each leave the contact a deterministic no-op.
    if dot(n, n) > EPS_LEN_SQ && w > 0.0 && dist < params.dhat {
        let jn = barrier_force_magnitude(dist);
        let inv_w = 1.0 / w;

        // Normal repulsion split by inverse mass (a pinned endpoint stays put).
        a_pos = a_pos + n * (jn * a_inv * inv_w);
        b_pos = b_pos - n * (jn * b_inv * inv_w);

        // Semi-implicit `Coulomb` friction in the contact tangent plane.
        let vrel = c.rel_velocity.xyz;
        let vn = dot(vrel, n);
        let vt = vrel - n * vn;
        let speed = length(vt);
        let t_hat = normalize_or_zero(vt);
        if dot(t_hat, t_hat) > EPS_LEN_SQ && jn > 0.0 {
            let jt_full = speed * inv_w;
            let jt_max = params.friction_mu * jn;
            let jt = min(jt_full, jt_max);
            if jt_full > jt_max + EPS_DENOM {
                slipping = 1.0;
            }
            friction_impulse = jt;
            a_vel = a_vel - t_hat * (jt * a_inv);
            b_vel = b_vel + t_hat * (jt * b_inv);
        }

        normal_impulse = jn;
    }

    out_results[i].a_pos = vec4<f32>(a_pos, 0.0);
    out_results[i].a_vel = vec4<f32>(a_vel, 0.0);
    out_results[i].b_pos = vec4<f32>(b_pos, 0.0);
    out_results[i].b_vel = vec4<f32>(b_vel, 0.0);
    out_results[i].scalars = vec4<f32>(normal_impulse, friction_impulse, slipping, 0.0);
}
