//! GPU compute twin of the closed-mesh pressure (volume-preservation) constraint.
//!
//! CPU golden reference:
//! prism_render_architecture::cloth::pressure::project_pressure -- the single
//! XPBD volume projection that keeps a closed, outward-wound triangle mesh at a
//! target enclosed volume by pushing its particles along the accumulated
//! per-vertex volume gradient (design section 6, the gas / pressure constraint
//! that inflates a garment shell into a balloon or a down jacket, the same model
//! shipped by Houdini Vellum and NVIDIA NvCloth, expressed here from first
//! principles without reusing their code).
//!
//! ## Math (mirrors the golden line for line)
//!
//! The signed enclosed volume of the shell is the divergence-theorem tetra sum
//!   V = (1/6) * sum_tri  p0 . (p1 x p2)
//! and the pressure constraint drives it toward target = overpressure * V_rest:
//!   C = V - target
//! Each triangle contributes a gradient to each of its three vertices,
//!   dV/dp0 = (1/6)(p1 x p2)   dV/dp1 = (1/6)(p2 x p0)   dV/dp2 = (1/6)(p0 x p1)
//! and shared vertices accumulate the sum. With alpha_tilde = compliance / dt^2,
//! one compliant XPBD update follows exactly as in the distance solver:
//!   d_lambda = -C / ( sum_i w_i |grad_i|^2 + alpha_tilde )
//!   x_i     += w_i * d_lambda * grad_i
//!
//! ## Two own-slot kernels (deterministic reduction, no float atomics)
//!
//! The volume and per-vertex gradient are a global reduction: every triangle
//! scatters into three shared vertices, and the single scalar d_lambda couples
//! the whole shell. To reproduce the golden's fixed accumulation order (and thus
//! its float32 rounding) bit-for-bit, cloth_pressure_solve runs as ONE
//! invocation (gid.x == 0) that walks the triangles and vertices in the same
//! sequence the CPU does; the parallel apply pass then folds the shared
//! d_lambda into each free particle independently. This trades the peak
//! throughput of a parallel reduction for exact parity with the golden -- a
//! production parallel-reduction variant is a separate, faster kernel and is out
//! of scope for this parity twin.
//!
//! * cloth_pressure_solve -- one invocation zeroes the gradient buffer, sums the
//!   signed volume and the per-vertex gradient in triangle order (skipping
//!   out-of-range triangles exactly like the golden), forms the volume error,
//!   accumulates the denominator sum_i w_i |grad_i|^2 (pinned particles, weight
//!   w <= 0, contribute nothing), adds alpha_tilde, and writes the single
//!   d_lambda scalar. A denominator below the degenerate-mesh floor writes a
//!   zero d_lambda so the apply pass is a no-op, matching the golden's guard.
//! * cloth_pressure_apply -- one invocation per particle adds w * d_lambda *
//!   grad_i to its position, skipping pinned particles (inverse_mass <= 0)
//!   exactly as the golden guards every write with a zero weight.
//!
//! The sandbox has no GPU, so this shader is validated by naga compilation only
//! (via cloth::shader_tests); real-machine bit-parity against the golden is
//! proven by cloth::pressure_gpu_tests on closed-mesh fixtures.

/// One sixth, the constant factor in the tetra-volume and gradient formulas.
/// Matches the CPU golden INV_SIX.
const CLOTH_PRESSURE_INV_SIX: f32 = 1.0 / 6.0;

/// Degenerate-mesh denominator floor. A denominator at or below this value (a
/// collapsed shell) yields a zero d_lambda so the update never divides by zero
/// or emits NaN. Mirrors the CPU EPS_LEN_SQ.
const CLOTH_PRESSURE_EPS_LEN_SQ: f32 = 1.0e-12;

/// Uniform parameters shared by the two pressure kernels. Flat 16-byte block:
/// two counts and the two derived scalars the host precomputes from the
/// sanitized golden PressureParams (target = overpressure * rest_volume) and the
/// substep (alpha_tilde = compliance / dt^2).
struct ClothPressureParams {
    /// Number of particles, bounding the solve reduction and the apply dispatch.
    particle_count: u32,
    /// Number of triangles in the closed shell.
    triangle_count: u32,
    /// Target enclosed volume, overpressure * rest_volume (already sanitized).
    target_volume: f32,
    /// XPBD compliance over dt squared, compliance / dt^2.
    alpha_tilde: f32,
}

// Current particle state: xyz = world position, w = inverse mass (<= 0 = pinned).
// cloth_pressure_apply is the only writer.
@group(0) @binding(0) var<storage, read_write> pressure_positions: array<vec4<f32>>;
// Triangle index list: xyz = vertex indices (outward winding), w unused.
@group(0) @binding(1) var<storage, read> pressure_triangles: array<vec4<u32>>;
// Per-vertex accumulated volume gradient: xyz = gradient, w unused. Written by
// the solve pass, read by the apply pass.
@group(0) @binding(2) var<storage, read_write> pressure_gradients: array<vec4<f32>>;
// Single shared scalar: element 0 holds the solved d_lambda.
@group(0) @binding(3) var<storage, read_write> pressure_lambda: array<f32>;
// Flat 16-byte uniform block.
@group(0) @binding(4) var<uniform> pressure_params: ClothPressureParams;

/// Effective inverse mass: zero when pinned (inverse_mass <= 0), else the stored
/// inverse mass. Mirrors the golden particle_weight.
fn cloth_pressure_weight(inv_mass: f32) -> f32 {
    if inv_mass <= 0.0 {
        return 0.0;
    }
    return inv_mass;
}

/// Reduction pass: sum the signed volume and per-vertex gradient in the golden's
/// fixed order, then solve the single compliant d_lambda. One invocation.
@compute @workgroup_size(1, 1, 1)
fn cloth_pressure_solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x != 0u {
        return;
    }
    let count = pressure_params.particle_count;

    // Zero the gradient accumulator exactly like the golden vec![ZERO; count].
    var i = 0u;
    loop {
        if i >= count { break; }
        pressure_gradients[i] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        i = i + 1u;
    }

    // Accumulate the signed volume and per-vertex gradient in one triangle pass.
    var volume = 0.0;
    let tri_count = pressure_params.triangle_count;
    var t = 0u;
    loop {
        if t >= tri_count { break; }
        let tri = pressure_triangles[t];
        t = t + 1u;
        let i0 = tri.x;
        let i1 = tri.y;
        let i2 = tri.z;
        if i0 >= count || i1 >= count || i2 >= count {
            continue;
        }
        let p0 = pressure_positions[i0].xyz;
        let p1 = pressure_positions[i1].xyz;
        let p2 = pressure_positions[i2].xyz;
        volume = volume + dot(p0, cross(p1, p2));
        pressure_gradients[i0] = vec4<f32>(
            pressure_gradients[i0].xyz + cross(p1, p2) * CLOTH_PRESSURE_INV_SIX, 0.0);
        pressure_gradients[i1] = vec4<f32>(
            pressure_gradients[i1].xyz + cross(p2, p0) * CLOTH_PRESSURE_INV_SIX, 0.0);
        pressure_gradients[i2] = vec4<f32>(
            pressure_gradients[i2].xyz + cross(p0, p1) * CLOTH_PRESSURE_INV_SIX, 0.0);
    }
    volume = volume * CLOTH_PRESSURE_INV_SIX;

    let error = volume - pressure_params.target_volume;

    // Denominator: sum_i w_i |grad_i|^2 + alpha_tilde.
    var denom = 0.0;
    var k = 0u;
    loop {
        if k >= count { break; }
        let w = cloth_pressure_weight(pressure_positions[k].w);
        if w > 0.0 {
            let g = pressure_gradients[k].xyz;
            denom = denom + w * dot(g, g);
        }
        k = k + 1u;
    }
    denom = denom + pressure_params.alpha_tilde;

    var d_lambda = 0.0;
    if denom >= CLOTH_PRESSURE_EPS_LEN_SQ {
        d_lambda = -error / denom;
    }
    pressure_lambda[0] = d_lambda;
}

/// Apply pass: fold the shared d_lambda into each free particle, skipping pinned
/// particles exactly as the golden guards each write with a zero weight.
@compute @workgroup_size(64, 1, 1)
fn cloth_pressure_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= pressure_params.particle_count {
        return;
    }
    let w = cloth_pressure_weight(pressure_positions[i].w);
    if w <= 0.0 {
        return;
    }
    let d_lambda = pressure_lambda[0];
    let delta = pressure_gradients[i].xyz * (w * d_lambda);
    pressure_positions[i] = vec4<f32>(pressure_positions[i].xyz + delta, pressure_positions[i].w);
}
