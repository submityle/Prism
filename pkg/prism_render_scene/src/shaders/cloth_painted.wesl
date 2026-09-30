//! GPU compute twins of the artist-painted per-vertex cloth constraint passes.
//!
//! CPU golden reference: prism_render_architecture::cloth::painted -- the four
//! stateless per-vertex projections that steer a garment toward its skinned
//! animation pose (following the UE5 Chaos Cloth painted-weight model, expressed
//! here from first principles without reusing their code):
//!
//!   * painted_anim_drive        -> drive_toward_anim   (pre-solve soft pull)
//!   * painted_clamp_max_distance-> clamp_max_distance  (post-solve drift cap)
//!   * painted_backstop          -> apply_painted_backstop (cushion push-out)
//!   * painted_blend_to_skin     -> blend_to_skin       (final skin blend)
//!
//! ## One own-slot kernel per pass (embarrassingly parallel, exact parity)
//!
//! Each pass reads only vertex i, its skinned anchor i and its painted weights i
//! and writes only vertex i, so there is no cross-vertex write hazard: the
//! golden iterates in order purely for determinism, and since no vertex reads a
//! neighbour's updated position the per-vertex dispatch reproduces the golden
//! bit-for-bit regardless of order. Every pass runs one invocation per vertex
//! with no reduction and no synchronisation.
//!
//! ## Weight sanitisation mirrors PaintedConstraint::clamped line for line
//!
//! The kernels take the raw authored weights and clamp them exactly as the
//! golden does: max_distance NaN -> +inf (uncapped) else max(0); backstop
//! max(0); blend_weight and anim_drive clamped to [0, 1]. A vertex is pinned
//! when its inverse mass is non-positive and is skipped, matching is_pinned.
//!
//! The sandbox has no GPU, so this shader is validated by naga compilation only
//! (via cloth::shader_tests); real-machine bit-parity against the golden is
//! proven by cloth::painted_gpu_tests on a real Metal device.

/// Squared-length floor below which a vector is treated as degenerate. Mirrors
/// the CPU EPS_LEN_SQ used by normalize_or_zero and the projection guards.
const CLOTH_PAINTED_EPS_LEN_SQ: f32 = 1.0e-12;

/// Uniform parameters shared by the painted passes. Flat 16-byte block: the
/// anim-drive master gain, the reciprocal timestep (1/dt, precomputed host-side
/// exactly as the golden does), the effective vertex count and one pad word.
struct ClothPaintedParams {
    /// Master anim-drive gain in [0, 1] (sanitized host-side); scales anim_drive.
    anim_gain: f32,
    /// Reciprocal timestep 1/dt for the anim-drive velocity update.
    inv_dt: f32,
    /// Number of vertices processed (min of positions/anchors/painted lengths).
    vertex_count: u32,
    /// Padding to a 16-byte uniform stride.
    pad0: u32,
}

// Vertex state: xyz = world position, w = inverse mass (<= 0 means pinned).
// Mutated in place by every pass except the anim-drive velocity write.
@group(0) @binding(0) var<storage, read_write> painted_positions: array<vec4<f32>>;
// Vertex velocity: xyz used; only the anim-drive pass writes it.
@group(0) @binding(1) var<storage, read_write> painted_velocities: array<vec4<f32>>;
// Skinned anchor position (xyz); the animated character-surface target point.
@group(0) @binding(2) var<storage, read> painted_anchor_positions: array<vec4<f32>>;
// Skinned anchor outward normal (xyz); zero disables only the backstop.
@group(0) @binding(3) var<storage, read> painted_anchor_normals: array<vec4<f32>>;
// Raw authored weights: x = max_distance, y = backstop, z = blend_weight,
// w = anim_drive. Clamped inside each kernel exactly like the golden.
@group(0) @binding(4) var<storage, read> painted_weights: array<vec4<f32>>;
// Flat 16-byte uniform block.
@group(0) @binding(5) var<uniform> painted_params: ClothPaintedParams;

/// Normalizes v, or returns zero when it is shorter than the epsilon. Mirrors
/// the CPU Vec3::normalize_or_zero, including the 1.0 / sqrt (not inverseSqrt)
/// so the result matches the golden to the same ULP.
fn painted_normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if len_sq > CLOTH_PAINTED_EPS_LEN_SQ {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

/// Pre-solve soft pull toward the animated anchor. One invocation per vertex;
/// skips pinned vertices and a zero follow fraction exactly like the golden.
@compute @workgroup_size(64, 1, 1)
fn painted_anim_drive(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= painted_params.vertex_count {
        return;
    }
    let state = painted_positions[idx];
    if state.w <= 0.0 {
        return;
    }
    // paint.clamped(): anim_drive clamped to [0, 1].
    let anim_drive = clamp(painted_weights[idx].w, 0.0, 1.0);
    let follow = clamp(anim_drive * painted_params.anim_gain, 0.0, 1.0);
    if follow <= 0.0 {
        return;
    }
    let position = state.xyz;
    let anchor = painted_anchor_positions[idx].xyz;
    let delta = (anchor - position) * follow;
    painted_positions[idx] = vec4<f32>(position + delta, state.w);
    let velocity = painted_velocities[idx].xyz + delta * painted_params.inv_dt;
    painted_velocities[idx] = vec4<f32>(velocity, painted_velocities[idx].w);
}

/// Post-solve hard cap on drift from the skinned anchor: projects an over-limit
/// vertex back onto the max-distance sphere. Skips pinned and uncapped vertices.
@compute @workgroup_size(64, 1, 1)
fn painted_clamp_max_distance(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= painted_params.vertex_count {
        return;
    }
    let state = painted_positions[idx];
    if state.w <= 0.0 {
        return;
    }
    // paint.clamped(): max_distance NaN -> +inf, else max(0). The golden then
    // skips the non-finite (uncapped) case. After the clamp the only non-finite
    // value is +inf, so detect it by its bit pattern (0x7f800000) and treat a
    // NaN input as the same uncapped skip -- no infinity literal required.
    let raw_md = painted_weights[idx].x;
    let max_distance = max(raw_md, 0.0);
    let raw_is_nan = raw_md != raw_md;
    let md_is_pos_inf = bitcast<u32>(max_distance) == 0x7f800000u;
    if raw_is_nan || md_is_pos_inf {
        return;
    }
    let position = state.xyz;
    let anchor = painted_anchor_positions[idx].xyz;
    let drift = position - anchor;
    let dist_sq = dot(drift, drift);
    let max_sq = max_distance * max_distance;
    if dist_sq <= max_sq {
        return;
    }
    if dist_sq > CLOTH_PAINTED_EPS_LEN_SQ {
        let dist = sqrt(dist_sq);
        painted_positions[idx] = vec4<f32>(anchor + drift * (max_distance / dist), state.w);
    } else {
        painted_positions[idx] = vec4<f32>(anchor, state.w);
    }
}

/// Post-solve push-out off the cushion sphere sunk one radius behind the anchor
/// along its normal. Skips pinned vertices, a non-positive backstop and a
/// (near-)zero anchor normal exactly like the golden.
@compute @workgroup_size(64, 1, 1)
fn painted_backstop(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= painted_params.vertex_count {
        return;
    }
    let state = painted_positions[idx];
    if state.w <= 0.0 {
        return;
    }
    // paint.clamped(): backstop max(0).
    let backstop = max(painted_weights[idx].y, 0.0);
    if backstop <= 0.0 {
        return;
    }
    let normal = painted_normalize_or_zero(painted_anchor_normals[idx].xyz);
    if dot(normal, normal) <= CLOTH_PAINTED_EPS_LEN_SQ {
        return;
    }
    let position = state.xyz;
    let anchor = painted_anchor_positions[idx].xyz;
    let center = anchor - normal * backstop;
    let to_vertex = position - center;
    let dist_sq = dot(to_vertex, to_vertex);
    let radius = backstop;
    if dist_sq >= radius * radius {
        return;
    }
    if dist_sq > CLOTH_PAINTED_EPS_LEN_SQ {
        let dist = sqrt(dist_sq);
        painted_positions[idx] = vec4<f32>(center + to_vertex * (radius / dist), state.w);
    } else {
        painted_positions[idx] = vec4<f32>(center + normal * radius, state.w);
    }
}

/// Final blend of the simulated position toward the skinned anchor by the
/// painted blend_weight. Skips pinned vertices; velocity is left untouched.
@compute @workgroup_size(64, 1, 1)
fn painted_blend_to_skin(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= painted_params.vertex_count {
        return;
    }
    let state = painted_positions[idx];
    if state.w <= 0.0 {
        return;
    }
    // paint.clamped(): blend_weight clamped to [0, 1].
    let blend_weight = clamp(painted_weights[idx].z, 0.0, 1.0);
    let anchor = painted_anchor_positions[idx].xyz;
    let offset = state.xyz - anchor;
    painted_positions[idx] = vec4<f32>(anchor + offset * blend_weight, state.w);
}
