//! GPU compute twin of the cloth plastic rest-length creep pass.
//!
//! CPU golden reference:
//! prism_render_architecture::cloth::tearing::apply_plasticity -- the permanent
//! deformation model that lets a distance edge stretched (or compressed) past a
//! yield strain creep its rest length toward the current length, capturing the
//! plastic set of wrinkles and sag while retaining a bounded residual elastic
//! strain (the same plastic-flow model used by production cloth in Chaos Cloth
//! and Houdini Vellum, expressed here from first principles without reusing
//! their code).
//!
//! ## Math (mirrors the golden line for line)
//!
//! For a two-sided fabric edge with current length len and rest length rest the
//! tensile strain is
//!   e = (len - rest) / rest
//! An edge within the yield band (abs(e) <= yield_strain), a one-sided
//! attachment (LRA or tether), a degenerate rest length, or an out-of-range
//! endpoint is left untouched. Otherwise the beyond-yield excess is
//!   excess = e - sign(e) * yield_strain
//! and the rest length creeps toward the current length by the creep fraction
//!   new_rest = rest * (1 + creep * excess)
//! floored at the degenerate rest epsilon, then clamped so the residual elastic
//! strain magnitude abs((len - new_rest) / new_rest) never exceeds max_strain
//!   new_rest = len / (1 + sign(residual) * max_strain)
//! and finally written back when it stays above the epsilon.
//!
//! ## One own-slot kernel (embarrassingly parallel, exact parity)
//!
//! Every edge only ever mutates its own rest_length, so unlike the color-batched
//! distance projection this pass has no write hazard: the golden visits edges in
//! order purely for determinism, and since no edge reads another edge's updated
//! rest length the parallel per-edge kernel reproduces the golden bit-for-bit
//! regardless of dispatch order. cloth_apply_plasticity therefore runs one
//! invocation per constraint with no reduction and no synchronisation.
//!
//! The sandbox has no GPU, so this shader is validated by naga compilation only
//! (via cloth::shader_tests); real-machine bit-parity against the golden is
//! proven by cloth::plasticity_gpu_tests on stretched and compressed edges.

/// Long-range attachment leash tag. One-sided, so plasticity skips it exactly as
/// the golden edge_strain returns None for a one-sided constraint.
const CLOTH_PLASTICITY_KIND_LRA: u32 = 3u;
/// One-sided tether leash tag, likewise skipped.
const CLOTH_PLASTICITY_KIND_TETHER: u32 = 4u;

/// Numerical floor below which a rest length is treated as degenerate and the
/// edge is skipped or clamped up. Mirrors the CPU EPS_REST.
const CLOTH_PLASTICITY_EPS_REST: f32 = 1.0e-9;

/// One distance constraint. Byte-compatible with the CPU CONSTRAINT_STRIDE
/// (20 bytes) and the GpuClothConstraint ABI: only rest_length is mutated here.
struct ClothConstraint {
    /// First endpoint particle index.
    a: u32,
    /// Second endpoint particle index.
    b: u32,
    /// Rest length the plastic creep mutates in place.
    rest_length: f32,
    /// XPBD compliance (inverse stiffness); carried through untouched.
    compliance: f32,
    /// Constraint-kind tag; the LRA and tether tags mark one-sided edges skipped.
    kind: u32,
}

/// Uniform parameters shared by the plasticity pass. Flat 32-byte block: two
/// counts, the three sanitized golden scalars, and padding to a 16-byte stride.
struct ClothPlasticityParams {
    /// Number of constraints bounding the per-edge dispatch.
    constraint_count: u32,
    /// Number of particles, for the endpoint range guard.
    particle_count: u32,
    /// Strain magnitude beyond which plastic flow begins (sanitized, >= 0).
    yield_strain: f32,
    /// Fraction in [0, 1] of the beyond-yield excess converted to permanent set.
    creep: f32,
    /// Cap on the residual elastic strain magnitude left after creep (>= 0).
    max_strain: f32,
    /// Padding to a 16-byte uniform stride.
    pad0: u32,
    /// Padding to a 16-byte uniform stride.
    pad1: u32,
    /// Padding to a 16-byte uniform stride.
    pad2: u32,
}

// Current particle state: xyz = world position, w = inverse mass. Read only; the
// plasticity pass never moves particles, it only edits rest lengths.
@group(0) @binding(0) var<storage, read> plasticity_positions: array<vec4<f32>>;
// Distance constraints; rest_length is mutated in place, all else preserved.
@group(0) @binding(1) var<storage, read_write> plasticity_constraints: array<ClothConstraint>;
// Flat 32-byte uniform block.
@group(0) @binding(2) var<uniform> plasticity_params: ClothPlasticityParams;

/// Per-edge plastic creep. One invocation per constraint. Skips one-sided,
/// degenerate, out-of-range, and within-yield edges exactly like the golden.
@compute @workgroup_size(64, 1, 1)
fn cloth_apply_plasticity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= plasticity_params.constraint_count {
        return;
    }
    let con = plasticity_constraints[idx];

    // edge_strain guards: one-sided attachments never creep.
    if con.kind == CLOTH_PLASTICITY_KIND_LRA || con.kind == CLOTH_PLASTICITY_KIND_TETHER {
        return;
    }
    // Degenerate rest length is skipped to avoid a divide-by-zero.
    if con.rest_length <= CLOTH_PLASTICITY_EPS_REST {
        return;
    }
    let count = plasticity_params.particle_count;
    if con.a >= count || con.b >= count {
        return;
    }

    let len = distance(plasticity_positions[con.a].xyz, plasticity_positions[con.b].xyz);
    let strain = (len - con.rest_length) / con.rest_length;
    let magnitude = abs(strain);
    if magnitude <= plasticity_params.yield_strain {
        return;
    }

    var sign = -1.0;
    if strain >= 0.0 {
        sign = 1.0;
    }
    let excess = strain - sign * plasticity_params.yield_strain;

    // Move the rest length by creep fraction of the excess strain.
    var new_rest = con.rest_length * (1.0 + plasticity_params.creep * excess);
    if new_rest <= CLOTH_PLASTICITY_EPS_REST {
        new_rest = CLOTH_PLASTICITY_EPS_REST;
    }

    // Clamp so the residual elastic strain magnitude stays within max_strain.
    let residual = (len - new_rest) / new_rest;
    if abs(residual) > plasticity_params.max_strain {
        var residual_sign = -1.0;
        if residual >= 0.0 {
            residual_sign = 1.0;
        }
        new_rest = len / (1.0 + residual_sign * plasticity_params.max_strain);
    }

    if new_rest > CLOTH_PLASTICITY_EPS_REST {
        plasticity_constraints[idx].rest_length = new_rest;
    }
}
