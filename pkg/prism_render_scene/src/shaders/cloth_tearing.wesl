//! GPU compute twin of the cloth tearing (constraint-break) detection pass.
//!
//! CPU golden reference:
//! prism_render_architecture::cloth::tearing::tear_flags -- the per-edge break
//! decision that marks every two-sided fabric edge whose tensile strain exceeds
//! the break threshold, so an over-stretched garment rips instead of stretching
//! without bound (the same constraint-breaking used by production cloth in Chaos
//! Cloth and Houdini Vellum, expressed here from first principles without reusing
//! their code). The host consumes the flags to compact the constraint graph
//! (apply_tearing); this kernel only produces the parallel per-edge mask.
//!
//! ## Math (mirrors the golden line for line)
//!
//! For a two-sided fabric edge with current length len and rest length rest the
//! tensile strain is
//!   e = (len - rest) / rest
//! The edge breaks (flag 1) exactly when e > break_strain. A one-sided
//! attachment (LRA or tether), a degenerate rest length (rest <= EPS_REST), or an
//! out-of-range endpoint never breaks (flag 0), matching the golden edge_strain
//! guards that return None for those edges. Compression (negative strain) never
//! breaks because break_strain is non-negative.
//!
//! ## One own-slot kernel (embarrassingly parallel, exact parity)
//!
//! Every invocation writes only its own edge flag and reads only shared read-only
//! positions and constraints, so there is no write hazard and the parallel
//! per-edge kernel reproduces the golden bit-for-bit regardless of dispatch
//! order. cloth_tearing_flag therefore runs one invocation per constraint with no
//! reduction and no synchronisation.
//!
//! The sandbox has no GPU, so this shader is validated by naga compilation only
//! (via cloth::shader_tests); real-machine bit-parity against the golden is
//! proven by cloth::tearing_gpu_tests on stretched, slack, one-sided and
//! out-of-range edges.

/// Long-range attachment leash tag. One-sided, so tearing skips it exactly as the
/// golden edge_strain returns None for a one-sided constraint.
const CLOTH_TEARING_KIND_LRA: u32 = 3u;
/// One-sided tether leash tag, likewise never torn.
const CLOTH_TEARING_KIND_TETHER: u32 = 4u;

/// Numerical floor below which a rest length is treated as degenerate and the
/// edge is skipped. Mirrors the CPU EPS_REST.
const CLOTH_TEARING_EPS_REST: f32 = 1.0e-9;

/// One distance constraint. Byte-compatible with the CPU CONSTRAINT_STRIDE
/// (20 bytes) and the GpuClothConstraint ABI; read only here.
struct ClothConstraint {
    /// First endpoint particle index.
    a: u32,
    /// Second endpoint particle index.
    b: u32,
    /// Rest length the tensile strain is measured against.
    rest_length: f32,
    /// XPBD compliance (inverse stiffness); unused by tearing.
    compliance: f32,
    /// Constraint-kind tag; the LRA and tether tags mark one-sided edges skipped.
    kind: u32,
}

/// Uniform parameters for the tearing pass. Flat 16-byte block: the sanitized
/// break threshold, two counts, and padding to a 16-byte stride.
struct ClothTearingParams {
    /// Tensile strain above which a two-sided edge breaks (sanitized, >= 0).
    break_strain: f32,
    /// Number of constraints bounding the per-edge dispatch.
    constraint_count: u32,
    /// Number of particles, for the endpoint range guard.
    particle_count: u32,
    /// Padding to a 16-byte uniform stride.
    pad0: u32,
}

// Current particle state: xyz = world position, w = inverse mass. Read only; the
// tearing pass never moves particles.
@group(0) @binding(0) var<storage, read> tearing_positions: array<vec4<f32>>;
// Distance constraints; inspected read only.
@group(0) @binding(1) var<storage, read> tearing_constraints: array<ClothConstraint>;
// Per-edge break flags: 1 = torn, 0 = kept. One slot per constraint.
@group(0) @binding(2) var<storage, read_write> tearing_flags: array<u32>;
// Flat 16-byte uniform block.
@group(0) @binding(3) var<uniform> tearing_params: ClothTearingParams;

/// Per-edge tensile break test. One invocation per constraint. Writes 1 when the
/// edge is a valid two-sided fabric edge whose strain exceeds break_strain, else
/// 0. Skips one-sided, degenerate and out-of-range edges exactly like the golden.
@compute @workgroup_size(64, 1, 1)
fn cloth_tearing_flag(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= tearing_params.constraint_count {
        return;
    }
    let con = tearing_constraints[idx];

    // Default: the edge is kept.
    var torn = 0u;

    // edge_strain guards: one-sided attachments never tear.
    let one_sided = con.kind == CLOTH_TEARING_KIND_LRA || con.kind == CLOTH_TEARING_KIND_TETHER;
    let degenerate = con.rest_length <= CLOTH_TEARING_EPS_REST;
    let count = tearing_params.particle_count;
    let out_of_range = con.a >= count || con.b >= count;

    if !one_sided && !degenerate && !out_of_range {
        let len = distance(tearing_positions[con.a].xyz, tearing_positions[con.b].xyz);
        let strain = (len - con.rest_length) / con.rest_length;
        if strain > tearing_params.break_strain {
            torn = 1u;
        }
    }

    tearing_flags[idx] = torn;
}
