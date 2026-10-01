//! GPU compute twin of the continuous cloth self-collision (self-CCD) tier.
//!
//! CPU golden reference:
//! prism_render_architecture::cloth::self_ccd::resolve_self_ccd -- the swept
//! spatial-hash resolver that snaps a tunnelling cloth-vs-cloth pair back to its
//! time-of-impact (TOI) contact instead of letting a fast layer pass clean
//! through another in a single step (design section 6.3, mirroring the body
//! sweep in ccd::resolve_ccd). Each particle is swept prev -> curr over the unit
//! frame interval; its thickness-expanded swept box is bucketed into every
//! integer cell it overlaps, and every pair sharing a cell is a TOI candidate.
//!
//! The sandbox has no GPU, so this shader is validated by naga compilation only
//! (via cloth::shader_tests); real-machine bit-parity against the golden is
//! proven by cloth::self_ccd_gpu_tests on single-contact geometry (where the
//! Gauss-Seidel golden and this own-slot Jacobi twin agree value-for-value).
//!
//! ## Three own-slot kernels (no float atomics, no read/write races)
//!
//! * cloth_ccd_hash_build -- one invocation per particle atomically prepends the
//!   particle into every hash cell its thickness-expanded swept box overlaps,
//!   mirroring the golden's ascending-index bucket insertion. The linked-list
//!   prepend uses atomicExchange (bucket order is unspecified, exactly like the
//!   point tier's cloth_self_collision_hash_build); correctness does not depend
//!   on bucket order because the resolve pass canonicalises each pair.
//! * cloth_ccd_resolve -- one invocation per particle i walks the cells of its
//!   own swept box, and for every co-occupant j != i solves the swept-pair TOI.
//!   To resolve each unordered pair exactly once (the golden visits a BTreeSet
//!   of unique pairs), the pair is only processed while scanning the
//!   lexicographically minimum cell of the intersection of the two swept-box
//!   cell ranges. On a hit, i writes ONLY its own half of the position snap and
//!   restitution impulse into ccd_pos_delta[i] / ccd_vel_delta[i]; the read
//!   state (positions / prev / velocities) is never mutated in this pass, so the
//!   Jacobi input is race-free.
//! * cloth_ccd_apply -- one invocation per particle adds its accumulated deltas
//!   into positions / velocities, skipping pinned particles (inverse_mass <= 0),
//!   exactly as the golden guards every write with is_pinned().

/// Relative-motion floor: below this squared closing speed the swept quadratic
/// has no appreciable leading term and the gap is treated as never closing.
/// Mirrors the CPU EPS_REL_MOTION.
const CLOTH_CCD_EPS_REL_MOTION: f32 = 1.0e-12;

/// Squared-length floor for treating the TOI separation as degenerate before
/// normalizing the contact normal. Mirrors the CPU `EPS_LEN_SQ` guard inside
/// `Vec3::normalize_or_zero`, so a near-coincident pair falls back to the fixed
/// +X axis on the GPU exactly as it does in the golden.
const CLOTH_CCD_EPS_LEN_SQ: f32 = 1.0e-12;

/// Linked-list terminator for the spatial-hash cell chains.
const CLOTH_CCD_SENTINEL: u32 = 0xffffffffu;

/// Per-bucket linked-list head + occupancy count (identical layout to the point
/// tier's ClothHashCell in cloth_collision.wesl).
struct ClothCcdHashCell {
    /// Index of the first particle chained into this cell, or the sentinel.
    head: atomic<u32>,
    /// Number of particles chained into this cell.
    count: atomic<u32>,
}

/// Uniform parameters shared by the three self-CCD kernels. Flat 32-byte block.
struct ClothCcdParams {
    /// Number of particles bounding every dispatch.
    particle_count: u32,
    /// Number of hash buckets in cell_table (the modulus for the cell hash).
    table_size: u32,
    /// Uniform grid cell edge, world units (assumed > 0; matches sanitized
    /// CPU cell_size).
    cell_size: f32,
    /// Minimum enforced separation; the TOI is the instant the swept distance
    /// first reaches this value (matches CPU thickness).
    thickness: f32,
    /// Normal restitution in [0, 1] (matches CPU restitution).
    restitution: f32,
    /// Inverse frame time 1/dt, or 0 when |dt| is negligible (matches the CPU
    /// inv_dt guard). Velocity is recovered from the TOI motion over dt.
    inv_dt: f32,
    /// Trailing pad words so the block is a flat 32-byte uniform.
    pad0: u32,
    pad1: u32,
}

// Shared bindings. Each entry point uses a disjoint subset (naga validates per
// entry point), matching how cloth_collision.wesl shares a group across kernels.

// Current particle state: xyz = current world position, w = inverse mass
// (<= 0 = pinned). cloth_ccd_apply is the only writer.
@group(0) @binding(0) var<storage, read_write> ccd_positions: array<vec4<f32>>;
// Frame-start positions: xyz = prev world position (w unused).
@group(0) @binding(1) var<storage, read> ccd_prev_positions: array<vec4<f32>>;
// Current velocities: xyz = velocity (w unused). cloth_ccd_apply is the only writer.
@group(0) @binding(2) var<storage, read_write> ccd_velocities: array<vec4<f32>>;
// Per-bucket linked-list heads + occupancy counts.
@group(0) @binding(3) var<storage, read_write> ccd_cell_table: array<ClothCcdHashCell>;
// Per-particle "next in chain" pointer, forming each bucket's linked list.
@group(0) @binding(4) var<storage, read_write> ccd_particle_next: array<u32>;
// Phase-1 output: accumulated own-half position delta for particle i.
@group(0) @binding(5) var<storage, read_write> ccd_pos_delta: array<vec4<f32>>;
// Phase-1 output: accumulated own-half velocity delta for particle i.
@group(0) @binding(6) var<storage, read_write> ccd_vel_delta: array<vec4<f32>>;
// Shared uniform block.
@group(0) @binding(7) var<uniform> ccd_params: ClothCcdParams;

/// Maps a world position to its integer spatial-hash cell (floor(pos / cell_size)),
/// mirroring the CPU cell_of and the point tier's cloth_cell_of.
fn cloth_ccd_cell_of(pos: vec3<f32>, cell_size: f32) -> vec3<i32> {
    let inv = 1.0 / cell_size;
    return vec3<i32>(floor(pos * inv));
}

/// Folds an integer cell into a bucket in [0, table_size) with the standard
/// large-prime multiply-xor uniform-grid hash (identical to the point tier).
fn cloth_ccd_cell_hash(cell: vec3<i32>, table_size: u32) -> u32 {
    let x = bitcast<u32>(cell.x) * 73856093u;
    let y = bitcast<u32>(cell.y) * 19349663u;
    let z = bitcast<u32>(cell.z) * 83492791u;
    return (x ^ y ^ z) % table_size;
}

/// The thickness-expanded swept-box cell range [lo, hi] (inclusive) a particle
/// occupies. Mirrors the CPU broad-phase: lo/hi span min/max of prev and curr,
/// grown by thickness on every side, then floored to cells.
fn cloth_ccd_cell_lo(index: u32, cell_size: f32, thickness: f32) -> vec3<i32> {
    let prev = ccd_prev_positions[index].xyz;
    let curr = ccd_positions[index].xyz;
    let margin = vec3<f32>(thickness, thickness, thickness);
    let lo = min(prev, curr) - margin;
    return cloth_ccd_cell_of(lo, cell_size);
}

fn cloth_ccd_cell_hi(index: u32, cell_size: f32, thickness: f32) -> vec3<i32> {
    let prev = ccd_prev_positions[index].xyz;
    let curr = ccd_positions[index].xyz;
    let margin = vec3<f32>(thickness, thickness, thickness);
    let hi = max(prev, curr) + margin;
    return cloth_ccd_cell_of(hi, cell_size);
}

/// Solves the earliest time of impact for a swept particle pair, mirroring the
/// CPU swept_pair_toi value-for-value. Returns the entry root in [0, 1] via the
/// out-param toi; the bool result is whether a valid TOI exists.
fn cloth_ccd_swept_toi(
    prev_a: vec3<f32>,
    curr_a: vec3<f32>,
    prev_b: vec3<f32>,
    curr_b: vec3<f32>,
    thickness: f32,
    toi: ptr<function, f32>,
) -> bool {
    let d0 = prev_a - prev_b;
    let d1 = curr_a - curr_b;
    let dv = d1 - d0;
    let c = dot(d0, d0) - thickness * thickness;
    // Already within the contact band at frame start: contact is immediate.
    if c <= 0.0 {
        *toi = 0.0;
        return true;
    }
    let a = dot(dv, dv);
    // No appreciable relative motion and not already overlapping: no crossing.
    if a <= CLOTH_CCD_EPS_REL_MOTION {
        return false;
    }
    let b = 2.0 * dot(d0, dv);
    let disc = b * b - 4.0 * a * c;
    // The swept separation never reaches thickness.
    if disc < 0.0 {
        return false;
    }
    // Entry root (the smaller of the two): first moment the gap closes.
    let t = (-b - sqrt(disc)) / (2.0 * a);
    if t >= 0.0 && t <= 1.0 {
        *toi = t;
        return true;
    }
    return false;
}

/// Broad phase: prepend particle i into every cell its swept box overlaps.
@compute @workgroup_size(64, 1, 1)
fn cloth_ccd_hash_build(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= ccd_params.particle_count {
        return;
    }
    let lo = cloth_ccd_cell_lo(i, ccd_params.cell_size, ccd_params.thickness);
    let hi = cloth_ccd_cell_hi(i, ccd_params.cell_size, ccd_params.thickness);
    var cx = lo.x;
    loop {
        if cx > hi.x { break; }
        var cy = lo.y;
        loop {
            if cy > hi.y { break; }
            var cz = lo.z;
            loop {
                if cz > hi.z { break; }
                let bucket = cloth_ccd_cell_hash(vec3<i32>(cx, cy, cz), ccd_params.table_size);
                // Prepend i into the bucket's linked list and bump its count.
                let prev_head = atomicExchange(&ccd_cell_table[bucket].head, i);
                ccd_particle_next[i] = prev_head;
                atomicAdd(&ccd_cell_table[bucket].count, 1u);
                cz = cz + 1;
            }
            cy = cy + 1;
        }
        cx = cx + 1;
    }
}

/// The lexicographically minimum cell shared by two particles' swept-box cell
/// ranges, returned via out-param and a bool for "the ranges intersect". Used to
/// canonicalise each unordered pair so it is resolved exactly once, standing in
/// for the golden's BTreeSet-of-unique-pairs visitation.
fn cloth_ccd_min_shared_cell(
    i: u32,
    j: u32,
    cell_size: f32,
    thickness: f32,
    out_cell: ptr<function, vec3<i32>>,
) -> bool {
    let lo_i = cloth_ccd_cell_lo(i, cell_size, thickness);
    let hi_i = cloth_ccd_cell_hi(i, cell_size, thickness);
    let lo_j = cloth_ccd_cell_lo(j, cell_size, thickness);
    let hi_j = cloth_ccd_cell_hi(j, cell_size, thickness);
    let lo = max(lo_i, lo_j);
    let hi = min(hi_i, hi_j);
    if lo.x > hi.x || lo.y > hi.y || lo.z > hi.z {
        return false;
    }
    *out_cell = lo;
    return true;
}

/// Narrow phase: resolve particle i's own half of every TOI contact.
@compute @workgroup_size(64, 1, 1)
fn cloth_ccd_resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= ccd_params.particle_count {
        return;
    }
    let cell_size = ccd_params.cell_size;
    let thickness = ccd_params.thickness;

    let prev_a = ccd_prev_positions[i].xyz;
    let curr_a = ccd_positions[i].xyz;
    let wa = max(ccd_positions[i].w, 0.0);

    var pos_delta = vec3<f32>(0.0, 0.0, 0.0);
    var vel_delta = vec3<f32>(0.0, 0.0, 0.0);

    let lo = cloth_ccd_cell_lo(i, cell_size, thickness);
    let hi = cloth_ccd_cell_hi(i, cell_size, thickness);
    var cx = lo.x;
    loop {
        if cx > hi.x { break; }
        var cy = lo.y;
        loop {
            if cy > hi.y { break; }
            var cz = lo.z;
            loop {
                if cz > hi.z { break; }
                let cell = vec3<i32>(cx, cy, cz);
                let bucket = cloth_ccd_cell_hash(cell, ccd_params.table_size);
                var node = atomicLoad(&ccd_cell_table[bucket].head);
                loop {
                    if node == CLOTH_CCD_SENTINEL { break; }
                    let j = node;
                    node = ccd_particle_next[j];
                    if j == i { continue; }

                    // Resolve each unordered pair once: only while scanning the
                    // minimum shared cell of the two swept boxes, and only from
                    // the lower-indexed particle's canonical cell match. Both i
                    // and j run this; the guard keeps the contribution single.
                    var min_cell = vec3<i32>(0, 0, 0);
                    if !cloth_ccd_min_shared_cell(i, j, cell_size, thickness, &min_cell) {
                        continue;
                    }
                    if cell.x != min_cell.x || cell.y != min_cell.y || cell.z != min_cell.z {
                        continue;
                    }

                    let wb = max(ccd_positions[j].w, 0.0);
                    let wsum = wa + wb;
                    if wsum <= 0.0 { continue; }

                    let prev_b = ccd_prev_positions[j].xyz;
                    let curr_b = ccd_positions[j].xyz;
                    var t = 0.0;
                    if !cloth_ccd_swept_toi(prev_a, curr_a, prev_b, curr_b, thickness, &t) {
                        continue;
                    }

                    // Positions at the time of impact.
                    let a_c = prev_a + (curr_a - prev_a) * t;
                    let b_c = prev_b + (curr_b - prev_b) * t;
                    let delta = a_c - b_c;
                    // normalize_or_zero: guard on dot(delta, delta) > EPS_LEN_SQ,
                    // then scale by 1.0 / sqrt(len_sq) (reciprocal-mul, not a
                    // component-wise divide) to mirror the golden
                    // Vec3::normalize_or_zero op order; the trailing dot>0 check
                    // and +X fallback reproduce the golden's degenerate branch
                    // bit-for-bit. penetration uses sqrt(len_sq) == delta.length().
                    let len_sq = dot(delta, delta);
                    var normal = vec3<f32>(1.0, 0.0, 0.0);
                    if len_sq > CLOTH_CCD_EPS_LEN_SQ {
                        let unit = delta * (1.0 / sqrt(len_sq));
                        if dot(unit, unit) > 0.0 {
                            normal = unit;
                        }
                    }
                    let penetration = max(thickness - sqrt(len_sq), 0.0);
                    let inv_wsum = 1.0 / wsum;

                    // i's own half: snap to TOI then push by inverse-mass share.
                    // Delta from the current position so cloth_ccd_apply can add.
                    let pos_a = a_c + normal * (wa * inv_wsum * penetration);
                    pos_delta = pos_delta + (pos_a - curr_a);

                    // Restitution impulse from the inbound relative normal speed.
                    let va_in = (a_c - prev_a) * ccd_params.inv_dt;
                    let vb_in = (b_c - prev_b) * ccd_params.inv_dt;
                    let vrel_n = dot(va_in - vb_in, normal);
                    if vrel_n < 0.0 {
                        let impulse = -(1.0 + ccd_params.restitution) * vrel_n * inv_wsum;
                        let new_va = va_in + normal * (wa * impulse);
                        vel_delta = vel_delta + (new_va - ccd_velocities[i].xyz);
                    }

                }
                cz = cz + 1;
            }
            cy = cy + 1;
        }
        cx = cx + 1;
    }

    ccd_pos_delta[i] = vec4<f32>(pos_delta, 0.0);
    ccd_vel_delta[i] = vec4<f32>(vel_delta, 0.0);
}

/// Apply phase: fold the accumulated deltas into position / velocity, skipping
/// pinned particles exactly as the golden guards each write with is_pinned().
@compute @workgroup_size(64, 1, 1)
fn cloth_ccd_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= ccd_params.particle_count {
        return;
    }
    // Pinned particles (inverse_mass <= 0) are never written.
    if ccd_positions[i].w <= 0.0 {
        return;
    }
    ccd_positions[i] = vec4<f32>(ccd_positions[i].xyz + ccd_pos_delta[i].xyz, ccd_positions[i].w);
    ccd_velocities[i] = vec4<f32>(ccd_velocities[i].xyz + ccd_vel_delta[i].xyz, ccd_velocities[i].w);
}
