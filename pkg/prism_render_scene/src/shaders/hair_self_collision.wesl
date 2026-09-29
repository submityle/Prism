//! GPU strand self-collision post-pass (parallel Jacobi over a uniform grid).
//!
//! Compute twin of the CPU golden reference
//! `prism_render_architecture::hair::self_collision_jacobi`. Body-proxy and SDF
//! collision keep hair off the character, but nothing stops a groom from
//! passing through *itself*; production strand solvers (`AMD` `TressFX` 4) add a
//! spatial-hash self-collision post-pass that treats particles as small spheres
//! and softly repels overlapping neighbors (design §6.2 / §8 "自碰撞近似").
//!
//! The sequential CPU reference `hair::self_collision::resolve_self_collision`
//! is **Gauss-Seidel** (in-place, order-dependent) and does not map to a `GPU`
//! kernel, where every particle is updated in parallel from the same read-only
//! snapshot — a **Jacobi** iteration. This shader therefore mirrors the Jacobi
//! golden `accumulate_jacobi_corrections` / `apply_corrections`, not the
//! Gauss-Seidel one, so there is no fake parity (design §9).
//!
//! Neighbor lookup consumes the deterministic compressed-sparse-row grid built
//! by `hair::self_collision_grid::build_csr` (`cell_keys` sorted in the CPU
//! `BTreeMap`'s lexicographic order, `cell_starts` a prefix-sum offset table,
//! `cell_indices` the flat ascending buckets). One `accumulate` invocation owns
//! one particle: it hashes its cell, binary-searches the 27 surrounding cells in
//! `cell_keys`, and sums its inverse-mass-weighted share of every overlap into
//! `corrections[i]` from the read-only `positions` snapshot. A second `apply`
//! pass adds each correction in place. The per-particle skip conditions,
//! inverse-mass weighting, collision diameter `2 * particle_radius` and
//! zero-length epsilon match the CPU golden term for term; corrections are
//! summed in `cell_keys` iteration order (the CPU golden sorts globally
//! ascending, so the two converge to the same separated state but are not
//! bit-identical on a single pass — the same honest distinction as
//! Jacobi-vs-Gauss-Seidel above).
//!
//! Nothing samples a real random source and no input panics on device: a
//! particle never collides with itself, non-finite particles neither push nor
//! are pushed, coincident particles (no separating axis) are skipped, pinned
//! particles (`inverse_mass <= 0`) receive zero correction, and degenerate
//! parameters make every correction zero. A `GPU` is now available, so this
//! shader is compile- and type-checked through the same render-world
//! `ShaderCache` / `wesl` pipeline the runtime uses (see `shader_tests`), with
//! on-device dispatch/perf and numeric calibration still awaiting real hardware.

/// Squared-length epsilon: pairs closer than its square root are coincident and
/// have no separating axis. Mirrors the CPU `self_collision_jacobi::EPS_LEN_SQ`.
const HAIR_SC_EPS_LEN_SQ: f32 = 1.0e-24;

/// Magnitude bound standing in for `f32::is_finite`: `NaN` fails self-equality
/// and infinities fail this bound, matching the CPU golden's finite-only guard.
const HAIR_SC_FINITE_MAG: f32 = 3.0e38;

/// Dispatch parameters. `particle_count` bounds the dispatch, `cell_count` the
/// number of occupied grid cells (the length of `cell_keys`, and one less than
/// `cell_starts`).
struct HairSelfCollisionParams {
    particle_radius: f32,
    stiffness: f32,
    cell_size: f32,
    particle_count: u32,
    cell_count: u32,
}

// Particle state: `xyz` = world position, `w` = inverse mass (0 = pinned).
@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
// Per-particle accumulated correction: `xyz` = delta position, `w` unused.
@group(0) @binding(1) var<storage, read_write> corrections: array<vec4<f32>>;
// CSR grid: occupied cell coordinates (`xyz`), ascending lexicographic order.
@group(0) @binding(2) var<storage, read> cell_keys: array<vec4<i32>>;
// CSR prefix-sum offsets; length is `cell_count + 1`.
@group(0) @binding(3) var<storage, read> cell_starts: array<u32>;
// CSR flat buckets: ascending particle indices grouped by cell.
@group(0) @binding(4) var<storage, read> cell_indices: array<u32>;

var<immediate> params: HairSelfCollisionParams;

// Finite check mirroring CPU `is_finite`: reject NaN (self-inequality) and
// infinities (magnitude bound).
fn hair_sc_is_finite(v: vec3<f32>) -> bool {
    return v.x == v.x && v.y == v.y && v.z == v.z
        && abs(v.x) < HAIR_SC_FINITE_MAG
        && abs(v.y) < HAIR_SC_FINITE_MAG
        && abs(v.z) < HAIR_SC_FINITE_MAG;
}

// Maps a finite position to its integer grid cell. Mirrors CPU
// `self_collision_grid::grid_cell_of`: `floor(coord / cell_size)` per axis.
fn hair_sc_cell_of(position: vec3<f32>, cell_size: f32) -> vec3<i32> {
    let inv = 1.0 / cell_size;
    return vec3<i32>(
        i32(floor(position.x * inv)),
        i32(floor(position.y * inv)),
        i32(floor(position.z * inv)),
    );
}

// Lexicographic `a < b` over integer cell coordinates, matching the ordering of
// the CPU `BTreeMap<(i32, i32, i32)>` the CSR was flattened from.
fn hair_sc_cell_less(a: vec3<i32>, b: vec3<i32>) -> bool {
    if a.x != b.x {
        return a.x < b.x;
    }
    if a.y != b.y {
        return a.y < b.y;
    }
    return a.z < b.z;
}

// Binary-searches `cell_keys` for `wanted`, returning its bucket index or -1.
// The array is sorted (CSR invariant), so this is the `GPU` analogue of the
// CPU `BTreeMap::get` used by `UniformGrid::neighbors`.
fn hair_sc_find_cell(wanted: vec3<i32>) -> i32 {
    var lo = 0u;
    var hi = params.cell_count;
    while lo < hi {
        let mid = lo + (hi - lo) / 2u;
        let key = cell_keys[mid].xyz;
        if key.x == wanted.x && key.y == wanted.y && key.z == wanted.z {
            return i32(mid);
        }
        if hair_sc_cell_less(key, wanted) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return -1;
}

// Accumulates each particle's total Jacobi self-collision correction from the
// read-only position snapshot. Mirrors CPU `accumulate_jacobi_corrections`.
@compute @workgroup_size(64)
fn accumulate_self_collision(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= params.particle_count {
        return;
    }
    corrections[index] = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    if params.particle_radius <= 0.0 || params.stiffness <= 0.0 || params.cell_size <= 0.0 {
        return;
    }

    let pi4 = positions[index];
    let pi = pi4.xyz;
    if !hair_sc_is_finite(pi) {
        return;
    }
    let wi = max(pi4.w, 0.0);
    if wi <= 0.0 {
        return; // pinned: absorbs no correction.
    }

    let min_sep = 2.0 * params.particle_radius;
    let min_sep_sq = min_sep * min_sep;
    let stiffness = clamp(params.stiffness, 0.0, 1.0);
    let base = hair_sc_cell_of(pi, params.cell_size);

    var delta = vec3<f32>(0.0, 0.0, 0.0);
    for (var dx = -1; dx <= 1; dx = dx + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dz = -1; dz <= 1; dz = dz + 1) {
                let cell = vec3<i32>(base.x + dx, base.y + dy, base.z + dz);
                let k = hair_sc_find_cell(cell);
                if k < 0 {
                    continue;
                }
                let ku = u32(k);
                let start = cell_starts[ku];
                let end = cell_starts[ku + 1u];
                for (var t = start; t < end; t = t + 1u) {
                    let j = cell_indices[t];
                    if j == index {
                        continue; // never collide with self.
                    }
                    let pj4 = positions[j];
                    let pj = pj4.xyz;
                    if !hair_sc_is_finite(pj) {
                        continue;
                    }
                    let d = pi - pj;
                    let dist_sq = dot(d, d);
                    if dist_sq >= min_sep_sq || dist_sq < HAIR_SC_EPS_LEN_SQ {
                        continue; // far enough, or coincident.
                    }
                    let wj = max(pj4.w, 0.0);
                    let w = wi + wj;
                    if w <= 0.0 {
                        continue;
                    }
                    let dist = sqrt(dist_sq);
                    let overlap = min_sep - dist;
                    let normal = d * (1.0 / dist);
                    delta = delta + normal * (overlap * stiffness * (wi / w));
                }
            }
        }
    }
    corrections[index] = vec4<f32>(delta, 0.0);
}

// Adds each accumulated correction to its particle in place. Mirrors CPU
// `apply_corrections`: non-finite particles are left untouched.
@compute @workgroup_size(64)
fn apply_self_collision(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= params.particle_count {
        return;
    }
    let p4 = positions[index];
    let p = p4.xyz;
    if hair_sc_is_finite(p) {
        positions[index] = vec4<f32>(p + corrections[index].xyz, p4.w);
    }
}
