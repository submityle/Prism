// Jacobi (parallel-safe) strand self-collision twin: evaluates on-device the
// same read-only-snapshot correction accumulation the CPU golden
// `accumulate_jacobi_corrections` computes, one thread per particle.
//
// The Gauss-Seidel resolver (`resolve_self_collision`) walks particles in index
// order and applies each push in place, so a later particle already sees an
// earlier one's moved position - that is the sequential CPU reference and does
// not map to a GPU compute kernel. On the GPU every particle is updated in
// parallel from the *same* read-only snapshot, which is a Jacobi iteration. The
// CPU golden for that parallel pass is `accumulate_jacobi_corrections` (in
// `prism_render_architecture::hair::self_collision_jacobi`); this kernel mirrors
// its per-particle body value-for-value, so a passing real-device parity test
// is direct evidence the ported kernel accumulates the same corrections as the
// reference, not merely that it compiles.
//
// Neighbor gathering (the 27-cell uniform-grid query, sorted ascending) is done
// host-side and uploaded as a per-particle compressed-sparse-row slice
// (`neighbor_starts` + `neighbor_indices`); this kernel only walks each
// particle's own neighbor slice in the same ascending order, so the float
// reduction order matches the reference (device-side spatial hashing is a
// separate dispatch). Each finite, unpinned particle sums its inverse-mass
// share of the separating push against every neighbor closer than the collision
// diameter `min_sep`; non-finite or coincident pairs and pinned particles
// (inverse mass 0) contribute nothing, exactly as the golden guards.
//
// It uses only the portable core-WGSL subset (sqrt, min/max, dot, multiply/add)
// - no exp, no pow, no optional feature - so this twin runs unmodified on Metal,
// Vulkan and DX12. It is not bit-exact with the scalar CPU reference: a GPU may
// fuse a multiply-add, perturbing the low mantissa bits, so the parity test
// asserts a per-component tolerance rather than exact equality.
//
// Provenance: standard uniform-grid self-collision push (TressFX 4 style) plus
// Jacobi accumulation; no Unreal Engine source or derived code.

struct Params {
    // Number of particles in `positions` / `inv_mass` / `corrections`.
    particle_count: u32,
    // Collision diameter `2 * particle_radius`.
    min_sep: f32,
    // `min_sep * min_sep`, precomputed to match the reference's compare term.
    min_sep_sq: f32,
    // Push stiffness, already clamped to [0, 1] host-side.
    stiffness: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Particle positions, stride 3 f32 (xyz).
@group(0) @binding(1) var<storage, read> positions: array<f32>;
// Per-particle inverse mass (0 = pinned).
@group(0) @binding(2) var<storage, read> inv_mass: array<f32>;
// Per-particle neighbor-slice bounds; length `particle_count + 1`. Particle `i`
// owns `neighbor_indices[neighbor_starts[i] .. neighbor_starts[i + 1]]`.
@group(0) @binding(3) var<storage, read> neighbor_starts: array<u32>;
// Flat ascending neighbor particle indices for every particle, concatenated.
@group(0) @binding(4) var<storage, read> neighbor_indices: array<u32>;
// Per-particle accumulated correction out, stride 3 f32 (xyz).
@group(0) @binding(5) var<storage, read_write> corrections: array<f32>;

// Matches the dynamics solver's zero-length epsilon and the golden constant.
const EPS_LEN_SQ: f32 = 1.0e-24;

fn inf() -> f32 {
    return bitcast<f32>(0x7f800000u);
}

// Finite means not NaN (v == v) and not an infinity (|v| below +inf).
fn finite_scalar(v: f32) -> bool {
    return v == v && abs(v) < inf();
}

fn finite3(p: vec3<f32>) -> bool {
    return finite_scalar(p.x) && finite_scalar(p.y) && finite_scalar(p.z);
}

fn load_pos(i: u32) -> vec3<f32> {
    let b = 3u * i;
    return vec3<f32>(positions[b], positions[b + 1u], positions[b + 2u]);
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) {
        return;
    }
    let bi = 3u * i;

    var delta = vec3<f32>(0.0, 0.0, 0.0);
    let pi = load_pos(i);
    let wi = max(inv_mass[i], 0.0);
    // Non-finite particles neither push nor are pushed; pinned particles
    // (inverse mass 0) absorb no correction - both leave delta at zero, exactly
    // as the golden `continue`s before touching `out[i]`.
    if (finite3(pi) && wi > 0.0) {
        let start = neighbor_starts[i];
        let end = neighbor_starts[i + 1u];
        for (var k: u32 = start; k < end; k = k + 1u) {
            let j = neighbor_indices[k];
            if (j == i) {
                continue; // never collide with self.
            }
            let pj = load_pos(j);
            if (!finite3(pj)) {
                continue;
            }
            let d = pi - pj;
            let dist_sq = dot(d, d);
            if (dist_sq >= params.min_sep_sq || dist_sq < EPS_LEN_SQ) {
                continue; // far enough, or coincident (no separating axis).
            }
            let wj = max(inv_mass[j], 0.0);
            let w = wi + wj;
            if (w <= 0.0) {
                continue;
            }
            let dist = sqrt(dist_sq);
            let overlap = params.min_sep - dist;
            let normal = d * (1.0 / dist);
            // This particle's inverse-mass share of the full separating push.
            delta = delta + normal * (overlap * params.stiffness * (wi / w));
        }
    }

    corrections[bi] = delta.x;
    corrections[bi + 1u] = delta.y;
    corrections[bi + 2u] = delta.z;
}
