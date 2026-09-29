//! GPU compute twin of the `NvCloth`-style virtual-particle self-collision tier.
//!
//! CPU golden reference:
//! `prism_render_architecture::cloth::virtual_particles_jacobi` — the two-phase,
//! own-slot Jacobi resolver that closes the point-to-point tier's blind spot
//! (a lone vertex tunnelling through the interior of a large triangle without
//! ever coming within `thickness` of that triangle's three corners, design
//! §6.2). Every triangle is seeded with a few barycentric **virtual particles**
//! (centroid + edge midpoints); reals and virtuals fold into one sample list
//! fed through the *same* uniform spatial hash as the point tier
//! (`cloth_collision.wesl`), so a diving vertex now finds a virtual particle on
//! the face it is piercing and is pushed back out, with the correction
//! scattered onto the face's three real vertices by the barycentric weights.
//!
//! The sandbox has no `GPU`, so this shader is validated by naga compilation
//! only (via `cloth::shader_tests`); no dispatch is executed here. The CPU
//! Jacobi golden is the value-for-value reference the three kernels below mirror.
//!
//! ## Three own-slot kernels (no float atomics)
//!
//! * `cloth_vp_hash_build` — one invocation per **sample** (reals `0..real_count`
//!   first, then virtuals) atomically prepends the sample into its spatial-hash
//!   cell, mirroring the point tier's `cloth_self_collision_hash_build`. A sample's
//!   bucket is chosen from its barycentric world position `Σ wₖ · pos[vₖ]`.
//! * `cloth_vp_resolve` — one invocation per sample `a` walks its 27-cell
//!   neighbourhood and sums *only `a`'s own half* of the separating push against
//!   every penetrating neighbour `b != a`, writing one `sample_dp[a]` slot. This
//!   is Phase 1 of the CPU golden: read the frozen snapshot, write your own slot.
//! * `cloth_vp_scatter` — one invocation per **real vertex** `v` gathers, through
//!   a host-built `CSR` incidence table (`csr_offsets` / `csr_entries`, the same
//!   pattern `cloth_aerodynamics.wesl` uses for per-vertex triangle adjacency),
//!   every sample that carries `v` as an active corner and adds its barycentric
//!   share `wₖ · inverse_massᵥ / eff(a)` of `sample_dp[a]` into `positions[v]`.
//!   This is Phase 2 of the CPU golden: per-vertex own-slot scatter, no atomics.
//!
//! ## Determinism note (honest `GPU` trade-off)
//!
//! Like the point-to-point `cloth_self_collision_resolve`, the parallel hash
//! build uses `atomicExchange`, so a bucket's linked-list order is
//! unspecified and the float reductions in `resolve` / `scatter` are summed in
//! that order rather than the CPU golden's ascending-index order. The kernels
//! therefore converge to the *same separated state* as the Jacobi golden but are
//! not a bit-parity mirror of its ordered reduction — exactly the trade-off the
//! sibling point tier documents. Everything else matches value-for-value: the
//! `shares_active_vertex` skip, the `thickness` target, the effective-inverse-mass
//! split `wa / (wa + wb)`, the `+X` coincident fallback, and pinned samples
//! (`eff <= 0`) contributing and receiving nothing.

/// Squared-length floor below which two sample points are treated as coincident,
/// so a separating direction is never normalised from ~0. Mirrors the CPU
/// `EPS_LEN_SQ` shared across the cloth subsystem.
const CLOTH_VP_EPS_LEN_SQ: f32 = 1.0e-12;

/// Linked-list terminator for the spatial-hash cell chains.
const CLOTH_VP_SENTINEL: u32 = 0xffffffffu;

/// One virtual-particle-tier sample, packed into two 16-byte `std430` slots so
/// the record stride matches the host `GpuClothVpSample` (32 bytes). A real
/// vertex `i` is encoded as `verts = (i, i, i)`, `weights = (1, 0, 0)`; a
/// virtual particle carries its triangle's three corner indices and barycentric
/// weights.
struct ClothVpSample {
    /// First active vertex index.
    v0: u32,
    /// Second vertex index.
    v1: u32,
    /// Third vertex index.
    v2: u32,
    /// Barycentric weight for `v0`.
    w0: f32,
    /// Barycentric weight for `v1`.
    w1: f32,
    /// Barycentric weight for `v2`.
    w2: f32,
    /// Trailing pad word (keeps the flat 32-byte stride).
    pad0: f32,
    /// Trailing pad word.
    pad1: f32,
}

/// Per-bucket linked-list head + occupancy count (identical layout to the point
/// tier's `ClothHashCell` in `cloth_collision.wesl`).
struct ClothVpHashCell {
    /// Index of the first sample chained into this cell, or the sentinel.
    head: atomic<u32>,
    /// Number of samples chained into this cell.
    count: atomic<u32>,
}

/// Uniform parameters shared by the three virtual-particle kernels.
struct ClothVpParams {
    /// Total sample count (reals + in-range virtuals) bounding the hash/resolve
    /// dispatches.
    sample_count: u32,
    /// Number of real particles; samples `< real_count` are real vertices.
    real_count: u32,
    /// Number of real vertices bounding the scatter dispatch (equals
    /// `real_count`; named separately to match the CSR length `vertex_count`).
    vertex_count: u32,
    /// Number of hash buckets in `cell_table` (the modulus for the cell hash).
    table_size: u32,
    /// Uniform grid cell edge, world units (assumed `> 0`; matches CPU
    /// `cell_size`).
    cell_size: f32,
    /// Separation distance below which a sample pair is pushed apart (matches
    /// CPU `thickness`).
    thickness: f32,
    /// When non-zero, real-vs-real pairs are skipped (augment mode: the friction
    /// point tier owns those pairs); zero resolves every pair (self-contained
    /// tier). Mirrors the CPU `PairScope`.
    virtual_only: u32,
    /// Trailing pad word so the block is a flat 32-byte uniform.
    pad: u32,
}

// Shared bindings. Each entry point uses a disjoint subset (naga validates per
// entry point), matching how `cloth_collision.wesl` shares a group across its
// body/self/backstop kernels.

// Particle state: `xyz` = world position, `w` = inverse mass (`<= 0` = pinned).
@group(0) @binding(0) var<storage, read_write> vp_positions: array<vec4<f32>>;
// Sample list: reals first (`verts = (i,i,i)`, `weights = (1,0,0)`), then virtuals.
@group(0) @binding(1) var<storage, read> vp_samples: array<ClothVpSample>;
// Per-bucket linked-list heads + occupancy counts.
@group(0) @binding(2) var<storage, read_write> vp_cell_table: array<ClothVpHashCell>;
// Per-sample "next in chain" pointer, forming each bucket's linked list.
@group(0) @binding(3) var<storage, read_write> vp_sample_next: array<u32>;
// Phase-1 output: `sample_dp[a].xyz` is sample `a`'s own accumulated half-push.
@group(0) @binding(4) var<storage, read_write> vp_sample_dp: array<vec4<f32>>;
// Shared uniform block.
@group(0) @binding(5) var<uniform> vp_params: ClothVpParams;
// CSR incidence: `csr_offsets[v]..csr_offsets[v+1]` index into `csr_entries`,
// which lists the samples carrying vertex `v` as an active corner.
@group(0) @binding(6) var<storage, read> vp_csr_offsets: array<u32>;
@group(0) @binding(7) var<storage, read> vp_csr_entries: array<u32>;

/// Maps a world position to its integer spatial-hash cell (`floor(pos / cell_size)`),
/// mirroring the CPU `cell_of` and the point tier's `cloth_cell_of`.
fn cloth_vp_cell_of(pos: vec3<f32>, cell_size: f32) -> vec3<i32> {
    let inv = 1.0 / cell_size;
    return vec3<i32>(floor(pos * inv));
}

/// Folds an integer cell into a bucket in `[0, table_size)` with the standard
/// large-prime multiply-xor uniform-grid hash (identical to the point tier).
fn cloth_vp_cell_hash(cell: vec3<i32>, table_size: u32) -> u32 {
    let x = bitcast<u32>(cell.x) * 73856093u;
    let y = bitcast<u32>(cell.y) * 19349663u;
    let z = bitcast<u32>(cell.z) * 83492791u;
    return (x ^ y ^ z) % table_size;
}

/// The live barycentric world position of a sample, `Σ wₖ · pos[vₖ]` (mirrors the
/// CPU `Sample::position`). Zero-weight corners are skipped implicitly.
fn cloth_vp_sample_position(s: ClothVpSample) -> vec3<f32> {
    var pos = vec3<f32>(0.0, 0.0, 0.0);
    if s.w0 != 0.0 {
        pos = pos + vp_positions[s.v0].xyz * s.w0;
    }
    if s.w1 != 0.0 {
        pos = pos + vp_positions[s.v1].xyz * s.w1;
    }
    if s.w2 != 0.0 {
        pos = pos + vp_positions[s.v2].xyz * s.w2;
    }
    return pos;
}

/// The sample's effective inverse mass `Σ wₖ² · max(inverse_mass[vₖ], 0)` (mirrors
/// the CPU `Sample::inverse_mass_eff`). A sample whose active corners are all
/// pinned returns `0` and is immovable.
fn cloth_vp_sample_eff(s: ClothVpSample) -> f32 {
    var eff = 0.0;
    if s.w0 != 0.0 {
        eff = eff + s.w0 * s.w0 * max(vp_positions[s.v0].w, 0.0);
    }
    if s.w1 != 0.0 {
        eff = eff + s.w1 * s.w1 * max(vp_positions[s.v1].w, 0.0);
    }
    if s.w2 != 0.0 {
        eff = eff + s.w2 * s.w2 * max(vp_positions[s.v2].w, 0.0);
    }
    return eff;
}

/// Whether two samples share any active (positive-weight) vertex. Incident
/// samples (a triangle against its own or an adjacent triangle's corners) must
/// not be separated. Mirrors the CPU `shares_active_vertex`.
fn cloth_vp_shares_active_vertex(a: ClothVpSample, b: ClothVpSample) -> bool {
    var av = array<u32, 3>(a.v0, a.v1, a.v2);
    var aw = array<f32, 3>(a.w0, a.w1, a.w2);
    var bv = array<u32, 3>(b.v0, b.v1, b.v2);
    var bw = array<f32, 3>(b.w0, b.w1, b.w2);
    for (var ka = 0; ka < 3; ka = ka + 1) {
        if aw[ka] <= 0.0 {
            continue;
        }
        for (var kb = 0; kb < 3; kb = kb + 1) {
            if bw[kb] <= 0.0 {
                continue;
            }
            if av[ka] == bv[kb] {
                return true;
            }
        }
    }
    return false;
}

/// Sample `a`'s own half of the separating push against neighbour `b`. Returns
/// zero when the pair shares an active vertex, is farther apart than `thickness`,
/// or is jointly immovable. `dir` points from `a` toward `b`, so `a` is pushed
/// the opposite way by its inverse-mass share; coincident samples separate along
/// `+X`. Mirrors the CPU `half_correction`.
fn cloth_vp_half(a_idx: u32, b_idx: u32, thickness: f32, thickness_sq: f32) -> vec3<f32> {
    let sa = vp_samples[a_idx];
    let sb = vp_samples[b_idx];
    if cloth_vp_shares_active_vertex(sa, sb) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let pa = cloth_vp_sample_position(sa);
    let pb = cloth_vp_sample_position(sb);
    let diff = pb - pa;
    let dist_sq = dot(diff, diff);
    if dist_sq >= thickness_sq {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let wa = cloth_vp_sample_eff(sa);
    let wb = cloth_vp_sample_eff(sb);
    let w_sum = wa + wb;
    if w_sum <= 0.0 {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let share = wa / w_sum;
    if dist_sq <= CLOTH_VP_EPS_LEN_SQ {
        // Coincident: `dir = +X`, penetration = full thickness; `a` pushed -X.
        return vec3<f32>(-thickness * share, 0.0, 0.0);
    }
    let dist = sqrt(dist_sq);
    let penetration = thickness - dist;
    let dir = diff * (1.0 / dist);
    return dir * (-penetration * share);
}

/// Phase-0 hash build: one invocation per sample prepends it to its cell's
/// linked list. The host must clear `cell_table` heads to `CLOTH_VP_SENTINEL`
/// (a `0xff` byte-fill) and reset the counts before this dispatch, mirroring the
/// CPU rebuild of a fresh `BTreeMap` each call.
@compute @workgroup_size(64)
fn cloth_vp_hash_build(@builtin(global_invocation_id) gid: vec3<u32>) {
    let a = gid.x;
    if a >= vp_params.sample_count {
        return;
    }
    let pos = cloth_vp_sample_position(vp_samples[a]);
    let cell = cloth_vp_cell_of(pos, vp_params.cell_size);
    let bucket = cloth_vp_cell_hash(cell, vp_params.table_size);
    let prev_head = atomicExchange(&vp_cell_table[bucket].head, a);
    vp_sample_next[a] = prev_head;
    atomicAdd(&vp_cell_table[bucket].count, 1u);
}

/// Phase-1 resolve: one invocation per sample `a` accumulates its own half of
/// the separating push from every in-range neighbour across the 27-cell
/// neighbourhood and writes only `sample_dp[a]` (Jacobi own-slot). Mirrors the
/// CPU Phase 1.
@compute @workgroup_size(64)
fn cloth_vp_resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let a = gid.x;
    if a >= vp_params.sample_count {
        return;
    }
    let a_is_real = a < vp_params.real_count;
    let thickness = vp_params.thickness;
    let thickness_sq = thickness * thickness;
    let pos_a = cloth_vp_sample_position(vp_samples[a]);
    let base_cell = cloth_vp_cell_of(pos_a, vp_params.cell_size);
    var acc = vec3<f32>(0.0, 0.0, 0.0);

    for (var dx = -1; dx <= 1; dx = dx + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dz = -1; dz <= 1; dz = dz + 1) {
                let neighbor_cell = base_cell + vec3<i32>(dx, dy, dz);
                let bucket = cloth_vp_cell_hash(neighbor_cell, vp_params.table_size);
                var b = atomicLoad(&vp_cell_table[bucket].head);
                for (var step = 0u; step < vp_params.sample_count; step = step + 1u) {
                    if b == CLOTH_VP_SENTINEL {
                        break;
                    }
                    let cur = b;
                    b = vp_sample_next[cur];
                    if cur == a {
                        continue;
                    }
                    // Exact-cell recheck: hash collisions can chain foreign
                    // cells into this bucket, so only act on samples whose true
                    // cell is the one we are visiting (each pair once per cell).
                    let other_pos = cloth_vp_sample_position(vp_samples[cur]);
                    let other_cell = cloth_vp_cell_of(other_pos, vp_params.cell_size);
                    if other_cell.x != neighbor_cell.x
                        || other_cell.y != neighbor_cell.y
                        || other_cell.z != neighbor_cell.z {
                        continue;
                    }
                    // Augment mode skips real-vs-real pairs (the friction point
                    // tier owns them); the full tier resolves every pair.
                    if vp_params.virtual_only != 0u && a_is_real && cur < vp_params.real_count {
                        continue;
                    }
                    acc = acc + cloth_vp_half(a, cur, thickness, thickness_sq);
                }
            }
        }
    }

    vp_sample_dp[a] = vec4<f32>(acc, 0.0);
}

/// Phase-2 scatter: one invocation per real vertex `v` gathers, through the CSR
/// incidence table, the barycentric share of every incident sample's
/// displacement and applies it to `positions[v]` (own-slot, no atomics).
/// Mirrors the CPU Phase 2 / `Sample::scatter`.
@compute @workgroup_size(64)
fn cloth_vp_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    if v >= vp_params.vertex_count {
        return;
    }
    let im_v = max(vp_positions[v].w, 0.0);
    if im_v <= 0.0 {
        // Pinned vertex never moves.
        return;
    }

    var delta = vec3<f32>(0.0, 0.0, 0.0);
    let start = vp_csr_offsets[v];
    let end = vp_csr_offsets[v + 1u];
    for (var e = start; e < end; e = e + 1u) {
        let a = vp_csr_entries[e];
        let s = vp_samples[a];
        let dp = vp_sample_dp[a].xyz;
        let eff = cloth_vp_sample_eff(s);
        if eff <= 0.0 {
            continue;
        }
        var w = 0.0;
        if s.v0 == v && s.w0 > 0.0 {
            w = w + s.w0;
        }
        if s.v1 == v && s.w1 > 0.0 {
            w = w + s.w1;
        }
        if s.v2 == v && s.w2 > 0.0 {
            w = w + s.w2;
        }
        if w <= 0.0 {
            continue;
        }
        let coeff = w * im_v / eff;
        delta = delta + dp * coeff;
    }

    vp_positions[v] = vec4<f32>(vp_positions[v].xyz + delta, vp_positions[v].w);
}
