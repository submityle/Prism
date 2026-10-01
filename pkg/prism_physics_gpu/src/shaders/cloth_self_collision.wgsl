// GPU virtual-particle cloth self-collision — the parallel-safe Jacobi twin of
// `prism_physics_core`'s `resolve_self_collision_virtual_jacobi`.
//
// Two own-slot passes, each reading only a frozen position snapshot so the
// result is independent of invocation order (a Jacobi iteration, never
// Gauss-Seidel):
//
//   * `phase1_samples` — one invocation per sample (real vertex or virtual
//     particle). It walks the sample's 27-cell neighbourhood through a
//     host-built uniform hash and sums *its own half* of the separating push
//     against every penetrating neighbour, writing a single `sample_dp[a]`
//     slot and reading no other slot. No atomics.
//   * `phase2_apply` — one invocation per real vertex. It gathers the
//     barycentric share of every incident sample's displacement through a
//     host-built CSR incidence list, then writes `positions_out[v] =
//     positions[v] + out[v]`.
//
// Cell assignment is performed on the host (integer-exact) and uploaded, so the
// only floating-point work on the GPU is the separating-push arithmetic, which
// the parity suite checks against the CPU golden within a tight tolerance.
//
// Provenance: the virtual-particle technique is the published NvCloth method;
// the Jacobi own-slot accumulate/apply split is a standard parallel
// position-based-dynamics reformulation. No Unreal Engine source or derived
// code.

struct Params {
    // Number of real particles (the leading samples).
    real_count: u32,
    // Total samples: real particles followed by in-range virtual particles.
    sample_count: u32,
    // Number of occupied grid cells (length of `grid_keys`).
    key_count: u32,
    // 0 = PairScope::All (real-vs-real included); 1 = PairScope::VirtualOnly.
    virtual_only: u32,
    // Fabric thickness and its square.
    thickness: f32,
    thickness_sq: f32,
    // Coincidence guard: pairs closer than this (squared) separate along +X.
    eps_len_sq: f32,
    pad0: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Frozen position snapshot, xyz used.
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
// Index-aligned inverse masses (0 marks a pinned particle).
@group(0) @binding(2) var<storage, read> inverse_masses: array<f32>;
// Sample vertex indices, xyz used (real sample is [i, i, i]).
@group(0) @binding(3) var<storage, read> sample_verts: array<vec4<u32>>;
// Sample barycentric weights, xyz used (real sample is [1, 0, 0]).
@group(0) @binding(4) var<storage, read> sample_weights: array<vec4<f32>>;
// Per-sample integer cell coordinate, xyz used (host-assigned, integer-exact).
@group(0) @binding(5) var<storage, read> sample_cells: array<vec4<i32>>;
// Occupied cell coordinates sorted ascending by (x, y, z), xyz used.
@group(0) @binding(6) var<storage, read> grid_keys: array<vec4<i32>>;
// CSR offsets into `grid_members`, length `key_count + 1`.
@group(0) @binding(7) var<storage, read> grid_offsets: array<u32>;
// Sample indices bucketed by cell, ascending within each bucket.
@group(0) @binding(8) var<storage, read> grid_members: array<u32>;
// Phase-1 output: each sample's accumulated half-correction, xyz used.
@group(0) @binding(9) var<storage, read_write> sample_dp: array<vec4<f32>>;
// CSR offsets into `vert_entries`, length `real_count + 1`.
@group(0) @binding(10) var<storage, read> vert_offsets: array<u32>;
// Incident (sample index, weight slot) pairs per vertex, ascending by sample.
@group(0) @binding(11) var<storage, read> vert_entries: array<vec2<u32>>;
// Phase-2 output: the snapshot positions advanced by their corrections.
@group(0) @binding(12) var<storage, read_write> positions_out: array<vec4<f32>>;

// The sample's live world position `Σ weights[k] * position[verts[k]]`.
fn sample_position(a: u32) -> vec3<f32> {
    let w = sample_weights[a];
    let v = sample_verts[a];
    var pos = vec3<f32>(0.0, 0.0, 0.0);
    if (w.x != 0.0) { pos += positions[v.x].xyz * w.x; }
    if (w.y != 0.0) { pos += positions[v.y].xyz * w.y; }
    if (w.z != 0.0) { pos += positions[v.z].xyz * w.z; }
    return pos;
}

// The effective inverse mass `Σ weights[k]^2 * inverse_mass[verts[k]]`.
fn sample_eff(a: u32) -> f32 {
    let w = sample_weights[a];
    let v = sample_verts[a];
    var eff = 0.0;
    if (w.x != 0.0) { eff += w.x * w.x * max(inverse_masses[v.x], 0.0); }
    if (w.y != 0.0) { eff += w.y * w.y * max(inverse_masses[v.y], 0.0); }
    if (w.z != 0.0) { eff += w.z * w.z * max(inverse_masses[v.z], 0.0); }
    return eff;
}

// True when two samples share any active (positive-weight) vertex.
fn shares_active_vertex(a: u32, b: u32) -> bool {
    let wa = sample_weights[a];
    let va = sample_verts[a];
    let wb = sample_weights[b];
    let vb = sample_verts[b];
    for (var ka = 0u; ka < 3u; ka = ka + 1u) {
        if (wa[ka] <= 0.0) { continue; }
        for (var kb = 0u; kb < 3u; kb = kb + 1u) {
            if (wb[kb] <= 0.0) { continue; }
            if (va[ka] == vb[kb]) { return true; }
        }
    }
    return false;
}

// Sample `a`'s own half of the separating push against neighbour `b`.
fn half_correction(a: u32, b: u32) -> vec3<f32> {
    if (shares_active_vertex(a, b)) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let pa = sample_position(a);
    let pb = sample_position(b);
    let delta = pb - pa;
    let dist_sq = dot(delta, delta);
    if (dist_sq >= params.thickness_sq) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let wa = sample_eff(a);
    let wb = sample_eff(b);
    let w_sum = wa + wb;
    if (w_sum <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    var dir: vec3<f32>;
    var penetration: f32;
    if (dist_sq <= params.eps_len_sq) {
        dir = vec3<f32>(1.0, 0.0, 0.0);
        penetration = params.thickness;
    } else {
        let dist = sqrt(dist_sq);
        dir = delta / dist;
        penetration = params.thickness - dist;
    }
    // `dir` points from A toward B; A is pushed the opposite way by its share.
    return dir * (-penetration * (wa / w_sum));
}

// Compares two integer cell keys in the BTreeMap (x, then y, then z) order.
// Returns -1 when `key` sorts before `probe`, 1 when after, 0 when equal.
fn compare_cell(key: vec3<i32>, probe: vec3<i32>) -> i32 {
    if (key.x != probe.x) {
        if (key.x < probe.x) { return -1; }
        return 1;
    }
    if (key.y != probe.y) {
        if (key.y < probe.y) { return -1; }
        return 1;
    }
    if (key.z != probe.z) {
        if (key.z < probe.z) { return -1; }
        return 1;
    }
    return 0;
}

// Binary-searches `grid_keys` for `cell`; returns its index, or -1 if absent.
fn find_cell(cell: vec3<i32>) -> i32 {
    var lo = 0i;
    var hi = i32(params.key_count) - 1;
    while (lo <= hi) {
        let mid = lo + (hi - lo) / 2;
        let ord = compare_cell(grid_keys[u32(mid)].xyz, cell);
        if (ord == 0) {
            return mid;
        } else if (ord < 0) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return -1;
}

@compute @workgroup_size(64)
fn phase1_samples(@builtin(global_invocation_id) gid: vec3<u32>) {
    let a = gid.x;
    if (a >= params.sample_count) {
        return;
    }
    let cell = sample_cells[a].xyz;
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    for (var dx = -1i; dx <= 1i; dx = dx + 1i) {
        for (var dy = -1i; dy <= 1i; dy = dy + 1i) {
            for (var dz = -1i; dz <= 1i; dz = dz + 1i) {
                let neighbor = cell + vec3<i32>(dx, dy, dz);
                let idx = find_cell(neighbor);
                if (idx < 0) {
                    continue;
                }
                let ui = u32(idx);
                let start = grid_offsets[ui];
                let end = grid_offsets[ui + 1u];
                for (var m = start; m < end; m = m + 1u) {
                    let b = grid_members[m];
                    if (b == a) {
                        continue;
                    }
                    if (params.virtual_only == 1u
                        && a < params.real_count
                        && b < params.real_count) {
                        continue;
                    }
                    acc += half_correction(a, b);
                }
            }
        }
    }
    sample_dp[a] = vec4<f32>(acc, 0.0);
}

@compute @workgroup_size(64)
fn phase2_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    if (v >= params.real_count) {
        return;
    }
    let start = vert_offsets[v];
    let end = vert_offsets[v + 1u];
    var out = vec3<f32>(0.0, 0.0, 0.0);
    for (var e = start; e < end; e = e + 1u) {
        let entry = vert_entries[e];
        let a = entry.x;
        let k = entry.y;
        let dp = sample_dp[a].xyz;
        if (dp.x == 0.0 && dp.y == 0.0 && dp.z == 0.0) {
            continue;
        }
        let eff = sample_eff(a);
        if (eff <= 0.0) {
            continue;
        }
        let w = sample_weights[a][k];
        if (w == 0.0) {
            continue;
        }
        let im = max(inverse_masses[v], 0.0);
        if (im <= 0.0) {
            continue;
        }
        out += dp * (w * im / eff);
    }
    positions_out[v] = vec4<f32>(positions[v].xyz + out, positions[v].w);
}
