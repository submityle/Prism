// Batched Voronoi fragment-assignment classifier.
//
// One invocation per query point: find the nearest seed site (its Voronoi
// cell) and the distance to the nearest wall of that cell. This mirrors the
// `cpu_assign_cells` golden twin arithmetic-for-arithmetic so the emitted cell
// index is integer-exact and the clearance matches within a tight tolerance.
//
// Provenance: nearest-site Voronoi membership and perpendicular-bisector cell
// walls are standard, publicly documented computational-geometry results. No
// Unreal Engine source or derived code.

struct Params {
    // Number of seed sites.
    n_sites: u32,
    // Number of query points.
    n_points: u32,
    // Squared-length threshold below which a rival bisector is degenerate.
    degenerate_eps: f32,
    // Padding to a 16-byte boundary.
    _pad: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Seed sites; xyz used, w ignored.
@group(0) @binding(1) var<storage, read> sites: array<vec4<f32>>;
// Query points; xyz used, w ignored.
@group(0) @binding(2) var<storage, read> points: array<vec4<f32>>;
// Output owning cell index per point (0xFFFFFFFF when there are no sites).
@group(0) @binding(3) var<storage, read_write> out_cell: array<u32>;
// Output clearance to the nearest cell wall per point.
@group(0) @binding(4) var<storage, read_write> out_clearance: array<f32>;

const NO_CELL: u32 = 0xFFFFFFFFu;

// Squared distance with the same summation order as the CPU twin.
fn length_sq(d: vec3<f32>) -> f32 {
    return d.x * d.x + d.y * d.y + d.z * d.z;
}

@compute @workgroup_size(64)
fn assign(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_points) {
        return;
    }
    let p = points[idx].xyz;

    if (params.n_sites == 0u) {
        out_cell[idx] = NO_CELL;
        out_clearance[idx] = 1e30;
        return;
    }

    // Nearest site by squared distance; strict `<` keeps the lowest index on a
    // tie, matching the twin.
    var best: u32 = 0u;
    var best_sq: f32 = length_sq(p - sites[0].xyz);
    for (var i: u32 = 1u; i < params.n_sites; i = i + 1u) {
        let sq = length_sq(p - sites[i].xyz);
        if (sq < best_sq) {
            best_sq = sq;
            best = i;
        }
    }

    // Clearance: least-negative bisector signed distance over the rivals,
    // negated so a positive value means the point sits inside the cell.
    let owner = sites[best].xyz;
    var worst_sd: f32 = -1e30;
    var found_wall: bool = false;
    for (var i: u32 = 0u; i < params.n_sites; i = i + 1u) {
        if (i == best) {
            continue;
        }
        let s = sites[i].xyz;
        let n = s - owner;
        if (length_sq(n) <= params.degenerate_eps) {
            continue;
        }
        // Bisector between owner and rival: unit normal through the midpoint,
        // mirroring `Plane::from_point_normal((owner+s)*0.5, s-owner)`.
        let unit = normalize(n);
        let mid = (owner + s) * 0.5;
        let offset = dot(unit, mid);
        let sd = dot(unit, p) - offset;
        if (sd > worst_sd) {
            worst_sd = sd;
        }
        found_wall = true;
    }

    out_cell[idx] = best;
    if (found_wall) {
        out_clearance[idx] = -worst_sd;
    } else {
        out_clearance[idx] = 1e30;
    }
}
