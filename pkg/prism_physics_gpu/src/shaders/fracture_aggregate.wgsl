// Per-fragment rigid-body seed aggregator.
//
// Two kernels turn a classified debris point cloud into one rigid-body seed per
// fragment cell:
//
//   * `scatter` runs one invocation per point and atomically adds that point's
//     mass, first moment (mass * position), and second moment (mass * the
//     position outer product) into its fragment's fixed-point accumulators.
//   * `finalize` runs one invocation per cell, de-quantises the accumulators,
//     and forms the fragment mass, centre of mass, and inertia tensor about the
//     centroid via `I = trace(M2) * E - M2`.
//
// The two kernels mirror the `cpu_aggregate_fragments` golden twin: integer
// atomic accumulation is exact and order independent, so the raw sums are
// bit-identical across devices and the final centroid and inertia agree within
// a tight floating-point tolerance.
//
// Provenance: the rigid-body mass/centroid/inertia formulas are textbook
// mechanics and fixed-point atomic accumulation is a standard GPU reduction. No
// Unreal Engine source or derived code.

struct Params {
    // Number of fragment cells.
    n_cells: u32,
    // Number of query points.
    n_points: u32,
    // Padding to keep the scalar block 16-byte aligned.
    _pad0: u32,
    _pad1: u32,
    // Fixed-point scale for accumulated mass.
    mass_scale: f32,
    // Fixed-point scale for the accumulated first moment.
    moment_scale: f32,
    // Fixed-point scale for the accumulated second moment.
    second_moment_scale: f32,
    // Padding to a 16-byte boundary.
    _pad2: f32,
};

// One rigid-body seed per fragment, packed into vec4 lanes for a clean readback.
struct FragmentOut {
    // xyz = centre of mass, w = total mass.
    centroid_mass: vec4<f32>,
    // Inertia tensor components: x = Ixx, y = Iyy, z = Izz, w = Ixy.
    inertia0: vec4<f32>,
    // Inertia tensor components: x = Ixz, y = Iyz, z and w unused.
    inertia1: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
// Query points; xyz = position, w = mass.
@group(0) @binding(1) var<storage, read> points: array<vec4<f32>>;
// Owning fragment cell per point (0xFFFFFFFF marks an unassigned point).
@group(0) @binding(2) var<storage, read> cells: array<u32>;
// Fixed-point mass accumulator, one slot per cell.
@group(0) @binding(3) var<storage, read_write> acc_mass: array<atomic<i32>>;
// Fixed-point first-moment accumulator, three slots per cell (cell*3 + axis).
@group(0) @binding(4) var<storage, read_write> acc_moment: array<atomic<i32>>;
// Fixed-point second-moment accumulator, six slots per cell
// (cell*6 + {xx, yy, zz, xy, xz, yz}).
@group(0) @binding(5) var<storage, read_write> acc_second: array<atomic<i32>>;
// Finalised per-fragment rigid-body seeds.
@group(0) @binding(6) var<storage, read_write> out_frag: array<FragmentOut>;

const NO_CELL: u32 = 0xFFFFFFFFu;

// Quantises `value * scale` to the nearest integer (ties to even) for atomic
// accumulation, matching the CPU twin's `round_ties_even`.
fn quantise(value: f32, scale: f32) -> i32 {
    return i32(round(value * scale));
}

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_points) {
        return;
    }
    let cell = cells[idx];
    if (cell == NO_CELL || cell >= params.n_cells) {
        return;
    }

    let pm = points[idx];
    let p = pm.xyz;
    let m = pm.w;

    atomicAdd(&acc_mass[cell], quantise(m, params.mass_scale));

    let base3 = cell * 3u;
    atomicAdd(&acc_moment[base3 + 0u], quantise(m * p.x, params.moment_scale));
    atomicAdd(&acc_moment[base3 + 1u], quantise(m * p.y, params.moment_scale));
    atomicAdd(&acc_moment[base3 + 2u], quantise(m * p.z, params.moment_scale));

    let base6 = cell * 6u;
    let s = params.second_moment_scale;
    atomicAdd(&acc_second[base6 + 0u], quantise(m * p.x * p.x, s));
    atomicAdd(&acc_second[base6 + 1u], quantise(m * p.y * p.y, s));
    atomicAdd(&acc_second[base6 + 2u], quantise(m * p.z * p.z, s));
    atomicAdd(&acc_second[base6 + 3u], quantise(m * p.x * p.y, s));
    atomicAdd(&acc_second[base6 + 4u], quantise(m * p.x * p.z, s));
    atomicAdd(&acc_second[base6 + 5u], quantise(m * p.y * p.z, s));
}

@compute @workgroup_size(64)
fn finalize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cid = gid.x;
    if (cid >= params.n_cells) {
        return;
    }

    let mass = f32(atomicLoad(&acc_mass[cid])) / params.mass_scale;
    if (mass <= 0.0) {
        out_frag[cid].centroid_mass = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out_frag[cid].inertia0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out_frag[cid].inertia1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return;
    }

    let base3 = cid * 3u;
    let first = vec3<f32>(
        f32(atomicLoad(&acc_moment[base3 + 0u])),
        f32(atomicLoad(&acc_moment[base3 + 1u])),
        f32(atomicLoad(&acc_moment[base3 + 2u])),
    ) / params.moment_scale;
    let centroid = first / mass;

    let base6 = cid * 6u;
    let s = params.second_moment_scale;
    let s_xx = f32(atomicLoad(&acc_second[base6 + 0u])) / s;
    let s_yy = f32(atomicLoad(&acc_second[base6 + 1u])) / s;
    let s_zz = f32(atomicLoad(&acc_second[base6 + 2u])) / s;
    let s_xy = f32(atomicLoad(&acc_second[base6 + 3u])) / s;
    let s_xz = f32(atomicLoad(&acc_second[base6 + 4u])) / s;
    let s_yz = f32(atomicLoad(&acc_second[base6 + 5u])) / s;

    // Shift the second moment to the centroid: M2 = S2 - mass * (c (x) c).
    let m_xx = s_xx - mass * centroid.x * centroid.x;
    let m_yy = s_yy - mass * centroid.y * centroid.y;
    let m_zz = s_zz - mass * centroid.z * centroid.z;
    let m_xy = s_xy - mass * centroid.x * centroid.y;
    let m_xz = s_xz - mass * centroid.x * centroid.z;
    let m_yz = s_yz - mass * centroid.y * centroid.z;

    let i_xx = m_yy + m_zz;
    let i_yy = m_xx + m_zz;
    let i_zz = m_xx + m_yy;

    out_frag[cid].centroid_mass = vec4<f32>(centroid, mass);
    out_frag[cid].inertia0 = vec4<f32>(i_xx, i_yy, i_zz, -m_xy);
    out_frag[cid].inertia1 = vec4<f32>(-m_xz, -m_yz, 0.0, 0.0);
}
