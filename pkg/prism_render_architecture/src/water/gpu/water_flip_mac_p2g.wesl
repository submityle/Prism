// water_flip_mac_p2g.wesl
//
// FLIP/APIC MAC transfer stage: particle-to-grid (P2G) scatter.
//
// Each particle splats its (optionally affine, APIC) velocity onto the eight
// surrounding staggered MAC face nodes of every face family with trilinear
// weights. The deposit is accumulated as a momentum/mass pair per face:
//
//   face_scatter[face * 2 + 0] += w * v_comp     (weighted momentum)
//   face_scatter[face * 2 + 1] += w              (weighted mass)
//
// A later normalize pass (water_flip_mac_faces_normalize) divides momentum by
// mass to recover the face velocity v = sum(w * v) / sum(w). Because trilinear
// weights are a partition of unity, a single interior particle recovers its own
// velocity exactly and a uniform velocity field is reproduced exactly.
//
// For the x faces the deposited component is the particle x-velocity (APIC row
// 0 for the affine term), y faces take the y-velocity (row 1), z faces the
// z-velocity (row 2). The APIC velocity at a node is v_p + C_p (x_node - x_p);
// a zero affine matrix collapses to plain PIC splatting.
//
// Scatter is a many-writers-one-slot operation, so the accumulators are
// atomics. Metal has no float atomics, so momentum and mass are kept as signed
// fixed-point packed into atomic<u32>: value_fixed = i32(round-toward-zero of
// value * SCATTER_SCALE), added with atomicAdd (two's-complement add is
// identical for signed and unsigned), and decoded by the normalize pass as
// bitcast<i32>(bits) / SCATTER_SCALE. The CPU twin models the un-quantized f32
// momentum/mass this fixed-point sum approximates.
//
// ABI (WaterKernel::FlipMacP2G shared descriptor: 3 storage + 1 uniform,
// @workgroup_size(64, 1, 1), Particle domain). binding 2 (the normalized face
// buffer) is reserved for the shared face-normalize pass and is not touched
// here, so this shader declares only the two storage buffers it reads/writes
// plus the uniform block. Particles are a flat f32 array, five vec4 lanes
// (20 f32) each: position+active, velocity, and the three APIC affine rows.
//
// Packed face layout (identical to the divergence/projection stages): a single
// buffer holds the three families back to back, u then v then w, each
// i-fastest row-major over its own face lattice:
//   u (x faces): (nx+1) * ny     * nz,     u_idx = i + (nx+1)*(j + ny*k)
//   v (y faces): nx     * (ny+1) * nz,     v_idx = i + nx    *(j + (ny+1)*k)
//   w (z faces): nx     * ny     * (nz+1), w_idx = i + nx    *(j + ny*k)
// with offsets u_off = 0, v_off = u_count, w_off = u_count + v_count.

const P2G_EPS: f32 = 1.0e-6;
const SCATTER_SCALE: f32 = 65536.0;
const FLIP_PARTICLE_FLOATS: u32 = 20u;

struct FlipP2GParams {
    grid_nx: u32,
    grid_ny: u32,
    grid_nz: u32,
    particle_count: u32,
    dx: f32,
    use_affine: u32,
    pad0: f32,
    pad1: f32,
};

@group(0) @binding(0) var<storage, read> particles: array<f32>;
@group(0) @binding(1) var<storage, read_write> face_scatter: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> sim_params: FlipP2GParams;

// Trilinear weight along one axis for corner `c` (0 = low node, 1 = high node).
fn w1(f: f32, c: u32) -> f32 {
    if (c == 0u) {
        return 1.0 - f;
    }
    return f;
}

// Accumulate one weighted deposit into a face's momentum/mass pair.
fn scatter_face(face_flat: u32, weight: f32, vel_comp: f32) {
    let mom = bitcast<u32>(i32(weight * vel_comp * SCATTER_SCALE));
    let mass = bitcast<u32>(i32(weight * SCATTER_SCALE));
    atomicAdd(&face_scatter[face_flat * 2u + 0u], mom);
    atomicAdd(&face_scatter[face_flat * 2u + 1u], mass);
}

// Scatter one particle onto the eight surrounding nodes of a single face
// family. `lattice` is the particle position in that family's staggered index
// space, `node_half` the family's half-cell node offset within a cell, and
// `off`/`stride_j`/`stride_k`/`max_*` describe the packed block of this family.
fn scatter_family(
    lattice: vec3<f32>,
    node_half: vec3<f32>,
    g: vec3<f32>,
    base_comp: f32,
    affine_row: vec3<f32>,
    use_affine: bool,
    dx: f32,
    max_i: i32,
    max_j: i32,
    max_k: i32,
    stride_j: u32,
    stride_k: u32,
    off: u32,
) {
    let bi = i32(floor(lattice.x));
    let bj = i32(floor(lattice.y));
    let bk = i32(floor(lattice.z));
    let fx = lattice.x - f32(bi);
    let fy = lattice.y - f32(bj);
    let fz = lattice.z - f32(bk);

    var cz: u32 = 0u;
    loop {
        if (cz >= 2u) {
            break;
        }
        let gk = bk + i32(cz);
        var cy: u32 = 0u;
        loop {
            if (cy >= 2u) {
                break;
            }
            let gj = bj + i32(cy);
            var cx: u32 = 0u;
            loop {
                if (cx >= 2u) {
                    break;
                }
                let gi = bi + i32(cx);
                if (gi >= 0 && gi <= max_i && gj >= 0 && gj <= max_j && gk >= 0 && gk <= max_k) {
                    let weight = w1(fx, cx) * w1(fy, cy) * w1(fz, cz);
                    let node = vec3<f32>(
                        f32(gi) + node_half.x,
                        f32(gj) + node_half.y,
                        f32(gk) + node_half.z,
                    );
                    let offset = (node - g) * dx;
                    var vel_comp = base_comp;
                    if (use_affine) {
                        vel_comp = base_comp + dot(affine_row, offset);
                    }
                    let face_flat = off + u32(gi) + stride_j * u32(gj) + stride_k * u32(gk);
                    scatter_face(face_flat, weight, vel_comp);
                }
                cx = cx + 1u;
            }
            cy = cy + 1u;
        }
        cz = cz + 1u;
    }
}

@compute @workgroup_size(64, 1, 1)
fn water_flip_mac_p2g(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    let pc = sim_params.particle_count;
    let dx = sim_params.dx;
    if (pid >= pc || dx <= P2G_EPS) {
        return;
    }
    let nx = sim_params.grid_nx;
    let ny = sim_params.grid_ny;
    let nz = sim_params.grid_nz;
    if (nx == 0u || ny == 0u || nz == 0u) {
        return;
    }

    let base = pid * FLIP_PARTICLE_FLOATS;
    if (particles[base + 3u] < 0.5) {
        return;
    }
    let pos = vec3<f32>(particles[base + 0u], particles[base + 1u], particles[base + 2u]);
    let vel = vec3<f32>(particles[base + 4u], particles[base + 5u], particles[base + 6u]);
    let c0 = vec3<f32>(particles[base + 8u], particles[base + 9u], particles[base + 10u]);
    let c1 = vec3<f32>(particles[base + 12u], particles[base + 13u], particles[base + 14u]);
    let c2 = vec3<f32>(particles[base + 16u], particles[base + 17u], particles[base + 18u]);
    let use_aff = sim_params.use_affine != 0u;
    let g = pos / dx;

    let uc = (nx + 1u) * ny * nz;
    let vc = nx * (ny + 1u) * nz;

    // u (x) faces: node at (i, j + 0.5, k + 0.5); deposit x-velocity (row 0).
    scatter_family(
        vec3<f32>(g.x, g.y - 0.5, g.z - 0.5),
        vec3<f32>(0.0, 0.5, 0.5),
        g, vel.x, c0, use_aff, dx,
        i32(nx), i32(ny) - 1, i32(nz) - 1,
        nx + 1u, (nx + 1u) * ny, 0u,
    );

    // v (y) faces: node at (i + 0.5, j, k + 0.5); deposit y-velocity (row 1).
    scatter_family(
        vec3<f32>(g.x - 0.5, g.y, g.z - 0.5),
        vec3<f32>(0.5, 0.0, 0.5),
        g, vel.y, c1, use_aff, dx,
        i32(nx) - 1, i32(ny), i32(nz) - 1,
        nx, nx * (ny + 1u), uc,
    );

    // w (z) faces: node at (i + 0.5, j + 0.5, k); deposit z-velocity (row 2).
    scatter_family(
        vec3<f32>(g.x - 0.5, g.y - 0.5, g.z),
        vec3<f32>(0.5, 0.5, 0.0),
        g, vel.z, c2, use_aff, dx,
        i32(nx) - 1, i32(ny) - 1, i32(nz),
        nx, nx * ny, uc + vc,
    );
}
