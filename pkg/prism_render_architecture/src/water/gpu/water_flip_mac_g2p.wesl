// water_flip_mac_g2p.wesl
//
// FLIP/APIC MAC transfer stage: grid-to-particle (G2P) gather — the half that
// closes the FLIP/APIC loop against the staggered MAC pressure projection.
//
// The scatter pass (water_flip_mac_p2g) splats each particle's velocity
// component onto the face it is normal to; the normalize pass
// (water_flip_mac_faces_normalize) turns the accumulated momentum/mass into a
// face velocity; the projection pass removes the divergent part of those face
// velocities. This kernel gathers the projected face field back to each
// particle and rebuilds its velocity (and APIC affine matrix).
//
// FLIP/PIC blend and the FLIP delta
// ---------------------------------
// Classic FLIP carries the change the grid applied this step, not the grid
// velocity itself. On the staggered grid that change is read directly as
// faces_projected - faces_preprojection per sampled face, so no pressure
// gradient is recomputed on the particle side and the delta is exact even where
// the boundary stencil clamps the gradient. PIC gathers the projected face
// velocity. They are mixed as (1 - alpha)*PIC + alpha*FLIP with alpha clamped to
// 0..=1, mirroring the CPU blend_flip_pic: alpha = 0 is pure dissipative PIC,
// alpha = 1 is pure FLIP. When faces_projected == faces_preprojection the delta
// is zero and FLIP keeps the particle's own incoming velocity.
//
// APIC affine reconstruction
// --------------------------
// When the affine field is carried, each affine row is rebuilt from the
// projected face velocities of its own component family:
//   C_row_a = (3 / dx^2) * sum(w * v_proj * (x_face - x_p))
// using the scalar multilinear inertia inverse D^-1 = 3 / dx^2, exactly
// mirroring the inverse of the P2G apic_velocity forward map. A zero affine
// request writes zero rows (plain PIC/FLIP).
//
// ABI (WaterKernel::FlipMacG2P shared descriptor: 3 storage + 1 uniform,
// @workgroup_size(64, 1, 1), Particle domain). binding 0 is the live particle
// pool (read_write), binding 1 the projected faces (read), binding 2 the
// pre-projection faces (read), binding 3 the uniform block. Particles are a flat
// f32 array, five vec4 lanes (20 f32) each: position+active, velocity, three
// APIC affine rows — identical to the P2G layout.
//
// Packed face layout (identical to the scatter/normalize/projection stages): a
// single buffer holds the three families back to back, u then v then w, each
// i-fastest row-major over its own face lattice:
//   u (x faces): (nx+1) * ny     * nz,     u_idx = i + (nx+1)*(j + ny*k)
//   v (y faces): nx     * (ny+1) * nz,     v_idx = u_count + i + nx*(j + (ny+1)*k)
//   w (z faces): nx     * ny     * (nz+1), w_idx = u_count + v_count + i + nx*(j + ny*k)
// The grid origin is the lattice origin (g = pos / dx), matching the P2G twin.
//
// One invocation owns one particle; it only reads the two face buffers and
// writes its own particle slot, so the gather is race-free and order
// independent. Every sampled face is bounds-checked against its staggered
// lattice, so corners outside the domain are skipped and never read out of
// range. Only + - * / and comparisons (plus the floor builtin) appear.

const G2P_EPS: f32 = 1.0e-6;
const FLIP_PARTICLE_FLOATS: u32 = 20u;
// Multilinear APIC inertia inverse numerator; D^-1 = APIC_INV_D_NUM / dx^2.
const APIC_INV_D_NUM: f32 = 3.0;

struct FlipG2PParams {
    grid_nx: u32,
    grid_ny: u32,
    grid_nz: u32,
    particle_count: u32,
    dx: f32,
    flip_blend: f32,
    use_affine: u32,
    pad0: f32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<f32>;
@group(0) @binding(1) var<storage, read> faces_projected: array<f32>;
@group(0) @binding(2) var<storage, read> faces_preprojection: array<f32>;
@group(0) @binding(3) var<uniform> sim_params: FlipG2PParams;

struct AxisGather {
    pic: f32,
    delta: f32,
    row: vec3<f32>,
};

// Trilinear weight along one axis for corner `c` (0 = low node, 1 = high node).
fn w1(f: f32, c: u32) -> f32 {
    if (c == 0u) {
        return 1.0 - f;
    }
    return f;
}

// Gathers one velocity component from its staggered face family.
// `g` is the particle position in grid units (pos / dx); `shift` the family's
// half-cell node offset; `max_*` the inclusive max index of the family lattice;
// `stride_j`/`stride_k`/`off` the packed block layout. Returns the PIC
// component, the FLIP delta component, and the APIC affine row before the D^-1
// scale.
fn gather_axis(
    g: vec3<f32>,
    shift: vec3<f32>,
    dx: f32,
    max_i: i32,
    max_j: i32,
    max_k: i32,
    stride_j: u32,
    stride_k: u32,
    off: u32,
) -> AxisGather {
    var result: AxisGather;
    result.pic = 0.0;
    result.delta = 0.0;
    result.row = vec3<f32>(0.0, 0.0, 0.0);

    let lattice = g - shift;
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
                    let flat = off + u32(gi) + stride_j * u32(gj) + stride_k * u32(gk);
                    let v_proj = faces_projected[flat];
                    let v_pre = faces_preprojection[flat];
                    result.pic = result.pic + v_proj * weight;
                    result.delta = result.delta + (v_proj - v_pre) * weight;
                    let node = vec3<f32>(
                        f32(gi) + shift.x,
                        f32(gj) + shift.y,
                        f32(gk) + shift.z,
                    );
                    let offset = (node - g) * dx;
                    result.row = result.row + offset * (v_proj * weight);
                }
                cx = cx + 1u;
            }
            cy = cy + 1u;
        }
        cz = cz + 1u;
    }
    return result;
}

@compute @workgroup_size(64, 1, 1)
fn water_flip_mac_g2p(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    let pc = sim_params.particle_count;
    let dx = sim_params.dx;
    if (pid >= pc || dx <= G2P_EPS) {
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
    let g = pos / dx;

    let uc = (nx + 1u) * ny * nz;
    let vc = nx * (ny + 1u) * nz;

    // x-velocity <- u-faces (nodes at (i, j+0.5, k+0.5), i in 0..=nx).
    let gu = gather_axis(
        g, vec3<f32>(0.0, 0.5, 0.5), dx,
        i32(nx), i32(ny) - 1, i32(nz) - 1,
        nx + 1u, (nx + 1u) * ny, 0u,
    );
    // y-velocity <- v-faces (nodes at (i+0.5, j, k+0.5), j in 0..=ny).
    let gv = gather_axis(
        g, vec3<f32>(0.5, 0.0, 0.5), dx,
        i32(nx) - 1, i32(ny), i32(nz) - 1,
        nx, nx * (ny + 1u), uc,
    );
    // z-velocity <- w-faces (nodes at (i+0.5, j+0.5, k), k in 0..=nz).
    let gw = gather_axis(
        g, vec3<f32>(0.5, 0.5, 0.0), dx,
        i32(nx) - 1, i32(ny) - 1, i32(nz),
        nx, nx * ny, uc + vc,
    );

    let pic = vec3<f32>(gu.pic, gv.pic, gw.pic);
    let delta = vec3<f32>(gu.delta, gv.delta, gw.delta);
    let flip_vel = vel + delta;
    var alpha = sim_params.flip_blend;
    if (alpha < 0.0) {
        alpha = 0.0;
    }
    if (alpha > 1.0) {
        alpha = 1.0;
    }
    let new_vel = pic * (1.0 - alpha) + flip_vel * alpha;

    particles[base + 4u] = new_vel.x;
    particles[base + 5u] = new_vel.y;
    particles[base + 6u] = new_vel.z;

    if (sim_params.use_affine != 0u) {
        let inv_d = APIC_INV_D_NUM / (dx * dx);
        let c0 = gu.row * inv_d;
        let c1 = gv.row * inv_d;
        let c2 = gw.row * inv_d;
        particles[base + 8u] = c0.x;
        particles[base + 9u] = c0.y;
        particles[base + 10u] = c0.z;
        particles[base + 12u] = c1.x;
        particles[base + 13u] = c1.y;
        particles[base + 14u] = c1.z;
        particles[base + 16u] = c2.x;
        particles[base + 17u] = c2.y;
        particles[base + 18u] = c2.z;
    } else {
        particles[base + 8u] = 0.0;
        particles[base + 9u] = 0.0;
        particles[base + 10u] = 0.0;
        particles[base + 12u] = 0.0;
        particles[base + 13u] = 0.0;
        particles[base + 14u] = 0.0;
        particles[base + 16u] = 0.0;
        particles[base + 17u] = 0.0;
        particles[base + 18u] = 0.0;
    }
}
