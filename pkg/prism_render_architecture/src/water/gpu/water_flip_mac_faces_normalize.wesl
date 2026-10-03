// water_flip_mac_faces_normalize.wesl
//
// FLIP/APIC MAC transfer stage: face normalize (second half of P2G).
//
// The scatter pass (water_flip_mac_p2g) splats every particle onto the eight
// surrounding staggered MAC face nodes of each family, accumulating a
// momentum/mass pair per face:
//
//   face_scatter[face * 2 + 0] += w * v_comp     (weighted momentum)
//   face_scatter[face * 2 + 1] += w              (weighted mass)
//
// This pass divides momentum by mass to recover the grid face velocity
//
//   faces[face] = sum(w * v) / sum(w)
//
// per face. Because the trilinear weights are a partition of unity, a lone
// interior particle recovers its own velocity exactly and a uniform velocity
// field is reproduced exactly. A face no particle reached has zero mass; its
// velocity is defined to be zero (an empty face carries no momentum).
//
// Scatter accumulates on-device as signed fixed-point packed into atomic<u32>
// (value_fixed = i32(round-toward-zero of value * SCATTER_SCALE)). This pass
// reads that same buffer back as a plain signed-integer array and decodes each
// slot as f32(fixed) / SCATTER_SCALE before the divide. The scale cancels in
// momentum / mass; it is kept explicit only so the mass threshold is compared
// against the true accumulated weight. The CPU twin consumes the un-quantized
// f32 momentum/mass the scatter twin produces, so it divides those directly and
// does not model the device's fixed-point rounding.
//
// ABI (WaterKernel::FlipMacFacesNormalize shared descriptor: 3 storage + 1
// uniform, @workgroup_size(64, 1, 1), Faces domain). The pass shares the P2G
// bind group layout: binding 0 (particles) is unused here and left undeclared,
// binding 1 is the scatter accumulator (read), binding 2 is the normalized
// face-velocity output (read_write), binding 3 is the uniform block. One
// invocation owns one packed face.
//
// Packed face layout (identical to the scatter/divergence/projection stages): a
// single buffer holds the three families back to back, u then v then w, each
// i-fastest row-major over its own face lattice:
//   u (x faces): (nx+1) * ny     * nz
//   v (y faces): nx     * (ny+1) * nz
//   w (z faces): nx     * ny     * (nz+1)
// Normalize is family-agnostic: every face is divided the same way regardless
// of which block it falls in, so the pass sweeps the flat face index directly.

const FN_EPS: f32 = 1.0e-6;
const SCATTER_SCALE: f32 = 65536.0;

struct FacesNormalizeParams {
    grid_nx: u32,
    grid_ny: u32,
    grid_nz: u32,
    pad0: u32,
};

@group(0) @binding(1) var<storage, read> face_scatter: array<i32>;
@group(0) @binding(2) var<storage, read_write> faces: array<f32>;
@group(0) @binding(3) var<uniform> sim_params: FacesNormalizeParams;

@compute @workgroup_size(64, 1, 1)
fn water_flip_mac_faces_normalize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let f = gid.x;
    let nx = sim_params.grid_nx;
    let ny = sim_params.grid_ny;
    let nz = sim_params.grid_nz;
    if (nx == 0u || ny == 0u || nz == 0u) {
        return;
    }
    let uc = (nx + 1u) * ny * nz;
    let vc = nx * (ny + 1u) * nz;
    let wc = nx * ny * (nz + 1u);
    let faces_count = uc + vc + wc;
    if (f >= faces_count) {
        return;
    }

    let mom = f32(face_scatter[f * 2u + 0u]) / SCATTER_SCALE;
    let mass = f32(face_scatter[f * 2u + 1u]) / SCATTER_SCALE;
    if (mass > FN_EPS) {
        faces[f] = mom / mass;
    } else {
        faces[f] = 0.0;
    }
}
