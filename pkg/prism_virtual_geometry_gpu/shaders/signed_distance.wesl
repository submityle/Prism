// Prism virtual-geometry frustum-plane twin: evaluates the signed distance from
// a point to an inward-facing plane in closed form, mirroring the CPU golden
// `Plane::signed_distance` (in `prism_render_architecture::virtual_geometry`).
//
// Frustum culling keeps a cluster when its bounds' signed distance to every
// inward plane clears the negated support radius. The signed distance itself -
// `dot(normal, point) + distance` - is the scalar every plane test consumes.
// The discrete cluster-cull twin only checks the boolean verdict; this kernel
// mirrors the continuous scalar so the two backends' plane arithmetic is
// validated to the bit, catching any floating-point drift the boolean verdict
// would hide. One thread evaluates one query.
//
// Numerical model: the value is `n0*p0 + n1*p1 + n2*p2 + d`. A GPU may contract
// this into fused-multiply-adds and reassociate the sum, so it is bit-exact
// against the sequential CPU golden only when every product and every partial
// sum is exactly representable (no rounding occurs, so fusion and ordering are
// immaterial). The parity suite therefore drives it with small integer / dyadic
// operands whose products and running sums stay well inside the 24-bit
// mantissa, where the result is bit-for-bit identical regardless of FMA or
// evaluation order. The kernel uses only the portable core-WGSL subset - no
// optional feature - so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard inward-plane signed-distance test for frustum culling;
// no Unreal Engine source or derived code.

struct Params {
    // Number of queries in `inputs` / outputs in `distances`.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One plane/point query. `28`-byte stride matching the `bytemuck` upload struct.
struct PlaneQuery {
    // Inward-facing plane normal.
    nx: f32, ny: f32, nz: f32,
    // Query point.
    px: f32, py: f32, pz: f32,
    // Signed plane distance to the origin.
    d: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputs: array<PlaneQuery>;
@group(0) @binding(2) var<storage, read_write> distances: array<f32>;

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }

    let q = inputs[i];
    // n0*p0 + n1*p1 + n2*p2 + d - mirrors Plane::signed_distance left-to-right.
    distances[i] = q.nx * q.px + q.ny * q.py + q.nz * q.pz + q.d;
}
