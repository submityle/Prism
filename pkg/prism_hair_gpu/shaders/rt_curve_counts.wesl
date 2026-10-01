// Per-strand LSS segment-count twin.
//
// Device mirror of the CPU golden
// `prism_render_architecture::hair::rt_curve::lss_segment_counts`. One thread
// per strand emits the number of Linear Swept Spheres segments that strand
// contributes to a downstream ray-tracing BLAS primitive budget.
//
// A strand of `n` vertices yields `max(0, n - 1)` segments (adjacent vertices
// form one segment each; fewer than two vertices yield none). The kernel reads
// only each strand's vertex count, so it is pure integer arithmetic with no
// floating point, no rounding and no transcendental step - the twin is
// *bit-exact* with the scalar reference and the parity test asserts integer
// equality.
//
// Provenance: hardware ray-traced curve / LSS strand primitive BLAS contract
// (RTX DXR / OptiX LSS) plus wgpu compute dispatch; no Unreal Engine source or
// derived code.

struct Params {
    // Number of strands in `lengths` / `counts`.
    strand_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per strand: its vertex count.
@group(0) @binding(1) var<storage, read> lengths: array<u32>;
// Per strand: max(0, vertex_count - 1) LSS segments.
@group(0) @binding(2) var<storage, read_write> counts: array<u32>;

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.strand_count) {
        return;
    }
    let n = lengths[i];
    // u32 is unsigned, so guard the subtraction: < 2 vertices yield 0 segments.
    counts[i] = select(0u, n - 1u, n >= 2u);
}
