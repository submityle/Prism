// Projective-dynamics global dense matrix-vector twin: evaluates on-device the
// same dense product `A x` the CPU golden `GlobalSystem::mul` produces, one
// thread per matrix row.
//
// Projective Dynamics (`Bouaziz` 2014, as used in `TressFX` / UE5 Groom-class
// global hair solves) couples every per-constraint local projection through a
// single global symmetric positive-definite (`SPD`) system
// `(M/h^2 + sum_i w_i S_i^T S_i) x = rhs`. The system matrix `A` is constant for
// a fixed constraint graph, so it is assembled and factored once; the hot
// per-iteration primitive of the conjugate-gradient / global-solve path is the
// dense product `A x`. The CPU golden `GlobalSystem::mul` computes, for each row
// `i`, `out[i] = sum_j A[i*n+j] * x[j]` independently for the three coordinate
// components. This kernel is the on-device twin of exactly that product so a
// passing real-device parity test is direct evidence the ported kernel computes
// the same matvec as the reference, not merely that its shader compiles.
//
// Distinction from the sibling `projective_edge` twin: that twin reproduces the
// *local* edge projection (`local_project_edge`), a sparse per-edge position
// projection with one thread per edge. This twin is the complementary *global*
// step - a dense `n x n` matrix-vector product with one thread per row - and no
// existing twin computes a dense global matvec. The two are disjoint halves of
// the projective-dynamics local/global split.
//
// Each thread owns one row `i < n`, loops `j` over all `n` columns, and
// accumulates the three coordinate components `out[i].c = sum_j A[i*n+j]*x[j].c`
// in ascending `j` order, matching the golden's fixed accumulation order. Rows
// `i >= n` early-return. `x` is uploaded flattened as `n` groups of three f32
// (the host pads a short `x` with zeros, mirroring the golden's zero padding).
//
// The kernel uses only multiply and add in the portable core-`WGSL` subset - no
// `exp`, `pow` or optional device feature - so this twin runs unmodified on
// Metal, Vulkan and DX12.
//
// Provenance: standard dense matrix-vector product plus a `wgpu` compute
// dispatch; no Unreal Engine source or derived code.

struct Params {
    // Dimension `n`: the matrix is `n * n` row-major and `x` / `out` are `n`
    // groups of three f32.
    n: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Row-major `n * n` system matrix `A`.
@group(0) @binding(1) var<storage, read> matrix: array<f32>;
// Flattened input vector: `n` groups of three f32 (`x`, `y`, `z` per particle).
@group(0) @binding(2) var<storage, read> x: array<f32>;
// Flattened output vector: `n` groups of three f32.
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = params.n;
    if (i >= n) {
        return;
    }
    var sx = 0.0;
    var sy = 0.0;
    var sz = 0.0;
    let row = i * n;
    for (var j: u32 = 0u; j < n; j = j + 1u) {
        let a = matrix[row + j];
        let b = 3u * j;
        sx = sx + a * x[b];
        sy = sy + a * x[b + 1u];
        sz = sz + a * x[b + 2u];
    }
    let o = 3u * i;
    out[o] = sx;
    out[o + 1u] = sy;
    out[o + 2u] = sz;
}
