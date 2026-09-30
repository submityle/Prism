//! `wgpu` compute twin of Prism's froxel voxel forward-scatter decode
//! (`prism_render_architecture::hair::dual_scattering::voxel_forward_scatter`).
//!
//! The additive counterpart of `voxel_transmittance`: where that composites the
//! per-voxel densities into a transmittance product `T = product(1 - sigma_j)`
//! over voxels `0..=index`, this sums them into the cumulative coverage-weighted
//! crossing count `n = sum(sigma_j)` over the same slab — the monotonically
//! non-decreasing curve a dual-scattering shading pass samples for the `a_f^n`
//! exponent (design §6/§11 体素自阴影). Both decode the same `voxel_density`
//! slab but carry independent information: the product discards how coverage
//! splits across strands while the sum preserves the raw count.
//!
//! One invocation owns one light texel/ray: it walks that texel's compacted
//! density slice (`texel_ranges` into the shared `densities` pool), keeps a
//! running sum, floors each `sigma_j` at `0` (an out-of-contract negative slab
//! cannot break monotonicity), and writes the prefix crossing count for every
//! voxel into its own disjoint output slice. Because each thread owns a disjoint
//! row there is no cross-thread contention, and the per-voxel add order matches
//! the reference `..=end` fold (a `GPU` may still fuse a multiply-add and
//! perturb the low mantissa bits, so parity is asserted to a tolerance).
//!
//! Bit-faithfulness: each output voxel `v` holds `voxel_forward_scatter(row, v)`
//! — the reference's `sum(max(sigma_j, 0))` over `0..=v` — so the whole emitted
//! curve equals the golden evaluated at every index. A voxel may hold several
//! strands, so the running sum can exceed `1`; the count is deliberately *not*
//! capped. An empty texel row (`count == 0`) emits nothing; the golden's
//! `densities.is_empty()` → `0.0` scalar case has no per-voxel slot to write and
//! is exercised on the host side of the parity test.

/// Decode parameters. `texel_count` bounds the dispatch (one invocation per
/// light texel). `16`-byte scalar-packed so the uniform block is a multiple of
/// 16.
struct HairVoxelForwardScatterParams {
    texel_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One compacted per-texel slice into `densities`: entries `start .. start +
/// count`, in voxel order (voxel `0` nearest the light).
struct HairTexelRange {
    start: u32,
    count: u32,
}

@group(0) @binding(0) var<uniform> params: HairVoxelForwardScatterParams;

// Shared flat per-voxel optical density pool (texel-major); each entry is a
// `sigma_j` accumulated opacity, floored at `0` on read.
@group(0) @binding(1) var<storage, read> densities: array<f32>;

// Per-texel compacted ranges into `densities` (and, one-to-one, into the
// output `crossings` pool).
@group(0) @binding(2) var<storage, read> texel_ranges: array<HairTexelRange>;

// Flat prefix-sum crossing-count pool, same layout as `densities`:
// `crossings[start + v]` is the cumulative count over voxels `0..=v`.
@group(0) @binding(3) var<storage, read_write> crossings: array<f32>;

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if t >= params.texel_count {
        return;
    }

    let start = texel_ranges[t].start;
    let count = texel_ranges[t].count;

    // Running prefix sum n = sum(max(sigma_j, 0)) over 0..=v.
    var running = 0.0;
    for (var v = 0u; v < count; v = v + 1u) {
        let sigma = max(densities[start + v], 0.0);
        running = running + sigma;
        crossings[start + v] = running;
    }
}
