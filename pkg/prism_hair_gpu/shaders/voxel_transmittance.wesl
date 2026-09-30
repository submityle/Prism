//! `wgpu` compute twin of Prism's froxel voxel transmittance decode
//! (`prism_render_architecture::hair::deep_transmittance::voxel_transmittance`).
//!
//! Where `voxel_density` accumulates a per-voxel optical density `sigma`, this
//! kernel is the decode half: it composites that slab into the running
//! self-shadow transmittance `T = product(1 - sigma_j)` over voxels
//! `0..=index`, the monotonically non-increasing curve a shading pass samples
//! to attenuate light through the groom (design §6/§11 体素自阴影). Uniform
//! voxels give a constant stride, so the whole curve is a single prefix product
//! per light texel — no sort, no transcendental.
//!
//! One invocation owns one light texel/ray: it walks that texel's compacted
//! density slice (`texel_ranges` into the shared `densities` pool), keeps a
//! running product, clamps each `sigma_j` into `0..=1`, and writes the prefix
//! transmittance for every voxel into its own disjoint output slice. Because
//! each thread owns a disjoint row there is no cross-thread contention, and the
//! multiply order per voxel matches the reference `..=end` fold bit-for-bit (a
//! `GPU` may still fuse a multiply-add and perturb the low mantissa bits, so
//! parity is asserted to a tolerance).
//!
//! Bit-faithfulness: each output voxel `v` holds `voxel_transmittance(row, v)`
//! — the reference's `product(1 - clamp(sigma_j, 0, 1))` over `0..=v` — so the
//! whole emitted curve equals the golden evaluated at every index. An empty
//! texel row (`count == 0`) emits nothing; the golden's `densities.is_empty()`
//! → `1.0` scalar case has no per-voxel slot to write and is exercised on the
//! host side of the parity test.

/// Decode parameters. `texel_count` bounds the dispatch (one invocation per
/// light texel). `16`-byte scalar-packed so the uniform block is a multiple of
/// 16.
struct HairVoxelTransmittanceParams {
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

@group(0) @binding(0) var<uniform> params: HairVoxelTransmittanceParams;

// Shared flat per-voxel optical density pool (texel-major); each entry is a
// `sigma_j` accumulated opacity, clamped to `0..=1` on read.
@group(0) @binding(1) var<storage, read> densities: array<f32>;

// Per-texel compacted ranges into `densities` (and, one-to-one, into the
// output `transmittance` pool).
@group(0) @binding(2) var<storage, read> texel_ranges: array<HairTexelRange>;

// Flat prefix-product transmittance pool, same layout as `densities`:
// `transmittance[start + v]` is the composite over voxels `0..=v`.
@group(0) @binding(3) var<storage, read_write> transmittance: array<f32>;

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if t >= params.texel_count {
        return;
    }

    let start = texel_ranges[t].start;
    let count = texel_ranges[t].count;

    // Running prefix product T = product(1 - clamp(sigma_j, 0, 1)) over 0..=v.
    var running = 1.0;
    for (var v = 0u; v < count; v = v + 1u) {
        let sigma = clamp(densities[start + v], 0.0, 1.0);
        running = running * (1.0 - sigma);
        transmittance[start + v] = running;
    }
}
