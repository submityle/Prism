//! `wgpu` compute twin of Prism's per-ray voxel opacity accumulation
//! (`prism_render_architecture::hair::deep_transmittance::accumulate_voxel_density`).
//!
//! The froxel self-shadow tier is the voxel counterpart of the layered deep
//! opacity map: instead of sorting per texel and slicing adaptive layers, it
//! bins strand opacity into a *uniform* slab of voxels along each light ray,
//! forming a per-voxel optical density `sigma` a shading pass composites into
//! transmittance `T = product(1 - sigma_j)` (design §6/§11 体素自阴影). Uniform
//! voxels need no depth sort, so this is the cheaper, coarser sibling of the
//! `deep_opacity` packing — the same self-shadow input, resolved on a fixed
//! grid.
//!
//! One invocation owns one light texel/ray: it walks that texel's compacted
//! sample slice (`texel_ranges` into the shared `samples` pool, in the host's
//! bin order — no sort, matching the golden's input-order scatter), skips
//! samples outside `slab_start..slab_end`, and scatter-adds each in-slab
//! sample's clamped opacity into the voxel its depth falls in. Because each
//! thread owns a disjoint density row there is no cross-thread contention, so
//! the accumulation order per voxel matches the reference bucket order
//! bit-for-bit (a `GPU` may still fuse a multiply-add and perturb the low
//! mantissa bits, so parity is asserted to a tolerance).
//!
//! Bit-faithfulness: the voxel index is spelled `u32((depth - slab_start) /
//! width)` clamped to the last voxel (matching the `as usize` truncation and
//! `.min(voxel_count - 1)` of the golden), the skip test is `depth <
//! slab_start` or `depth >= slab_end`, `width = (slab_end - slab_start) /
//! f32(voxel_count)`, `voxel_count` is clamped to at least `1`, and a
//! degenerate slab (`slab_end <= slab_start`) leaves every density at `0`.

/// Voxel accumulation parameters. `voxel_count` is the fixed slab resolution
/// (clamped to at least `1`); `texel_count` bounds the dispatch (one invocation
/// per light texel). `16`-byte scalar-packed so the uniform block is a multiple
/// of 16.
struct HairVoxelParams {
    slab_start: f32,
    slab_end: f32,
    voxel_count: u32,
    texel_count: u32,
}

/// One compacted per-texel slice into `samples`: entries `start .. start +
/// count`, in the host's bin order (no sort needed for uniform voxels).
struct HairTexelRange {
    start: u32,
    count: u32,
}

@group(0) @binding(0) var<uniform> params: HairVoxelParams;

// Shared flat sample buffer: `.x` is light-space depth (`>= 0`), `.y` is the
// `alpha` opacity contribution (`0..=1`).
@group(0) @binding(1) var<storage, read> samples: array<vec2<f32>>;

// Per-texel compacted ranges into `samples`.
@group(0) @binding(2) var<storage, read> texel_ranges: array<HairTexelRange>;

// Flat texel-major density slab: texel `t` voxel `v` at
// `densities[t * voxel_count + v]`.
@group(0) @binding(3) var<storage, read_write> densities: array<f32>;

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if t >= params.texel_count {
        return;
    }

    // `voxel_count` is clamped to at least `1`.
    var voxels = params.voxel_count;
    if voxels < 1u {
        voxels = 1u;
    }

    let base = t * voxels;

    // Zero this texel's density row first (the readback slab is not cleared).
    for (var v = 0u; v < voxels; v = v + 1u) {
        densities[base + v] = 0.0;
    }

    let span = params.slab_end - params.slab_start;
    if span <= 0.0 {
        // Degenerate slab: every sample is skipped, densities stay zero.
        return;
    }

    let width = span / f32(voxels);
    let start = texel_ranges[t].start;
    let count = texel_ranges[t].count;

    for (var i = 0u; i < count; i = i + 1u) {
        let s = samples[start + i];
        let depth = s.x;
        if depth < params.slab_start {
            continue;
        }
        if depth >= params.slab_end {
            continue;
        }
        let raw = u32((depth - params.slab_start) / width);
        var index = raw;
        if index > voxels - 1u {
            index = voxels - 1u;
        }
        densities[base + index] = densities[base + index] + clamp(s.y, 0.0, 1.0);
    }
}
