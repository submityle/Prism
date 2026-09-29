//! GPU strand self-shadow transmittance kernel (shared deep-opacity service).
//!
//! Compute twin of the voxel path of the CPU golden
//! `prism_render_architecture::hair::deep_transmittance`
//! (`accumulate_voxel_density` + `voxel_transmittance`). Hair self-shadowing is
//! dominated by strand-on-strand occlusion, so — like AMD `TressFX`'s
//! approximate self-shadow and UE5 Groom's deep opacity map — this service
//! accumulates strand opacity along each light ray/texel into a monotone
//! transmittance curve the shading side samples by receiver depth.
//!
//! The *voxel density* path is chosen for the GPU twin because it is sort-free
//! and exp-free: each light texel owns a fixed slab of voxels, strand opacity is
//! scattered into the voxel its light-space depth lands in to form an optical
//! density `sigma`, and the transmittance curve is the running alpha-composite
//! `T = product(1 - sigma_j)` — bit-for-bit the CPU golden's arithmetic, so the
//! CPU golden tests validate this kernel's math. (The depth-sorted deep-opacity
//! layer layout stays a CPU/host build step; sorting thousands of per-texel
//! samples on-device is a separate scheduling concern.)
//!
//! One invocation owns one light texel/ray: it reads that texel's contiguous
//! slice of strand samples via `ranges[texel]` (offset + count into the flat
//! `samples` buffer) and writes `voxel_count` transmittance values into
//! `out_transmittance` at stride `voxel_count`.

/// Maximum voxels per light ray. Bounds the per-invocation scratch slab so the
/// kernel needs no dynamic allocation; callers clamp `params.voxel_count` to
/// this. 64 layers is well beyond the 8..16 a deep opacity map typically uses.
const MAX_HAIR_VOXELS: u32 = 64u;

/// Flat range of one light texel's strand samples in the shared `samples`
/// buffer. Mirrors the per-bucket slice `TransmittanceBins` hands to the CPU
/// voxel accumulator.
struct HairTexelRange {
    offset: u32,
    count: u32,
}

/// Global transmittance parameters. `slab_start..slab_end` is the light-space
/// depth range sliced into `voxel_count` equal voxels (mirrors the CPU
/// `accumulate_voxel_density` slab), and `texel_count` bounds the dispatch.
struct HairTransmittanceParams {
    slab_start: f32,
    slab_end: f32,
    voxel_count: u32,
    texel_count: u32,
}

// Per-texel slice descriptor into `samples`.
@group(0) @binding(0) var<storage, read> ranges: array<HairTexelRange>;
// Flat strand samples: `x` = light-space depth (>= 0), `y` = opacity (0..=1).
@group(0) @binding(1) var<storage, read> samples: array<vec2<f32>>;
// Output transmittance curve, `texel * voxel_count + voxel` layout.
@group(0) @binding(2) var<storage, read_write> out_transmittance: array<f32>;

var<immediate> params: HairTransmittanceParams;

@compute @workgroup_size(64)
fn hair_transmittance_voxel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let texel = gid.x;
    if texel >= params.texel_count {
        return;
    }
    let voxel_count = clamp(params.voxel_count, 1u, MAX_HAIR_VOXELS);

    // Per-invocation optical-density scratch, mirroring the CPU `densities`
    // vector (zero-initialized, one entry per voxel).
    var densities: array<f32, MAX_HAIR_VOXELS>;
    for (var j = 0u; j < voxel_count; j = j + 1u) {
        densities[j] = 0.0;
    }

    let span = params.slab_end - params.slab_start;
    let base = texel * voxel_count;

    // Degenerate slab: every sample is skipped, transmittance is fully lit.
    if span <= 0.0 {
        for (var j = 0u; j < voxel_count; j = j + 1u) {
            out_transmittance[base + j] = 1.0;
        }
        return;
    }

    let width = span / f32(voxel_count);
    let range = ranges[texel];
    // Scatter each in-slab sample's opacity into its voxel (matches CPU
    // `accumulate_voxel_density`: `< slab_start` or `>= slab_end` skipped, the
    // final voxel clamps the tail).
    for (var i = 0u; i < range.count; i = i + 1u) {
        let s = samples[range.offset + i];
        let depth = s.x;
        if depth < params.slab_start || depth >= params.slab_end {
            continue;
        }
        let raw = u32((depth - params.slab_start) / width);
        let index = min(raw, voxel_count - 1u);
        densities[index] = densities[index] + clamp(s.y, 0.0, 1.0);
    }

    // Running alpha-composite `T = product(1 - sigma_j)` — the CPU
    // `voxel_transmittance` curve, monotonically non-increasing in voxel index.
    var transmittance = 1.0;
    for (var j = 0u; j < voxel_count; j = j + 1u) {
        transmittance = transmittance * (1.0 - clamp(densities[j], 0.0, 1.0));
        out_transmittance[base + j] = transmittance;
    }
}
