// water_spectrum_assemble.wesl
//
// Ocean spectral cascade assembly compute kernel (entry
// `water_spectrum_assemble`). The final stage of the Tessendorf ocean pipeline:
// after each spectral cascade has been inverse-FFT'd into its own spatial
// displacement grid, this kernel sums the (up to four) weighted cascade grids
// into one combined world-space displacement texture and derives the matching
// surface normal from the assembled height field.
//
// Multi-scale cascades (large swell down to capillary ripples) tile at
// different spatial frequencies; stacking them breaks the visible tiling of any
// single grid (WaveWorks / Crest / UE5 projected-grid oceans). Every cascade
// here shares the output grid resolution N, so output texel (gx, gy) sums the
// same (gx, gy) texel of each cascade scaled by its per-cascade distance weight:
//
//   disp = sum_c weight_c * cascade_c[idx]              (xyz displacement, w = foam/jacobian)
//
// The surface normal comes from central differences of the assembled height
// (the y channel) over the tile, with wrap-around neighbours (the ocean tile is
// periodic) and world cell size dx = patch_size / N:
//
//   dH/dx = (h(x+1) - h(x-1)) * (0.5 / dx)
//   dH/dz = (h(z+1) - h(z-1)) * (0.5 / dx)
//   normal = normalize((-dH/dx, 1, -dH/dz))             (fallback (0,1,0) if degenerate)
//
// Deterministic classical numerics; no AI/ML. Only sqrt (via normalize) is used.
//
// This is the standalone contract twin of
// `prism_render_architecture::water::gpu::spectrum_assemble_kernel` (the
// bit-exact CPU reference). The entry-point name is byte-identical to the
// `WaterKernel::SpectrumAssemble::wesl_entry_point` contract, and the ABI
// matches its descriptor: four storage buffers (the cascade grids), one uniform
// param block, two `rgba32float` storage-texture outputs (displacement,
// normal), an 8x8 tile, and the `Grid2d` dispatch domain. The sandbox has no
// GPU, so `prism_render_architecture` (a zero-dependency crate) ships this
// source and covers it with the structural ABI test; the standalone naga/wesl
// compile check runs out of tree, and the CPU twin is the texel-for-texel proof.

// Squared-length floor below which the assembled slope vector is treated as
// degenerate and the normal falls back to the upright (0,1,0); mirrors the CPU
// `prism_render_architecture::water::EPS_LEN_SQ`.
const WATER_EPS_LEN_SQ: f32 = 1.0e-12;

// One assembled cascade texel: xyz displacement plus a foam/Jacobian lane that
// is summed through to the displacement output's w channel.
struct CascadeTexel {
    disp_x: f32,
    disp_y: f32,
    disp_z: f32,
    foam: f32,
}

// Per-dispatch cascade-assembly scalars.
struct AssembleParams {
    // Grid resolution N; cascade grids and output textures are N*N texels.
    grid_size: u32,
    // Number of active cascades to sum (1..=4).
    cascade_count: u32,
    // Physical patch size L (m) of the output tile; the finite-difference world
    // cell size is dx = patch_size / N.
    patch_size: f32,
    // Per-cascade distance weights (index 0 = finest cascade).
    weight0: f32,
    weight1: f32,
    weight2: f32,
    weight3: f32,
    // Padding lane for a 16-byte-aligned uniform tail; ignored.
    pad: f32,
}

@group(0) @binding(0) var<uniform> params: AssembleParams;
@group(0) @binding(1) var<storage, read> cascade0: array<CascadeTexel>;
@group(0) @binding(2) var<storage, read> cascade1: array<CascadeTexel>;
@group(0) @binding(3) var<storage, read> cascade2: array<CascadeTexel>;
@group(0) @binding(4) var<storage, read> cascade3: array<CascadeTexel>;
@group(0) @binding(5) var displacement_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(6) var normal_out: texture_storage_2d<rgba32float, write>;

// Per-cascade weight by index (0..3); out-of-range yields 0.
fn cascade_weight(c: u32) -> f32 {
    if c == 0u {
        return params.weight0;
    }
    if c == 1u {
        return params.weight1;
    }
    if c == 2u {
        return params.weight2;
    }
    if c == 3u {
        return params.weight3;
    }
    return 0.0;
}

// Reads cascade `c`'s texel at linear index `idx` (xyz displacement + foam).
fn cascade_texel(c: u32, idx: u32) -> CascadeTexel {
    if c == 0u {
        return cascade0[idx];
    }
    if c == 1u {
        return cascade1[idx];
    }
    if c == 2u {
        return cascade2[idx];
    }
    return cascade3[idx];
}

// Weighted sum of every active cascade's displacement at linear index `idx`.
fn assembled_disp(idx: u32) -> vec4<f32> {
    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    let count = min(params.cascade_count, 4u);
    for (var c: u32 = 0u; c < count; c = c + 1u) {
        let w = cascade_weight(c);
        let t = cascade_texel(c, idx);
        acc.x = acc.x + w * t.disp_x;
        acc.y = acc.y + w * t.disp_y;
        acc.z = acc.z + w * t.disp_z;
        acc.w = acc.w + w * t.foam;
    }
    return acc;
}

// Assembled height (summed y displacement) at wrapped texel (gx, gy).
fn assembled_height(gx: u32, gy: u32, n: u32) -> f32 {
    return assembled_disp(gy * n + gx).y;
}

@compute @workgroup_size(8, 8, 1)
fn water_spectrum_assemble(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.grid_size;
    if gid.x >= n || gid.y >= n {
        return;
    }

    let idx = gid.y * n + gid.x;
    let disp = assembled_disp(idx);

    // Central-difference surface normal over the periodic ocean tile.
    let xm1 = (gid.x + n - 1u) % n;
    let xp1 = (gid.x + 1u) % n;
    let ym1 = (gid.y + n - 1u) % n;
    let yp1 = (gid.y + 1u) % n;

    var normal = vec3<f32>(0.0, 1.0, 0.0);
    let dx = params.patch_size / max(f32(n), 1.0);
    if dx > 0.0 {
        let inv_2dx = 0.5 / dx;
        let dhdx = (assembled_height(xp1, gid.y, n) - assembled_height(xm1, gid.y, n)) * inv_2dx;
        let dhdz = (assembled_height(gid.x, yp1, n) - assembled_height(gid.x, ym1, n)) * inv_2dx;
        let slope = vec3<f32>(-dhdx, 1.0, -dhdz);
        if dot(slope, slope) > WATER_EPS_LEN_SQ {
            normal = normalize(slope);
        }
    }

    let texel = vec2<i32>(i32(gid.x), i32(gid.y));
    textureStore(displacement_out, texel, disp);
    textureStore(normal_out, texel, vec4<f32>(normal, 0.0));
}
