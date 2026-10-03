//! Ocean spectral cascade-assembly compute kernel: the `WESL` shader plus its
//! bit-exact `CPU` twin.
//!
//! This is the final stage of the `Tessendorf` ocean pipeline. After each
//! spectral cascade has been inverse-`FFT`'d into its own spatial displacement
//! grid, this kernel sums the (up to four) weighted cascade grids into one
//! combined world-space displacement texture and derives the surface normal
//! from the assembled height field.
//!
//! Multi-scale cascades (large swell down to capillary ripples) tile at
//! different spatial frequencies; stacking them breaks the visible tiling of
//! any single grid (the `WaveWorks` / `Crest` / `UE5` projected-grid ocean
//! trick). Every cascade shares the output resolution `N`, so output texel
//! `(gx, gy)` sums the same `(gx, gy)` texel of each cascade scaled by its
//! per-cascade distance weight:
//!   `disp = sum_c weight_c * cascade_c[idx]`  (xyz displacement, w = foam/jacobian).
//! The normal comes from central differences of the assembled height (the `y`
//! channel) over the periodic tile with world cell size `dx = patch_size / N`:
//!   `dH/dx = (h(x+1) - h(x-1)) * (0.5 / dx)`, likewise `dH/dz`,
//!   `normal = normalize((-dH/dx, 1, -dH/dz))`  (fallback `(0,1,0)`).
//!
//! [`WATER_SPECTRUM_ASSEMBLE_WESL`] is the shader (entry point
//! `water_spectrum_assemble`); [`dispatch_spectrum_assemble`] is its bit-exact
//! `CPU` twin. Because the sandbox has no `GPU`, the twin is the correctness
//! proof: it consumes the identical buffer `ABI`
//! ([`WaterKernel::SpectrumAssemble`](super::super::kernels::WaterKernel) — four
//! storage buffers of cascade texels, one uniform param block, two
//! `rgba32float` storage-texture outputs, 8x8 tile, `Grid2d` domain) and
//! reproduces the shader texel-for-texel, mirroring the per-cascade float order
//! (weight first) and the central-difference normal. Pure classical numerics —
//! no AI/ML; only `sqrt` (via the normal's normalize) is used.

use alloc::vec;
use alloc::vec::Vec;

use super::super::{Vec3, EPS_LEN_SQ};

/// `WESL` source of the ocean spectral cascade-assembly compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// test; the standalone `naga`/`wesl` compile check runs out of tree, because
/// `prism_render_architecture` is a zero-dependency crate.
pub const WATER_SPECTRUM_ASSEMBLE_WESL: &str = include_str!("water_spectrum_assemble.wesl");

/// Number of `f32` lanes per cascade texel (`disp_x, disp_y, disp_z, foam`).
pub const CASCADE_TEXEL_FLOATS: usize = 4;

/// Number of `f32` lanes per output texel (`xyz` + one summed foam/padding lane).
pub const ASSEMBLE_OUT_FLOATS: usize = 4;

/// Maximum number of cascades the kernel sums.
pub const ASSEMBLE_MAX_CASCADES: usize = 4;

/// Per-dispatch cascade-assembly scalars, mirroring the shader `AssembleParams`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AssembleParams {
    /// Grid resolution `N`; cascade grids and output textures are `N*N` texels.
    pub grid_size: u32,
    /// Number of active cascades to sum (clamped to `1..=4`).
    pub cascade_count: u32,
    /// Physical patch size `L` (m) of the output tile; the finite-difference
    /// world cell size is `dx = patch_size / N`.
    pub patch_size: f32,
    /// Per-cascade distance weights (index `0` is the finest cascade).
    pub weights: [f32; ASSEMBLE_MAX_CASCADES],
}

/// Output of one assembly dispatch: the combined per-texel displacement field
/// and the derived normals, each `grid_size * grid_size` texels of
/// [`ASSEMBLE_OUT_FLOATS`] lanes in row-major `(y, x)` order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AssembleFields {
    /// Displacement texels `(sum dx, sum dy, sum dz, sum foam)`.
    pub displacement: Vec<f32>,
    /// Derived normal texels `(n.x, n.y, n.z, 0)`.
    pub normal: Vec<f32>,
}

/// Weighted sum of every active cascade's `(dx, dy, dz, foam)` texel at linear
/// index `idx`. A cascade slice too short to hold `idx` contributes zero (the
/// shader's `array` bound guard), so a short buffer degrades gracefully.
#[inline]
fn assembled_disp(
    cascades: &[&[f32]],
    weights: &[f32; ASSEMBLE_MAX_CASCADES],
    count: usize,
    idx: usize,
) -> [f32; CASCADE_TEXEL_FLOATS] {
    let mut acc = [0.0f32; CASCADE_TEXEL_FLOATS];
    let base = idx * CASCADE_TEXEL_FLOATS;
    for c in 0..count {
        let w = weights[c];
        let grid = cascades[c];
        if base + CASCADE_TEXEL_FLOATS > grid.len() {
            continue;
        }
        acc[0] += w * grid[base];
        acc[1] += w * grid[base + 1];
        acc[2] += w * grid[base + 2];
        acc[3] += w * grid[base + 3];
    }
    acc
}

/// Assembled height (summed `y` displacement) at wrapped texel `(gx, gy)`.
#[inline]
fn assembled_height(
    cascades: &[&[f32]],
    weights: &[f32; ASSEMBLE_MAX_CASCADES],
    count: usize,
    gx: usize,
    gy: usize,
    n: usize,
) -> f32 {
    assembled_disp(cascades, weights, count, gy * n + gx)[1]
}

/// Bit-exact `CPU` twin of the `water_spectrum_assemble` shader.
///
/// `cascades` holds up to [`ASSEMBLE_MAX_CASCADES`] cascade grids, each a packed
/// `N*N` slice of [`CASCADE_TEXEL_FLOATS`]-lane texels; only the first
/// `min(params.cascade_count, cascades.len(), 4)` are summed. Returns the
/// combined displacement and derived normal fields; an empty grid
/// (`grid_size == 0`) returns empty fields.
#[must_use]
pub fn dispatch_spectrum_assemble(cascades: &[&[f32]], params: AssembleParams) -> AssembleFields {
    let n = params.grid_size as usize;
    let texels = n * n;
    let mut displacement = vec![0.0f32; texels * ASSEMBLE_OUT_FLOATS];
    let mut normal = vec![0.0f32; texels * ASSEMBLE_OUT_FLOATS];
    if n == 0 {
        return AssembleFields {
            displacement,
            normal,
        };
    }

    let count = (params.cascade_count as usize)
        .min(ASSEMBLE_MAX_CASCADES)
        .min(cascades.len());
    let weights = &params.weights;
    let dx = params.patch_size / (n as f32).max(1.0);

    for gy in 0..n {
        let ym1 = (gy + n - 1) % n;
        let yp1 = (gy + 1) % n;
        for gx in 0..n {
            let xm1 = (gx + n - 1) % n;
            let xp1 = (gx + 1) % n;

            let idx = gy * n + gx;
            let disp = assembled_disp(cascades, weights, count, idx);

            let mut nrm = Vec3::new(0.0, 1.0, 0.0);
            if dx > 0.0 {
                let inv_2dx = 0.5 / dx;
                let dhdx = (assembled_height(cascades, weights, count, xp1, gy, n)
                    - assembled_height(cascades, weights, count, xm1, gy, n))
                    * inv_2dx;
                let dhdz = (assembled_height(cascades, weights, count, gx, yp1, n)
                    - assembled_height(cascades, weights, count, gx, ym1, n))
                    * inv_2dx;
                let slope = Vec3::new(-dhdx, 1.0, -dhdz);
                let len_sq = slope.dot(slope);
                if len_sq > EPS_LEN_SQ {
                    nrm = slope.scale(1.0 / len_sq.sqrt());
                }
            }

            let out = idx * ASSEMBLE_OUT_FLOATS;
            displacement[out] = disp[0];
            displacement[out + 1] = disp[1];
            displacement[out + 2] = disp[2];
            displacement[out + 3] = disp[3];
            normal[out] = nrm.x;
            normal[out + 1] = nrm.y;
            normal[out + 2] = nrm.z;
            normal[out + 3] = 0.0;
        }
    }

    AssembleFields {
        displacement,
        normal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds an `N*N` cascade grid of `(dx, dy, dz, foam)` texels from a
    /// per-texel closure.
    fn grid<F: Fn(usize, usize) -> [f32; 4]>(n: usize, f: F) -> Vec<f32> {
        let mut out = Vec::with_capacity(n * n * CASCADE_TEXEL_FLOATS);
        for gy in 0..n {
            for gx in 0..n {
                out.extend_from_slice(&f(gx, gy));
            }
        }
        out
    }

    /// The shipped `WESL` honors the `SpectrumAssemble` descriptor's `ABI`:
    /// entry-point name, 8x8 tile, the four cascade storage buffers, the uniform
    /// block, and the two `rgba32float` storage-texture outputs.
    #[test]
    fn wesl_matches_descriptor_abi() {
        let src = WATER_SPECTRUM_ASSEMBLE_WESL;
        assert!(src.contains("fn water_spectrum_assemble("));
        assert!(src.contains("@workgroup_size(8, 8, 1)"));
        assert!(src.contains("var<uniform> params: AssembleParams"));
        assert_eq!(
            src.matches("var<storage, read> cascade").count(),
            ASSEMBLE_MAX_CASCADES
        );
        assert_eq!(
            src.matches("texture_storage_2d<rgba32float, write>")
                .count(),
            2
        );

        use super::super::super::kernels::{DispatchDomain, WaterKernel};
        let desc = WaterKernel::SpectrumAssemble.descriptor();
        assert_eq!(desc.layout.storage_buffers, 4);
        assert_eq!(desc.layout.uniform_buffers, 1);
        assert_eq!(desc.layout.storage_textures, 2);
        assert_eq!(desc.layout.sampled_textures, 0);
        assert_eq!(desc.workgroup.x, 8);
        assert_eq!(desc.workgroup.y, 8);
        assert_eq!(desc.workgroup.z, 1);
        assert_eq!(desc.domain, DispatchDomain::Grid2d);
        assert_eq!(
            WaterKernel::SpectrumAssemble.wesl_entry_point(),
            "water_spectrum_assemble"
        );
    }

    /// A single flat cascade (constant height, zero horizontal displacement)
    /// scaled by its weight yields a flat displacement and the upright normal.
    #[test]
    fn flat_cascade_is_scaled_and_upright() {
        let n = 4;
        let c0 = grid(n, |_, _| [0.0, 2.0, 0.0, 0.5]);
        let params = AssembleParams {
            grid_size: n as u32,
            cascade_count: 1,
            patch_size: 8.0,
            weights: [0.5, 0.0, 0.0, 0.0],
        };
        let fields = dispatch_spectrum_assemble(&[&c0], params);
        for t in 0..n * n {
            let i = t * ASSEMBLE_OUT_FLOATS;
            assert_eq!(fields.displacement[i].to_bits(), 0.0f32.to_bits());
            assert_eq!(fields.displacement[i + 1].to_bits(), 1.0f32.to_bits());
            assert_eq!(fields.displacement[i + 2].to_bits(), 0.0f32.to_bits());
            assert_eq!(fields.displacement[i + 3].to_bits(), 0.25f32.to_bits());
            // Flat height -> zero slope. The kernel forms (-dhdx, 1, -dhdz),
            // so negating the +0.0 derivatives yields signed -0.0 on x/z.
            assert_eq!(fields.normal[i].to_bits(), (-(0.0f32)).to_bits());
            assert_eq!(fields.normal[i + 1].to_bits(), 1.0f32.to_bits());
            assert_eq!(fields.normal[i + 2].to_bits(), (-(0.0f32)).to_bits());
        }
    }

    /// Four cascades sum lane-for-lane with their weights; the displacement is
    /// the exact weighted sum of every cascade's texel.
    #[test]
    fn four_cascades_sum_with_weights() {
        let n = 2;
        let c0 = grid(n, |_, _| [1.0, 0.0, 0.0, 1.0]);
        let c1 = grid(n, |_, _| [0.0, 1.0, 0.0, 2.0]);
        let c2 = grid(n, |_, _| [0.0, 0.0, 1.0, 3.0]);
        let c3 = grid(n, |_, _| [1.0, 1.0, 1.0, 4.0]);
        let w = [0.5f32, 0.25, 0.125, 0.0625];
        let params = AssembleParams {
            grid_size: n as u32,
            cascade_count: 4,
            patch_size: 4.0,
            weights: w,
        };
        let fields = dispatch_spectrum_assemble(&[&c0, &c1, &c2, &c3], params);
        let ex = w[0] * 1.0 + w[3] * 1.0;
        let ey = w[1] * 1.0 + w[3] * 1.0;
        let ez = w[2] * 1.0 + w[3] * 1.0;
        let ef = w[0] * 1.0 + w[1] * 2.0 + w[2] * 3.0 + w[3] * 4.0;
        for t in 0..n * n {
            let i = t * ASSEMBLE_OUT_FLOATS;
            assert_eq!(fields.displacement[i].to_bits(), ex.to_bits());
            assert_eq!(fields.displacement[i + 1].to_bits(), ey.to_bits());
            assert_eq!(fields.displacement[i + 2].to_bits(), ez.to_bits());
            assert_eq!(fields.displacement[i + 3].to_bits(), ef.to_bits());
        }
    }

    /// A non-flat height field drives the central-difference normal: for a tile
    /// whose height ramps with `gx`, the normal tilts in `-x` and is unit length.
    #[test]
    fn height_gradient_tilts_normal() {
        let n = 4;
        // Height = gx (0,1,2,3); horizontal displacement zero.
        let c0 = grid(n, |gx, _| [0.0, gx as f32, 0.0, 0.0]);
        let params = AssembleParams {
            grid_size: n as u32,
            cascade_count: 1,
            patch_size: 4.0, // dx = 1.0
            weights: [1.0, 0.0, 0.0, 0.0],
        };
        let fields = dispatch_spectrum_assemble(&[&c0], params);
        // Interior texel gx == 1: central diff over wrapped neighbours gx=2,gx=0.
        // Row-major texel (gx = 1, gy = 0) -> flat index 1.
        let texel = 1usize;
        let i = texel * ASSEMBLE_OUT_FLOATS;
        // dH/dx = (2 - 0) * 0.5 = 1.0 ; dH/dz = 0. slope = (-1, 1, 0).
        let slope = Vec3::new(-1.0, 1.0, -(0.0));
        let expect = slope.scale(1.0 / slope.dot(slope).sqrt());
        assert_eq!(fields.normal[i].to_bits(), expect.x.to_bits());
        assert_eq!(fields.normal[i + 1].to_bits(), expect.y.to_bits());
        assert_eq!(fields.normal[i + 2].to_bits(), expect.z.to_bits());
        // Normals are unit length (within f32).
        let len_sq = expect.dot(expect);
        assert!((len_sq - 1.0).abs() < 1e-6);
    }

    /// Independent inline recomputation of the full field, lane-for-lane, binds
    /// the twin to the documented algorithm (anti-vacuous: at least one normal
    /// must differ from the upright `(0,1,0)`).
    #[test]
    fn matches_independent_inline_recompute() {
        let n = 5;
        let c0 = grid(n, |gx, gy| {
            [
                0.1 * gx as f32,
                (gx as f32) * 0.3 - (gy as f32) * 0.2,
                0.05 * gy as f32,
                0.2,
            ]
        });
        let c1 = grid(n, |gx, gy| {
            [
                -0.2 * gy as f32,
                0.15 * (gx as f32 + gy as f32),
                0.1 * gx as f32,
                0.4,
            ]
        });
        let w = [0.75f32, 0.5, 0.0, 0.0];
        let params = AssembleParams {
            grid_size: n as u32,
            cascade_count: 2,
            patch_size: 10.0,
            weights: w,
        };
        let fields = dispatch_spectrum_assemble(&[&c0, &c1], params);

        let grids = [&c0, &c1];
        let dx = params.patch_size / (n as f32).max(1.0);
        let inv_2dx = 0.5 / dx;
        let h = |gx: usize, gy: usize| -> f32 {
            let base = (gy * n + gx) * CASCADE_TEXEL_FLOATS;
            w[0] * grids[0][base + 1] + w[1] * grids[1][base + 1]
        };
        let mut tilted = 0usize;
        for gy in 0..n {
            for gx in 0..n {
                let base = (gy * n + gx) * CASCADE_TEXEL_FLOATS;
                let ex = w[0] * c0[base] + w[1] * c1[base];
                let ey = w[0] * c0[base + 1] + w[1] * c1[base + 1];
                let ez = w[0] * c0[base + 2] + w[1] * c1[base + 2];
                let ef = w[0] * c0[base + 3] + w[1] * c1[base + 3];
                let xm1 = (gx + n - 1) % n;
                let xp1 = (gx + 1) % n;
                let ym1 = (gy + n - 1) % n;
                let yp1 = (gy + 1) % n;
                let dhdx = (h(xp1, gy) - h(xm1, gy)) * inv_2dx;
                let dhdz = (h(gx, yp1) - h(gx, ym1)) * inv_2dx;
                let sx = -dhdx;
                let sz = -dhdz;
                let len_sq = sx * sx + 1.0 + sz * sz;
                let (nx, ny, nz) = if len_sq > EPS_LEN_SQ {
                    let r = 1.0 / len_sq.sqrt();
                    (sx * r, r, sz * r)
                } else {
                    (0.0, 1.0, 0.0)
                };
                let o = (gy * n + gx) * ASSEMBLE_OUT_FLOATS;
                assert_eq!(fields.displacement[o].to_bits(), ex.to_bits());
                assert_eq!(fields.displacement[o + 1].to_bits(), ey.to_bits());
                assert_eq!(fields.displacement[o + 2].to_bits(), ez.to_bits());
                assert_eq!(fields.displacement[o + 3].to_bits(), ef.to_bits());
                assert_eq!(fields.normal[o].to_bits(), nx.to_bits());
                assert_eq!(fields.normal[o + 1].to_bits(), ny.to_bits());
                assert_eq!(fields.normal[o + 2].to_bits(), nz.to_bits());
                if nx.to_bits() != 0.0f32.to_bits() || nz.to_bits() != 0.0f32.to_bits() {
                    tilted += 1;
                }
            }
        }
        assert!(tilted > 0, "normals must respond to the height gradient");
    }

    /// `cascade_count` beyond the supplied slices or beyond four is clamped, and
    /// a short cascade slice contributes zero instead of panicking.
    #[test]
    fn over_count_and_short_slice_do_not_panic() {
        let n = 2;
        let full = grid(n, |_, _| [1.0, 1.0, 1.0, 1.0]);
        let short: Vec<f32> = alloc::vec![1.0, 1.0]; // too short for any texel
        let params = AssembleParams {
            grid_size: n as u32,
            cascade_count: 9, // clamped to the two supplied slices, then to 4
            patch_size: 4.0,
            weights: [1.0, 1.0, 0.0, 0.0],
        };
        let fields = dispatch_spectrum_assemble(&[&full, &short], params);
        // Only `full` contributes (short slice is skipped per-texel).
        for t in 0..n * n {
            let i = t * ASSEMBLE_OUT_FLOATS;
            assert_eq!(fields.displacement[i + 1].to_bits(), 1.0f32.to_bits());
        }

        // grid_size == 0 returns empty fields.
        let empty = dispatch_spectrum_assemble(
            &[&full],
            AssembleParams {
                grid_size: 0,
                ..params
            },
        );
        assert!(empty.displacement.is_empty());
        assert!(empty.normal.is_empty());
    }

    /// The dispatch is deterministic: identical inputs produce bit-identical
    /// fields on repeated calls.
    #[test]
    fn deterministic_across_calls() {
        let n = 3;
        let c0 = grid(n, |gx, gy| [0.1 * gx as f32, 0.2 * gy as f32, 0.3, 0.1]);
        let c1 = grid(n, |gx, gy| [0.0, 0.1 * (gx + gy) as f32, 0.0, 0.2]);
        let params = AssembleParams {
            grid_size: n as u32,
            cascade_count: 2,
            patch_size: 6.0,
            weights: [1.0, 0.5, 0.0, 0.0],
        };
        let a = dispatch_spectrum_assemble(&[&c0, &c1], params);
        let b = dispatch_spectrum_assemble(&[&c0, &c1], params);
        assert_eq!(a, b);
    }
}
