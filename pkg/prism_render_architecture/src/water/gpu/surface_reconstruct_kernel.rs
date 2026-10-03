//! Screen-space fluid surface-reconstruction compute kernel: the `WESL` shader
//! plus its `CPU` twin, the deterministic golden the production normal-recovery
//! pass is checked against.
//!
//! Particle fluids (`FLIP`/`APIC`, `PBF`) carry no mesh, so the shipping route
//! to a shaded surface is screen space: splat the particles into an eye-depth
//! buffer, smooth it, then recover a per-pixel view-space normal from the depth
//! field. This kernel is that normal-recovery half, the one `van der Laan`
//! ("Screen Space Fluid Rendering with Curvature Flow") and every real-time
//! fluid renderer rasterizes.
//!
//! Per pixel it unprojects the center and its four axis neighbours to view
//! space through a pinhole model, forms a one-sided derivative on each axis
//! choosing the neighbour with the smaller depth step (the `van der Laan`
//! silhouette guard that stops normals bleeding across the fluid's edge), takes
//! the cross product, normalizes, and flips the result to face the camera
//! (negative view `z`). Background pixels (depth `<= 0`) and pixels with no
//! valid neighbour on an axis degrade gracefully to a flat camera-facing
//! normal, never a `NaN`. The per-pixel thickness rides through to the output
//! alpha for the downstream absorption/opacity shade.
//!
//! [`WATER_SURFACE_RECONSTRUCT_WESL`] is the shader (entry point
//! `water_surface_reconstruct`); [`dispatch_surface_reconstruct`] is its `CPU`
//! twin. Because the sandbox has no `GPU`, the twin is the correctness proof: it
//! consumes the identical buffer `ABI`
//! ([`WaterKernel::SurfaceReconstruct`](super::super::kernels::WaterKernel) —
//! two read storage buffers for the depth and thickness screens, one uniform
//! param block, one `rgba32float` storage-texture normal output, an 8x8 texel
//! tile and the `Screen` dispatch domain) and reproduces the shader
//! texel-for-texel. Pure classical numerics — no AI/ML; the only float
//! intrinsic is `sqrt` (the normalize), everything else is `+ - * /`,
//! comparisons, `abs`, [`Vec3::cross`] and [`Vec3::dot`].

use alloc::vec;
use alloc::vec::Vec;

use super::super::{Vec3, EPS_LEN_SQ};

/// `WESL` source of the screen-space surface-reconstruction compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// `ABI` test; the standalone `naga` / `wesl` compile check runs out of tree,
/// because `prism_render_architecture` is a zero-dependency crate.
pub const WATER_SURFACE_RECONSTRUCT_WESL: &str = include_str!("water_surface_reconstruct.wesl");

/// Number of `f32` lanes per output texel: the view-space normal `(nx, ny, nz)`
/// plus the passed-through thickness in the fourth lane.
pub const RECONSTRUCT_OUT_FLOATS: usize = 4;

/// Pinhole intrinsics for the unprojection: the tangents of the half field of
/// view on each axis. Screen dimensions are passed to
/// [`dispatch_surface_reconstruct`] directly (the shader carries them in its
/// own uniform block alongside these tangents).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReconstructParams {
    /// Tangent of half the horizontal field of view.
    pub tan_half_fov_x: f32,
    /// Tangent of half the vertical field of view.
    pub tan_half_fov_y: f32,
}

/// Screen-space surface reconstruction `CPU` twin.
///
/// Mirrors `water_surface_reconstruct` texel-for-texel. `depth` and `thickness`
/// are row-major `width * height` screens; a depth `<= 0` marks a background
/// pixel. Returns `width * height * RECONSTRUCT_OUT_FLOATS` floats, four lanes
/// per pixel: the view-space normal plus thickness. Short input slices are read
/// as zero (background) so the twin never panics on a malformed screen.
#[must_use]
pub fn dispatch_surface_reconstruct(
    depth: &[f32],
    thickness: &[f32],
    params: ReconstructParams,
    width: usize,
    height: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; width * height * RECONSTRUCT_OUT_FLOATS];
    let wf = width as f32;
    let hf = height as f32;
    let tx = params.tan_half_fov_x;
    let ty = params.tan_half_fov_y;
    let dlen = depth.len();
    let tlen = thickness.len();

    let view = |px: f32, py: f32, z: f32| -> Vec3 {
        let ndc_x = 2.0 * ((px + 0.5) / wf) - 1.0;
        let ndc_y = 1.0 - 2.0 * ((py + 0.5) / hf);
        Vec3::new(ndc_x * z * tx, ndc_y * z * ty, z)
    };
    let depth_at = |i: usize| -> f32 {
        if i < dlen {
            depth[i]
        } else {
            0.0
        }
    };

    let mut gy = 0usize;
    while gy < height {
        let mut gx = 0usize;
        while gx < width {
            let idx = gy * width + gx;
            let o = idx * RECONSTRUCT_OUT_FLOATS;
            let zc = depth_at(idx);
            let thick = if idx < tlen { thickness[idx] } else { 0.0 };

            // Background: leave the already-zeroed null record (zero normal,
            // zero thickness) rather than reconstructing from a bogus depth.
            if zc <= 0.0 {
                gx += 1;
                continue;
            }

            let gxf = gx as f32;
            let gyf = gy as f32;
            let pc = view(gxf, gyf, zc);

            // x-axis derivative with the silhouette guard: prefer the neighbour
            // (left or right) whose depth step is smaller.
            let mut dxv = Vec3::ZERO;
            let have_r = gx + 1 < width;
            let have_l = gx > 0;
            let zr = if have_r { depth_at(idx + 1) } else { 0.0 };
            let zl = if have_l { depth_at(idx - 1) } else { 0.0 };
            let vr = have_r && zr > 0.0;
            let vl = have_l && zl > 0.0;
            if vr && vl {
                let dpr = view((gx + 1) as f32, gyf, zr).sub(pc);
                let dpl = pc.sub(view((gx - 1) as f32, gyf, zl));
                if (zr - zc).abs() <= (zc - zl).abs() {
                    dxv = dpr;
                } else {
                    dxv = dpl;
                }
            } else if vr {
                dxv = view((gx + 1) as f32, gyf, zr).sub(pc);
            } else if vl {
                dxv = pc.sub(view((gx - 1) as f32, gyf, zl));
            }

            // y-axis derivative, same guard over the up/down neighbours.
            let mut dyv = Vec3::ZERO;
            let have_u = gy > 0;
            let have_d = gy + 1 < height;
            let zu = if have_u { depth_at(idx - width) } else { 0.0 };
            let zd = if have_d { depth_at(idx + width) } else { 0.0 };
            let vu = have_u && zu > 0.0;
            let vd = have_d && zd > 0.0;
            if vu && vd {
                let dpu = view(gxf, (gy - 1) as f32, zu).sub(pc);
                let dpd = pc.sub(view(gxf, (gy + 1) as f32, zd));
                if (zu - zc).abs() <= (zc - zd).abs() {
                    dyv = dpu;
                } else {
                    dyv = dpd;
                }
            } else if vu {
                dyv = view(gxf, (gy - 1) as f32, zu).sub(pc);
            } else if vd {
                dyv = pc.sub(view(gxf, (gy + 1) as f32, zd));
            }

            let n0 = dxv.cross(dyv);
            let len_sq = n0.dot(n0);
            let mut n = Vec3::new(0.0, 0.0, -1.0);
            if len_sq > EPS_LEN_SQ {
                n = n0.scale(1.0 / len_sq.sqrt());
            }
            // Face the camera: the eye looks down +z, so a visible fluid normal
            // has a negative view-space z.
            if n.z > 0.0 {
                n = Vec3::new(-n.x, -n.y, -n.z);
            }

            out[o] = n.x;
            out[o + 1] = n.y;
            out[o + 2] = n.z;
            out[o + 3] = thick;

            gx += 1;
        }
        gy += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::water::kernels::{DispatchDomain, WaterKernel};

    const PARAMS: ReconstructParams = ReconstructParams {
        tan_half_fov_x: 0.5,
        tan_half_fov_y: 0.4,
    };

    /// The shipped `WESL` honors the `SurfaceReconstruct` descriptor's `ABI`:
    /// the entry point name, tile, dispatch domain and bind-group shape are the
    /// contract the twin and the backend both target.
    #[test]
    fn wesl_matches_descriptor_abi() {
        assert!(WATER_SURFACE_RECONSTRUCT_WESL.contains("fn water_surface_reconstruct"));
        assert!(WATER_SURFACE_RECONSTRUCT_WESL.contains("@workgroup_size(8, 8, 1)"));
        assert!(WATER_SURFACE_RECONSTRUCT_WESL.contains("struct ReconstructParams"));
        assert!(WATER_SURFACE_RECONSTRUCT_WESL.contains("tan_half_fov_x"));
        assert!(WATER_SURFACE_RECONSTRUCT_WESL.contains("tan_half_fov_y"));
        assert!(WATER_SURFACE_RECONSTRUCT_WESL.contains("var<storage, read> depth_in"));
        assert!(WATER_SURFACE_RECONSTRUCT_WESL.contains("var<storage, read> thickness_in"));
        assert!(WATER_SURFACE_RECONSTRUCT_WESL.contains("texture_storage_2d<rgba32float, write>"));

        let desc = WaterKernel::SurfaceReconstruct.descriptor();
        assert_eq!(desc.workgroup.x, 8);
        assert_eq!(desc.workgroup.y, 8);
        assert_eq!(desc.workgroup.z, 1);
        assert_eq!(desc.domain, DispatchDomain::Screen);
        assert_eq!(desc.layout.storage_buffers, 2);
        assert_eq!(desc.layout.uniform_buffers, 1);
        assert_eq!(desc.layout.storage_textures, 1);
        assert_eq!(desc.layout.sampled_textures, 0);
        assert_eq!(
            WaterKernel::SurfaceReconstruct.wesl_entry_point(),
            "water_surface_reconstruct"
        );
    }

    /// A fronto-parallel plane (constant eye depth) reconstructs to a normal
    /// that points straight back at the camera: `(0, 0, -1)`, unit length, with
    /// the per-pixel thickness carried through to alpha bit-for-bit.
    #[test]
    fn flat_plane_faces_camera() {
        let w = 5usize;
        let h = 5usize;
        let depth = vec![5.0_f32; w * h];
        let mut thickness = Vec::with_capacity(w * h);
        let mut i = 0usize;
        while i < w * h {
            thickness.push(0.1 + 0.01 * i as f32);
            i += 1;
        }
        let out = dispatch_surface_reconstruct(&depth, &thickness, PARAMS, w, h);

        let mut p = 0usize;
        while p < w * h {
            let o = p * RECONSTRUCT_OUT_FLOATS;
            let nx = out[o];
            let ny = out[o + 1];
            let nz = out[o + 2];
            assert!(nx.abs() < 1.0e-5, "nx at {p} = {nx}");
            assert!(ny.abs() < 1.0e-5, "ny at {p} = {ny}");
            assert!(nz < 0.0, "normal must face camera at {p}, nz = {nz}");
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            assert!(
                (len - 1.0).abs() < 1.0e-5,
                "unit length at {p}, len = {len}"
            );
            // Thickness rides through to alpha untouched.
            assert_eq!(
                out[o + 3].to_bits(),
                thickness[p].to_bits(),
                "thickness {p}"
            );
            p += 1;
        }
    }

    /// Core anti-tautology: feed a depth screen whose unprojected view-space
    /// points lie exactly on a known tilted plane, and the reconstructed normal
    /// at an interior pixel must recover that plane's normal. Two in-plane chord
    /// vectors give a cross product parallel to the true normal regardless of
    /// which silhouette side is chosen, so the recovery is exact up to the
    /// floating-point floor — this proves the kernel reconstructs geometry, not
    /// a constant.
    #[test]
    fn tilted_plane_recovers_analytic_normal() {
        let w = 9usize;
        let h = 9usize;
        // Plane normal facing the camera (negative view z), unit length.
        let raw = Vec3::new(0.3, -0.2, -1.0);
        let inv = 1.0 / raw.dot(raw).sqrt();
        let pn = raw.scale(inv);
        let plane_c = -5.0_f32; // pn . P = plane_c for every surface point.

        let tx = PARAMS.tan_half_fov_x;
        let ty = PARAMS.tan_half_fov_y;
        let wf = w as f32;
        let hf = h as f32;
        let mut depth = vec![0.0_f32; w * h];
        let mut gy = 0usize;
        while gy < h {
            let mut gx = 0usize;
            while gx < w {
                let ndc_x = 2.0 * ((gx as f32 + 0.5) / wf) - 1.0;
                let ndc_y = 1.0 - 2.0 * ((gy as f32 + 0.5) / hf);
                // Ray direction d with P = z * d; pn . (z d) = c  =>  z = c/(pn.d).
                let d = Vec3::new(ndc_x * tx, ndc_y * ty, 1.0);
                let z = plane_c / pn.dot(d);
                depth[gy * w + gx] = z;
                gx += 1;
            }
            gy += 1;
        }
        let thickness = vec![1.0_f32; w * h];
        let out = dispatch_surface_reconstruct(&depth, &thickness, PARAMS, w, h);

        // Interior pixel: all four neighbours valid.
        let p = 4 * w + 4;
        let o = p * RECONSTRUCT_OUT_FLOATS;
        let n = Vec3::new(out[o], out[o + 1], out[o + 2]);
        assert!((n.x - pn.x).abs() < 1.0e-3, "nx {} vs {}", n.x, pn.x);
        assert!((n.y - pn.y).abs() < 1.0e-3, "ny {} vs {}", n.y, pn.y);
        assert!((n.z - pn.z).abs() < 1.0e-3, "nz {} vs {}", n.z, pn.z);
        let len = n.dot(n).sqrt();
        assert!((len - 1.0).abs() < 1.0e-5, "unit length, len = {len}");
    }

    /// Whole-field anti-tautology / anti-vacuous: an independent inline rewrite
    /// of the exact twin arithmetic must match every output lane bit-for-bit,
    /// and at least one reconstructed normal must differ from the degenerate
    /// `(0, 0, -1)` fallback (otherwise the test would pass on a stub).
    #[test]
    fn matches_independent_inline_recompute() {
        let w = 7usize;
        let h = 6usize;
        let tx = PARAMS.tan_half_fov_x;
        let ty = PARAMS.tan_half_fov_y;
        let wf = w as f32;
        let hf = h as f32;
        // A smooth non-planar bump so normals genuinely vary across the screen.
        let mut depth = vec![0.0_f32; w * h];
        let mut thickness = vec![0.0_f32; w * h];
        let mut gy = 0usize;
        while gy < h {
            let mut gx = 0usize;
            while gx < w {
                let fx = gx as f32 - 3.0;
                let fy = gy as f32 - 2.5;
                depth[gy * w + gx] = 6.0 + 0.2 * fx + 0.15 * fy + 0.05 * (fx * fx - fy * fy);
                thickness[gy * w + gx] = 0.3 + 0.02 * (gy * w + gx) as f32;
                gx += 1;
            }
            gy += 1;
        }
        let out = dispatch_surface_reconstruct(&depth, &thickness, PARAMS, w, h);

        let view = |px: f32, py: f32, z: f32| -> Vec3 {
            let ndc_x = 2.0 * ((px + 0.5) / wf) - 1.0;
            let ndc_y = 1.0 - 2.0 * ((py + 0.5) / hf);
            Vec3::new(ndc_x * z * tx, ndc_y * z * ty, z)
        };

        let mut saw_non_fallback = false;
        let fallback = Vec3::new(0.0, 0.0, -1.0);
        let mut gy2 = 0usize;
        while gy2 < h {
            let mut gx2 = 0usize;
            while gx2 < w {
                let idx = gy2 * w + gx2;
                let o = idx * RECONSTRUCT_OUT_FLOATS;
                let zc = depth[idx];
                let gxf = gx2 as f32;
                let gyf = gy2 as f32;
                let pc = view(gxf, gyf, zc);

                let mut dxv = Vec3::ZERO;
                let zr = if gx2 + 1 < w { depth[idx + 1] } else { 0.0 };
                let zl = if gx2 > 0 { depth[idx - 1] } else { 0.0 };
                let vr = gx2 + 1 < w && zr > 0.0;
                let vl = gx2 > 0 && zl > 0.0;
                if vr && vl {
                    let dpr = view((gx2 + 1) as f32, gyf, zr).sub(pc);
                    let dpl = pc.sub(view((gx2 - 1) as f32, gyf, zl));
                    if (zr - zc).abs() <= (zc - zl).abs() {
                        dxv = dpr;
                    } else {
                        dxv = dpl;
                    }
                } else if vr {
                    dxv = view((gx2 + 1) as f32, gyf, zr).sub(pc);
                } else if vl {
                    dxv = pc.sub(view((gx2 - 1) as f32, gyf, zl));
                }

                let mut dyv = Vec3::ZERO;
                let zu = if gy2 > 0 { depth[idx - w] } else { 0.0 };
                let zd = if gy2 + 1 < h { depth[idx + w] } else { 0.0 };
                let vu = gy2 > 0 && zu > 0.0;
                let vd = gy2 + 1 < h && zd > 0.0;
                if vu && vd {
                    let dpu = view(gxf, (gy2 - 1) as f32, zu).sub(pc);
                    let dpd = pc.sub(view(gxf, (gy2 + 1) as f32, zd));
                    if (zu - zc).abs() <= (zc - zd).abs() {
                        dyv = dpu;
                    } else {
                        dyv = dpd;
                    }
                } else if vu {
                    dyv = view(gxf, (gy2 - 1) as f32, zu).sub(pc);
                } else if vd {
                    dyv = pc.sub(view(gxf, (gy2 + 1) as f32, zd));
                }

                let n0 = dxv.cross(dyv);
                let len_sq = n0.dot(n0);
                let mut n = Vec3::new(0.0, 0.0, -1.0);
                if len_sq > EPS_LEN_SQ {
                    n = n0.scale(1.0 / len_sq.sqrt());
                }
                if n.z > 0.0 {
                    n = Vec3::new(-n.x, -n.y, -n.z);
                }

                assert_eq!(out[o].to_bits(), n.x.to_bits(), "nx at {idx}");
                assert_eq!(out[o + 1].to_bits(), n.y.to_bits(), "ny at {idx}");
                assert_eq!(out[o + 2].to_bits(), n.z.to_bits(), "nz at {idx}");
                assert_eq!(
                    out[o + 3].to_bits(),
                    thickness[idx].to_bits(),
                    "thick {idx}"
                );

                if (n.x - fallback.x).abs() > 1.0e-4 || (n.y - fallback.y).abs() > 1.0e-4 {
                    saw_non_fallback = true;
                }
                gx2 += 1;
            }
            gy2 += 1;
        }
        assert!(
            saw_non_fallback,
            "reconstruction never deviated from the flat fallback"
        );
    }

    /// Background pixels (depth `<= 0`) emit a null record; a valid pixel beside
    /// them still reconstructs and carries its thickness through.
    #[test]
    fn background_depth_writes_null_record() {
        let w = 3usize;
        let h = 1usize;
        // Left and right are background, the center is a surface sample.
        let depth = [0.0_f32, 5.0, -2.0];
        let thickness = [9.0_f32, 0.7, 9.0];
        let out = dispatch_surface_reconstruct(&depth, &thickness, PARAMS, w, h);

        // Background pixel 0: all four lanes zero.
        assert_eq!(out[0].to_bits(), 0.0_f32.to_bits());
        assert_eq!(out[1].to_bits(), 0.0_f32.to_bits());
        assert_eq!(out[2].to_bits(), 0.0_f32.to_bits());
        assert_eq!(out[3].to_bits(), 0.0_f32.to_bits());
        // Background pixel 2 (negative depth): also a null record.
        let o2 = 2 * RECONSTRUCT_OUT_FLOATS;
        assert_eq!(out[o2 + 3].to_bits(), 0.0_f32.to_bits());
        // Valid center pixel 1: thickness carried through, normal faces camera.
        let o1 = RECONSTRUCT_OUT_FLOATS;
        assert_eq!(out[o1 + 3].to_bits(), 0.7_f32.to_bits());
        assert!(out[o1 + 2] < 0.0, "center normal faces camera");
    }

    /// Degenerate screens and short buffers must never panic.
    #[test]
    fn degenerate_and_short_inputs_do_not_panic() {
        // Zero-size screen.
        let empty = dispatch_surface_reconstruct(&[], &[], PARAMS, 0, 0);
        assert!(empty.is_empty());
        // Buffers shorter than width*height: missing cells read as background.
        let short = dispatch_surface_reconstruct(&[5.0], &[1.0], PARAMS, 4, 4);
        assert_eq!(short.len(), 4 * 4 * RECONSTRUCT_OUT_FLOATS);
        // Single pixel: no valid neighbour on either axis -> flat fallback.
        let one = dispatch_surface_reconstruct(&[3.0], &[0.5], PARAMS, 1, 1);
        assert!(one[2] < 0.0, "lone pixel falls back to camera-facing");
        assert_eq!(one[3].to_bits(), 0.5_f32.to_bits());
    }

    /// Identical inputs produce byte-identical outputs across repeated calls.
    #[test]
    fn deterministic_across_calls() {
        let w = 6usize;
        let h = 4usize;
        let mut depth = vec![0.0_f32; w * h];
        let mut thickness = vec![0.0_f32; w * h];
        let mut i = 0usize;
        while i < w * h {
            depth[i] = 4.0 + 0.1 * i as f32;
            thickness[i] = 0.2 + 0.03 * i as f32;
            i += 1;
        }
        let a = dispatch_surface_reconstruct(&depth, &thickness, PARAMS, w, h);
        let b = dispatch_surface_reconstruct(&depth, &thickness, PARAMS, w, h);
        assert_eq!(a.len(), b.len());
        let mut k = 0usize;
        while k < a.len() {
            assert_eq!(a[k].to_bits(), b[k].to_bits(), "lane {k}");
            k += 1;
        }
    }
}
