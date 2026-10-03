//! Water-surface meshing compute kernel: the `WESL` shader plus its bit-exact
//! `CPU` twin.
//!
//! This is the bridge between the spectral/assemble stages (which end by
//! writing *textures*: an assembled displacement field and a normal/foam field,
//! see [`super::spectrum_assemble_kernel`]) and the raster surface draw (which
//! reads *per-vertex storage arrays*, see [`super::surface_bindings`]). One
//! invocation assembles one surface vertex of a regular `verts_x` by `verts_z`
//! lattice: it writes the undisplaced base position and the patch `UV`, then
//! point-samples the two assembled textures at that `UV` and scatters the
//! results into the per-vertex displacement and normal/foam arrays the draw
//! consumes.
//!
//! Shipping oceans (`WaveWorks` / `Crest` / `UE5` Water) run the identical
//! producer->consumer split: evaluate the displacement/normal fields into `GPU`
//! textures once per cascade, then sample them while building the
//! clipmap/projected-grid vertices. This kernel is that sampling pass factored
//! into its own dispatch so the draw node stays a dumb index-buffer executor.
//! The pure-integer dispatch *sizing* contract already lives in
//! [`super::surface_mesh`]; this module adds the shader and the texel-for-texel
//! numeric twin.
//!
//! Sampling is a nearest-texel point fetch (clamp-to-edge on the `UV`->texel
//! map), chosen so the result is fully deterministic and reproducible by the
//! `CPU` twin without depending on host sampler state. This matches a
//! nearest/clamp sampler but removes the sampler-state ambiguity.
//!
//! [`WATER_SURFACE_MESH_WESL`] is the shader (entry point `water_surface_mesh`);
//! [`dispatch_surface_mesh`] is its bit-exact `CPU` twin. Because the sandbox has
//! no `GPU`, the twin is the correctness proof: it consumes the identical buffer
//! `ABI` ([`WaterKernel::SurfaceMesh`](super::super::kernels::WaterKernel) — four
//! storage buffers of per-vertex `vec4` records, one uniform param block, two
//! sampled textures, no storage textures, a 64-lane linear tile, `Vertices`
//! domain) and reproduces the shader vertex-for-vertex, mirroring the base/`UV`
//! float order and the nearest-texel fetch. Pure classical numerics — no AI/ML;
//! no transcendental math (only a divide for the `UV` and a floor for the texel
//! map).

use alloc::vec;
use alloc::vec::Vec;

/// `WESL` source of the water-surface meshing compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// test; the standalone `naga`/`wesl` compile check runs out of tree, because
/// `prism_render_architecture` is a zero-dependency crate.
pub const WATER_SURFACE_MESH_WESL: &str = include_str!("water_surface_mesh.wesl");

/// Number of `f32` lanes per per-vertex output record (one `vec4`: `xyzw`).
pub const SURFACE_MESH_VERTEX_FLOATS: usize = 4;

/// Number of `f32` lanes per sampled source texel (`rgba`).
pub const SURFACE_TEXEL_FLOATS: usize = 4;

/// Per-dispatch surface-meshing scalars, mirroring the shader `MeshParams`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshParams {
    /// Lattice vertex count along x.
    pub verts_x: u32,
    /// Lattice vertex count along z.
    pub verts_z: u32,
    /// Assembled source texture width in texels (0 disables sampling).
    pub tex_width: u32,
    /// Assembled source texture height in texels (0 disables sampling).
    pub tex_height: u32,
    /// World spacing between adjacent lattice vertices (m).
    pub cell_size: f32,
    /// World-space x of the patch's `(vx = 0)` edge.
    pub origin_x: f32,
    /// World-space z of the patch's `(vz = 0)` edge.
    pub origin_z: f32,
}

/// The four per-vertex arrays the meshing pass writes, each one `vec4<f32>`
/// (`SURFACE_MESH_VERTEX_FLOATS` lanes) per surface vertex, in the raster draw's
/// storage-binding order.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceMeshFields {
    /// Undisplaced base lattice positions (`xyz` world, `w = 1`).
    pub base_positions: Vec<f32>,
    /// Per-vertex patch `UV` (`xy` in `[0, 1]`, `zw` pad).
    pub surface_uvs: Vec<f32>,
    /// Sampled displacement (`xyz` world offset, `w` foldover/Jacobian).
    pub displacement: Vec<f32>,
    /// Sampled normal/foam (`xyz` surface normal, `w` foam coverage).
    pub normal_foam: Vec<f32>,
}

/// Nearest-texel point fetch of `tex` (`rgba`, row-major, `tex_width` by
/// `tex_height`) at normalised `UV` `(u, w)`, clamped to the last texel on each
/// axis. A zero-sized texture, or a slice too short to hold the fetched texel,
/// yields a zero fetch rather than a panic. Mirrors the shader `sample_texel`,
/// including the `u32(floor(uv * dim))` texel map.
fn sample_texel(tex: &[f32], params: &MeshParams, u: f32, w: f32) -> [f32; SURFACE_TEXEL_FLOATS] {
    if params.tex_width == 0 || params.tex_height == 0 {
        return [0.0; SURFACE_TEXEL_FLOATS];
    }
    let tx = ((u * params.tex_width as f32).floor() as u32).min(params.tex_width - 1);
    let ty = ((w * params.tex_height as f32).floor() as u32).min(params.tex_height - 1);
    let idx = (ty as usize * params.tex_width as usize + tx as usize) * SURFACE_TEXEL_FLOATS;
    if idx + SURFACE_TEXEL_FLOATS <= tex.len() {
        [tex[idx], tex[idx + 1], tex[idx + 2], tex[idx + 3]]
    } else {
        [0.0; SURFACE_TEXEL_FLOATS]
    }
}

/// Run the `water_surface_mesh` kernel on the `CPU`: assemble every surface
/// vertex of the `verts_x` by `verts_z` lattice, writing the four per-vertex
/// arrays.
///
/// `disp_tex` and `normal_tex` are the assembled source textures (`rgba`,
/// row-major). A degenerate lattice (`verts_x == 0` or `verts_z == 0`) assembles
/// no vertices and returns four empty arrays. Short or zero-sized source
/// textures yield zero fetches, never a panic.
#[must_use]
pub fn dispatch_surface_mesh(
    disp_tex: &[f32],
    normal_tex: &[f32],
    params: MeshParams,
) -> SurfaceMeshFields {
    let vx_count = params.verts_x as usize;
    let vz_count = params.verts_z as usize;
    let total = vx_count * vz_count;

    let mut base_positions = vec![0.0f32; total * SURFACE_MESH_VERTEX_FLOATS];
    let mut surface_uvs = vec![0.0f32; total * SURFACE_MESH_VERTEX_FLOATS];
    let mut displacement = vec![0.0f32; total * SURFACE_MESH_VERTEX_FLOATS];
    let mut normal_foam = vec![0.0f32; total * SURFACE_MESH_VERTEX_FLOATS];

    if total == 0 {
        return SurfaceMeshFields {
            base_positions,
            surface_uvs,
            displacement,
            normal_foam,
        };
    }

    // Mirror the shader's `max(verts - 1u, 1u)` single-vertex-axis guard.
    let denom_x = params.verts_x.saturating_sub(1).max(1);
    let denom_z = params.verts_z.saturating_sub(1).max(1);

    for v in 0..total {
        let vx = (v as u32) % params.verts_x;
        let vz = (v as u32) / params.verts_x;

        let world_x = params.origin_x + (vx as f32) * params.cell_size;
        let world_z = params.origin_z + (vz as f32) * params.cell_size;

        let u = (vx as f32) / (denom_x as f32);
        let w = (vz as f32) / (denom_z as f32);

        let disp = sample_texel(disp_tex, &params, u, w);
        let nf = sample_texel(normal_tex, &params, u, w);

        let b = v * SURFACE_MESH_VERTEX_FLOATS;
        base_positions[b] = world_x;
        base_positions[b + 1] = 0.0;
        base_positions[b + 2] = world_z;
        base_positions[b + 3] = 1.0;
        surface_uvs[b] = u;
        surface_uvs[b + 1] = w;
        surface_uvs[b + 2] = 0.0;
        surface_uvs[b + 3] = 0.0;
        displacement[b] = disp[0];
        displacement[b + 1] = disp[1];
        displacement[b + 2] = disp[2];
        displacement[b + 3] = disp[3];
        normal_foam[b] = nf[0];
        normal_foam[b + 1] = nf[1];
        normal_foam[b + 2] = nf[2];
        normal_foam[b + 3] = nf[3];
    }

    SurfaceMeshFields {
        base_positions,
        surface_uvs,
        displacement,
        normal_foam,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::water::kernels::{DispatchDomain, WaterKernel};

    /// Builds a `width` by `height` `rgba` source texture from a per-texel
    /// closure, row-major.
    fn texture<F: Fn(u32, u32) -> [f32; 4]>(width: u32, height: u32, f: F) -> Vec<f32> {
        let mut out = Vec::with_capacity((width * height) as usize * SURFACE_TEXEL_FLOATS);
        for ty in 0..height {
            for tx in 0..width {
                out.extend_from_slice(&f(tx, ty));
            }
        }
        out
    }

    /// The shipped `WESL` honors the `SurfaceMesh` descriptor's `ABI`:
    /// entry-point name, 64-lane linear tile, the four storage output arrays,
    /// the uniform block, the two sampled textures, and no storage textures.
    #[test]
    fn wesl_matches_descriptor_abi() {
        let src = WATER_SURFACE_MESH_WESL;
        assert!(src.contains("fn water_surface_mesh("));
        assert!(src.contains("@workgroup_size(64, 1, 1)"));
        assert!(src.contains("var<uniform> params: MeshParams"));
        assert!(src.contains("var disp_tex: texture_2d<f32>"));
        assert!(src.contains("var normal_tex: texture_2d<f32>"));
        assert!(src.contains("var<storage, read_write> base_positions: array<vec4<f32>>"));
        assert!(src.contains("var<storage, read_write> surface_uvs: array<vec4<f32>>"));
        assert!(src.contains("var<storage, read_write> out_displacement: array<vec4<f32>>"));
        assert!(src.contains("var<storage, read_write> out_normal_foam: array<vec4<f32>>"));
        assert!(src.contains("textureLoad("));

        let desc = WaterKernel::SurfaceMesh.descriptor();
        assert_eq!(desc.workgroup.x, 64);
        assert_eq!(desc.workgroup.y, 1);
        assert_eq!(desc.workgroup.z, 1);
        assert_eq!(desc.domain, DispatchDomain::Vertices);
        assert_eq!(desc.layout.storage_buffers, 4);
        assert_eq!(desc.layout.uniform_buffers, 1);
        assert_eq!(desc.layout.sampled_textures, 2);
        assert_eq!(desc.layout.storage_textures, 0);
    }

    /// Base positions follow the world lattice and `UV`s span `[0, 1]`; with a
    /// zero-sized texture the sampled arrays stay zero but the lattice still
    /// fills in.
    #[test]
    fn base_lattice_and_uv_without_sampling() {
        let params = MeshParams {
            verts_x: 3,
            verts_z: 2,
            tex_width: 0,
            tex_height: 0,
            cell_size: 2.0,
            origin_x: -1.0,
            origin_z: 5.0,
        };
        let fields = dispatch_surface_mesh(&[], &[], params);
        // 3*2 vertices, 4 lanes each.
        assert_eq!(fields.base_positions.len(), 6 * SURFACE_MESH_VERTEX_FLOATS);

        // Vertex (vx=2, vz=1) -> flat index vz*verts_x + vx = 1*3 + 2 = 5.
        let v = 5usize;
        let b = v * SURFACE_MESH_VERTEX_FLOATS;
        // world_x = -1 + 2*2 = 3; world_z = 5 + 1*2 = 7.
        assert_eq!(fields.base_positions[b].to_bits(), 3.0f32.to_bits());
        assert_eq!(fields.base_positions[b + 1].to_bits(), 0.0f32.to_bits());
        assert_eq!(fields.base_positions[b + 2].to_bits(), 7.0f32.to_bits());
        assert_eq!(fields.base_positions[b + 3].to_bits(), 1.0f32.to_bits());
        // u = 2 / (3-1) = 1.0; w = 1 / (2-1) = 1.0.
        assert_eq!(fields.surface_uvs[b].to_bits(), 1.0f32.to_bits());
        assert_eq!(fields.surface_uvs[b + 1].to_bits(), 1.0f32.to_bits());
        assert_eq!(fields.surface_uvs[b + 2].to_bits(), 0.0f32.to_bits());
        assert_eq!(fields.surface_uvs[b + 3].to_bits(), 0.0f32.to_bits());

        // Zero-sized texture -> every sampled lane is zero.
        for lane in &fields.displacement {
            assert_eq!(lane.to_bits(), 0.0f32.to_bits());
        }
        for lane in &fields.normal_foam {
            assert_eq!(lane.to_bits(), 0.0f32.to_bits());
        }
    }

    /// A single-vertex axis pins its `UV` to 0 (no divide by zero) and still
    /// produces a valid base position.
    #[test]
    fn single_vertex_axis_pins_uv_to_zero() {
        let params = MeshParams {
            verts_x: 1,
            verts_z: 4,
            tex_width: 0,
            tex_height: 0,
            cell_size: 1.0,
            origin_x: 0.0,
            origin_z: 0.0,
        };
        let fields = dispatch_surface_mesh(&[], &[], params);
        for v in 0..4usize {
            let b = v * SURFACE_MESH_VERTEX_FLOATS;
            // vx == 0 -> u == 0 for every vertex.
            assert_eq!(fields.surface_uvs[b].to_bits(), 0.0f32.to_bits());
            // w = vz / (4-1).
            let expect_w = (v as f32) / 3.0f32;
            assert_eq!(fields.surface_uvs[b + 1].to_bits(), expect_w.to_bits());
        }
    }

    /// Nearest-texel sampling maps each lattice vertex to its matching source
    /// texel; the fetched `rgba` passes through to the displacement and
    /// normal/foam arrays unchanged.
    #[test]
    fn nearest_sampling_maps_vertices_to_texels() {
        // 2x2 textures: texel (tx, ty) carries a distinct, recognisable value.
        let disp = texture(2, 2, |tx, ty| {
            let idx = (ty * 2 + tx) as f32;
            [idx, idx + 0.5, idx + 0.25, idx + 0.125]
        });
        let nrm = texture(2, 2, |tx, ty| {
            let idx = (ty * 2 + tx) as f32;
            [
                idx * 10.0,
                idx * 10.0 + 1.0,
                idx * 10.0 + 2.0,
                idx * 10.0 + 3.0,
            ]
        });
        let params = MeshParams {
            verts_x: 2,
            verts_z: 2,
            tex_width: 2,
            tex_height: 2,
            cell_size: 1.0,
            origin_x: 0.0,
            origin_z: 0.0,
        };
        let fields = dispatch_surface_mesh(&disp, &nrm, params);
        // With 2 verts per axis, u/w are 0 or 1; floor(0*2)=0, floor(1*2)=2->clamp 1.
        // So vertex (vx, vz) samples texel (vx, vz).
        for vz in 0..2u32 {
            for vx in 0..2u32 {
                let v = (vz * 2 + vx) as usize;
                let b = v * SURFACE_MESH_VERTEX_FLOATS;
                let idx = (vz * 2 + vx) as f32;
                assert_eq!(fields.displacement[b].to_bits(), idx.to_bits());
                assert_eq!(fields.displacement[b + 1].to_bits(), (idx + 0.5).to_bits());
                assert_eq!(fields.displacement[b + 2].to_bits(), (idx + 0.25).to_bits());
                assert_eq!(
                    fields.displacement[b + 3].to_bits(),
                    (idx + 0.125).to_bits()
                );
                assert_eq!(fields.normal_foam[b].to_bits(), (idx * 10.0).to_bits());
                assert_eq!(
                    fields.normal_foam[b + 1].to_bits(),
                    (idx * 10.0 + 1.0).to_bits()
                );
                assert_eq!(
                    fields.normal_foam[b + 2].to_bits(),
                    (idx * 10.0 + 2.0).to_bits()
                );
                assert_eq!(
                    fields.normal_foam[b + 3].to_bits(),
                    (idx * 10.0 + 3.0).to_bits()
                );
            }
        }
    }

    /// Full-field independent inline recomputation, vertex-for-vertex, binds the
    /// twin to the documented algorithm (anti-vacuous: at least one sampled
    /// displacement lane differs from the texel-0 value, proving the `UV`->texel
    /// map actually varies across the lattice).
    #[test]
    fn matches_independent_inline_recompute() {
        let tw = 4u32;
        let th = 4u32;
        let disp = texture(tw, th, |tx, ty| {
            [tx as f32, ty as f32, (tx + ty) as f32, 0.0]
        });
        let nrm = texture(tw, th, |tx, ty| [0.0, 1.0, 0.0, (tx * ty) as f32]);
        let params = MeshParams {
            verts_x: 5,
            verts_z: 3,
            tex_width: tw,
            tex_height: th,
            cell_size: 0.5,
            origin_x: 2.0,
            origin_z: -3.0,
        };
        let fields = dispatch_surface_mesh(&disp, &nrm, params);

        let total = (params.verts_x * params.verts_z) as usize;
        let denom_x = params.verts_x.saturating_sub(1).max(1);
        let denom_z = params.verts_z.saturating_sub(1).max(1);
        let mut saw_nonzero_offset = false;
        for v in 0..total {
            let vx = (v as u32) % params.verts_x;
            let vz = (v as u32) / params.verts_x;
            let world_x = params.origin_x + (vx as f32) * params.cell_size;
            let world_z = params.origin_z + (vz as f32) * params.cell_size;
            let u = (vx as f32) / (denom_x as f32);
            let w = (vz as f32) / (denom_z as f32);
            let tx = ((u * tw as f32).floor() as u32).min(tw - 1);
            let ty = ((w * th as f32).floor() as u32).min(th - 1);
            let b = v * SURFACE_MESH_VERTEX_FLOATS;
            assert_eq!(fields.base_positions[b].to_bits(), world_x.to_bits());
            assert_eq!(fields.base_positions[b + 2].to_bits(), world_z.to_bits());
            assert_eq!(fields.surface_uvs[b].to_bits(), u.to_bits());
            assert_eq!(fields.surface_uvs[b + 1].to_bits(), w.to_bits());
            // Expected displacement texel passthrough.
            assert_eq!(fields.displacement[b].to_bits(), (tx as f32).to_bits());
            assert_eq!(fields.displacement[b + 1].to_bits(), (ty as f32).to_bits());
            assert_eq!(
                fields.displacement[b + 2].to_bits(),
                ((tx + ty) as f32).to_bits()
            );
            assert_eq!(
                fields.normal_foam[b + 3].to_bits(),
                ((tx * ty) as f32).to_bits()
            );
            if fields.displacement[b].to_bits() != 0.0f32.to_bits() {
                saw_nonzero_offset = true;
            }
        }
        assert!(
            saw_nonzero_offset,
            "UV->texel map must vary across the lattice"
        );
    }

    /// A degenerate lattice assembles nothing; a source slice too short to hold
    /// the mapped texel yields a zero fetch, never a panic.
    #[test]
    fn degenerate_grid_and_short_texture_do_not_panic() {
        for params in [
            MeshParams {
                verts_x: 0,
                verts_z: 4,
                tex_width: 2,
                tex_height: 2,
                cell_size: 1.0,
                origin_x: 0.0,
                origin_z: 0.0,
            },
            MeshParams {
                verts_x: 4,
                verts_z: 0,
                tex_width: 2,
                tex_height: 2,
                cell_size: 1.0,
                origin_x: 0.0,
                origin_z: 0.0,
            },
        ] {
            let fields = dispatch_surface_mesh(&[], &[], params);
            assert!(fields.base_positions.is_empty());
            assert!(fields.surface_uvs.is_empty());
            assert!(fields.displacement.is_empty());
            assert!(fields.normal_foam.is_empty());
        }

        // Texture claims to be 4x4 but the slice holds only one texel: every
        // out-of-range fetch returns zero instead of panicking.
        let params = MeshParams {
            verts_x: 3,
            verts_z: 3,
            tex_width: 4,
            tex_height: 4,
            cell_size: 1.0,
            origin_x: 0.0,
            origin_z: 0.0,
        };
        let short = vec![9.0f32; SURFACE_TEXEL_FLOATS];
        let fields = dispatch_surface_mesh(&short, &short, params);
        // Vertex (0,0) maps to texel (0,0), which the slice does hold.
        assert_eq!(fields.displacement[0].to_bits(), 9.0f32.to_bits());
        // A later vertex maps past the slice -> zero fetch.
        let last = (params.verts_x * params.verts_z - 1) as usize * SURFACE_MESH_VERTEX_FLOATS;
        assert_eq!(fields.displacement[last].to_bits(), 0.0f32.to_bits());
    }

    /// The twin is deterministic: identical inputs yield bit-identical output.
    #[test]
    fn dispatch_is_deterministic() {
        let disp = texture(3, 3, |tx, ty| [tx as f32, ty as f32, 1.0, 2.0]);
        let nrm = texture(3, 3, |tx, ty| [0.0, 1.0, 0.0, (tx + ty) as f32]);
        let params = MeshParams {
            verts_x: 6,
            verts_z: 4,
            tex_width: 3,
            tex_height: 3,
            cell_size: 0.25,
            origin_x: 1.0,
            origin_z: 1.0,
        };
        let a = dispatch_surface_mesh(&disp, &nrm, params);
        let b = dispatch_surface_mesh(&disp, &nrm, params);
        assert_eq!(a, b);
    }
}
