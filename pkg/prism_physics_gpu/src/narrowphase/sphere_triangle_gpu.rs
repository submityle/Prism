//! Real-device `wgpu` compute implementation of the sphere-versus-triangle
//! narrow phase.
//!
//! [`GpuSphereTriangleNarrowphase`] compiles
//! `shaders/narrowphase_sphere_triangle.wgsl` once and exposes
//! [`GpuSphereTriangleNarrowphase::query`], which turns a batch of candidate
//! sphere-versus-triangle pairs into one contact slot each on the device. One
//! invocation handles one pair, reading the sphere's bounding sphere and the
//! triangle's three vertices, and running the same closest-point query and
//! manifold construction the
//! [`cpu_sphere_triangle_narrowphase`](super::sphere_triangle::cpu_sphere_triangle_narrowphase)
//! golden twin runs, so a passing real-device parity test is direct evidence the
//! kernel builds the same contacts as the reference.
//!
//! # Buffer layout
//!
//! The `CPU`-side packing here is byte-for-byte with the `WGSL` structs:
//!
//! * spheres upload as one `vec4<f32>` each (`xyz` centre, `w` radius), reusing
//!   the [`Particle`] packing;
//! * triangles upload as three `vec4<f32>` each ([`GpuTriangle`]): one vertex in
//!   the `xyz` of each row, the `w` lanes unused padding;
//! * pairs upload as one `vec2<u32>` each (sphere index, triangle index);
//! * contacts read back as two `vec4<f32>` each ([`GpuContact`]):
//!   `(normal.xyz, depth)` then `(point.xyz, valid)`.
//!
//! # Dense output
//!
//! The kernel writes a slot per input pair, marking non-penetrating pairs with a
//! zero validity flag rather than compacting them away. This keeps the contact
//! index aligned with the pair index (which the parity test relies on) and lets
//! a downstream [`crate::scan`] compaction stream the survivors without a second
//! pass over the pairs.
//!
//! Provenance: closest-point-on-triangle is the Voronoi-region method from
//! Christer Ericson, *Real-Time Collision Detection* (2004), section 5.1.5; the
//! sphere manifold is textbook. No Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::broadphase::Particle;
use crate::buffer;
use crate::context::GpuContext;

use super::contact::Contact;
use super::layout::{buffer_entry, entry};
use super::sphere_triangle::{SphereTrianglePair, Triangle};

/// Lanes per workgroup; must match `@workgroup_size` in the kernel.
const WORKGROUP: usize = 64;

/// Bytes each output [`GpuContact`] slot occupies, mirroring the `WGSL` `Contact`
/// stride (two `vec4<f32>`).
const CONTACT_BYTES: u64 = size_of::<GpuContact>() as u64;

/// Uniform parameters shared with `Params` in
/// `shaders/narrowphase_sphere_triangle.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of candidate pairs queued in the pair buffer.
    num_pairs: u32,
    /// Padding to a 16-byte boundary.
    pad0: u32,
    /// Padding to a 16-byte boundary.
    pad1: u32,
    /// Padding to a 16-byte boundary.
    pad2: u32,
}

/// Upload form of one [`Triangle`], matching the `WGSL` `Triangle` struct: each
/// vertex in the `xyz` of its row, the `w` lanes unused padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTriangle {
    /// First vertex in `xyz`; `w` is unused padding.
    a: [f32; 4],
    /// Second vertex in `xyz`; `w` is unused padding.
    b: [f32; 4],
    /// Third vertex in `xyz`; `w` is unused padding.
    c: [f32; 4],
}

/// Readback form of one contact slot, matching the `WGSL` `Contact` layout:
/// `(normal.xyz, depth)` then `(point.xyz, valid)`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuContact {
    /// Unit contact normal in `xyz`, penetration depth in `w`.
    normal_depth: [f32; 4],
    /// World contact point in `xyz`, validity flag in `w` (nonzero means a
    /// reported contact).
    point_valid: [f32; 4],
}

/// A compiled, reusable `GPU` sphere-versus-triangle narrow-phase pipeline.
pub struct GpuSphereTriangleNarrowphase {
    /// Kept alive so the pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring params, spheres, triangles, pairs, and
    /// contacts.
    layout: BindGroupLayout,
    /// The contact kernel: one invocation per candidate pair.
    contacts: ComputePipeline,
}

impl GpuSphereTriangleNarrowphase {
    /// Compiles the sphere-versus-triangle contact kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSphereTriangleNarrowphase {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_narrowphase_sphere_triangle"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/narrowphase_sphere_triangle.wgsl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_narrowphase_sphere_triangle_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_narrowphase_sphere_triangle_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let contacts = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_narrowphase_sphere_triangle_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("narrowphase_sphere_triangle"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSphereTriangleNarrowphase {
            module,
            layout,
            contacts,
        }
    }

    /// Generates sphere-versus-triangle contacts for `pairs` on the `GPU`.
    ///
    /// Returns one slot per pair, in input order: [`Some`] carrying the manifold
    /// when the sphere penetrates the triangle, or [`None`] when it is separated
    /// or exactly touching. The output matches the
    /// [`cpu_sphere_triangle_narrowphase`](super::sphere_triangle::cpu_sphere_triangle_narrowphase)
    /// twin: the validity flag agrees exactly and the normal, depth, and point
    /// agree within the tight tolerance the square root and reciprocals impose.
    ///
    /// An empty `pairs` batch returns an empty vector without touching the
    /// device.
    #[must_use]
    pub fn query(
        &self,
        ctx: &GpuContext,
        spheres: &[Particle],
        triangles: &[Triangle],
        pairs: &[SphereTrianglePair],
    ) -> Vec<Option<Contact>> {
        if pairs.is_empty() {
            return Vec::new();
        }

        let device = ctx.device();
        let num_pairs = pairs.len();

        let params = Params {
            num_pairs: u32::try_from(num_pairs).unwrap_or(u32::MAX),
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf =
            buffer::uniform(device, "prism_narrowphase_sphere_triangle_params", &params);

        // Spheres pack to a single vec4 each: xyz centre, w radius.
        let packed_spheres: Vec<[f32; 4]> = spheres
            .iter()
            .map(|s| [s.position.x, s.position.y, s.position.z, s.radius])
            .collect();
        let spheres_buf = buffer::storage_read(
            device,
            "prism_narrowphase_sphere_triangle_spheres",
            &packed_spheres,
        );

        // Triangles pack to three vec4 each: one vertex per row, w unused.
        let packed_triangles: Vec<GpuTriangle> = triangles
            .iter()
            .map(|t| GpuTriangle {
                a: [t.a.x, t.a.y, t.a.z, 0.0],
                b: [t.b.x, t.b.y, t.b.z, 0.0],
                c: [t.c.x, t.c.y, t.c.z, 0.0],
            })
            .collect();
        let triangles_buf = buffer::storage_read(
            device,
            "prism_narrowphase_sphere_triangle_triangles",
            &packed_triangles,
        );

        // Pairs pack to a vec2<u32> each: the sphere index then the triangle index.
        let packed_pairs: Vec<[u32; 2]> = pairs
            .iter()
            .map(|pair| [pair.sphere, pair.triangle])
            .collect();
        let pairs_buf = buffer::storage_read(
            device,
            "prism_narrowphase_sphere_triangle_pairs",
            &packed_pairs,
        );

        let contacts_bytes = CONTACT_BYTES * num_pairs as u64;
        let contacts_buf = buffer::storage_rw_zeroed(
            device,
            "prism_narrowphase_sphere_triangle_contacts",
            contacts_bytes,
        );

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_narrowphase_sphere_triangle_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &spheres_buf),
                entry(2, &triangles_buf),
                entry(3, &pairs_buf),
                entry(4, &contacts_buf),
            ],
        });

        let contacts_stage = buffer::staging(
            device,
            "prism_narrowphase_sphere_triangle_contacts_stage",
            contacts_bytes,
        );

        let groups = u32::try_from(num_pairs.div_ceil(WORKGROUP)).unwrap_or(u32::MAX);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_narrowphase_sphere_triangle_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_narrowphase_sphere_triangle_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.contacts);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        buffer::copy(&mut encoder, &contacts_buf, &contacts_stage, contacts_bytes);
        ctx.queue().submit([encoder.finish()]);

        let raw = buffer::read_back::<GpuContact>(ctx, &contacts_stage);
        raw.iter()
            .zip(pairs)
            .map(|(slot, pair)| {
                if slot.point_valid[3] == 0.0 {
                    None
                } else {
                    let nd = slot.normal_depth;
                    let pv = slot.point_valid;
                    Some(Contact::new(
                        pair.sphere,
                        pair.triangle,
                        Vec3::new(nd[0], nd[1], nd[2]),
                        nd[3],
                        Vec3::new(pv[0], pv[1], pv[2]),
                    ))
                }
            })
            .collect()
    }
}
