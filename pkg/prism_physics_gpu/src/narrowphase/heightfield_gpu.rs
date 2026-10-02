//! Real-device `wgpu` compute implementation of the sphere-versus-heightfield
//! narrow phase.
//!
//! [`GpuSphereHeightfieldNarrowphase`] compiles
//! `shaders/narrowphase_sphere_heightfield.wgsl` once and exposes
//! [`GpuSphereHeightfieldNarrowphase::query`], which turns a batch of candidate
//! sphere-versus-heightfield pairs into one contact slot each on the device. One
//! invocation handles one pair: it reads the sphere's bounding sphere and the
//! heightfield's metadata, finds the grid cells overlapping the sphere's `XZ`
//! footprint, and runs the same closest-point query, manifold construction, and
//! deepest-contact reduction the
//! [`cpu_sphere_heightfield_narrowphase`](super::heightfield::cpu_sphere_heightfield_narrowphase)
//! golden twin runs, so a passing real-device parity test is direct evidence the
//! kernel builds the same contacts as the reference.
//!
//! # Buffer layout
//!
//! The `CPU`-side packing here is byte-for-byte with the `WGSL` structs:
//!
//! * spheres upload as one `vec4<f32>` each (`xyz` centre, `w` radius), reusing
//!   the [`Particle`] packing;
//! * heightfields upload as one [`GpuField`] each: `meta = (rows, cols,
//!   heights_offset, pad)` as a `vec4<u32>`, then `geom = (cell_size, origin.x,
//!   origin.y, origin.z)` as a `vec4<f32>`;
//! * every field's row-major height samples are concatenated into one `f32`
//!   storage array, each field's slice starting at its `heights_offset`;
//! * pairs upload as one `vec2<u32>` each (sphere index, field index);
//! * contacts read back as two `vec4<f32>` each ([`GpuContact`]):
//!   `(normal.xyz, depth)` then `(point.xyz, valid)`.
//!
//! # Dense output
//!
//! The kernel writes a slot per input pair, marking pairs that touch no cell
//! triangle with a zero validity flag rather than compacting them away. This
//! keeps the contact index aligned with the pair index (which the parity test
//! relies on) and lets a downstream [`crate::scan`] compaction stream the
//! survivors without a second pass over the pairs.
//!
//! Provenance: closest-point-on-triangle is the Voronoi-region method from
//! Christer Ericson, *Real-Time Collision Detection* (2004), section 5.1.5; the
//! heightfield cell triangulation is textbook. No Unreal Engine source or
//! derived code.

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
use super::heightfield::{Heightfield, HeightfieldSpherePair};
use super::layout::{buffer_entry, entry};

/// Lanes per workgroup; must match `@workgroup_size` in the kernel.
const WORKGROUP: usize = 64;

/// Bytes each output [`GpuContact`] slot occupies, mirroring the `WGSL` `Contact`
/// stride (two `vec4<f32>`).
const CONTACT_BYTES: u64 = size_of::<GpuContact>() as u64;

/// Uniform parameters shared with `Params` in
/// `shaders/narrowphase_sphere_heightfield.wgsl`.
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

/// Upload form of one [`Heightfield`], matching the `WGSL` `Field` struct:
/// `meta = (rows, cols, heights_offset, pad)` then `geom = (cell_size,
/// origin.x, origin.y, origin.z)`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuField {
    /// `(rows, cols, heights_offset, pad)`.
    meta: [u32; 4],
    /// `(cell_size, origin.x, origin.y, origin.z)`.
    geom: [f32; 4],
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

/// A compiled, reusable `GPU` sphere-versus-heightfield narrow-phase pipeline.
pub struct GpuSphereHeightfieldNarrowphase {
    /// Kept alive so the pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring params, spheres, fields, heights, pairs, and
    /// contacts.
    layout: BindGroupLayout,
    /// The contact kernel: one invocation per candidate pair.
    contacts: ComputePipeline,
}

impl GpuSphereHeightfieldNarrowphase {
    /// Compiles the sphere-versus-heightfield contact kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSphereHeightfieldNarrowphase {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_narrowphase_sphere_heightfield"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/narrowphase_sphere_heightfield.wgsl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_narrowphase_sphere_heightfield_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_narrowphase_sphere_heightfield_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let contacts = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_narrowphase_sphere_heightfield_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("narrowphase_sphere_heightfield"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSphereHeightfieldNarrowphase {
            module,
            layout,
            contacts,
        }
    }

    /// Generates sphere-versus-heightfield contacts for `pairs` on the `GPU`.
    ///
    /// Returns one slot per pair, in input order: [`Some`] carrying the deepest
    /// manifold when the sphere penetrates the terrain, or [`None`] when it is
    /// clear of every candidate cell. The output matches the
    /// [`cpu_sphere_heightfield_narrowphase`](super::heightfield::cpu_sphere_heightfield_narrowphase)
    /// twin: the validity flag agrees exactly and the normal, depth, and point
    /// agree within the tight tolerance the square root, floor, and reciprocals
    /// impose.
    ///
    /// An empty `pairs` batch returns an empty vector without touching the
    /// device.
    #[must_use]
    pub fn query(
        &self,
        ctx: &GpuContext,
        spheres: &[Particle],
        fields: &[Heightfield],
        pairs: &[HeightfieldSpherePair],
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
        let params_buf = buffer::uniform(
            device,
            "prism_narrowphase_sphere_heightfield_params",
            &params,
        );

        // Spheres pack to a single vec4 each: xyz centre, w radius.
        let packed_spheres: Vec<[f32; 4]> = spheres
            .iter()
            .map(|s| [s.position.x, s.position.y, s.position.z, s.radius])
            .collect();
        let spheres_buf = buffer::storage_read(
            device,
            "prism_narrowphase_sphere_heightfield_spheres",
            &packed_spheres,
        );

        // Fields pack to a GpuField each; their samples concatenate into one f32
        // array, each field remembering where its slice begins.
        let mut packed_heights: Vec<f32> = Vec::new();
        let packed_fields: Vec<GpuField> = fields
            .iter()
            .map(|f| {
                let offset = u32::try_from(packed_heights.len()).unwrap_or(u32::MAX);
                packed_heights.extend_from_slice(f.heights());
                let origin = f.origin();
                GpuField {
                    meta: [f.rows(), f.cols(), offset, 0],
                    geom: [f.cell_size(), origin.x, origin.y, origin.z],
                }
            })
            .collect();
        // A zero-length storage buffer is invalid; a single padding sample is
        // never read because such a field has no cells.
        if packed_heights.is_empty() {
            packed_heights.push(0.0);
        }
        let fields_buf = buffer::storage_read(
            device,
            "prism_narrowphase_sphere_heightfield_fields",
            &packed_fields,
        );
        let heights_buf = buffer::storage_read(
            device,
            "prism_narrowphase_sphere_heightfield_heights",
            &packed_heights,
        );

        // Pairs pack to a vec2<u32> each: the sphere index then the field index.
        let packed_pairs: Vec<[u32; 2]> = pairs
            .iter()
            .map(|pair| [pair.sphere, pair.heightfield])
            .collect();
        let pairs_buf = buffer::storage_read(
            device,
            "prism_narrowphase_sphere_heightfield_pairs",
            &packed_pairs,
        );

        let contacts_bytes = CONTACT_BYTES * num_pairs as u64;
        let contacts_buf = buffer::storage_rw_zeroed(
            device,
            "prism_narrowphase_sphere_heightfield_contacts",
            contacts_bytes,
        );

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_narrowphase_sphere_heightfield_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &spheres_buf),
                entry(2, &fields_buf),
                entry(3, &heights_buf),
                entry(4, &pairs_buf),
                entry(5, &contacts_buf),
            ],
        });

        let contacts_stage = buffer::staging(
            device,
            "prism_narrowphase_sphere_heightfield_contacts_stage",
            contacts_bytes,
        );

        let groups = u32::try_from(num_pairs.div_ceil(WORKGROUP)).unwrap_or(u32::MAX);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_narrowphase_sphere_heightfield_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_narrowphase_sphere_heightfield_pass"),
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
                        pair.heightfield,
                        Vec3::new(nd[0], nd[1], nd[2]),
                        nd[3],
                        Vec3::new(pv[0], pv[1], pv[2]),
                    ))
                }
            })
            .collect()
    }
}
