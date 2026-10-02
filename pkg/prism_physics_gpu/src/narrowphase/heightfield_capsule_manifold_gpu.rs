//! Real-device `wgpu` compute implementation of the multi-point
//! capsule-versus-heightfield narrow phase.
//!
//! [`GpuCapsuleHeightfieldManifoldNarrowphase`] compiles
//! `shaders/narrowphase_capsule_heightfield_manifold.wgsl` once and exposes
//! [`GpuCapsuleHeightfieldManifoldNarrowphase::query`], which turns a batch of
//! `(capsule, heightfield)` pairs into one contact manifold each on the device.
//! One invocation handles one pair: it finds the grid cells overlapping the
//! capsule's `XZ` footprint, runs the shared per-triangle capsule clip over
//! each candidate cell triangle, picks the deepest sub-manifold as the reference
//! plane, pools the coplanar points, and reduces them to the widest, deepest
//! four exactly as the
//! [`cpu_capsule_heightfield_manifold`](super::heightfield_capsule_manifold::cpu_capsule_heightfield_manifold)
//! golden twin does, so a passing real-device parity test is direct evidence the
//! kernel builds the same manifolds as the reference.
//!
//! # Buffer layout
//!
//! The `CPU`-side packing here is byte-for-byte with the `WGSL` structs:
//!
//! * capsules upload as two `vec4<f32>` each ([`GpuCapsule`]): the first axis
//!   endpoint in row 0 with the swept radius riding in its `w` lane, then the
//!   second endpoint in row 1 (`w` unused);
//! * heightfields upload as one [`GpuField`] each: `dims = (rows, cols,
//!   heights_offset, pad)` as a `vec4<u32>`, then `geom = (cell_size,
//!   origin.x, origin.y, origin.z)` as a `vec4<f32>`;
//! * every field's row-major height samples are concatenated into one `f32`
//!   storage array, each field's slice starting at its `heights_offset`;
//! * pairs upload as one `vec2<u32>` each (capsule index, field index);
//! * manifolds read back as five `vec4<f32>` each ([`GpuManifold`]):
//!   `(normal.xyz, count)` then four `(position.xyz, depth)` points.
//!
//! # Dense output
//!
//! The kernel writes a slot per input pair, marking pairs that touch no cell
//! triangle with a zero point count rather than compacting them away. This keeps
//! the manifold index aligned with the pair index, which the parity test relies
//! on, and lets a downstream [`crate::scan`] compaction stream the survivors
//! without a second pass over the pairs.
//!
//! Provenance: textbook capsule-versus-triangle clipping (closest-point and
//! segment-triangle clip from Christer Ericson, *Real-Time Collision Detection*,
//! 2004) plus the standard four-point manifold reduction; the heightfield cell
//! triangulation is textbook. No Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::buffer;
use crate::context::GpuContext;

use super::capsule::Capsule;
use super::heightfield::Heightfield;
use super::heightfield_capsule_manifold::HeightfieldCapsulePair;
use super::layout::{buffer_entry, entry};
use super::manifold::{ContactManifold, ManifoldPoint, MAX_MANIFOLD_POINTS};

/// Lanes per workgroup; must match `@workgroup_size` in the kernel.
const WORKGROUP: usize = 64;

/// Bytes each output [`GpuManifold`] slot occupies, mirroring the `WGSL`
/// `Manifold` stride (five `vec4<f32>`).
const MANIFOLD_BYTES: u64 = size_of::<GpuManifold>() as u64;

/// Uniform parameters shared with `Params` in
/// `shaders/narrowphase_capsule_heightfield_manifold.wgsl`.
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

/// Upload form of one [`Capsule`], matching the `WGSL` `Capsule` struct: the
/// first axis endpoint in `p0_radius.xyz` with the swept radius in `w`, then
/// the second endpoint in `p1_pad.xyz` (`w` unused).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCapsule {
    /// First axis endpoint in `xyz`, swept radius in `w`.
    p0_radius: [f32; 4],
    /// Second axis endpoint in `xyz`; `w` is unused padding.
    p1_pad: [f32; 4],
}

/// Upload form of one [`Heightfield`], matching the `WGSL` `Field` struct:
/// `dims = (rows, cols, heights_offset, pad)` then `geom = (cell_size,
/// origin.x, origin.y, origin.z)`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuField {
    /// `(rows, cols, heights_offset, pad)`.
    dims: [u32; 4],
    /// `(cell_size, origin.x, origin.y, origin.z)`.
    geom: [f32; 4],
}

/// Readback form of one manifold slot, matching the `WGSL` `Manifold` layout:
/// `(normal.xyz, count)` then four `(position.xyz, depth)` points.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuManifold {
    /// Shared contact normal in `xyz` (from the heightfield toward the
    /// capsule), live point count in `w`.
    normal_count: [f32; 4],
    /// First contact point in `xyz`, its penetration depth in `w`.
    p0: [f32; 4],
    /// Second contact point in `xyz`, its penetration depth in `w`.
    p1: [f32; 4],
    /// Third contact point in `xyz`, its penetration depth in `w`.
    p2: [f32; 4],
    /// Fourth contact point in `xyz`, its penetration depth in `w`.
    p3: [f32; 4],
}

/// A compiled, reusable `GPU` capsule-versus-heightfield manifold pipeline.
pub struct GpuCapsuleHeightfieldManifoldNarrowphase {
    /// Kept alive so the pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring params, capsules, fields, heights, pairs, and
    /// manifolds.
    layout: BindGroupLayout,
    /// The manifold kernel: one invocation per candidate pair.
    manifolds: ComputePipeline,
}

impl GpuCapsuleHeightfieldManifoldNarrowphase {
    /// Compiles the capsule-versus-heightfield manifold kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCapsuleHeightfieldManifoldNarrowphase {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_narrowphase_capsule_heightfield_manifold"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/narrowphase_capsule_heightfield_manifold.wgsl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_narrowphase_capsule_heightfield_manifold_layout"),
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
            label: Some("prism_narrowphase_capsule_heightfield_manifold_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let manifolds = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_narrowphase_capsule_heightfield_manifold_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("narrowphase_capsule_heightfield_manifold"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCapsuleHeightfieldManifoldNarrowphase {
            module,
            layout,
            manifolds,
        }
    }

    /// Generates capsule-versus-heightfield contact manifolds for `pairs` over
    /// `capsules` and `fields` on the `GPU`.
    ///
    /// Returns one slot per pair, in input order: [`Some`] carrying a one- to
    /// four-point manifold when the capsule penetrates the terrain, or [`None`]
    /// when it is clear of every candidate cell. The output matches the
    /// [`cpu_capsule_heightfield_manifold`](super::heightfield_capsule_manifold::cpu_capsule_heightfield_manifold)
    /// twin: the point count agrees exactly and the normal, positions, and depths
    /// agree to within the tight float tolerance the square root, floor, and
    /// reciprocals impose.
    ///
    /// An empty `pairs` batch returns an empty vector without touching the
    /// device.
    #[must_use]
    pub fn query(
        &self,
        ctx: &GpuContext,
        capsules: &[Capsule],
        fields: &[Heightfield],
        pairs: &[HeightfieldCapsulePair],
    ) -> Vec<Option<ContactManifold>> {
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
            "prism_narrowphase_capsule_heightfield_manifold_params",
            &params,
        );

        // Capsules pack to two vec4 each: xyz endpoint plus radius, then xyz
        // endpoint with an unused w.
        let packed_capsules: Vec<GpuCapsule> = capsules
            .iter()
            .map(|cap| GpuCapsule {
                p0_radius: [cap.p0.x, cap.p0.y, cap.p0.z, cap.radius],
                p1_pad: [cap.p1.x, cap.p1.y, cap.p1.z, 0.0],
            })
            .collect();
        let capsules_buf = buffer::storage_read(
            device,
            "prism_narrowphase_capsule_heightfield_manifold_capsules",
            &packed_capsules,
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
                    dims: [f.rows(), f.cols(), offset, 0],
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
            "prism_narrowphase_capsule_heightfield_manifold_fields",
            &packed_fields,
        );
        let heights_buf = buffer::storage_read(
            device,
            "prism_narrowphase_capsule_heightfield_manifold_heights",
            &packed_heights,
        );

        // Pairs pack to a vec2<u32> each: the capsule index then the field index.
        let packed_pairs: Vec<[u32; 2]> = pairs
            .iter()
            .map(|pair| [pair.capsule, pair.heightfield])
            .collect();
        let pairs_buf = buffer::storage_read(
            device,
            "prism_narrowphase_capsule_heightfield_manifold_pairs",
            &packed_pairs,
        );

        let manifolds_bytes = MANIFOLD_BYTES * num_pairs as u64;
        let manifolds_buf = buffer::storage_rw_zeroed(
            device,
            "prism_narrowphase_capsule_heightfield_manifold_out",
            manifolds_bytes,
        );

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_narrowphase_capsule_heightfield_manifold_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &capsules_buf),
                entry(2, &fields_buf),
                entry(3, &heights_buf),
                entry(4, &pairs_buf),
                entry(5, &manifolds_buf),
            ],
        });

        let manifolds_stage = buffer::staging(
            device,
            "prism_narrowphase_capsule_heightfield_manifold_stage",
            manifolds_bytes,
        );

        let groups = u32::try_from(num_pairs.div_ceil(WORKGROUP)).unwrap_or(u32::MAX);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_narrowphase_capsule_heightfield_manifold_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_narrowphase_capsule_heightfield_manifold_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.manifolds);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        buffer::copy(
            &mut encoder,
            &manifolds_buf,
            &manifolds_stage,
            manifolds_bytes,
        );
        ctx.queue().submit([encoder.finish()]);

        let raw = buffer::read_back::<GpuManifold>(ctx, &manifolds_stage);
        raw.iter()
            .zip(pairs)
            .map(|(slot, pair)| decode_manifold(slot, pair))
            .collect()
    }
}

/// Rebuilds a [`ContactManifold`] from one raw `GPU` slot, or [`None`] when
/// the slot reports zero live points.
fn decode_manifold(slot: &GpuManifold, pair: &HeightfieldCapsulePair) -> Option<ContactManifold> {
    let count = slot.normal_count[3] as usize;
    if count == 0 {
        return None;
    }
    let normal = Vec3::new(
        slot.normal_count[0],
        slot.normal_count[1],
        slot.normal_count[2],
    );
    let rows = [slot.p0, slot.p1, slot.p2, slot.p3];
    let mut points = [ManifoldPoint::new(Vec3::ZERO, 0.0); MAX_MANIFOLD_POINTS];
    for (point, row) in points.iter_mut().zip(rows.iter()).take(count) {
        *point = ManifoldPoint::new(Vec3::new(row[0], row[1], row[2]), row[3]);
    }
    Some(ContactManifold::new(
        pair.capsule,
        pair.heightfield,
        normal,
        count,
        &points,
    ))
}
