//! `wgpu` compute twin of the capsule volume accessor from the `CPU` golden
//! `prism_physics_core::collider::bounding_capsule`'s `BoundingCapsule::volume`.
//!
//! A capsule is a cylinder capped by two hemispheres, so its volume is the sum
//! of a cylinder `pi r^2 h` and a full sphere `4/3 pi r^3`, where `h` is the
//! segment length between the two cap centres `center_a` and `center_b`. The
//! kernel evaluates one capsule per thread and never divides at runtime, so it
//! has no degenerate guard: a zero-length segment collapses to the pure-sphere
//! term and `valid` is always `1`.
//!
//! The host oracle in the parity test independently reimplements the same
//! closed form in plain `f32`; it does not pull in `prism_physics_core`,
//! `prism_render_architecture` or `glam`.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::bounding_capsule`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` capsule-volume kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `BoundingCapsule::volume`; see the module documentation for the
/// algorithm.
///
/// `PI` is written as `3.1415927`, which rounds to the identical `f32` bit
/// pattern as Rust's `core::f32::consts::PI` (`0x40490FDB`). `FOUR_THIRDS` is a
/// typed `f32` constant holding the correctly-rounded value of `4.0 / 3.0`
/// (`0x3FAAAAAB`), matching Rust's `f32` division, so both terms share the exact
/// same constants as the oracle.
const BOUNDING_CAPSULE_VOLUME_WGSL: &str = r#"
// Capsule-volume twin: one thread per capsule computes the cylinder term plus
// the full-sphere term from the segment length and radius. No runtime division
// and no transcendental calls. All vectors are flattened to scalars.
// Provenance: 孪生自本仓 prism_physics_core::collider::bounding_capsule；无第三方引擎源码或衍生代码。

const PI: f32 = 3.1415927;
const FOUR_THIRDS: f32 = 1.3333334;

struct Params {
    // Number of capsules in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Capsule {
    // First cap centre (x, y, z).
    cax: f32,
    cay: f32,
    caz: f32,
    // Second cap centre (x, y, z).
    cbx: f32,
    cby: f32,
    cbz: f32,
    // Capsule radius.
    radius: f32,
    pad0: f32,
}

struct Volume {
    // Capsule volume.
    volume: f32,
    // Always 1; the closed form has no degenerate rejection.
    valid: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> capsules: array<Capsule>;
@group(0) @binding(2) var<storage, read_write> results: array<Volume>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = capsules[idx];

    let ca = vec3<f32>(q.cax, q.cay, q.caz);
    let cb = vec3<f32>(q.cbx, q.cby, q.cbz);
    let radius = q.radius;

    let axis = cb - ca;
    let h = sqrt(dot(axis, axis));

    // Cylinder term pi r^2 h, then the sphere term (4/3) pi r^3, in that order.
    let cylinder = PI * radius * radius * h;
    let sphere = FOUR_THIRDS * PI * radius * radius * radius;

    var out: Volume;
    out.volume = cylinder + sphere;
    out.valid = 1u;
    out.pad0 = 0u;
    out.pad1 = 0u;

    results[idx] = out;
}
"#;

/// `repr(C)` `std430` layout of the dispatch parameters.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid capsules in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one capsule, matching the `WGSL` `Capsule`
/// struct. Seven payload words plus one padding word keep the stride a flat `32`
/// bytes, a multiple of `16`, with every `vec3` flattened to scalars so no
/// vector-alignment surprise can appear.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCapsule {
    cax: f32,
    cay: f32,
    caz: f32,
    cbx: f32,
    cby: f32,
    cbz: f32,
    radius: f32,
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Volume`
/// struct. The volume word plus the `valid` word plus two padding words keep the
/// stride a flat `16` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    volume: f32,
    valid: u32,
    pad0: u32,
    pad1: u32,
}

/// One capsule query: the two cap centres and the capsule radius. Every vector
/// is flattened to scalars so the `std430` stride stays an unambiguous flat
/// layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundingCapsuleVolumeQuery {
    /// First cap centre, x component.
    pub cax: f32,
    /// First cap centre, y component.
    pub cay: f32,
    /// First cap centre, z component.
    pub caz: f32,
    /// Second cap centre, x component.
    pub cbx: f32,
    /// Second cap centre, y component.
    pub cby: f32,
    /// Second cap centre, z component.
    pub cbz: f32,
    /// Capsule radius.
    pub radius: f32,
}

impl BoundingCapsuleVolumeQuery {
    /// Builds a capsule-volume query from the two cap centres and the radius.
    #[must_use]
    pub fn new(center_a: [f32; 3], center_b: [f32; 3], radius: f32) -> BoundingCapsuleVolumeQuery {
        BoundingCapsuleVolumeQuery {
            cax: center_a[0],
            cay: center_a[1],
            caz: center_a[2],
            cbx: center_b[0],
            cby: center_b[1],
            cbz: center_b[2],
            radius,
        }
    }
}

/// One resolved capsule volume: the volume scalar and a `valid` flag that is
/// always `1`, since the closed form never rejects an input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundingCapsuleVolumeResult {
    /// Capsule volume.
    pub volume: f32,
    /// Always `1`; the closed form has no degenerate rejection.
    pub valid: u32,
}

/// Encodes one [`BoundingCapsuleVolumeQuery`] into its `std430` [`GpuCapsule`].
fn encode_query(q: &BoundingCapsuleVolumeQuery) -> GpuCapsule {
    GpuCapsule {
        cax: q.cax,
        cay: q.cay,
        caz: q.caz,
        cbx: q.cbx,
        cby: q.cby,
        cbz: q.cbz,
        radius: q.radius,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`BoundingCapsuleVolumeResult`].
fn decode_result(raw: &GpuResult) -> BoundingCapsuleVolumeResult {
    BoundingCapsuleVolumeResult {
        volume: raw.volume,
        valid: raw.valid,
    }
}

/// Builds a compute-visible buffer binding layout entry.
fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A compiled, reusable capsule-volume compute pipeline, twinning the `CPU`
/// golden `BoundingCapsule::volume`.
pub struct GpuBoundingCapsuleVolume {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBoundingCapsuleVolume {
    /// Compiles the capsule-volume kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBoundingCapsuleVolume {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_bounding_capsule_volume"),
            source: ShaderSource::Wgsl(BOUNDING_CAPSULE_VOLUME_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_bounding_capsule_volume_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_bounding_capsule_volume_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_bounding_capsule_volume_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBoundingCapsuleVolume {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`BoundingCapsuleVolumeResult`] per input, in order.
    ///
    /// An empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[BoundingCapsuleVolumeQuery],
    ) -> Vec<BoundingCapsuleVolumeResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bounding_capsule_volume_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuCapsule> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bounding_capsule_volume_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bounding_capsule_volume_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_bounding_capsule_volume_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bounding_capsule_volume_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_bounding_capsule_volume_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_bounding_capsule_volume_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per capsule, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter().map(decode_result).collect()
    }
}
