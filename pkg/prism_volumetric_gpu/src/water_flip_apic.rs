//! `wgpu` compute twin of the `APIC` velocity reconstruction
//! ([`apic_velocity`](prism_render_architecture::water::flip::apic_velocity)).
//!
//! Affine Particle-in-Cell transfer reconstructs the velocity a particle
//! carries to a nearby grid node from the particle's base velocity `v_p` and
//! its affine velocity matrix `C_p`: `v(x) = v_p + C_p (x - x_p)`. The matrix
//! is supplied as its three rows, and `offset = x - x_p` is the node position
//! relative to the particle. The reconstruction is three independent
//! dot-products plus an add, a pure multiply-add kernel that ports to the
//! device unchanged, so a passing real-device parity run is direct evidence the
//! ported kernel folds the same arithmetic the reference does, not merely that
//! the shader compiles.
//!
//! # What is twinned
//!
//! One thread serves one query. It reproduces
//! [`apic_velocity`](prism_render_architecture::water::flip::apic_velocity)
//! exactly: for each output component `i`,
//! `result[i] = base[i] + dot(affine_rows[i], offset)`. A zero affine matrix
//! collapses to plain `PIC` transfer (`result == base`), and a linear velocity
//! field is reproduced exactly.
//!
//! # What stays on the host
//!
//! The gather of the affine matrix `C_p`, the particle base velocity, and the
//! node offsets are produced by the transfer stage upstream; this twin only
//! reconstructs the per-node velocity from those inputs. The reference performs
//! no sanitization (no `NaN`/`inf` guards), so neither does this twin.
//!
//! # Correctness model
//!
//! Every output is a continuous `f32` built from a bounded multiply-add
//! sequence (three products and an add per component), so the parity test uses
//! a tolerance of `abs <= 1e-5 || rel <= 1e-5` with a relative floor. There are
//! no branches, no transcendental calls, and no `sqrt`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — scalar multiply-add
//! — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry,
//! no `smoothstep`, no `round`, no `cbrt`, and no `sqrt`. Each thread performs a
//! bounded, branch-free sequence, so the kernel provably terminates. No
//! optional device feature is required, so it runs unmodified across backends.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::flip`；无第三方引擎源码或衍生代码。
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

/// Number of threads per workgroup. The reconstruction is a
/// single-thread-per-element kernel, so a one-dimensional dispatch of this
/// width keeps every lane busy on real hardware.
const WORKGROUP_SIZE: u32 = 64;

/// The inlined `WGSL` twin of
/// [`apic_velocity`](prism_render_architecture::water::flip::apic_velocity):
/// one thread per query, computing each output component as the base velocity
/// plus the dot-product of the matching affine row with the offset.
const WATER_FLIP_APIC_WGSL: &str = r#"
// Twin of water::flip::apic_velocity. One thread per query:
//   out.x = base.x + dot(row0, offset)
//   out.y = base.y + dot(row1, offset)
//   out.z = base.z + dot(row2, offset)
// Pure scalar multiply-add; no transcendental and no sqrt.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::flip；无第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this stop.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Particle base velocity v_p.
    base_x: f32,
    base_y: f32,
    base_z: f32,
    // Affine matrix C_p row 0.
    r0x: f32,
    r0y: f32,
    r0z: f32,
    // Affine matrix C_p row 1.
    r1x: f32,
    r1y: f32,
    r1z: f32,
    // Affine matrix C_p row 2.
    r2x: f32,
    r2y: f32,
    r2z: f32,
    // Node offset x - x_p.
    off_x: f32,
    off_y: f32,
    off_z: f32,
}

struct Result {
    // Reconstructed velocity components.
    vx: f32,
    vy: f32,
    vz: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let q = queries[idx];
    results[idx].vx = q.base_x + q.r0x * q.off_x + q.r0y * q.off_y + q.r0z * q.off_z;
    results[idx].vy = q.base_y + q.r1x * q.off_x + q.r1y * q.off_y + q.r1z * q.off_z;
    results[idx].vz = q.base_z + q.r2x * q.off_x + q.r2y * q.off_y + q.r2z * q.off_z;
}
"#;

/// Uniform parameters for one dispatch: the count plus three pad words to fill
/// a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_FLIP_APIC_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// the base velocity, the three affine rows, and the offset (alignment `4`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Base velocity `x`.
    base_x: f32,
    /// Base velocity `y`.
    base_y: f32,
    /// Base velocity `z`.
    base_z: f32,
    /// Affine row `0`, component `x`.
    r0x: f32,
    /// Affine row `0`, component `y`.
    r0y: f32,
    /// Affine row `0`, component `z`.
    r0z: f32,
    /// Affine row `1`, component `x`.
    r1x: f32,
    /// Affine row `1`, component `y`.
    r1y: f32,
    /// Affine row `1`, component `z`.
    r1z: f32,
    /// Affine row `2`, component `x`.
    r2x: f32,
    /// Affine row `2`, component `y`.
    r2y: f32,
    /// Affine row `2`, component `z`.
    r2z: f32,
    /// Offset `x`.
    off_x: f32,
    /// Offset `y`.
    off_y: f32,
    /// Offset `z`.
    off_z: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the three reconstructed velocity components (alignment `4`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Reconstructed velocity `x`.
    vx: f32,
    /// Reconstructed velocity `y`.
    vy: f32,
    /// Reconstructed velocity `z`.
    vz: f32,
}

/// One `APIC` reconstruction query to run on the device: the particle base
/// velocity, the three rows of the affine matrix `C_p`, and the node offset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterFlipApicQuery {
    /// Particle base velocity `v_p`, as `[x, y, z]`.
    pub base: [f32; 3],
    /// Affine matrix `C_p`, as three rows of `[x, y, z]`.
    pub affine_rows: [[f32; 3]; 3],
    /// Node offset `x - x_p`, as `[x, y, z]`.
    pub offset: [f32; 3],
}

/// One `APIC` reconstruction result: the velocity `v = base + C_p * offset`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterFlipApicResult {
    /// Reconstructed velocity, as `[x, y, z]`.
    pub velocity: [f32; 3],
}

/// Encodes one [`WaterFlipApicQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &WaterFlipApicQuery) -> GpuQuery {
    GpuQuery {
        base_x: q.base[0],
        base_y: q.base[1],
        base_z: q.base[2],
        r0x: q.affine_rows[0][0],
        r0y: q.affine_rows[0][1],
        r0z: q.affine_rows[0][2],
        r1x: q.affine_rows[1][0],
        r1y: q.affine_rows[1][1],
        r1z: q.affine_rows[1][2],
        r2x: q.affine_rows[2][0],
        r2y: q.affine_rows[2][1],
        r2z: q.affine_rows[2][2],
        off_x: q.offset[0],
        off_y: q.offset[1],
        off_z: q.offset[2],
    }
}

/// Decodes one packed [`GpuResult`] into the public [`WaterFlipApicResult`].
fn decode_result(raw: &GpuResult) -> WaterFlipApicResult {
    WaterFlipApicResult {
        velocity: [raw.vx, raw.vy, raw.vz],
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

/// A compiled, reusable `APIC` velocity compute pipeline, twinning the `CPU`
/// golden
/// [`apic_velocity`](prism_render_architecture::water::flip::apic_velocity).
pub struct GpuWaterFlipApic {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterFlipApic {
    /// Compiles the `APIC` velocity kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterFlipApic {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_flip_apic"),
            source: ShaderSource::Wgsl(WATER_FLIP_APIC_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_flip_apic_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_flip_apic_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_flip_apic_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterFlipApic {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs every query in `queries` and returns one [`WaterFlipApicResult`]
    /// per input, in order.
    ///
    /// An empty `queries` batch returns an empty vector with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterFlipApicQuery],
    ) -> Vec<WaterFlipApicResult> {
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
            label: Some("prism_volumetric_water_flip_apic_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_flip_apic_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_flip_apic_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_flip_apic_bind_group"),
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
            label: Some("prism_volumetric_water_flip_apic_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_flip_apic_encoder"),
        });
        {
            // One thread per query.
            let threads = count as u32;
            let groups = threads.div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_flip_apic_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
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
