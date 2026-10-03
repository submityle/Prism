//! `wgpu` compute twin of the two stateless `MAC`-grid numerics inside the
//! `FLIP`/`PIC` fluid transfer contract
//! ([`flip`](prism_render_architecture::water::flip)).
//!
//! The staggered `MAC` fluid loop splats particle velocities to a grid,
//! projects out divergence, and gathers the corrected velocities back. Two of
//! its per-cell/per-particle kernels are pure, deterministic closed forms with
//! fixed-width inputs, and this module is their on-device twin:
//!
//! - [`blend_flip_pic`](prism_render_architecture::water::flip::blend_flip_pic):
//!   the `PIC`/`FLIP` velocity blend `(1 - a) * pic + a * flip` with the blend
//!   factor `a` clamped to `0..=1`. At `a = 0` it is the stable `PIC` velocity,
//!   at `a = 1` the low-dissipation `FLIP` velocity, varying monotonically
//!   between them.
//! - [`cell_divergence`](prism_render_architecture::water::flip::cell_divergence):
//!   the discrete velocity divergence
//!   `((x_pos - x_neg) + (y_pos - y_neg) + (z_pos - z_neg)) / dx` of one `MAC`
//!   cell's six staggered face velocities, the right-hand side the pressure
//!   projection drives to zero. A non-positive spacing `dx <= EPS` yields `0`.
//!
//! [`GpuWaterFlipMac`] evaluates both for one query per thread, reproducing the
//! reference's exact closed form, so a passing real-device parity test is
//! direct evidence the ported kernel computes the same blend and divergence the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each thread reads one [`WaterFlipMacQuery`] — a `pic` and `flip` velocity, a
//! blend factor `alpha`, the six staggered face velocities and the grid spacing
//! `dx` — and writes one [`WaterFlipMacResult`] holding the blended velocity and
//! the cell divergence. Both are pure multiply/add/clamp/divide with a single
//! ordered `dx <= EPS` guard; there is no loop and no transcendental.
//!
//! # What stays on the host
//!
//! The surrounding `FLIP`/`APIC` schedule — the trilinear transfer weights, the
//! affine (`APIC`) reconstruction, the pressure-solver choice, the sub-step
//! budget and the variable-length grid walk — stays on the host. The device
//! sees only the two fixed-width per-cell numerics, one query at a time, so a
//! storage buffer is never zero-sized.
//!
//! # Correctness model
//!
//! Both outputs thread through only multiply, add, subtract, `clamp` and a
//! single divide, so the `CPU` reference and the `GPU` agree to within a few
//! units in the last place. The parity test asserts a tolerance
//! (`abs_diff <= 1e-6` or `rel_diff <= 1e-5`), tight enough to catch a wrong
//! port (a dropped `1 -`, a swapped face pair, a missing `clamp`) yet loose
//! enough to admit a legal last-place rounding difference. The degenerate
//! `dx <= EPS` branch returns an exact `0`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `+ - * /`
//! and unsigned index arithmetic — with no `sin`, `cos`, `exp`, `log`, `pow`,
//! `sqrt`, `round` or `ceil`, and no `u64`/`u16`/`i64`/`f64`. It runs unmodified
//! on `Metal`, `Vulkan` and `DX12`.
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

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` `MAC`-grid blend-and-divergence kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden
/// [`blend_flip_pic`](prism_render_architecture::water::flip::blend_flip_pic)
/// and
/// [`cell_divergence`](prism_render_architecture::water::flip::cell_divergence)
/// closed forms; see the module documentation for the algorithm.
const WATER_FLIP_MAC_WGSL: &str = r#"
// MAC-grid FLIP/PIC twin: one thread blends one PIC/FLIP velocity pair and
// computes one MAC cell's divergence, mirroring the CPU golden
// `water::flip::{blend_flip_pic, cell_divergence}` with only clamp and
// + - * /. It owns no transfer weights, no affine reconstruction and no grid
// walk; those stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::flip；无第三方引擎
// 源码或衍生代码。

// Grid-spacing floor below which the divergence is inert, matching the golden
// `water::EPS`.
const EPS: f32 = 1e-6;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // The PIC velocity components.
    pic_x: f32,
    pic_y: f32,
    pic_z: f32,
    // The FLIP velocity components.
    flip_x: f32,
    flip_y: f32,
    flip_z: f32,
    // The FLIP/PIC blend factor, clamped to 0..=1 on device.
    alpha: f32,
    // The six staggered MAC face velocities.
    x_pos: f32,
    x_neg: f32,
    y_pos: f32,
    y_neg: f32,
    z_pos: f32,
    z_neg: f32,
    // The MAC grid cell edge length.
    dx: f32,
    pad0: f32,
    pad1: f32,
}

struct Result {
    // The blended velocity (1 - a) * pic + a * flip.
    blend_x: f32,
    blend_y: f32,
    blend_z: f32,
    // The discrete cell divergence.
    divergence: f32,
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

    // Blend: alpha clamped to 0..=1, then a componentwise lerp from pic to
    // flip, matching pic.scale(1 - a).add(flip.scale(a)).
    let a = clamp(q.alpha, 0.0, 1.0);
    let one_minus_a = 1.0 - a;

    var out: Result;
    out.blend_x = q.pic_x * one_minus_a + q.flip_x * a;
    out.blend_y = q.pic_y * one_minus_a + q.flip_y * a;
    out.blend_z = q.pic_z * one_minus_a + q.flip_z * a;

    // Divergence: a non-positive spacing is inert (exact zero), mirroring the
    // golden `if dx <= EPS { return 0.0 }` branch with an ordered compare.
    var div: f32 = 0.0;
    if (q.dx <= EPS) {
        div = 0.0;
    } else {
        let sum = (q.x_pos - q.x_neg) + (q.y_pos - q.y_neg) + (q.z_pos - q.z_neg);
        div = sum / q.dx;
    }
    out.divergence = div;

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_FLIP_MAC_WGSL`].
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
/// the `pic` and `flip` velocity components, the blend factor, the six staggered
/// face velocities and the grid spacing, plus two pad words to a `64`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// `PIC` velocity `x`.
    pic_x: f32,
    /// `PIC` velocity `y`.
    pic_y: f32,
    /// `PIC` velocity `z`.
    pic_z: f32,
    /// `FLIP` velocity `x`.
    flip_x: f32,
    /// `FLIP` velocity `y`.
    flip_y: f32,
    /// `FLIP` velocity `z`.
    flip_z: f32,
    /// `FLIP`/`PIC` blend factor.
    alpha: f32,
    /// Velocity through the `+x` face.
    x_pos: f32,
    /// Velocity through the `-x` face.
    x_neg: f32,
    /// Velocity through the `+y` face.
    y_pos: f32,
    /// Velocity through the `-y` face.
    y_neg: f32,
    /// Velocity through the `+z` face.
    z_pos: f32,
    /// Velocity through the `-z` face.
    z_neg: f32,
    /// `MAC` grid cell edge length.
    dx: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the blended velocity components and the cell divergence, a `16`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Blended velocity `x`.
    blend_x: f32,
    /// Blended velocity `y`.
    blend_y: f32,
    /// Blended velocity `z`.
    blend_z: f32,
    /// Discrete cell divergence.
    divergence: f32,
}

/// One query for the `MAC`-grid blend-and-divergence twin.
///
/// `pic` and `flip` are the two candidate velocities, `alpha` the `FLIP`/`PIC`
/// blend factor (clamped to `0..=1` by the kernel), `faces` the six staggered
/// `MAC` face velocities ordered `[x_pos, x_neg, y_pos, y_neg, z_pos, z_neg]`,
/// and `dx` the grid cell edge length. The host owns the surrounding
/// `FLIP`/`APIC` schedule and enqueues one [`WaterFlipMacQuery`] per cell or
/// particle it needs resolved.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterFlipMacQuery {
    /// The `PIC` velocity components `[x, y, z]`.
    pub pic: [f32; 3],
    /// The `FLIP` velocity components `[x, y, z]`.
    pub flip: [f32; 3],
    /// The `FLIP`/`PIC` blend factor, clamped to `0..=1` by the kernel.
    pub alpha: f32,
    /// The six staggered face velocities
    /// `[x_pos, x_neg, y_pos, y_neg, z_pos, z_neg]`.
    pub faces: [f32; 6],
    /// The `MAC` grid cell edge length `dx`.
    pub dx: f32,
}

impl WaterFlipMacQuery {
    /// Builds a query from a `pic` and `flip` velocity, a blend factor, the six
    /// staggered `faces` and the grid spacing `dx`.
    #[must_use]
    pub const fn new(
        pic: [f32; 3],
        flip: [f32; 3],
        alpha: f32,
        faces: [f32; 6],
        dx: f32,
    ) -> WaterFlipMacQuery {
        WaterFlipMacQuery {
            pic,
            flip,
            alpha,
            faces,
            dx,
        }
    }
}

/// One resolved query of the `MAC`-grid twin: the blended velocity and the cell
/// divergence.
///
/// `blended` is `(1 - a) * pic + a * flip` with `a` the clamped blend factor,
/// mirroring
/// [`blend_flip_pic`](prism_render_architecture::water::flip::blend_flip_pic);
/// `divergence` is the discrete
/// [`cell_divergence`](prism_render_architecture::water::flip::cell_divergence),
/// `0` for a non-positive spacing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterFlipMacResult {
    /// The blended velocity components `[x, y, z]`.
    pub blended: [f32; 3],
    /// The discrete cell divergence.
    pub divergence: f32,
}

/// Encodes one [`WaterFlipMacQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &WaterFlipMacQuery) -> GpuQuery {
    GpuQuery {
        pic_x: q.pic[0],
        pic_y: q.pic[1],
        pic_z: q.pic[2],
        flip_x: q.flip[0],
        flip_y: q.flip[1],
        flip_z: q.flip[2],
        alpha: q.alpha,
        x_pos: q.faces[0],
        x_neg: q.faces[1],
        y_pos: q.faces[2],
        y_neg: q.faces[3],
        z_pos: q.faces[4],
        z_neg: q.faces[5],
        dx: q.dx,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`WaterFlipMacResult`].
fn decode_result(raw: &GpuResult) -> WaterFlipMacResult {
    WaterFlipMacResult {
        blended: [raw.blend_x, raw.blend_y, raw.blend_z],
        divergence: raw.divergence,
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

/// A compiled, reusable `MAC`-grid blend-and-divergence compute pipeline,
/// twinning the `CPU` golden
/// [`blend_flip_pic`](prism_render_architecture::water::flip::blend_flip_pic)
/// and
/// [`cell_divergence`](prism_render_architecture::water::flip::cell_divergence).
pub struct GpuWaterFlipMac {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterFlipMac {
    /// Compiles the `MAC`-grid blend-and-divergence kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterFlipMac {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_flip_mac"),
            source: ShaderSource::Wgsl(WATER_FLIP_MAC_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_flip_mac_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_flip_mac_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_flip_mac_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterFlipMac {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`WaterFlipMacResult`]
    /// per input, in order.
    ///
    /// The blended velocity and the divergence match the reference to within the
    /// tolerance documented on this module. An empty `queries` batch returns an
    /// empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterFlipMacQuery],
    ) -> Vec<WaterFlipMacResult> {
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
            label: Some("prism_volumetric_water_flip_mac_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_flip_mac_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_flip_mac_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_flip_mac_bind_group"),
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
            label: Some("prism_volumetric_water_flip_mac_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_flip_mac_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_flip_mac_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
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
