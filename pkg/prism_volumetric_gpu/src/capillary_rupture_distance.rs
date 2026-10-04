//! `wgpu` compute twin of the capillary-bridge rupture-distance closed form,
//! from the `CPU` golden
//! `prism_physics_core::collider::capillary_bridge`'s
//! `CapillaryBridgeModel::rupture_distance`.
//!
//! The rupture distance is the gap at which a liquid bridge between two grains
//! snaps, modelled by the standard wet-`DEM` closed form
//! `H_rupture = (1 + 0.5 * theta) * V^(1/3)`, where `theta` is the contact
//! angle and `V` the bridge liquid volume. This module ports that single
//! stateless closed form onto the device: one thread resolves one query, so a
//! passing real-device parity test is direct evidence the ported kernel
//! computes the same rupture distance the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `rupture_distance` for one bridge with
//! an explicit `contact_angle` and `liquid_volume`:
//!
//! * The pair is valid only when both inputs are finite, the contact angle is
//!   in `[0, pi/2]` and the volume is strictly positive — the same admissible
//!   domain the golden `CapillaryBridgeModel::new` enforces.
//! * When valid, `rupture = (1.0 + 0.5 * theta) * V^(1/3)`, evaluated in the
//!   golden operator order (the cube root first, then scaled by the angle
//!   factor).
//! * When invalid, `valid = 0` and `rupture = 0`.
//!
//! # Correctness model
//!
//! The golden computes the cube root in `f64` (`(V as f64).cbrt() as f32`),
//! whereas `WGSL` has no `f64` and no `cbrt`, so the twin evaluates
//! `pow(V, 1.0 / 3.0)` in `f32`. The two therefore agree only up to a small
//! numerical difference, which the relative tolerance absorbs: the valid
//! `rupture` scalar is compared with an `abs <= 1e-4 || rel <= 1e-3` tolerance
//! (`REL_FLOOR = 1e-6`). The discrete `valid` flag is compared exactly; the
//! parity test keeps the random validity decision well clear of its knees so
//! round-off cannot flip it.
//!
//! # Degenerate inputs
//!
//! A non-finite input, a contact angle outside `[0, pi/2]` or a volume `<= 0`
//! yields `valid = 0` with `rupture = 0`. The `pow` base is fed through a
//! `select` guard so the un-taken (invalid) branch never raises `pow` on a
//! non-positive base; when valid the volume is strictly positive, so the cube
//! root is well defined. An empty query batch short-circuits on the host with
//! no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `pow`,
//! `+ - *`, `select` and unsigned index arithmetic — with no `sin`, `cos`,
//! `tan`, `exp`, `log`, no `round`, no `f32` remainder and no `sqrt`, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. Finiteness is tested with the
//! ordered compare `abs(x) < 3.0e38` (which rejects both infinities and `NaN`)
//! rather than a bare `x == x`, and the angle range and positivity with ordered
//! `>=`, `<=` and `>`; there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::capillary_bridge`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` rupture-distance kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `CapillaryBridgeModel::rupture_distance`; see the module
/// documentation for the closed form.
const CAPILLARY_RUPTURE_DISTANCE_WGSL: &str = r#"
// Rupture-distance twin: one thread per query reproduces rupture_distance. It
// uses only the portable core-WGSL subset (abs, pow, + - *, select plus
// unsigned index math), takes no optional feature, and has no loop and no
// branch, so it provably terminates. Finiteness is an ordered abs < 3.0e38
// compare (rejecting infinities and NaN), the angle range an ordered
// [0, HALF_PI] compare and positivity an ordered > 0 compare, all fed to
// select. There is no f32 equality anywhere.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Contact angle theta (radians).
    contact_angle: f32,
    // Bridge liquid volume V.
    liquid_volume: f32,
    // Padding words to a 16-byte-friendly stride.
    pad0: f32,
    pad1: f32,
}

struct Result {
    // Rupture distance (1 + 0.5*theta) * V^(1/3) when valid, else 0.
    rupture: f32,
    // 1 when the inputs are finite, theta in [0, pi/2] and V > 0, else 0.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const FINITE_LIMIT: f32 = 3.0e38;
const HALF_PI: f32 = 1.5707964;
const ONE_THIRD: f32 = 0.33333334;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let theta = q.contact_angle;
    let vol = q.liquid_volume;

    // Finiteness via ordered abs < 3.0e38 (rejects +/-inf and NaN, since every
    // comparison with NaN is false). The admissible domain mirrors the golden
    // CapillaryBridgeModel::new gate: theta in [0, pi/2] and V strictly > 0.
    let finite = (abs(theta) < FINITE_LIMIT) && (abs(vol) < FINITE_LIMIT);
    let angle_ok = (theta >= 0.0) && (theta <= HALF_PI);
    let positive = vol > 0.0;
    let ok = finite && angle_ok && positive;

    // Guard the pow base so the un-taken (invalid) branch never raises pow on a
    // non-positive base; when ok the volume is strictly positive.
    let base = select(1.0, vol, ok);
    let cube_root = pow(base, ONE_THIRD);
    let r = (1.0 + 0.5 * theta) * cube_root;

    var out: Result;
    out.rupture = select(0.0, r, ok);
    out.valid = select(0u, 1u, ok);
    results[idx] = out;
}
"#;

/// `repr(C)` `std430` dispatch parameters: the query count plus padding to a
/// 16-byte uniform block.
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// The angle and volume are padded to `4` `f32` words (`16` bytes), aligned to
/// `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    contact_angle: f32,
    liquid_volume: f32,
    pad0: f32,
    pad1: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the rupture distance and the validity flag — `2` words (`8` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    rupture: f32,
    valid: u32,
}

/// One rupture-distance query: the contact angle and the bridge liquid volume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapillaryRuptureDistanceQuery {
    /// Contact angle `theta` (radians); admissible in `[0, pi/2]`.
    pub contact_angle: f32,
    /// Bridge liquid volume `V`; admissible when strictly positive.
    pub liquid_volume: f32,
}

impl CapillaryRuptureDistanceQuery {
    /// Builds a query from the contact angle and the liquid volume.
    #[must_use]
    pub fn new(contact_angle: f32, liquid_volume: f32) -> CapillaryRuptureDistanceQuery {
        CapillaryRuptureDistanceQuery {
            contact_angle,
            liquid_volume,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `CapillaryBridgeModel::rupture_distance` output for that bridge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapillaryRuptureDistanceResult {
    /// The rupture distance `(1 + 0.5 * theta) * V^(1/3)` when valid, else `0`.
    pub rupture: f32,
    /// `1` when the inputs are finite, `theta` in `[0, pi/2]` and `V > 0`, else
    /// `0`.
    pub valid: u32,
}

/// Encodes one [`CapillaryRuptureDistanceQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &CapillaryRuptureDistanceQuery) -> GpuQuery {
    GpuQuery {
        contact_angle: q.contact_angle,
        liquid_volume: q.liquid_volume,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`CapillaryRuptureDistanceResult`].
fn decode_result(raw: &GpuResult) -> CapillaryRuptureDistanceResult {
    CapillaryRuptureDistanceResult {
        rupture: raw.rupture,
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

/// A compiled, reusable rupture-distance compute pipeline, twinning the `CPU`
/// golden `CapillaryBridgeModel::rupture_distance`.
pub struct GpuCapillaryRuptureDistance {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCapillaryRuptureDistance {
    /// Compiles the rupture-distance kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCapillaryRuptureDistance {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_capillary_rupture_distance"),
            source: ShaderSource::Wgsl(CAPILLARY_RUPTURE_DISTANCE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_capillary_rupture_distance_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_capillary_rupture_distance_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_capillary_rupture_distance_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCapillaryRuptureDistance {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`CapillaryRuptureDistanceResult`] per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and the `rupture` scalar
    /// to the module's tolerance. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[CapillaryRuptureDistanceQuery],
    ) -> Vec<CapillaryRuptureDistanceResult> {
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
            label: Some("prism_volumetric_capillary_rupture_distance_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_capillary_rupture_distance_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_capillary_rupture_distance_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_capillary_rupture_distance_bind_group"),
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
            label: Some("prism_volumetric_capillary_rupture_distance_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_capillary_rupture_distance_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_capillary_rupture_distance_pass"),
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
