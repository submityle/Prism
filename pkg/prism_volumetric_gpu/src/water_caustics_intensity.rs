//! `wgpu` compute twin of the water caustic *intensity* functions
//! [`jacobian_caustic_gain`](prism_render_architecture::water::caustics::jacobian_caustic_gain),
//! [`project_caustic_intensity`](prism_render_architecture::water::caustics::project_caustic_intensity),
//! and
//! [`photon_splat_density`](prism_render_architecture::water::caustics::photon_splat_density)
//! that the water subsystem uses to shade refracted-light caustics.
//!
//! When light refracts through a wavy surface the refracted rays converge and
//! diverge, concentrating irradiance into the bright dancing caustic patterns.
//! The per-receiver *intensity* of those patterns is a small set of stateless,
//! closed-form, non-negative functions, so they port cleanly to the device: a
//! passing real-device parity run is direct evidence the ported kernel honors
//! the same gain clamp, projection, and splat-density estimate the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! One thread computes one caustic query, dispatched by an operation code:
//!
//! - `JacobianGain` mirrors
//!   [`jacobian_caustic_gain`](prism_render_architecture::water::caustics::jacobian_caustic_gain):
//!   `cap = max(max_gain, 0)`, `mag = abs(jacobian)`; a near-zero `mag`
//!   saturates to `cap`, otherwise the gain is `min(1 / mag, cap)`.
//! - `ProjectIntensity` mirrors
//!   [`project_caustic_intensity`](prism_render_architecture::water::caustics::project_caustic_intensity):
//!   `max(incident, 0)` times the Jacobian gain.
//! - `PhotonDensity` mirrors
//!   [`photon_splat_density`](prism_render_architecture::water::caustics::photon_splat_density):
//!   a degenerate radius returns `0`, otherwise
//!   `photon_count * max(photon_power, 0) / (PI * radius * radius)`.
//!
//! Only `+`, `*`, `/`, `abs`, `min`, `max`, `select`, and an unsigned-to-float
//! cast appear, so the kernel maps directly onto the portable core-`WGSL`
//! subset.
//!
//! # What stays on the host
//!
//! The route selection
//! ([`select_caustics`](prism_render_architecture::water::caustics::select_caustics))
//! and the [`CausticsMethod`](prism_render_architecture::water::caustics::CausticsMethod)
//! cost ranking are classification policy, not the per-receiver intensity
//! arithmetic this module targets, so they stay on the host. The actual photon
//! emission, refraction, and splat gathering that supply `photon_count` and the
//! ray-scene traversal that supplies `incident` are stateful, variable-length
//! host passes and are not twinned here.
//!
//! # Correctness model
//!
//! Each result is a short sequence of guarded divides and clamps with a single
//! reciprocal — no transcendental — so the `CPU` and `GPU` agree to within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+`, `*`, `/`,
//! `abs`, `min`, `max`, `select`, and `f32(u32)` — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `smoothstep`, and no
//! `round`. The degenerate-input legs are selected with `select`, so a divide
//! by zero is discarded rather than observed. Each thread performs a fixed,
//! bounded sequence of arithmetic, so the kernel provably terminates. No
//! optional device feature is required, so it runs unmodified across backends.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::caustics`；无第三方引擎源码或衍生代码。
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

/// Operation code selecting the
/// [`jacobian_caustic_gain`](prism_render_architecture::water::caustics::jacobian_caustic_gain)
/// route in [`WATER_CAUSTICS_INTENSITY_WGSL`].
const OP_JACOBIAN_GAIN: u32 = 0;

/// Operation code selecting the
/// [`project_caustic_intensity`](prism_render_architecture::water::caustics::project_caustic_intensity)
/// route in [`WATER_CAUSTICS_INTENSITY_WGSL`].
const OP_PROJECT_INTENSITY: u32 = 1;

/// Operation code selecting the
/// [`photon_splat_density`](prism_render_architecture::water::caustics::photon_splat_density)
/// route in [`WATER_CAUSTICS_INTENSITY_WGSL`].
const OP_PHOTON_DENSITY: u32 = 2;

/// The portable core-`WGSL` caustic-intensity kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden functions in
/// [`prism_render_architecture::water::caustics`]; see the module documentation
/// for the algorithm.
const WATER_CAUSTICS_INTENSITY_WGSL: &str = r#"
// Water caustic intensity twin: one thread evaluates one caustic query, chosen
// by an operation code, mirroring the CPU golden `jacobian_caustic_gain`,
// `project_caustic_intensity`, and `photon_splat_density` with only + * /
// abs min max select on f32 (plus f32(u32) for the photon count).
//
// Provenance: 孪生自本仓 prism_render_architecture::water::caustics；无第三方引擎源码或衍生代码。

// Shared small epsilon guarding the reciprocal and the splat-disk divide,
// matching the golden `water::EPS`.
const EPS: f32 = 1.0e-6;
// Matches the golden `water::PI` (`core::f32::consts::PI`); the long decimal
// rounds to the same f32 bit pattern.
const PI: f32 = 3.14159265358979;

const OP_JACOBIAN_GAIN: u32 = 0u;
const OP_PROJECT_INTENSITY: u32 = 1u;
const OP_PHOTON_DENSITY: u32 = 2u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Operation code: 0 gain, 1 projection, 2 photon density.
    op: u32,
    // Photon count for the photon-density route.
    photon_count: u32,
    // Refracted-ray Jacobian determinant.
    jacobian: f32,
    // Upper clamp on the Jacobian gain.
    max_gain: f32,
    // Incident irradiance for the projection route.
    incident: f32,
    // Per-photon power for the photon-density route.
    photon_power: f32,
    // Splat gather radius for the photon-density route.
    radius: f32,
    pad0: f32,
}

struct Result {
    // Resolved caustic intensity (non-negative).
    value: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Reciprocal area distortion clamped to `cap`; a near-singular Jacobian
// saturates to `cap` instead of dividing by zero, mirroring the golden.
fn jacobian_caustic_gain(jacobian: f32, max_gain: f32) -> f32 {
    let cap = max(max_gain, 0.0);
    let mag = abs(jacobian);
    let gain = min(1.0 / mag, cap);
    return select(gain, cap, mag <= EPS);
}

// Incident irradiance scaled by the Jacobian gain, non-negative and bounded.
fn project_caustic_intensity(incident: f32, jacobian: f32, max_gain: f32) -> f32 {
    return max(incident, 0.0) * jacobian_caustic_gain(jacobian, max_gain);
}

// Photon power inside the splat disk divided by the disk area; a degenerate
// radius returns 0 rather than dividing by zero.
fn photon_splat_density(photon_count: u32, photon_power: f32, radius: f32) -> f32 {
    let r = max(radius, 0.0);
    let area = PI * r * r;
    let density = f32(photon_count) * max(photon_power, 0.0) / area;
    return select(density, 0.0, r <= EPS);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    var value: f32 = 0.0;
    if (q.op == OP_JACOBIAN_GAIN) {
        value = jacobian_caustic_gain(q.jacobian, q.max_gain);
    } else if (q.op == OP_PROJECT_INTENSITY) {
        value = project_caustic_intensity(q.incident, q.jacobian, q.max_gain);
    } else {
        value = photon_splat_density(q.photon_count, q.photon_power, q.radius);
    }
    var out: Result;
    out.value = value;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    out.pad2 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the operation count plus three pad words
/// to fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_CAUSTICS_INTENSITY_WGSL`].
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

/// `repr(C)` `std430` layout of one caustic query, matching the `WGSL` `Query`
/// struct: the operation code and photon count followed by the six `f32`
/// operands (one unused as padding).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Operation code: `0` gain, `1` projection, `2` photon density.
    op: u32,
    /// Photon count for the photon-density route.
    photon_count: u32,
    /// Refracted-ray Jacobian determinant.
    jacobian: f32,
    /// Upper clamp on the Jacobian gain.
    max_gain: f32,
    /// Incident irradiance for the projection route.
    incident: f32,
    /// Per-photon power for the photon-density route.
    photon_power: f32,
    /// Splat gather radius for the photon-density route.
    radius: f32,
    /// Padding word.
    pad0: f32,
}

/// `repr(C)` `std430` layout of one caustic result, matching the `WGSL`
/// `Result` struct: the resolved intensity plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Resolved caustic intensity (non-negative).
    value: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
    /// Padding word.
    pad2: f32,
}

/// One caustic-intensity query to run on the device, mirroring the golden
/// functions in [`prism_render_architecture::water::caustics`].
///
/// Each variant selects one closed-form caustic route; see the module
/// documentation for the arithmetic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WaterCausticsIntensityQuery {
    /// Jacobian caustic gain from the refracted-ray `jacobian` determinant,
    /// clamped to `max_gain`, mirroring
    /// [`jacobian_caustic_gain`](prism_render_architecture::water::caustics::jacobian_caustic_gain).
    JacobianGain {
        /// Refracted-ray Jacobian determinant.
        jacobian: f32,
        /// Upper clamp on the gain.
        max_gain: f32,
    },
    /// Projected caustic irradiance from `incident` light, the `jacobian`
    /// determinant, and the `max_gain` clamp, mirroring
    /// [`project_caustic_intensity`](prism_render_architecture::water::caustics::project_caustic_intensity).
    ProjectIntensity {
        /// Incident irradiance at the receiver.
        incident: f32,
        /// Refracted-ray Jacobian determinant.
        jacobian: f32,
        /// Upper clamp on the gain.
        max_gain: f32,
    },
    /// Photon-map irradiance estimate from `photon_count` photons each carrying
    /// `photon_power`, gathered inside `radius`, mirroring
    /// [`photon_splat_density`](prism_render_architecture::water::caustics::photon_splat_density).
    PhotonDensity {
        /// Number of photons gathered in the splat disk.
        photon_count: u32,
        /// Per-photon power.
        photon_power: f32,
        /// Splat gather radius.
        radius: f32,
    },
}

/// One resolved caustic-intensity result, mirroring the golden intensity
/// (always non-negative).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterCausticsIntensityResult {
    /// Resolved caustic intensity.
    pub value: f32,
}

/// Encodes one [`WaterCausticsIntensityQuery`] into its `std430` [`GpuQuery`]
/// slot, zero-filling the operands the chosen route does not read.
fn encode_query(q: &WaterCausticsIntensityQuery) -> GpuQuery {
    match *q {
        WaterCausticsIntensityQuery::JacobianGain { jacobian, max_gain } => GpuQuery {
            op: OP_JACOBIAN_GAIN,
            photon_count: 0,
            jacobian,
            max_gain,
            incident: 0.0,
            photon_power: 0.0,
            radius: 0.0,
            pad0: 0.0,
        },
        WaterCausticsIntensityQuery::ProjectIntensity {
            incident,
            jacobian,
            max_gain,
        } => GpuQuery {
            op: OP_PROJECT_INTENSITY,
            photon_count: 0,
            jacobian,
            max_gain,
            incident,
            photon_power: 0.0,
            radius: 0.0,
            pad0: 0.0,
        },
        WaterCausticsIntensityQuery::PhotonDensity {
            photon_count,
            photon_power,
            radius,
        } => GpuQuery {
            op: OP_PHOTON_DENSITY,
            photon_count,
            jacobian: 0.0,
            max_gain: 0.0,
            incident: 0.0,
            photon_power,
            radius,
            pad0: 0.0,
        },
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`WaterCausticsIntensityResult`].
fn decode_result(raw: &GpuResult) -> WaterCausticsIntensityResult {
    WaterCausticsIntensityResult { value: raw.value }
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

/// A compiled, reusable caustic-intensity compute pipeline, twinning the
/// numeric core of the `CPU` golden functions in
/// [`prism_render_architecture::water::caustics`].
pub struct GpuWaterCausticsIntensity {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterCausticsIntensity {
    /// Compiles the caustic-intensity kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterCausticsIntensity {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_caustics_intensity"),
            source: ShaderSource::Wgsl(WATER_CAUSTICS_INTENSITY_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_caustics_intensity_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_caustics_intensity_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_caustics_intensity_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterCausticsIntensity {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs every query in `queries` and returns one
    /// [`WaterCausticsIntensityResult`] per input, in order.
    ///
    /// The outputs match the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterCausticsIntensityQuery],
    ) -> Vec<WaterCausticsIntensityResult> {
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
            label: Some("prism_volumetric_water_caustics_intensity_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_caustics_intensity_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_caustics_intensity_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_caustics_intensity_bind_group"),
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
            label: Some("prism_volumetric_water_caustics_intensity_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_caustics_intensity_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_caustics_intensity_pass"),
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
