//! `wgpu` compute twin of the `CPU` golden combustion-coupling pure functions
//! ([`step_combustion`](prism_render_architecture::particle::fluid::step_combustion),
//! [`buoyancy_force`](prism_render_architecture::particle::fluid::buoyancy_force),
//! and
//! [`heat_haze_distortion`](prism_render_architecture::particle::fluid::heat_haze_distortion),
//! design §10, §17).
//!
//! The stable-fluids pipeline carries a three-channel combustion state
//! (temperature / fuel / smoke) per voxel. Each frame the solver advances that
//! state, derives the buoyancy body force that lifts hot gas and sinks dense
//! soot, and forms a screen-space heat-haze refraction offset from the local
//! temperature gradient. The `CPU` golden
//! [`fluid`](prism_render_architecture::particle::fluid) module owns that math;
//! [`GpuFluidCombustion`] is the on-device twin, validated against that
//! reference so a passing real-device parity test is direct evidence the ported
//! kernel evaluates the same update, not merely that its shader compiles.
//!
//! # Scope
//!
//! This twin deliberately reproduces only the three multiply-add pure functions
//! above. The sibling
//! [`blackbody_emission`](prism_render_architecture::particle::fluid::blackbody_emission)
//! is intentionally excluded: although the golden implementation itself is
//! polynomial, this twin keeps a tight, verifiable surface around the three
//! functions the solver's force and advection passes consume per voxel.
//!
//! # Algorithm
//!
//! One thread owns one voxel. For each element the kernel:
//!
//! 1. Advances the combustion state exactly as
//!    [`step_combustion`](prism_render_architecture::particle::fluid::step_combustion):
//!    when the voxel is at or above the ignition temperature and still has
//!    fuel, it burns `burn_rate*dt` (clamped to the remaining fuel), converting
//!    the burned amount into smoke and heat, then relaxes temperature toward
//!    ambient with the linear `Newton`-style cooling term. Multiply-add only.
//! 2. Forms the buoyancy force from the *input* state, matching
//!    [`buoyancy_force`](prism_render_architecture::particle::fluid::buoyancy_force):
//!    `lift = (temperature − ambient)*alpha − smoke*beta`, returned as
//!    `vec3(0, lift, 0)`.
//! 3. Scales the temperature gradient by `strength`, matching
//!    [`heat_haze_distortion`](prism_render_architecture::particle::fluid::heat_haze_distortion).
//!
//! The buoyancy term reads the per-element input state (not the stepped state)
//! so it mirrors the golden `buoyancy_force` call applied to the same input.
//! The arithmetic order of every expression matches the reference term for
//! term.
//!
//! # Degenerate inputs
//!
//! An empty query dispatches nothing and returns empty output vectors, matching
//! the host guard. The ignition branch and the burn clamp depend only on the
//! uploaded `f32` inputs (never on previously computed values), so the two sides
//! take the same branch for bit-identical inputs; the saturating `burn > fuel`
//! clamp is reproduced exactly.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — integer index
//! arithmetic plus `+ − × ÷` and comparison on `f32` — with no `sin`, `cos`,
//! `exp`, `log`, `pow` or optional device feature, so it runs unmodified on
//! Metal, Vulkan and `DX12`.
//!
//! # Correctness model
//!
//! The update contains no transcendental call, so `CPU` and `GPU` evaluate the
//! same closed-form algebra. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few `ULP`. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`), tight enough to catch a genuinely
//! wrong port (a swapped term, a dropped clamp, a missing cooling step) yet
//! loose enough to admit legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid` 的燃烧耦合
//! 纯函数 `step_combustion` / `buoyancy_force` / `heat_haze_distortion` 加 `wgpu`
//! 计算下发；无第三方引擎源码或衍生代码。

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::fluid::{CombustionParams, CombustionState};
use prism_render_architecture::particle::Vec3;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// The number of threads per workgroup. `64` is a portable, warp-friendly size
/// used across this crate's kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` combustion-coupling kernel, embedded inline so the
/// twin ships as a single source file. Mirrors the `CPU` pure functions exactly;
/// see the module documentation for the algorithm.
const FLUID_COMBUSTION_WGSL: &str = r#"
// Combustion-coupling twin: one thread per voxel advances the combustion state
// (step_combustion), forms the buoyancy body force from the input state
// (buoyancy_force), and scales the temperature gradient for the heat-haze
// offset (heat_haze_distortion). It mirrors the CPU golden
// `particle::fluid` pure functions, uses only the portable core-WGSL subset
// (integer index math plus + - * / and comparison on f32), and takes no
// optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture 的燃烧耦合纯函数；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Temperature at or above which fuel ignites.
    ignition_temperature: f32,
    // Fuel burned per unit time while ignited.
    burn_rate: f32,
    // Smoke produced per unit of burned fuel.
    smoke_yield: f32,
    // Temperature added per unit of burned fuel.
    heat_yield: f32,
    // Linear cooling coefficient toward ambient.
    cooling_rate: f32,
    // Ambient temperature the field relaxes toward.
    ambient_temperature: f32,
    // Upward buoyancy per degree above ambient.
    buoyancy_alpha: f32,
    // Downward drag per unit of smoke density.
    buoyancy_beta: f32,
    // Integration step shared by every voxel this dispatch.
    dt: f32,
    // Number of live elements; threads past this return early.
    count: u32,
    // Padding to keep the uniform a multiple of 16 bytes.
    pad0: u32,
    pad1: u32,
}

// Per-voxel input: the combustion state plus the heat-haze gradient/strength.
struct Element {
    temperature: f32,
    fuel: f32,
    smoke: f32,
    grad_x: f32,
    grad_y: f32,
    grad_z: f32,
    strength: f32,
    pad: f32,
}

// Per-voxel output: the stepped state, the buoyancy force, the haze offset.
struct OutElement {
    temperature: f32,
    fuel: f32,
    smoke: f32,
    pad0: f32,
    buoy_x: f32,
    buoy_y: f32,
    buoy_z: f32,
    pad1: f32,
    haze_x: f32,
    haze_y: f32,
    haze_z: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> elements: array<Element>;
@group(0) @binding(2) var<storage, read_write> out_elements: array<OutElement>;

@compute @workgroup_size(64)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let e = elements[idx];

    // Advance the combustion state (step_combustion), term for term.
    var temperature = e.temperature;
    var fuel = e.fuel;
    var smoke = e.smoke;
    if (temperature >= params.ignition_temperature && fuel > 0.0) {
        var burned = params.burn_rate * params.dt;
        if (burned > fuel) {
            burned = fuel;
        }
        fuel = fuel - burned;
        smoke = smoke + burned * params.smoke_yield;
        temperature = temperature + burned * params.heat_yield;
    }
    let cooled = params.cooling_rate * params.dt
        * (temperature - params.ambient_temperature);
    temperature = temperature - cooled;

    // Buoyancy from the input state (buoyancy_force), matching term order.
    let lift = (e.temperature - params.ambient_temperature) * params.buoyancy_alpha
        - e.smoke * params.buoyancy_beta;

    // Heat-haze offset: temperature gradient scaled by strength.
    let haze_x = e.grad_x * e.strength;
    let haze_y = e.grad_y * e.strength;
    let haze_z = e.grad_z * e.strength;

    var out: OutElement;
    out.temperature = temperature;
    out.fuel = fuel;
    out.smoke = smoke;
    out.pad0 = 0.0;
    out.buoy_x = 0.0;
    out.buoy_y = lift;
    out.buoy_z = 0.0;
    out.pad1 = 0.0;
    out.haze_x = haze_x;
    out.haze_y = haze_y;
    out.haze_z = haze_z;
    out.pad2 = 0.0;
    out_elements[idx] = out;
}
"#;

/// Uniform parameters for one combustion dispatch. `repr(C)` `std430` layout
/// matching `Params` in [`FLUID_COMBUSTION_WGSL`]: the eight
/// [`CombustionParams`] coefficients, the shared `dt`, the live `count`, then
/// two pad words — `48` bytes total with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Temperature at or above which fuel ignites.
    ignition_temperature: f32,
    /// Fuel burned per unit time while ignited.
    burn_rate: f32,
    /// Smoke produced per unit of burned fuel.
    smoke_yield: f32,
    /// Temperature added per unit of burned fuel.
    heat_yield: f32,
    /// Linear cooling coefficient toward ambient.
    cooling_rate: f32,
    /// Ambient temperature the field relaxes toward.
    ambient_temperature: f32,
    /// Upward buoyancy per degree above ambient.
    buoyancy_alpha: f32,
    /// Downward drag per unit of smoke density.
    buoyancy_beta: f32,
    /// Integration step shared by every voxel this dispatch.
    dt: f32,
    /// Number of live elements.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One combustion voxel as uploaded. `32`-byte `std430` stride matching
/// `Element` in the shader: the three combustion channels, the three gradient
/// components, the haze `strength`, then one pad lane held at zero.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuElement {
    /// Temperature channel.
    temperature: f32,
    /// Remaining fuel.
    fuel: f32,
    /// Accumulated smoke density.
    smoke: f32,
    /// Temperature-gradient `x` component.
    grad_x: f32,
    /// Temperature-gradient `y` component.
    grad_y: f32,
    /// Temperature-gradient `z` component.
    grad_z: f32,
    /// Heat-haze strength scalar.
    strength: f32,
    /// Padding lane, held at zero so it never perturbs the arithmetic.
    pad: f32,
}

/// One combustion voxel as read back. `48`-byte `std430` stride matching
/// `OutElement` in the shader: the stepped state, the buoyancy force, the haze
/// offset, each padded to a `16`-byte lane boundary.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuOutElement {
    /// Stepped temperature.
    temperature: f32,
    /// Stepped fuel.
    fuel: f32,
    /// Stepped smoke.
    smoke: f32,
    /// Padding lane.
    pad0: f32,
    /// Buoyancy force `x` component (always zero).
    buoy_x: f32,
    /// Buoyancy force `y` component (the lift).
    buoy_y: f32,
    /// Buoyancy force `z` component (always zero).
    buoy_z: f32,
    /// Padding lane.
    pad1: f32,
    /// Heat-haze offset `x` component.
    haze_x: f32,
    /// Heat-haze offset `y` component.
    haze_y: f32,
    /// Heat-haze offset `z` component.
    haze_z: f32,
    /// Padding lane.
    pad2: f32,
}

/// A batch of combustion voxels to advance in one dispatch.
///
/// Every field vector is parallel and indexed by voxel: `states[i]` is advanced
/// by [`step_combustion`](prism_render_architecture::particle::fluid::step_combustion),
/// `gradients[i]` and `strengths[i]` feed
/// [`heat_haze_distortion`](prism_render_architecture::particle::fluid::heat_haze_distortion),
/// and the shared `params`/`dt` apply to all. The three vectors must share a
/// length.
///
/// Provenance: 本模块新建的 GPU 批量查询类型；无第三方引擎源码或衍生代码。
#[derive(Clone, Debug, PartialEq)]
pub struct GpuFluidCombustionQuery {
    /// The per-voxel combustion states to advance.
    pub states: Vec<CombustionState>,
    /// The per-voxel temperature gradients for the heat-haze offset.
    pub gradients: Vec<Vec3>,
    /// The per-voxel heat-haze strength scalars.
    pub strengths: Vec<f32>,
    /// The combustion coupling coefficients shared by every voxel.
    pub params: CombustionParams,
    /// The integration step shared by every voxel.
    pub dt: f32,
}

/// The outcome of a `GPU` combustion dispatch, one entry per input voxel.
///
/// Provenance: 本模块新建的 GPU 批量结果类型；无第三方引擎源码或衍生代码。
#[derive(Clone, Debug, PartialEq)]
pub struct GpuFluidCombustionResult {
    /// The advanced combustion states, matching the golden
    /// [`step_combustion`](prism_render_architecture::particle::fluid::step_combustion).
    pub states: Vec<CombustionState>,
    /// The buoyancy body forces, matching the golden
    /// [`buoyancy_force`](prism_render_architecture::particle::fluid::buoyancy_force).
    pub buoyancy: Vec<Vec3>,
    /// The heat-haze refraction offsets, matching the golden
    /// [`heat_haze_distortion`](prism_render_architecture::particle::fluid::heat_haze_distortion).
    pub haze: Vec<Vec3>,
}

/// A compiled, reusable combustion-coupling pipeline.
///
/// Provenance: 本模块新建的 GPU 管线封装类型；无第三方引擎源码或衍生代码。
pub struct GpuFluidCombustion {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuFluidCombustion {
    /// Compiles the combustion kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 本模块新建的管线构造；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFluidCombustion {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_fluid_combustion"),
            source: ShaderSource::Wgsl(FLUID_COMBUSTION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_fluid_combustion_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_fluid_combustion_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_fluid_combustion_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("step"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuFluidCombustion {
            module,
            layout,
            pipeline,
        }
    }

    /// Advances every voxel in `query` on the device and reads the results back.
    ///
    /// For voxel `i` the returned `states[i]`, `buoyancy[i]` and `haze[i]` equal
    /// [`step_combustion`](prism_render_architecture::particle::fluid::step_combustion)`(query.states[i], query.params, query.dt)`,
    /// [`buoyancy_force`](prism_render_architecture::particle::fluid::buoyancy_force)`(query.states[i], query.params)`,
    /// and
    /// [`heat_haze_distortion`](prism_render_architecture::particle::fluid::heat_haze_distortion)`(query.gradients[i], query.strengths[i])`
    /// to within the tolerance documented on this module. An empty batch
    /// dispatches nothing and returns empty vectors.
    ///
    /// # Panics
    ///
    /// Panics if `query.states`, `query.gradients` and `query.strengths` do not
    /// share a length, since the three are parallel per-voxel inputs.
    ///
    /// Provenance: 本模块新建的下发/回读流程；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn step(
        &self,
        ctx: &GpuContext,
        query: &GpuFluidCombustionQuery,
    ) -> GpuFluidCombustionResult {
        let count = query.states.len();
        assert_eq!(
            count,
            query.gradients.len(),
            "states and gradients must share a length"
        );
        assert_eq!(
            count,
            query.strengths.len(),
            "states and strengths must share a length"
        );
        if count == 0 {
            return GpuFluidCombustionResult {
                states: Vec::new(),
                buoyancy: Vec::new(),
                haze: Vec::new(),
            };
        }

        let device = ctx.device();
        let packed: Vec<GpuElement> = (0..count)
            .map(|i| {
                let s = query.states[i];
                let g = query.gradients[i];
                GpuElement {
                    temperature: s.temperature,
                    fuel: s.fuel,
                    smoke: s.smoke,
                    grad_x: g.x,
                    grad_y: g.y,
                    grad_z: g.z,
                    strength: query.strengths[i],
                    pad: 0.0,
                }
            })
            .collect();
        let gpu_params = GpuParams {
            ignition_temperature: query.params.ignition_temperature,
            burn_rate: query.params.burn_rate,
            smoke_yield: query.params.smoke_yield,
            heat_yield: query.params.heat_yield,
            cooling_rate: query.params.cooling_rate,
            ambient_temperature: query.params.ambient_temperature,
            buoyancy_alpha: query.params.buoyancy_alpha,
            buoyancy_beta: query.params.buoyancy_beta,
            dt: query.dt,
            count: count as u32,
            pad0: 0,
            pad1: 0,
        };

        let out_bytes = (count * size_of::<GpuOutElement>()) as u64;
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fluid_combustion_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let element_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fluid_combustion_elements"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fluid_combustion_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fluid_combustion_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_fluid_combustion_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: element_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_fluid_combustion_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_fluid_combustion_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
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
        let gpu_out = bytemuck::cast_slice::<u8, GpuOutElement>(&view).to_vec();
        drop(view);
        stage.unmap();
        debug_assert_eq!(gpu_out.len(), count);

        let mut states = Vec::with_capacity(count);
        let mut buoyancy = Vec::with_capacity(count);
        let mut haze = Vec::with_capacity(count);
        for o in gpu_out {
            states.push(CombustionState::new(o.temperature, o.fuel, o.smoke));
            buoyancy.push(Vec3::new(o.buoy_x, o.buoy_y, o.buoy_z));
            haze.push(Vec3::new(o.haze_x, o.haze_y, o.haze_z));
        }
        GpuFluidCombustionResult {
            states,
            buoyancy,
            haze,
        }
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
