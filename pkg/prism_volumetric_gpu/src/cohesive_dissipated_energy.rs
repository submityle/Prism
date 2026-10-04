//! `wgpu` compute twin of the bilinear cohesive-zone dissipated-energy closed
//! form, from the `CPU` golden `prism_physics_core::collider::cohesive_zone`'s
//! `dissipated_energy` combined with the `CohesiveModel::new` derived
//! quantities.
//!
//! A cohesive zone loses cohesion gradually as an interface separates. The
//! irreversible energy per unit area dissipated by decohesion for a history
//! variable `κ` is the area between the bilinear envelope and the current
//! secant unloading line; it ranges from `0` at `κ <= δ₀` to the full fracture
//! energy `G_c` at `κ >= δ_f`. This module ports that single stateless closed
//! form onto the device: one thread resolves one query, so a passing
//! real-device parity test is direct evidence the ported kernel computes the
//! same dissipation the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel first derives the model's onset and final
//! separations exactly as `CohesiveModel::new` does, then evaluates
//! `dissipated_energy`, in the golden operator order:
//!
//! * `onset = strength / stiffness`.
//! * `final = 2 * fracture_energy / strength`.
//! * `energy = 0` when `κ <= onset`.
//! * `energy = fracture_energy` when `κ >= final`.
//! * `energy = 0.5 * strength * final * (κ - onset) / (final - onset)`
//!   otherwise.
//!
//! The reference model also carries a shear mode-mixity weight `β`; it does not
//! enter the dissipated-energy closed form, so this twin omits it entirely.
//!
//! # Correctness model
//!
//! The softening-branch arithmetic (two derived divisions, a multiply chain and
//! a division) threads through operators a `GPU` may contract, so `CPU` and
//! `GPU` are not necessarily bit-exact; the `energy` scalar is compared with an
//! `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`). The discrete
//! `valid` flag is compared exactly.
//!
//! # Degenerate inputs
//!
//! A query is valid only when `stiffness`, `strength`, `fracture_energy` and
//! `kappa` are all finite, when `stiffness`, `strength` and `fracture_energy`
//! are strictly positive, and when the derived softening branch exists
//! (`final > onset`). An invalid query yields `valid = 0` with `energy = 0`.
//! The onset and final divisors (`stiffness`, `strength`) and the softening
//! divisor (`final - onset`) are each guarded by a `select` that substitutes
//! `1` when the query is not valid, so no `inf`/`NaN` survives into the
//! discarded branch. An empty query batch short-circuits on the host with no
//! dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `select`,
//! `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`, `tan`,
//! `exp`, `log`, `pow`, `sqrt`, no `round` and no `f32` remainder, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. Finiteness is tested with the
//! ordered compare `abs(x) < 3.0e38` (which rejects both infinities and `NaN`)
//! rather than a bare `x == x`; there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::cohesive_zone`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` cohesive dissipated-energy kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden `CohesiveModel::new` derivation plus
/// `dissipated_energy`; see the module documentation for the closed form.
const COHESIVE_DISSIPATED_ENERGY_WGSL: &str = r#"
// Cohesive dissipated-energy twin: one thread per query derives onset/final and
// evaluates the bilinear dissipation. It uses only the portable core-WGSL
// subset (abs, select, + - * / plus unsigned index math), takes no optional
// feature, and has no loop and no branch, so it provably terminates.
// Finiteness is an ordered abs < 3.0e38 compare (rejecting infinities and NaN)
// fed to select; no bare f32 equality anywhere. The divisors are select-guarded
// so no inf/NaN survives into the discarded invalid branch.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Penalty stiffness K.
    stiffness: f32,
    // Peak traction (strength) sigma_c.
    strength: f32,
    // Fracture energy G_c.
    fracture_energy: f32,
    // History variable kappa (largest effective separation reached).
    kappa: f32,
}

struct Result {
    // Dissipated energy per unit area when valid, else 0.
    energy: f32,
    // 1 when the query is finite, positive and has a softening branch, else 0.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const FINITE_LIMIT: f32 = 3.0e38;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Finiteness via ordered abs < 3.0e38 (rejects +/-inf and NaN, since every
    // comparison with NaN is false). No bare f32 equality anywhere.
    let k_finite = abs(q.stiffness) < FINITE_LIMIT;
    let sc_finite = abs(q.strength) < FINITE_LIMIT;
    let gc_finite = abs(q.fracture_energy) < FINITE_LIMIT;
    let kap_finite = abs(q.kappa) < FINITE_LIMIT;
    let positive = (q.stiffness > 0.0) && (q.strength > 0.0) && (q.fracture_energy > 0.0);
    let base_ok = k_finite && sc_finite && gc_finite && kap_finite && positive;

    // Guard the derived divisors: when base_ok, K>0 and sigma_c>0, so onset and
    // final are well defined; otherwise substitute 1 to avoid inf/NaN.
    let k_den = select(1.0, q.stiffness, base_ok);
    let sc_den = select(1.0, q.strength, base_ok);
    let onset = q.strength / k_den;
    let final_sep = 2.0 * q.fracture_energy / sc_den;

    // The softening branch exists only when final > onset (ordered compare).
    let softening_exists = final_sep > onset;
    let ok = base_ok && softening_exists;

    // Softening-branch dissipation with a select-guarded span denominator.
    let span = final_sep - onset;
    let span_den = select(1.0, span, ok);
    let linear = 0.5 * q.strength * final_sep * (q.kappa - onset) / span_den;

    // Piecewise: kappa <= onset -> 0; kappa >= final -> G_c; else linear. The
    // below override is applied last to match the golden precedence.
    let below = q.kappa <= onset;
    let above = q.kappa >= final_sep;
    var energy = linear;
    energy = select(energy, q.fracture_energy, above);
    energy = select(energy, 0.0, below);

    var out: Result;
    out.energy = select(0.0, energy, ok);
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// the four raw model scalars — `4` `f32` words (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    stiffness: f32,
    strength: f32,
    fracture_energy: f32,
    kappa: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the dissipated energy and the validity flag — `2` words (`8` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    energy: f32,
    valid: u32,
}

/// One dissipated-energy query: the cohesive model's penalty stiffness, peak
/// strength, fracture energy and the history variable `kappa`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CohesiveDissipatedEnergyQuery {
    /// Penalty stiffness `K`.
    pub stiffness: f32,
    /// Peak traction (strength) `sigma_c`.
    pub strength: f32,
    /// Fracture energy `G_c`.
    pub fracture_energy: f32,
    /// History variable `kappa` (largest effective separation reached).
    pub kappa: f32,
}

impl CohesiveDissipatedEnergyQuery {
    /// Builds a query from the model's stiffness, strength, fracture energy and
    /// the history variable `kappa`.
    #[must_use]
    pub fn new(
        stiffness: f32,
        strength: f32,
        fracture_energy: f32,
        kappa: f32,
    ) -> CohesiveDissipatedEnergyQuery {
        CohesiveDissipatedEnergyQuery {
            stiffness,
            strength,
            fracture_energy,
            kappa,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `dissipated_energy` output for that model and history variable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CohesiveDissipatedEnergyResult {
    /// The dissipated energy per unit area when valid, else `0`.
    pub energy: f32,
    /// `true` when the query is finite, positive and has a softening branch.
    pub valid: bool,
}

/// Encodes one [`CohesiveDissipatedEnergyQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &CohesiveDissipatedEnergyQuery) -> GpuQuery {
    GpuQuery {
        stiffness: q.stiffness,
        strength: q.strength,
        fracture_energy: q.fracture_energy,
        kappa: q.kappa,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`CohesiveDissipatedEnergyResult`].
fn decode_result(raw: &GpuResult) -> CohesiveDissipatedEnergyResult {
    CohesiveDissipatedEnergyResult {
        energy: raw.energy,
        valid: raw.valid != 0,
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

/// A compiled, reusable cohesive dissipated-energy compute pipeline, twinning
/// the `CPU` golden `CohesiveModel::new` derivation plus `dissipated_energy`.
pub struct GpuCohesiveDissipatedEnergy {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCohesiveDissipatedEnergy {
    /// Compiles the cohesive dissipated-energy kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCohesiveDissipatedEnergy {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cohesive_dissipated_energy"),
            source: ShaderSource::Wgsl(COHESIVE_DISSIPATED_ENERGY_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cohesive_dissipated_energy_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cohesive_dissipated_energy_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cohesive_dissipated_energy_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCohesiveDissipatedEnergy {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`CohesiveDissipatedEnergyResult`] per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and the `energy` scalar to
    /// the module's tolerance. An empty `queries` batch returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[CohesiveDissipatedEnergyQuery],
    ) -> Vec<CohesiveDissipatedEnergyResult> {
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
            label: Some("prism_volumetric_cohesive_dissipated_energy_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cohesive_dissipated_energy_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cohesive_dissipated_energy_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cohesive_dissipated_energy_bind_group"),
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
            label: Some("prism_volumetric_cohesive_dissipated_energy_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cohesive_dissipated_energy_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cohesive_dissipated_energy_pass"),
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
