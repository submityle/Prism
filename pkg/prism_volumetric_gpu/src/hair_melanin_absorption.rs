//! `wgpu` compute twin of the stateless physically based hair-colour pigment
//! map
//! [`melanin_absorption`](prism_render_architecture::hair::melanin::melanin_absorption).
//!
//! Real hair colour comes from two pigments — eumelanin (brown-black) and
//! pheomelanin (red-yellow) — and the renderer's hair `BSDF` is driven by a
//! per-channel RGB absorption coefficient `sigma_a`, not a free tint. The golden
//! [`melanin_absorption`](prism_render_architecture::hair::melanin::melanin_absorption)
//! maps a [`MelaninProfile`](prism_render_architecture::hair::melanin::MelaninProfile)
//! (two non-negative concentrations) to that RGB `sigma_a` as a pure
//! non-negative linear combination of the two canonical per-pigment spectra
//! [`EUMELANIN_SIGMA_A`](prism_render_architecture::hair::melanin::EUMELANIN_SIGMA_A)
//! and
//! [`PHEOMELANIN_SIGMA_A`](prism_render_architecture::hair::melanin::PHEOMELANIN_SIGMA_A).
//! It performs no transcendental math, so the port is a direct closed-form
//! twin.
//!
//! [`GpuHairMelaninAbsorption`] is the on-device twin of that one map. One
//! thread solves one query, reproducing the golden's `sanitized` concentration
//! guard (negative or non-finite concentrations collapse to `0`) and the
//! per-channel multiply-add, so a passing real-device parity test is direct
//! evidence the ported kernel computes the same `sigma_a` the reference does,
//! not merely that the shader compiles.
//!
//! # What is twinned
//!
//! The kernel reproduces the golden exactly: each concentration is passed
//! through the `sanitized` guard (a value is kept only when it is finite and
//! strictly positive, otherwise `0`), then the RGB `sigma_a` is
//! `eumelanin * EUMELANIN_SIGMA_A + pheomelanin * PHEOMELANIN_SIGMA_A` per
//! channel. The golden's `is_finite() && value > 0.0` guard is reproduced with
//! ordered comparisons: a value passes when `value > 0.0 && value < +inf`, which
//! rejects `NaN`, `-inf`, `+inf` and non-positive values identically to the
//! reference.
//!
//! # What stays on the host
//!
//! The slice form
//! [`melanin_absorption_map`](prism_render_architecture::hair::melanin::melanin_absorption_map)
//! is a variable-length array-in/array-out aggregate owned by the host, which
//! enqueues one [`HairMelaninAbsorptionQuery`] per fibre. The named
//! [`NaturalHairColor`](prism_render_architecture::hair::melanin::NaturalHairColor)
//! presets are host-side enum-to-concentration lookups; only the stateless
//! numeric pigment map is twinned.
//!
//! # Correctness model
//!
//! The map is a pure linear combination guarded by ordered comparisons, so
//! `CPU` and `GPU` agree to floating-point rounding and the parity test asserts
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` on every channel.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — multiply, add, ordered
//! comparison, and a `bitcast` to construct `+inf` — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, no inverse trigonometry, no `sqrt` and no `u64`. No
//! optional device feature is required, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. There is no loop: each thread performs a fixed, bounded
//! sequence of arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::hair::melanin::melanin_absorption`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` melanin-absorption kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`melanin_absorption`](prism_render_architecture::hair::melanin::melanin_absorption);
/// see the module documentation for the algorithm.
const HAIR_MELANIN_ABSORPTION_WGSL: &str = r#"
// Hair pigment-to-absorption twin: one thread maps one fibre's eumelanin and
// pheomelanin concentrations to an RGB sigma_a as a non-negative linear
// combination of the two canonical per-pigment spectra, after collapsing any
// negative or non-finite concentration to zero — mirroring the CPU golden
// `hair::melanin::melanin_absorption` with only multiply, add and ordered
// comparison. The variable-length slice form stays on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::hair::melanin::melanin_absorption
// ；无第三方引擎源码或衍生代码。

// Per-unit-concentration eumelanin absorption at the renderer's RGB primaries
// (golden EUMELANIN_SIGMA_A).
const EU_R: f32 = 0.419;
const EU_G: f32 = 0.697;
const EU_B: f32 = 1.37;

// Per-unit-concentration pheomelanin absorption at the renderer's RGB primaries
// (golden PHEOMELANIN_SIGMA_A).
const PH_R: f32 = 0.187;
const PH_G: f32 = 0.4;
const PH_B: f32 = 1.05;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Eumelanin (brown-black pigment) concentration before sanitation.
    eumelanin: f32,
    // Pheomelanin (red-yellow pigment) concentration before sanitation.
    pheomelanin: f32,
    pad0: f32,
    pad1: f32,
}

struct Result {
    // RGB absorption coefficient sigma_a.
    r: f32,
    g: f32,
    b: f32,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Collapses one concentration to a finite, non-negative value, mirroring the
// golden `clamp_concentration`: keep the value only when it is finite and
// strictly positive, otherwise 0. The `value < +inf` guard rejects +inf (which
// would otherwise pass `value > 0.0`), while NaN, -inf and non-positive values
// are rejected by `value > 0.0`, matching `is_finite() && value > 0.0`.
fn clamp_concentration(value: f32) -> f32 {
    let pos_inf = bitcast<f32>(0x7f800000u);
    if (value > 0.0 && value < pos_inf) {
        return value;
    }
    return 0.0;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let eu = clamp_concentration(q.eumelanin);
    let ph = clamp_concentration(q.pheomelanin);

    var out: Result;
    out.r = eu * EU_R + ph * PH_R;
    out.g = eu * EU_G + ph * PH_G;
    out.b = eu * EU_B + ph * PH_B;
    out.pad0 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`HAIR_MELANIN_ABSORPTION_WGSL`].
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

/// `repr(C)` `std430` layout of one melanin query, matching the `WGSL` `Query`
/// struct: the two raw concentrations and two pad words to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Eumelanin concentration before sanitation.
    eumelanin: f32,
    /// Pheomelanin concentration before sanitation.
    pheomelanin: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// `repr(C)` `std430` layout of one melanin result, matching the `WGSL` `Result`
/// struct: the RGB `sigma_a` and one pad word to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Red-channel absorption coefficient.
    r: f32,
    /// Green-channel absorption coefficient.
    g: f32,
    /// Blue-channel absorption coefficient.
    b: f32,
    /// Padding word.
    pad0: f32,
}

/// One melanin-absorption query: the raw eumelanin and pheomelanin
/// concentrations, mirroring the inputs the reference `melanin_absorption` reads
/// from a [`MelaninProfile`](prism_render_architecture::hair::melanin::MelaninProfile).
///
/// Concentrations are stored exactly as supplied; the kernel reproduces the
/// golden's `sanitized` guard, so negative or non-finite inputs are valid and
/// collapse to `0` on device just as they do on the host.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairMelaninAbsorptionQuery {
    /// Eumelanin (brown-black pigment) concentration (the golden `eumelanin`).
    pub eumelanin: f32,
    /// Pheomelanin (red-yellow pigment) concentration (the golden
    /// `pheomelanin`).
    pub pheomelanin: f32,
}

impl HairMelaninAbsorptionQuery {
    /// Builds a query from explicit eumelanin and pheomelanin concentrations,
    /// mirroring the reference
    /// [`MelaninProfile::new`](prism_render_architecture::hair::melanin::MelaninProfile::new).
    #[must_use]
    pub const fn new(eumelanin: f32, pheomelanin: f32) -> HairMelaninAbsorptionQuery {
        HairMelaninAbsorptionQuery {
            eumelanin,
            pheomelanin,
        }
    }
}

/// One resolved melanin-absorption query: the RGB absorption coefficient
/// `sigma_a`, mirroring the reference
/// [`melanin_absorption`](prism_render_architecture::hair::melanin::melanin_absorption)
/// return value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairMelaninAbsorptionResult {
    /// The RGB absorption coefficient `sigma_a` (the golden `[f32; 3]` return).
    pub sigma_a: [f32; 3],
}

/// Encodes one [`HairMelaninAbsorptionQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &HairMelaninAbsorptionQuery) -> GpuQuery {
    GpuQuery {
        eumelanin: q.eumelanin,
        pheomelanin: q.pheomelanin,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`HairMelaninAbsorptionResult`].
fn decode_result(raw: &GpuResult) -> HairMelaninAbsorptionResult {
    HairMelaninAbsorptionResult {
        sigma_a: [raw.r, raw.g, raw.b],
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

/// A compiled, reusable melanin-absorption compute pipeline, twinning the
/// stateless RGB `sigma_a` of the `CPU` golden
/// [`melanin_absorption`](prism_render_architecture::hair::melanin::melanin_absorption).
pub struct GpuHairMelaninAbsorption {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairMelaninAbsorption {
    /// Compiles the melanin-absorption kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairMelaninAbsorption {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_hair_melanin_absorption"),
            source: ShaderSource::Wgsl(HAIR_MELANIN_ABSORPTION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_hair_melanin_absorption_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_hair_melanin_absorption_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_hair_melanin_absorption_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairMelaninAbsorption {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`HairMelaninAbsorptionResult`] per input, in order.
    ///
    /// Each `sigma_a` equals the reference to floating-point rounding. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[HairMelaninAbsorptionQuery],
    ) -> Vec<HairMelaninAbsorptionResult> {
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
            label: Some("prism_volumetric_hair_melanin_absorption_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_hair_melanin_absorption_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_hair_melanin_absorption_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_hair_melanin_absorption_bind_group"),
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
            label: Some("prism_volumetric_hair_melanin_absorption_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_hair_melanin_absorption_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_hair_melanin_absorption_pass"),
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
