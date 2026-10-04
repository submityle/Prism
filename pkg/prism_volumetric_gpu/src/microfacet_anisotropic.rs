//! `wgpu` compute twin of the anisotropic `GGX` (Trowbridge-Reitz) microfacet
//! evaluation terms from the `CPU` golden
//! `prism_render_architecture::reference_pt::microfacet_aniso::GgxAnisotropic`.
//!
//! Brushed metal, hair, vinyl records and machined bezels have a *grain*:
//! parallel grooves that stretch a specular highlight into a streak. The
//! reference `GgxAnisotropic` captures that by giving the micro-slope its own
//! width `alpha_x`/`alpha_y` along the two local tangent axes. This module
//! ports the lobe's stateless, no-`RNG` evaluation terms onto the device: the
//! normal distribution `D(h)`, the single-direction masking `G1(wo)`, the
//! height-correlated masking-shadowing `G2(wo, wi)`, and the solid-angle
//! reflection density `reflection_pdf`. The importance-sampling
//! `sample_half_vector`, which needs a random stream, is deliberately *not*
//! twinned.
//!
//! [`GpuGgxAnisotropic`] is the on-device twin: one thread solves one query, so
//! a passing real-device parity test is direct evidence the ported kernel
//! computes the same four terms and takes the same below-horizon /
//! back-facing-half-vector branch the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces the reference closed forms. The widths
//! `alpha_x`/`alpha_y` arrive already clamped to `MIN_ALPHA` on the host, so the
//! kernel consumes them directly:
//!
//! - `D(h) = 1/pi / (ax*ay*q*q)` with `q = (h.x/ax)^2 + (h.y/ay)^2 + h.z^2`,
//!   zero for a back-facing half vector (`h.z <= 0`);
//! - the Smith `lambda(w)` auxiliary feeding `g1(w) = 1/(1+lambda)` and
//!   `g2(wo, wi) = 1/(1+lambda(wo)+lambda(wi))`;
//! - `reflection_pdf(wo, h) = g1(wo)*D(h)/(4*wo.z)`, zero when `wo.z <= 0`.
//!
//! There is no loop: each thread performs a fixed, bounded sequence of
//! arithmetic plus a handful of `sqrt` calls, so the kernel provably
//! terminates.
//!
//! # Correctness model
//!
//! Every quantity threads through multiplies, adds, guarded divisions and
//! `sqrt`, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate. The parity test asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on
//! every continuous quantity, tight enough to catch a genuinely wrong port yet
//! loose enough to admit legal fused multiply-add contraction. The discrete
//! `valid` flag is compared exactly.
//!
//! # Degenerate inputs
//!
//! When the view direction is at or below the surface (`wo.z <= 0`) or the half
//! vector is back-facing (`h.z <= 0`), the reference distribution and density
//! return zero; the twin reports `valid = 0` with all four continuous outputs
//! cleared. Fixtures and the sweep keep both `wo.z` and `h.z` comfortably
//! positive and the widths away from the `cz >= 1` branch so parity never sits
//! on a knife edge. An empty query batch short-circuits on the host with no
//! dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `clamp`, `select`, `sqrt`, `+ - * /` and unsigned index arithmetic — with no
//! `sin`, `cos`, `tan`, `exp`, `log`, `pow`, no inverse trigonometry, no
//! `smoothstep`, no `round` and no optional device feature, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::microfacet_aniso`；无第三方引擎源码或衍生代码。
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

/// Minimum `GGX` lobe width, matching the reference `MIN_ALPHA`. The host clamps
/// each width to at least this value before upload, mirroring
/// `GgxAnisotropic::new`.
pub const MIN_ALPHA: f32 = 0.001;

/// The portable core-`WGSL` anisotropic-`GGX` kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `GgxAnisotropic::distribution`, `g1`, `g2` and
/// `reflection_pdf` branch for branch; see the module documentation for the
/// algorithm.
const MICROFACET_ANISOTROPIC_WGSL: &str = r#"
// Anisotropic GGX twin: one thread per query reproduces the four stateless
// evaluation terms GgxAnisotropic exposes — D(h), G1(wo), G2(wo, wi) and the
// reflection density G1(wo)*D(h)/(4 wo.z) — from the two lobe widths and the
// local-frame half, outgoing and incoming directions. It mirrors the CPU
// golden branch for branch, uses only the portable core-WGSL subset
// (abs/min/max/clamp/select/sqrt and + - * / plus unsigned index math), takes
// no optional feature, and has no loop, so the kernel provably terminates. The
// RNG importance-sampling sample_half_vector is intentionally not twinned.
//
// Provenance: 孪生自本仓 prism_render_architecture::reference_pt::microfacet_aniso；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // GGX widths along the local tangent (x) and bitangent (y), each already
    // clamped to MIN_ALPHA on the host.
    alpha_x: f32,
    alpha_y: f32,
    // Half vector in the local shading frame.
    hx: f32,
    hy: f32,
    hz: f32,
    // Outgoing (view) direction in the local shading frame.
    wox: f32,
    woy: f32,
    woz: f32,
    // Incoming (light) direction in the local shading frame.
    wix: f32,
    wiy: f32,
    wiz: f32,
    // Trailing pad word so this std430 struct is 48 bytes (12 lanes), matching
    // the host `#[repr(C)]` `GpuQuery`. Without it naga lays the array out at a
    // 44-byte stride on device, so every element after the first reads shifted
    // and lanes such as wo.z alias a neighboring (possibly negative) component.
    pad0: f32,
}

struct Result {
    // Anisotropic GGX normal distribution D(h); zero when invalid.
    d: f32,
    // Smith single-direction masking G1(wo); zero when invalid.
    g1_wo: f32,
    // Height-correlated masking-shadowing G2(wo, wi); zero when invalid.
    g2: f32,
    // Solid-angle reflection density; zero when invalid.
    pdf: f32,
    // 1 when wo.z > 0 and h.z > 0, 0 for a degenerate query.
    valid: u32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const INV_PI: f32 = 0.31830988618;

// Anisotropic GGX normal distribution D(h); zero for a back-facing half vector.
fn ggx_distribution(h: vec3<f32>, ax: f32, ay: f32) -> f32 {
    if (h.z <= 0.0) {
        return 0.0;
    }
    let hx = h.x / ax;
    let hy = h.y / ay;
    let hz = h.z;
    let q = hx * hx + hy * hy + hz * hz;
    return INV_PI / (ax * ay * q * q);
}

// Smith Lambda auxiliary for a local-frame direction.
fn ggx_lambda(w: vec3<f32>, ax: f32, ay: f32) -> f32 {
    let cz = abs(w.z);
    if (cz >= 1.0) {
        return 0.0;
    }
    let axx = ax * w.x;
    let ayy = ay * w.y;
    let numer = axx * axx + ayy * ayy;
    if (numer <= 0.0) {
        return 0.0;
    }
    let ratio = numer / (cz * cz);
    return 0.5 * (sqrt(1.0 + ratio) - 1.0);
}

fn ggx_g1(w: vec3<f32>, ax: f32, ay: f32) -> f32 {
    return 1.0 / (1.0 + ggx_lambda(w, ax, ay));
}

fn ggx_g2(wo: vec3<f32>, wi: vec3<f32>, ax: f32, ay: f32) -> f32 {
    return 1.0 / (1.0 + ggx_lambda(wo, ax, ay) + ggx_lambda(wi, ax, ay));
}

fn ggx_reflection_pdf(wo: vec3<f32>, h: vec3<f32>, ax: f32, ay: f32) -> f32 {
    if (wo.z <= 0.0) {
        return 0.0;
    }
    return ggx_g1(wo, ax, ay) * ggx_distribution(h, ax, ay) / (4.0 * wo.z);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let h = vec3<f32>(q.hx, q.hy, q.hz);
    let wo = vec3<f32>(q.wox, q.woy, q.woz);
    let wi = vec3<f32>(q.wix, q.wiy, q.wiz);
    let ax = q.alpha_x;
    let ay = q.alpha_y;

    var out: Result;
    out.d = 0.0;
    out.g1_wo = 0.0;
    out.g2 = 0.0;
    out.pdf = 0.0;
    out.valid = 0u;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    out.pad2 = 0.0;

    let valid = (wo.z > 0.0) && (h.z > 0.0);
    if (!valid) {
        results[idx] = out;
        return;
    }

    out.d = ggx_distribution(h, ax, ay);
    out.g1_wo = ggx_g1(wo, ax, ay);
    out.g2 = ggx_g2(wo, wi, ax, ay);
    out.pdf = ggx_reflection_pdf(wo, h, ax, ay);
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`MICROFACET_ANISOTROPIC_WGSL`].
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
/// The `vec3` inputs are flattened to scalar lanes so the layout never trips a
/// `16`-byte vector-alignment rule; the kernel rebuilds each `vec3<f32>`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    alpha_x: f32,
    alpha_y: f32,
    hx: f32,
    hy: f32,
    hz: f32,
    wox: f32,
    woy: f32,
    woz: f32,
    wix: f32,
    wiy: f32,
    wiz: f32,
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    d: f32,
    g1_wo: f32,
    g2: f32,
    pdf: f32,
    valid: u32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

/// One query for the anisotropic-`GGX` twin: the two lobe widths and the local
/// half, outgoing and incoming directions.
///
/// All four evaluation terms are derived from this one tuple, so a single query
/// exercises the whole twinned core at once. The widths are clamped to
/// [`MIN_ALPHA`] when the query is built, mirroring `GgxAnisotropic::new`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GgxAnisotropicQuery {
    /// `GGX` width along the local tangent (`x`) axis, clamped to [`MIN_ALPHA`].
    pub alpha_x: f32,
    /// `GGX` width along the local bitangent (`y`) axis, clamped to
    /// [`MIN_ALPHA`].
    pub alpha_y: f32,
    /// Half vector in the local shading frame.
    pub h: [f32; 3],
    /// Outgoing (view) direction in the local shading frame.
    pub wo: [f32; 3],
    /// Incoming (light) direction in the local shading frame.
    pub wi: [f32; 3],
}

impl GgxAnisotropicQuery {
    /// Builds a query from the two widths and the three local-frame directions,
    /// clamping each width to at least [`MIN_ALPHA`] as `GgxAnisotropic::new`
    /// does.
    #[must_use]
    pub fn new(
        alpha_x: f32,
        alpha_y: f32,
        h: [f32; 3],
        wo: [f32; 3],
        wi: [f32; 3],
    ) -> GgxAnisotropicQuery {
        GgxAnisotropicQuery {
            alpha_x: alpha_x.max(MIN_ALPHA),
            alpha_y: alpha_y.max(MIN_ALPHA),
            h,
            wo,
            wi,
        }
    }

    /// Builds a query from a perceptual `roughness` in `[0, 1]` and an
    /// `anisotropy` in `[0, 1]`, using the Disney / `UE` (Burley) remap that
    /// `GgxAnisotropic::from_roughness_anisotropy` applies, then clamping each
    /// resulting width to [`MIN_ALPHA`].
    #[must_use]
    pub fn from_roughness_anisotropy(
        roughness: f32,
        anisotropy: f32,
        h: [f32; 3],
        wo: [f32; 3],
        wi: [f32; 3],
    ) -> GgxAnisotropicQuery {
        let r = roughness.clamp(0.0, 1.0);
        let alpha = r * r;
        let aniso = anisotropy.clamp(0.0, 1.0);
        let aspect = (1.0 - 0.9 * aniso).max(1.0e-4).sqrt();
        GgxAnisotropicQuery::new(alpha / aspect, alpha * aspect, h, wo, wi)
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `GgxAnisotropic::distribution`, `g1`, `g2` and `reflection_pdf` outputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GgxAnisotropicResult {
    /// The anisotropic `GGX` normal distribution `D(h)`; zero when invalid.
    pub d: f32,
    /// The Smith single-direction masking `G1(wo)`; zero when invalid.
    pub g1_wo: f32,
    /// The height-correlated masking-shadowing `G2(wo, wi)`; zero when invalid.
    pub g2: f32,
    /// The solid-angle reflection density; zero when invalid.
    pub pdf: f32,
    /// `1` when `wo.z > 0` and `h.z > 0`, `0` for a degenerate query.
    pub valid: u32,
}

/// Encodes one [`GgxAnisotropicQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &GgxAnisotropicQuery) -> GpuQuery {
    GpuQuery {
        alpha_x: q.alpha_x,
        alpha_y: q.alpha_y,
        hx: q.h[0],
        hy: q.h[1],
        hz: q.h[2],
        wox: q.wo[0],
        woy: q.wo[1],
        woz: q.wo[2],
        wix: q.wi[0],
        wiy: q.wi[1],
        wiz: q.wi[2],
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`GgxAnisotropicResult`].
fn decode_result(raw: &GpuResult) -> GgxAnisotropicResult {
    GgxAnisotropicResult {
        d: raw.d,
        g1_wo: raw.g1_wo,
        g2: raw.g2,
        pdf: raw.pdf,
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

/// A compiled, reusable anisotropic-`GGX` compute pipeline, twinning the `CPU`
/// golden `GgxAnisotropic::distribution`, `g1`, `g2` and `reflection_pdf`.
pub struct GpuGgxAnisotropic {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuGgxAnisotropic {
    /// Compiles the anisotropic-`GGX` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuGgxAnisotropic {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_microfacet_anisotropic"),
            source: ShaderSource::Wgsl(MICROFACET_ANISOTROPIC_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_microfacet_anisotropic_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_microfacet_anisotropic_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_microfacet_anisotropic_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuGgxAnisotropic {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`GgxAnisotropicResult`]
    /// per input, in order.
    ///
    /// Each continuous channel matches the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[GgxAnisotropicQuery],
    ) -> Vec<GgxAnisotropicResult> {
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
            label: Some("prism_volumetric_microfacet_anisotropic_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_microfacet_anisotropic_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_microfacet_anisotropic_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_microfacet_anisotropic_bind_group"),
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
            label: Some("prism_volumetric_microfacet_anisotropic_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_microfacet_anisotropic_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_microfacet_anisotropic_pass"),
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
