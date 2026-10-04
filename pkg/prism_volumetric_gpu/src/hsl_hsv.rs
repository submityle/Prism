//! `wgpu` compute twin of the cylindrical `HSL` / `HSV` color conversions, from
//! the `CPU` golden `prism_math::color::hsl` (`Hsla` and `Hsva`
//! `from_srgb` / `to_srgb`, plus the private `rgb_to_hue` / `hue_to_rgb`
//! helpers).
//!
//! Both color models are defined over non-linear `sRGB` components (the color
//! picker convention), with hue in degrees `[0, 360)` and the remaining axes in
//! `[0, 1]`. A single `dir_id` selects one of four stateless conversions, so
//! one thread resolves one query and a passing real-device parity test is
//! direct evidence the ported kernel computes the same colors the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces one conversion, chosen by `dir_id`:
//!
//! * `0` — `rgb -> hsl`: inputs `(r, g, b, alpha)`, output `(hue, saturation,
//!   lightness, alpha)` via `Hsla::from_srgb`.
//! * `1` — `hsl -> rgb`: inputs `(hue, saturation, lightness, alpha)`, output
//!   `(r, g, b, alpha)` via `Hsla::to_srgb`.
//! * `2` — `rgb -> hsv`: inputs `(r, g, b, alpha)`, output `(hue, saturation,
//!   value, alpha)` via `Hsva::from_srgb`.
//! * `3` — `hsv -> rgb`: inputs `(hue, saturation, value, alpha)`, output
//!   `(r, g, b, alpha)` via `Hsva::to_srgb`.
//!
//! Any `dir_id > 3` is invalid (`valid = 0`, all outputs `0`). All in-range
//! directions are valid (`valid = 1`), matching the total golden functions.
//!
//! # Correctness model
//!
//! Both sides evaluate the same pure-`f32` closed form, so `CPU` and `GPU` need
//! not be bit-exact under reassociation; every continuous output is compared
//! with `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`) and the discrete
//! `valid` word is compared exactly. The hue decomposition picks a channel
//! branch; the parity fixtures and sweep keep inputs off the `60`-degree sector
//! knees and off exact channel ties so the branch choice cannot be flipped by
//! round-off.
//!
//! # Degenerate inputs
//!
//! A grayscale input (`r = g = b`) has zero chroma and hue `0` on both sides.
//! The chroma, lightness-edge and value divisors are all fed through a `select`
//! guard so the un-taken branch never divides by zero. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot
//! be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `floor`, `abs`,
//! `min`, `max`, `select`, `+ - * /` and unsigned index arithmetic — with no
//! `sin`, `cos`, `tan`, `exp`, `log`, `pow`, no `round`, no `f32` remainder and
//! no `sqrt`. The Euclidean remainder `WGSL` lacks is reconstructed as
//! `a - b * floor(a / b)`. There is no bare `f32` equality: the chroma and
//! divisor guards use ordered compares fed to `select`, and only `u32` sector
//! indices are compared with `==`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_math::color::hsl`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` `HSL` / `HSV` conversion kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden `prism_math::color::hsl` conversions; see the
/// module documentation for the per-`dir_id` closed forms.
const HSL_HSV_WGSL: &str = r#"
// HSL / HSV conversion twin: one thread per query reproduces one of four
// sRGB <-> cylindrical conversions, chosen by dir_id. It uses only the portable
// core-WGSL subset (floor, abs, min, max, select, + - * / plus unsigned index
// math) with no transcendental, no round, no f32 remainder and no sqrt. The
// Euclidean remainder is reconstructed as a - b * floor(a / b). There is no
// bare f32 equality: divisor guards use ordered compares fed to select and only
// u32 sector indices are compared with ==.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Direction selector: 0 rgb->hsl, 1 hsl->rgb, 2 rgb->hsv, 3 hsv->rgb.
    dir_id: u32,
    // Four input components, interpreted per direction.
    c0: f32,
    c1: f32,
    c2: f32,
    c3: f32,
    // Padding words to a 32-byte stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct ColorResult {
    // Four output components, interpreted per direction.
    out0: f32,
    out1: f32,
    out2: f32,
    out3: f32,
    // 1 when dir_id is in range, else 0.
    valid: u32,
    // Padding words to a 32-byte stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<ColorResult>;

// Euclidean remainder, which WGSL has no builtin for.
fn rem_euclid(a: f32, b: f32) -> f32 {
    let r = a - b * floor(a / b);
    return r;
}

// Decompose sRGB into (max, min, chroma, hue-in-degrees), vec4 packed.
fn rgb_to_hue(r: f32, g: f32, b: f32) -> vec4<f32> {
    let mx = max(r, max(g, b));
    let mn = min(r, min(g, b));
    let chroma = mx - mn;
    // Guard the divisor so the zero-chroma branch never divides by zero; the
    // ordered compares below reproduce the golden tie-break order
    // (max==r first, then max==g, else blue).
    let safe_chroma = select(1.0, chroma, chroma > 0.0);
    let hue_r = 60.0 * rem_euclid((g - b) / safe_chroma, 6.0);
    let hue_g = 60.0 * ((b - r) / safe_chroma + 2.0);
    let hue_b = 60.0 * ((r - g) / safe_chroma + 4.0);
    let r_is_max = (r >= g) && (r >= b);
    let g_is_max = (g >= b);
    let hue_nonzero = select(hue_b, hue_g, g_is_max);
    let hue_rgb = select(hue_nonzero, hue_r, r_is_max);
    let hue = select(hue_rgb, 0.0, chroma <= 0.0);
    return vec4<f32>(mx, mn, chroma, hue);
}

// Reconstruct sRGB from hue/chroma plus a per-channel offset m, vec4 packed.
fn hue_to_rgb(hue: f32, chroma: f32, m: f32, alpha: f32) -> vec4<f32> {
    let h = rem_euclid(hue, 360.0) / 60.0;
    let x = chroma * (1.0 - abs(rem_euclid(h, 2.0) - 1.0));
    let sector = u32(floor(h));
    var r1: f32;
    var g1: f32;
    var b1: f32;
    if (sector == 0u) {
        r1 = chroma;
        g1 = x;
        b1 = 0.0;
    } else if (sector == 1u) {
        r1 = x;
        g1 = chroma;
        b1 = 0.0;
    } else if (sector == 2u) {
        r1 = 0.0;
        g1 = chroma;
        b1 = x;
    } else if (sector == 3u) {
        r1 = 0.0;
        g1 = x;
        b1 = chroma;
    } else if (sector == 4u) {
        r1 = x;
        g1 = 0.0;
        b1 = chroma;
    } else {
        r1 = chroma;
        g1 = 0.0;
        b1 = x;
    }
    return vec4<f32>(r1 + m, g1 + m, b1 + m, alpha);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let dir_id = q.dir_id;
    let c0 = q.c0;
    let c1 = q.c1;
    let c2 = q.c2;
    let c3 = q.c3;

    var o0 = 0.0;
    var o1 = 0.0;
    var o2 = 0.0;
    var o3 = 0.0;

    let valid = dir_id <= 3u;

    if (dir_id == 0u) {
        // rgb -> hsl: inputs (r, g, b, alpha).
        let dec = rgb_to_hue(c0, c1, c2);
        let mx = dec.x;
        let mn = dec.y;
        let chroma = dec.z;
        let hue = dec.w;
        let lightness = 0.5 * (mx + mn);
        let denom = 1.0 - abs(2.0 * lightness - 1.0);
        let sat_denom = select(1.0, denom, denom > 0.0);
        let sat_full = chroma / sat_denom;
        let light_edge = (lightness <= 0.0) || (lightness >= 1.0);
        let saturation = select(sat_full, 0.0, light_edge);
        o0 = hue;
        o1 = saturation;
        o2 = lightness;
        o3 = c3;
    } else if (dir_id == 1u) {
        // hsl -> rgb: inputs (hue, saturation, lightness, alpha).
        let chroma = (1.0 - abs(2.0 * c2 - 1.0)) * c1;
        let m = c2 - 0.5 * chroma;
        let rgb = hue_to_rgb(c0, chroma, m, c3);
        o0 = rgb.x;
        o1 = rgb.y;
        o2 = rgb.z;
        o3 = rgb.w;
    } else if (dir_id == 2u) {
        // rgb -> hsv: inputs (r, g, b, alpha).
        let dec = rgb_to_hue(c0, c1, c2);
        let mx = dec.x;
        let chroma = dec.z;
        let hue = dec.w;
        let value = mx;
        let val_denom = select(1.0, value, value > 0.0);
        let sat_full = chroma / val_denom;
        let saturation = select(sat_full, 0.0, value <= 0.0);
        o0 = hue;
        o1 = saturation;
        o2 = value;
        o3 = c3;
    } else if (dir_id == 3u) {
        // hsv -> rgb: inputs (hue, saturation, value, alpha).
        let chroma = c2 * c1;
        let m = c2 - chroma;
        let rgb = hue_to_rgb(c0, chroma, m, c3);
        o0 = rgb.x;
        o1 = rgb.y;
        o2 = rgb.z;
        o3 = rgb.w;
    }

    var out: ColorResult;
    out.out0 = select(0.0, o0, valid);
    out.out1 = select(0.0, o1, valid);
    out.out2 = select(0.0, o2, valid);
    out.out3 = select(0.0, o3, valid);
    out.valid = select(0u, 1u, valid);
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;
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
/// a direction selector, four input components and three padding words — `8`
/// words (`32` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    dir_id: u32,
    c0: f32,
    c1: f32,
    c2: f32,
    c3: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `ColorResult`
/// struct: four output components, the validity flag and three padding words —
/// `8` words (`32` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    out0: f32,
    out1: f32,
    out2: f32,
    out3: f32,
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One color-conversion query: a direction selector plus four input components,
/// interpreted as `rgba` or `hsla` / `hsva` according to `dir_id`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HslHsvQuery {
    /// Direction selector: `0` `rgb->hsl`, `1` `hsl->rgb`, `2` `rgb->hsv`,
    /// `3` `hsv->rgb`. Any other value is invalid.
    pub dir_id: u32,
    /// First input component.
    pub c0: f32,
    /// Second input component.
    pub c1: f32,
    /// Third input component.
    pub c2: f32,
    /// Fourth input component (alpha, passed through unchanged).
    pub c3: f32,
}

impl HslHsvQuery {
    /// Builds a query from a direction selector and four input components.
    #[must_use]
    pub fn new(dir_id: u32, c0: f32, c1: f32, c2: f32, c3: f32) -> HslHsvQuery {
        HslHsvQuery {
            dir_id,
            c0,
            c1,
            c2,
            c3,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference color
/// conversion for that direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HslHsvResult {
    /// First output component.
    pub out0: f32,
    /// Second output component.
    pub out1: f32,
    /// Third output component.
    pub out2: f32,
    /// Fourth output component (alpha, passed through unchanged).
    pub out3: f32,
    /// `1` when `dir_id` is in range (`<= 3`), else `0`.
    pub valid: u32,
}

/// Encodes one [`HslHsvQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &HslHsvQuery) -> GpuQuery {
    GpuQuery {
        dir_id: q.dir_id,
        c0: q.c0,
        c1: q.c1,
        c2: q.c2,
        c3: q.c3,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`HslHsvResult`].
fn decode_result(raw: &GpuResult) -> HslHsvResult {
    HslHsvResult {
        out0: raw.out0,
        out1: raw.out1,
        out2: raw.out2,
        out3: raw.out3,
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

/// A compiled, reusable `HSL` / `HSV` conversion compute pipeline, twinning the
/// `CPU` golden `prism_math::color::hsl` conversions.
pub struct GpuHslHsv {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHslHsv {
    /// Compiles the conversion kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHslHsv {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_hsl_hsv"),
            source: ShaderSource::Wgsl(HSL_HSV_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_hsl_hsv_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_hsl_hsv_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_hsl_hsv_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHslHsv {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`HslHsvResult`] per
    /// input, in order.
    ///
    /// The `valid` flag matches the reference exactly and each output component
    /// to the module's tolerance. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[HslHsvQuery]) -> Vec<HslHsvResult> {
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
            label: Some("prism_volumetric_hsl_hsv_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_hsl_hsv_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_hsl_hsv_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_hsl_hsv_bind_group"),
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
            label: Some("prism_volumetric_hsl_hsv_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_hsl_hsv_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_hsl_hsv_pass"),
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
