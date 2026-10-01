//! `wgpu` compute twin of the `RGB` ↔ `YCoCg` / `YCoCg-R` colour-space
//! transforms
//! ([`rgb_ycocg`](prism_render_architecture::particle::rgb_ycocg), design §21).
//!
//! `YCoCg` splits a colour into one luma channel (`Y`) and two chroma channels
//! (`Co`, orange–blue; `Cg`, green–magenta); a particle temporal-antialiasing
//! (`TAA`) resolve runs its history clamp in that basis because the
//! neighbourhood colour-clamp that kills ghosting is far tighter in luma/chroma
//! than in raw `RGB`. The `CPU` golden
//! [`rgb_ycocg`](prism_render_architecture::particle::rgb_ycocg) owns that math;
//! [`GpuRgbYCoCg`] is the on-device twin that runs one thread per sample and
//! reproduces the same value, so a passing real-device parity test is direct
//! evidence the ported kernels compute the same transform the reference does,
//! not merely that the shaders compile.
//!
//! # What is twinned
//!
//! All four transforms the golden exposes are reproduced:
//!
//! 1. **Lossy `YCoCg`** — [`GpuRgbYCoCg::rgb_to_ycocg`] /
//!    [`GpuRgbYCoCg::ycocg_to_rgb`] mirror
//!    [`rgb_to_ycocg`](prism_render_architecture::particle::rgb_ycocg::rgb_to_ycocg)
//!    / [`ycocg_to_rgb`](prism_render_architecture::particle::rgb_ycocg::ycocg_to_rgb),
//!    the classic quarter/half-weight linear transform evaluated in `f32`.
//! 2. **Lossless `YCoCg-R`** — [`GpuRgbYCoCg::rgb_to_ycocg_r`] /
//!    [`GpuRgbYCoCg::ycocg_r_to_rgb`] mirror
//!    [`rgb_to_ycocg_r`](prism_render_architecture::particle::rgb_ycocg::rgb_to_ycocg_r)
//!    / [`ycocg_r_to_rgb`](prism_render_architecture::particle::rgb_ycocg::ycocg_r_to_rgb),
//!    the integer *lifting* scheme of `H.264` lossless mode. The lifting uses
//!    arithmetic right shifts (floor-division by two) on signed integers, which
//!    is exactly what makes the round trip bit-exact reversible for negative
//!    chroma; the twin reproduces that integer path, shift for shift, in `WGSL`
//!    `i32` arithmetic.
//!
//! # Correctness model
//!
//! The lossy transform touches only `+ - * /` with the same power-of-two
//! coefficients and the same evaluation order the reference uses, so `CPU` and
//! `GPU` evaluate the same closed form. They are not guaranteed bit-exact — a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few units in the last place — so the
//! parity test asserts a tight tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`), loose enough to admit a legal fused multiply-add yet
//! tight enough to fail a swapped coefficient or sign. The lossless `YCoCg-R`
//! path is pure integer arithmetic — subtraction, addition and arithmetic right
//! shift — so it *is* bit-exact and the parity test compares it with `==`.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — `+ - * /`, `clamp`,
//! signed integer subtraction and the arithmetic right shift `>>` — with no
//! transcendental function (`sin`, `cos`, `exp`, `log`, `pow`), no `sqrt` and no
//! optional device feature, so they run unmodified on `Metal`, `Vulkan` and
//! `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `YCoCg` / `YCoCg-R` reversible colour transform
//! (`H.264` lossless mode) plus `wgpu` compute dispatch; no third-party engine
//! source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::rgb_ycocg::{Rgb, YCoCg, YCoCgR};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// the sibling twins use; one thread evaluates one sample.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` kernels for the lossy `f32` `YCoCg` transform,
/// embedded inline so the twin ships as a single source file. The two entry
/// points `rgb_to_ycocg` and `ycocg_to_rgb` mirror the `CPU` golden
/// [`rgb_to_ycocg`](prism_render_architecture::particle::rgb_ycocg::rgb_to_ycocg)
/// and
/// [`ycocg_to_rgb`](prism_render_architecture::particle::rgb_ycocg::ycocg_to_rgb)
/// term for term and in the same evaluation order.
const RGB_YCOCG_F32_WGSL: &str = r#"
// Lossy YCoCg transform twin: one thread per sample. Each sample is a vec4<f32>
// slot (xyz carry the triple, w is unused padding so the storage buffer stays
// 16-byte aligned). Uses only + - * / in the portable core-WGSL subset, mirrors
// the CPU golden `particle::rgb_ycocg`, and takes no optional feature, so it
// runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard YCoCg colour transform; no third-party engine source or
// derived code.

struct Params {
    // Number of samples in the dispatch (one thread each).
    count: u32,
    // Padding to a 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;

// Lossy forward transform: Y = R/4 + G/2 + B/4, Co = R/2 - B/2,
// Cg = -R/4 + G/2 - B/4. The term order matches the reference exactly so the
// low mantissa bits agree.
@compute @workgroup_size(64)
fn rgb_to_ycocg(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let c = src[idx];
    let r = c.x;
    let g = c.y;
    let b = c.z;
    let y = r * 0.25 + g * 0.5 + b * 0.25;
    let co = r * 0.5 - b * 0.5;
    let cg = -r * 0.25 + g * 0.5 - b * 0.25;
    dst[idx] = vec4<f32>(y, co, cg, 0.0);
}

// Lossy inverse transform: R = Y + Co - Cg, G = Y + Cg, B = Y - Co - Cg.
@compute @workgroup_size(64)
fn ycocg_to_rgb(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let c = src[idx];
    let y = c.x;
    let co = c.y;
    let cg = c.z;
    let r = y + co - cg;
    let g = y + cg;
    let b = y - co - cg;
    dst[idx] = vec4<f32>(r, g, b, 0.0);
}
"#;

/// The portable core-`WGSL` kernels for the lossless integer `YCoCg-R` lifting
/// transform. The two entry points `rgb_to_ycocg_r` and `ycocg_r_to_rgb` mirror
/// the `CPU` golden
/// [`rgb_to_ycocg_r`](prism_render_architecture::particle::rgb_ycocg::rgb_to_ycocg_r)
/// and
/// [`ycocg_r_to_rgb`](prism_render_architecture::particle::rgb_ycocg::ycocg_r_to_rgb)
/// shift for shift, so the integer round trip is bit-exact.
const RGB_YCOCG_R_INT_WGSL: &str = r#"
// Lossless YCoCg-R lifting twin: one thread per sample, all signed-integer
// arithmetic. Each sample is a vec4<i32> slot (xyz carry the triple, w unused).
// The `>> 1u` are arithmetic shifts on i32, i.e. floor-division by two, exactly
// what makes the lifting bit-exact reversible for negative chroma. Uses only
// integer subtract/add, the arithmetic right shift and clamp in the portable
// core-WGSL subset, mirrors the CPU golden `particle::rgb_ycocg`, and takes no
// optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard YCoCg-R colour transform (H.264 lossless mode); no
// third-party engine source or derived code.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<i32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<i32>>;

// Lossless integer forward transform (lifting):
// Co = R - B, t = B + (Co >> 1), Cg = G - t, Y = t + (Cg >> 1).
@compute @workgroup_size(64)
fn rgb_to_ycocg_r(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let c = src[idx];
    let r = c.x;
    let g = c.y;
    let b = c.z;
    let co = r - b;
    let t = b + (co >> 1u);
    let cg = g - t;
    let y = t + (cg >> 1u);
    dst[idx] = vec4<i32>(y, co, cg, 0);
}

// Lossless integer inverse transform (lifting):
// t = Y - (Cg >> 1), G = Cg + t, B = t - (Co >> 1), R = B + Co.
// The final channels are clamped into [0, 255] as a defensive guard against
// out-of-contract inputs, matching the reference `clamp_u8`.
@compute @workgroup_size(64)
fn ycocg_r_to_rgb(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let c = src[idx];
    let y = c.x;
    let co = c.y;
    let cg = c.z;
    let t = y - (cg >> 1u);
    let g = cg + t;
    let b = t - (co >> 1u);
    let r = b + co;
    dst[idx] = vec4<i32>(clamp(r, 0, 255), clamp(g, 0, 255), clamp(b, 0, 255), 0);
}
"#;

/// Uniform parameters for one dispatch: the sample count plus three pad words,
/// a `16`-byte `repr(C)` block matching `Params` in both kernels.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of samples in the dispatch.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One `f32` sample slot: a `vec4<f32>` whose `xyz` carry an `RGB` or `YCoCg`
/// triple and whose `w` is unused padding keeping the slot `16`-byte aligned.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuF32x4 {
    /// First channel (`R` or `Y`).
    x: f32,
    /// Second channel (`G` or `Co`).
    y: f32,
    /// Third channel (`B` or `Cg`).
    z: f32,
    /// Unused padding word.
    w: f32,
}

/// One integer sample slot: a `vec4<i32>` whose `xyz` carry an `RGB` or
/// `YCoCg-R` triple and whose `w` is unused padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuI32x4 {
    /// First channel (`R` or `Y`).
    x: i32,
    /// Second channel (`G` or `Co`).
    y: i32,
    /// Third channel (`B` or `Cg`).
    z: i32,
    /// Unused padding word.
    w: i32,
}

/// A compiled, reusable `RGB` ↔ `YCoCg` / `YCoCg-R` pipeline set.
///
/// All four pipelines share one bind-group layout (a uniform `params` plus a
/// read-only source and a read-write destination storage buffer); only the
/// `WGSL` element type differs between the `f32` and integer kernels, which the
/// layout does not constrain, so a single layout backs both shader modules.
pub struct GpuRgbYCoCg {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    f32_module: ShaderModule,
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    int_module: ShaderModule,
    layout: BindGroupLayout,
    rgb_to_ycocg: ComputePipeline,
    ycocg_to_rgb: ComputePipeline,
    rgb_to_ycocg_r: ComputePipeline,
    ycocg_r_to_rgb: ComputePipeline,
}

impl GpuRgbYCoCg {
    /// Compiles the four `RGB` ↔ `YCoCg` / `YCoCg-R` kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRgbYCoCg {
        let device = ctx.device();
        let f32_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_rgb_ycocg_f32"),
            source: ShaderSource::Wgsl(RGB_YCOCG_F32_WGSL.into()),
        });
        let int_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_rgb_ycocg_r_int"),
            source: ShaderSource::Wgsl(RGB_YCOCG_R_INT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_rgb_ycocg_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_rgb_ycocg_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make = |module: &ShaderModule, entry: &str, label: &str| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module,
                entry_point: Some(entry),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let rgb_to_ycocg = make(
            &f32_module,
            "rgb_to_ycocg",
            "prism_volumetric_rgb_to_ycocg_pipeline",
        );
        let ycocg_to_rgb = make(
            &f32_module,
            "ycocg_to_rgb",
            "prism_volumetric_ycocg_to_rgb_pipeline",
        );
        let rgb_to_ycocg_r = make(
            &int_module,
            "rgb_to_ycocg_r",
            "prism_volumetric_rgb_to_ycocg_r_pipeline",
        );
        let ycocg_r_to_rgb = make(
            &int_module,
            "ycocg_r_to_rgb",
            "prism_volumetric_ycocg_r_to_rgb_pipeline",
        );
        GpuRgbYCoCg {
            f32_module,
            int_module,
            layout,
            rgb_to_ycocg,
            ycocg_to_rgb,
            rgb_to_ycocg_r,
            ycocg_r_to_rgb,
        }
    }

    /// Applies the lossy forward transform to every sample in `inputs`,
    /// returning one [`YCoCg`] per input in order.
    ///
    /// The returned triple for input `c` equals
    /// [`rgb_to_ycocg`](prism_render_architecture::particle::rgb_ycocg::rgb_to_ycocg)`(c)`
    /// to within the tolerance documented on this module. An empty slice yields
    /// an empty result (storage buffers cannot be zero-sized, so it is handled
    /// by an early return).
    #[must_use]
    pub fn rgb_to_ycocg(&self, ctx: &GpuContext, inputs: &[Rgb]) -> Vec<YCoCg> {
        if inputs.is_empty() {
            return Vec::new();
        }
        let packed: Vec<GpuF32x4> = inputs
            .iter()
            .map(|c| GpuF32x4 {
                x: c.r,
                y: c.g,
                z: c.b,
                w: 0.0,
            })
            .collect();
        let out = self.run_f32(ctx, &self.rgb_to_ycocg, &packed);
        out.into_iter().map(|v| YCoCg::new(v.x, v.y, v.z)).collect()
    }

    /// Applies the lossy inverse transform to every sample in `inputs`,
    /// returning one [`Rgb`] per input in order.
    ///
    /// The returned triple for input `c` equals
    /// [`ycocg_to_rgb`](prism_render_architecture::particle::rgb_ycocg::ycocg_to_rgb)`(c)`
    /// to within the tolerance documented on this module. An empty slice yields
    /// an empty result.
    #[must_use]
    pub fn ycocg_to_rgb(&self, ctx: &GpuContext, inputs: &[YCoCg]) -> Vec<Rgb> {
        if inputs.is_empty() {
            return Vec::new();
        }
        let packed: Vec<GpuF32x4> = inputs
            .iter()
            .map(|c| GpuF32x4 {
                x: c.y,
                y: c.co,
                z: c.cg,
                w: 0.0,
            })
            .collect();
        let out = self.run_f32(ctx, &self.ycocg_to_rgb, &packed);
        out.into_iter().map(|v| Rgb::new(v.x, v.y, v.z)).collect()
    }

    /// Applies the lossless integer forward lifting transform to every `[u8; 3]`
    /// sample in `inputs`, returning one [`YCoCgR`] per input in order.
    ///
    /// The result for input `rgb` equals
    /// [`rgb_to_ycocg_r`](prism_render_architecture::particle::rgb_ycocg::rgb_to_ycocg_r)`(rgb)`
    /// bit-for-bit. An empty slice yields an empty result.
    #[must_use]
    pub fn rgb_to_ycocg_r(&self, ctx: &GpuContext, inputs: &[[u8; 3]]) -> Vec<YCoCgR> {
        if inputs.is_empty() {
            return Vec::new();
        }
        let packed: Vec<GpuI32x4> = inputs
            .iter()
            .map(|c| GpuI32x4 {
                x: i32::from(c[0]),
                y: i32::from(c[1]),
                z: i32::from(c[2]),
                w: 0,
            })
            .collect();
        let out = self.run_int(ctx, &self.rgb_to_ycocg_r, &packed);
        out.into_iter()
            .map(|v| YCoCgR {
                y: v.x,
                co: v.y,
                cg: v.z,
            })
            .collect()
    }

    /// Applies the lossless integer inverse lifting transform to every
    /// [`YCoCgR`] sample in `inputs`, returning one `[u8; 3]` per input in
    /// order.
    ///
    /// The result for input `c` equals
    /// [`ycocg_r_to_rgb`](prism_render_architecture::particle::rgb_ycocg::ycocg_r_to_rgb)`(c)`
    /// bit-for-bit, including the defensive clamp into `[0, 255]`. An empty
    /// slice yields an empty result.
    #[must_use]
    pub fn ycocg_r_to_rgb(&self, ctx: &GpuContext, inputs: &[YCoCgR]) -> Vec<[u8; 3]> {
        if inputs.is_empty() {
            return Vec::new();
        }
        let packed: Vec<GpuI32x4> = inputs
            .iter()
            .map(|c| GpuI32x4 {
                x: c.y,
                y: c.co,
                z: c.cg,
                w: 0,
            })
            .collect();
        let out = self.run_int(ctx, &self.ycocg_r_to_rgb, &packed);
        out.into_iter().map(channel_i32x4_to_u8).collect()
    }

    /// Runs an `f32` kernel over `inputs`, returning the per-sample output
    /// slots. `inputs` is never empty (callers early-return on an empty slice).
    fn run_f32(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        inputs: &[GpuF32x4],
    ) -> Vec<GpuF32x4> {
        let bytes = self.dispatch(ctx, pipeline, bytemuck::cast_slice(inputs), inputs.len());
        bytemuck::cast_slice::<u8, GpuF32x4>(&bytes).to_vec()
    }

    /// Runs an integer kernel over `inputs`, returning the per-sample output
    /// slots. `inputs` is never empty (callers early-return on an empty slice).
    fn run_int(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        inputs: &[GpuI32x4],
    ) -> Vec<GpuI32x4> {
        let bytes = self.dispatch(ctx, pipeline, bytemuck::cast_slice(inputs), inputs.len());
        bytemuck::cast_slice::<u8, GpuI32x4>(&bytes).to_vec()
    }

    /// Uploads `input` (one `16`-byte slot per sample), runs `pipeline` with one
    /// thread per sample and reads the equal-sized output back as raw bytes.
    ///
    /// The input and output slots are the same `16`-byte stride for every
    /// kernel, so `out_bytes` equals the input length. `count` is always
    /// positive here, so no storage buffer is zero-sized.
    fn dispatch(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        input: &[u8],
        count: usize,
    ) -> Vec<u8> {
        let device = ctx.device();
        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let out_bytes = input.len() as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_rgb_ycocg_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let src_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_rgb_ycocg_src"),
            contents: input,
            usage: BufferUsages::STORAGE,
        });
        let dst_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_rgb_ycocg_dst"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_rgb_ycocg_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_rgb_ycocg_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: src_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: dst_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_rgb_ycocg_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_rgb_ycocg_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per sample, flattened to a 1-D dispatch.
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&dst_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let bytes = view.to_vec();
        drop(view);
        stage.unmap();
        bytes
    }
}

/// Converts an integer output slot into a clamped `[u8; 3]`.
///
/// The kernel already clamps each channel into `[0, 255]`, so the cast is
/// lossless; the extra `clamp` is a defensive mirror of the reference
/// `clamp_u8` and keeps the conversion total without a fallible cast.
fn channel_i32x4_to_u8(v: GpuI32x4) -> [u8; 3] {
    [clamp_u8(v.x), clamp_u8(v.y), clamp_u8(v.z)]
}

/// Clamps an `i32` into the `u8` range without a lossy cast, mirroring the
/// reference `clamp_u8`.
fn clamp_u8(v: i32) -> u8 {
    let clamped = v.clamp(0, 255);
    u8::try_from(clamped).unwrap_or(0)
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
