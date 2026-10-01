//! `wgpu` compute twin of the premultiplied-alpha conversion and the
//! `Porter-Duff` compositing algebra
//! ([`premultiply_alpha`](prism_render_architecture::particle::premultiply_alpha),
//! design §5, §12).
//!
//! Blending is where a particle system finally decides *how* a fragment's
//! colour is written over what is already in the target. The classic reference
//! is the `Porter-Duff` operator family: twelve ways two coverage-weighted
//! colours can be combined, each expressed as `result = src * Fa + dst * Fb`
//! for a pair of per-operator blend factors that depend only on the two alphas,
//! plus the additive `lighter` blend (`Plus` / `Add`) used for glows and
//! sparks. The `CPU` golden
//! [`premultiply_alpha`](prism_render_architecture::particle::premultiply_alpha)
//! owns that math; [`GpuPremultiplyAlpha`] is the on-device twin that runs one
//! thread per colour and reproduces the same results
//! [`straight_to_premul`](prism_render_architecture::particle::premultiply_alpha::straight_to_premul),
//! [`premul_to_straight`](prism_render_architecture::particle::premultiply_alpha::premul_to_straight),
//! [`composite`](prism_render_architecture::particle::premultiply_alpha::composite)
//! and
//! [`composite_over_batch`](prism_render_architecture::particle::premultiply_alpha::composite_over_batch)
//! produce. A passing real-device parity test is therefore direct evidence the
//! ported kernels fold coverage, divide it back out, select the same blend
//! factors and clamp the same additive sum the reference does, not merely that
//! the shaders compile.
//!
//! # What is twinned
//!
//! Three kernels cover the whole module:
//!
//! * `straight_to_premul` folds coverage into the colour (`rgb *= a`), the
//!   alpha lane preserved.
//! * `premul_to_straight` divides coverage back out (`rgb /= a`) with the exact
//!   same [`ALPHA_FLOOR`](Self::ALPHA_FLOOR) degenerate guard: an alpha at or
//!   below the floor yields a fully transparent (all-zero) colour instead of a
//!   division by (near-)zero, bit-for-bit the same branch the reference takes.
//! * `composite` reproduces every [`BlendOp`] variant. A per-element `u32` tag
//!   selects the operator so one dispatch can mix all fourteen variants: the
//!   twelve `Porter-Duff` operators evaluate `src * Fa + dst * Fb` with the same
//!   alpha-dependent `(Fa, Fb)` the reference's `porter_duff_factors` returns,
//!   and `Plus` / `Add` instead take the per-channel sum clamped to `1.0`.
//!   [`GpuPremultiplyAlpha::composite_over_batch`] is the common
//!   [`BlendOp::SrcOver`] workhorse expressed as a batch dispatch over that same
//!   kernel.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — `clamp`, the four
//! arithmetic operators `+ - * /` on scalars and vectors, an unsigned equality
//! test and a `switch` on the integer tag — with no `sin`, `cos`, `exp`, `log`,
//! `pow` or `sqrt` and no optional device feature, so they run unmodified on
//! `Metal`, `Vulkan` and `DX12`. The only division is the single `1.0 / a`
//! reciprocal in `premul_to_straight`, guarded by the alpha floor so it can
//! never divide by (near-)zero.
//!
//! # Correctness model
//!
//! Each colour is a fixed, non-reorderable closed-form expression — a handful
//! of multiplies, adds, one guarded reciprocal and one clamp — so `CPU` and
//! `GPU` evaluate the same algebra in the same order. They are not bit-exact: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few units in the last place. The
//! parity test therefore asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`), tight enough to catch a genuinely wrong port (a swapped
//! blend factor, a dropped clamp, a missing degenerate guard) yet loose enough
//! to admit legal fused multiply-add contraction. The degenerate `alpha = 0`
//! branch is exact on both sides because the comparison against the floor is an
//! ordinary ordered compare on identical `f32` bits, so the two never diverge
//! on which side of the floor they land.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `Porter-Duff` compositing algebra (`Porter` and `Duff`,
//! "Compositing Digital Images", `SIGGRAPH` 1984) plus `wgpu` compute dispatch;
//! no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::premultiply_alpha::{BlendOp, PremulRgba, Rgba};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// used across this crate; one thread handles one colour and the dispatch is
/// flattened to a single linear index so it stays one-dimensional.
const WORKGROUP_SIZE: u32 = 64;

/// Scalar components per colour: an `RGBA` quad packed as one `std430`
/// `vec4<f32>` slot (`[r, g, b, a]`), the same layout
/// [`PremulRgba::to_std430`](prism_render_architecture::particle::premultiply_alpha::PremulRgba::to_std430)
/// writes.
const CHANNELS: usize = 4;

/// The portable core-`WGSL` conversion and compositing kernels, embedded inline
/// so the twin ships as a single source file. The three entry points
/// `straight_to_premul`, `premul_to_straight` and `composite` mirror the `CPU`
/// golden functions of the same name; see the module documentation for the
/// algebra.
const PREMULTIPLY_ALPHA_WGSL: &str = r#"
// Premultiplied-alpha / Porter-Duff twin: one thread per colour. The two
// conversion kernels fold coverage into the colour and divide it back out (with
// a degenerate alpha-floor guard); the composite kernel selects a Porter-Duff
// blend-factor pair by per-element tag or takes the clamped additive sum. All
// mirror the CPU golden `particle::premultiply_alpha`, use only the portable
// core-WGSL subset (clamp, + - * /, unsigned compare and a switch on the tag),
// and take no optional feature, so they run unmodified on Metal, Vulkan and
// DX12.
//
// Provenance: standard Porter-Duff compositing (Porter & Duff, SIGGRAPH 1984);
// no third-party engine source or derived code.

struct Params {
    // Number of colours this dispatch processes; one thread each.
    count: u32,
    // Padding to a 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// Alphas at or below this magnitude are treated as fully transparent when
// dividing colour back out, so the reciprocal can never explode. This matches
// the reference `ALPHA_FLOOR` exactly.
const ALPHA_FLOOR: f32 = 1e-6;

// Blend-op tags, matching the host `blend_op_tag` mapping one-for-one.
const OP_CLEAR: u32 = 0u;
const OP_SRC: u32 = 1u;
const OP_DST: u32 = 2u;
const OP_SRC_OVER: u32 = 3u;
const OP_DST_OVER: u32 = 4u;
const OP_SRC_IN: u32 = 5u;
const OP_DST_IN: u32 = 6u;
const OP_SRC_OUT: u32 = 7u;
const OP_DST_OUT: u32 = 8u;
const OP_SRC_ATOP: u32 = 9u;
const OP_DST_ATOP: u32 = 10u;
const OP_XOR: u32 = 11u;
const OP_PLUS: u32 = 12u;
const OP_ADD: u32 = 13u;

@group(0) @binding(0) var<uniform> params: Params;

// Conversion kernels bind a single input and a single output colour array.
@group(0) @binding(1) var<storage, read> conv_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> conv_out: array<vec4<f32>>;

@compute @workgroup_size(64)
fn straight_to_premul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let c = conv_in[idx];
    // Fold coverage into the colour (`rgb *= a`); the alpha lane is preserved.
    conv_out[idx] = vec4<f32>(c.x * c.w, c.y * c.w, c.z * c.w, c.w);
}

@compute @workgroup_size(64)
fn premul_to_straight(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let c = conv_in[idx];
    // Divide coverage back out (`rgb /= a`); below the floor the straight colour
    // is undefined, so return a zeroed colour instead of dividing by zero. The
    // ordered compare on identical bits takes the same branch the reference does.
    if (c.w > ALPHA_FLOOR) {
        let inv = 1.0 / c.w;
        conv_out[idx] = vec4<f32>(c.x * inv, c.y * inv, c.z * inv, c.w);
    } else {
        conv_out[idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
}

// The composite kernel binds a per-element op tag plus the two input colours
// and one output. Rebinding `conv_in`/`conv_out` would collide, so the
// composite layout uses its own bindings 1..=4.
@group(0) @binding(1) var<storage, read> ops: array<u32>;
@group(0) @binding(2) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> dst: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> composited: array<vec4<f32>>;

// Returns the `(Fa, Fb)` Porter-Duff blend factors for the two alphas. The
// additive operators are handled by the caller and never reach this switch.
fn porter_duff_factors(tag: u32, alpha_src: f32, alpha_dst: f32) -> vec2<f32> {
    switch tag {
        case OP_CLEAR: { return vec2<f32>(0.0, 0.0); }
        case OP_SRC: { return vec2<f32>(1.0, 0.0); }
        case OP_DST: { return vec2<f32>(0.0, 1.0); }
        case OP_SRC_OVER: { return vec2<f32>(1.0, 1.0 - alpha_src); }
        case OP_DST_OVER: { return vec2<f32>(1.0 - alpha_dst, 1.0); }
        case OP_SRC_IN: { return vec2<f32>(alpha_dst, 0.0); }
        case OP_DST_IN: { return vec2<f32>(0.0, alpha_src); }
        case OP_SRC_OUT: { return vec2<f32>(1.0 - alpha_dst, 0.0); }
        case OP_DST_OUT: { return vec2<f32>(0.0, 1.0 - alpha_src); }
        case OP_SRC_ATOP: { return vec2<f32>(alpha_dst, 1.0 - alpha_src); }
        case OP_DST_ATOP: { return vec2<f32>(1.0 - alpha_dst, alpha_src); }
        case OP_XOR: { return vec2<f32>(1.0 - alpha_dst, 1.0 - alpha_src); }
        default: { return vec2<f32>(0.0, 0.0); }
    }
}

@compute @workgroup_size(64)
fn composite(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let s = src[idx];
    let d = dst[idx];
    let tag = ops[idx];
    if (tag == OP_PLUS || tag == OP_ADD) {
        // Additive `lighter` blend: per-channel sum clamped to `1.0`, every lane
        // (including alpha) treated identically to the reference.
        composited[idx] = clamp(s + d, vec4<f32>(0.0), vec4<f32>(1.0));
    } else {
        // `result = src * Fa + dst * Fb`, the same scale-and-sum order and the
        // same factors applied to all four lanes the reference `combine` uses.
        let f = porter_duff_factors(tag, s.w, d.w);
        composited[idx] = s * f.x + d * f.y;
    }
}
"#;

/// Maps a [`BlendOp`] to the `u32` tag the `composite` kernel branches on. The
/// tag values match the `OP_*` constants in [`PREMULTIPLY_ALPHA_WGSL`]
/// one-for-one; [`BlendOp::Add`] shares [`BlendOp::Plus`]'s tag only in that
/// both select the additive branch, but each keeps a distinct value so the
/// mapping is total.
#[must_use]
fn blend_op_tag(op: BlendOp) -> u32 {
    match op {
        BlendOp::Clear => 0,
        BlendOp::Src => 1,
        BlendOp::Dst => 2,
        BlendOp::SrcOver => 3,
        BlendOp::DstOver => 4,
        BlendOp::SrcIn => 5,
        BlendOp::DstIn => 6,
        BlendOp::SrcOut => 7,
        BlendOp::DstOut => 8,
        BlendOp::SrcAtop => 9,
        BlendOp::DstAtop => 10,
        BlendOp::Xor => 11,
        BlendOp::Plus => 12,
        BlendOp::Add => 13,
    }
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`PREMULTIPLY_ALPHA_WGSL`]: the colour `count` and three pad
/// words — `16` bytes, each field at the `std140` uniform offset the shader
/// expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of colours this dispatch processes.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

impl GpuParams {
    /// Packs the colour count for one dispatch.
    fn new(count: usize) -> GpuParams {
        GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        }
    }
}

/// A compiled, reusable premultiplied-alpha / `Porter-Duff` pipeline set: the
/// two conversion kernels and the operator-tagged composite kernel.
pub struct GpuPremultiplyAlpha {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    convert_layout: BindGroupLayout,
    composite_layout: BindGroupLayout,
    pipeline_straight_to_premul: ComputePipeline,
    pipeline_premul_to_straight: ComputePipeline,
    pipeline_composite: ComputePipeline,
}

impl GpuPremultiplyAlpha {
    /// Alpha floor below which [`GpuPremultiplyAlpha::premul_to_straight`]
    /// returns a fully transparent colour instead of dividing coverage out,
    /// matching the reference `ALPHA_FLOOR` and the `ALPHA_FLOOR` constant in
    /// the kernel source.
    pub const ALPHA_FLOOR: f32 = 1e-6;

    /// Compiles the conversion and composite kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPremultiplyAlpha {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_premultiply_alpha"),
            source: ShaderSource::Wgsl(PREMULTIPLY_ALPHA_WGSL.into()),
        });

        // Conversion kernels: uniform + one input colour array + one output.
        let convert_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_premultiply_alpha_convert_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        // Composite kernel: uniform + op tags + two input colour arrays + output.
        let composite_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_premultiply_alpha_composite_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });

        let convert_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_premultiply_alpha_convert_pipeline_layout"),
            bind_group_layouts: &[Some(&convert_layout)],
            immediate_size: 0,
        });
        let composite_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_premultiply_alpha_composite_pipeline_layout"),
            bind_group_layouts: &[Some(&composite_layout)],
            immediate_size: 0,
        });

        let pipeline_straight_to_premul =
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("prism_volumetric_premultiply_alpha_straight_to_premul_pipeline"),
                layout: Some(&convert_pipeline_layout),
                module: &module,
                entry_point: Some("straight_to_premul"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            });
        let pipeline_premul_to_straight =
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("prism_volumetric_premultiply_alpha_premul_to_straight_pipeline"),
                layout: Some(&convert_pipeline_layout),
                module: &module,
                entry_point: Some("premul_to_straight"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            });
        let pipeline_composite = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_premultiply_alpha_composite_pipeline"),
            layout: Some(&composite_pipeline_layout),
            module: &module,
            entry_point: Some("composite"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        GpuPremultiplyAlpha {
            module,
            convert_layout,
            composite_layout,
            pipeline_straight_to_premul,
            pipeline_premul_to_straight,
            pipeline_composite,
        }
    }

    /// Converts each straight [`Rgba`] to premultiplied form on-device
    /// (`rgb *= a`), returning the premultiplied colours in input order.
    ///
    /// The result equals mapping
    /// [`straight_to_premul`](prism_render_architecture::particle::premultiply_alpha::straight_to_premul)
    /// over `colors` to within the tolerance documented on this module. An empty
    /// input returns an empty vector with no dispatch issued, since a storage
    /// buffer cannot be zero-sized.
    #[must_use]
    pub fn straight_to_premul(&self, ctx: &GpuContext, colors: &[Rgba]) -> Vec<PremulRgba> {
        if colors.is_empty() {
            return Vec::new();
        }
        let flat = flatten_rgba(colors);
        let quads = self.run_convert(ctx, &self.pipeline_straight_to_premul, &flat, colors.len());
        quads
            .into_iter()
            .map(|q| PremulRgba::new(q[0], q[1], q[2], q[3]))
            .collect()
    }

    /// Converts each premultiplied [`PremulRgba`] back to straight form
    /// on-device (`rgb /= a`), returning the straight colours in input order.
    ///
    /// The result equals mapping
    /// [`premul_to_straight`](prism_render_architecture::particle::premultiply_alpha::premul_to_straight)
    /// over `colors` to within the tolerance documented on this module,
    /// including the degenerate [`ALPHA_FLOOR`](Self::ALPHA_FLOOR) branch: an
    /// alpha at or below the floor yields a fully transparent colour. An empty
    /// input returns an empty vector with no dispatch issued.
    #[must_use]
    pub fn premul_to_straight(&self, ctx: &GpuContext, colors: &[PremulRgba]) -> Vec<Rgba> {
        if colors.is_empty() {
            return Vec::new();
        }
        let flat = flatten_premul(colors);
        let quads = self.run_convert(ctx, &self.pipeline_premul_to_straight, &flat, colors.len());
        quads
            .into_iter()
            .map(|q| Rgba::new(q[0], q[1], q[2], q[3]))
            .collect()
    }

    /// Composites each `src[i]` with `dst[i]` under `ops[i]` on-device,
    /// returning the premultiplied results.
    ///
    /// Each element equals
    /// [`composite(ops[i], src[i], dst[i])`](prism_render_architecture::particle::premultiply_alpha::composite)
    /// to within the tolerance documented on this module, so a single dispatch
    /// can mix every [`BlendOp`] variant. The result length is the shortest of
    /// the three inputs so a length mismatch can never index out of bounds; an
    /// empty effective length returns an empty vector with no dispatch issued.
    #[must_use]
    pub fn composite(
        &self,
        ctx: &GpuContext,
        ops: &[BlendOp],
        src: &[PremulRgba],
        dst: &[PremulRgba],
    ) -> Vec<PremulRgba> {
        let count = ops.len().min(src.len()).min(dst.len());
        if count == 0 {
            return Vec::new();
        }
        let tags: Vec<u32> = ops[..count].iter().map(|&op| blend_op_tag(op)).collect();
        let src_flat = flatten_premul(&src[..count]);
        let dst_flat = flatten_premul(&dst[..count]);
        let quads = self.run_composite(ctx, &tags, &src_flat, &dst_flat, count);
        quads
            .into_iter()
            .map(|q| PremulRgba::new(q[0], q[1], q[2], q[3]))
            .collect()
    }

    /// Composites each `src[i]` over the matching `dst[i]` with
    /// [`BlendOp::SrcOver`], the workhorse particle blend, as a single batch
    /// dispatch.
    ///
    /// Mirrors the reference
    /// [`composite_over_batch`](prism_render_architecture::particle::premultiply_alpha::composite_over_batch):
    /// the result length is the shorter of the two inputs. An empty effective
    /// length returns an empty vector with no dispatch issued.
    #[must_use]
    pub fn composite_over_batch(
        &self,
        ctx: &GpuContext,
        src: &[PremulRgba],
        dst: &[PremulRgba],
    ) -> Vec<PremulRgba> {
        let count = src.len().min(dst.len());
        if count == 0 {
            return Vec::new();
        }
        let ops = vec![BlendOp::SrcOver; count];
        self.composite(ctx, &ops, src, dst)
    }

    /// Dispatches a conversion kernel over `count` colours and reads the result
    /// back as `[r, g, b, a]` quads. `flat_in` is the row-major, 4-per-colour
    /// `f32` input the device storage buffer expects.
    fn run_convert(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        flat_in: &[f32],
        count: usize,
    ) -> Vec<[f32; 4]> {
        let device = ctx.device();
        let in_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_premultiply_alpha_convert_in"),
            contents: bytemuck::cast_slice(flat_in),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = empty_color_buffer(device, count);
        let params_buf = self.params_buffer(device, count);
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_premultiply_alpha_convert_bind_group"),
            layout: &self.convert_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: in_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        dispatch_and_read(ctx, pipeline, &bind_group, &out_buf, count)
    }

    /// Dispatches the composite kernel over `count` colours and reads the result
    /// back as `[r, g, b, a]` quads.
    fn run_composite(
        &self,
        ctx: &GpuContext,
        tags: &[u32],
        src_flat: &[f32],
        dst_flat: &[f32],
        count: usize,
    ) -> Vec<[f32; 4]> {
        let device = ctx.device();
        let ops_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_premultiply_alpha_composite_ops"),
            contents: bytemuck::cast_slice(tags),
            usage: BufferUsages::STORAGE,
        });
        let src_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_premultiply_alpha_composite_src"),
            contents: bytemuck::cast_slice(src_flat),
            usage: BufferUsages::STORAGE,
        });
        let dst_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_premultiply_alpha_composite_dst"),
            contents: bytemuck::cast_slice(dst_flat),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = empty_color_buffer(device, count);
        let params_buf = self.params_buffer(device, count);
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_premultiply_alpha_composite_bind_group"),
            layout: &self.composite_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: ops_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: src_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: dst_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        dispatch_and_read(ctx, &self.pipeline_composite, &bind_group, &out_buf, count)
    }

    /// Builds the single-word uniform parameter buffer for one dispatch.
    fn params_buffer(&self, device: &wgpu::Device, count: usize) -> Buffer {
        device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_premultiply_alpha_params"),
            contents: bytemuck::bytes_of(&GpuParams::new(count)),
            usage: BufferUsages::UNIFORM,
        })
    }
}

/// Records one dispatch of `pipeline` with `bind_group`, copies `out_buf` to a
/// mappable staging buffer, waits for the device and reads the result back as
/// `count` `[r, g, b, a]` quads.
fn dispatch_and_read(
    ctx: &GpuContext,
    pipeline: &ComputePipeline,
    bind_group: &BindGroup,
    out_buf: &Buffer,
    count: usize,
) -> Vec<[f32; 4]> {
    let device = ctx.device();
    let out_bytes = (count * CHANNELS * size_of::<f32>()) as u64;
    let stage = device.create_buffer(&BufferDescriptor {
        label: Some("prism_volumetric_premultiply_alpha_stage"),
        size: out_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("prism_volumetric_premultiply_alpha_encoder"),
    });
    {
        let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism_volumetric_premultiply_alpha_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        // One thread per colour, flattened to a 1-D dispatch.
        pass.dispatch_workgroups(groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(out_buf, 0, &stage, 0, out_bytes);
    ctx.queue().submit([encoder.finish()]);

    stage.slice(..).map_async(MapMode::Read, |_| {});
    ctx.wait();
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    stage.unmap();

    let mut quads: Vec<[f32; 4]> = Vec::with_capacity(count);
    for chunk in flat.chunks_exact(CHANNELS) {
        quads.push([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    debug_assert_eq!(quads.len(), count);
    quads
}

/// Flattens straight [`Rgba`] colours into the row-major, 4-per-colour `f32`
/// layout the device storage buffers expect.
fn flatten_rgba(colors: &[Rgba]) -> Vec<f32> {
    let mut data = Vec::with_capacity(colors.len() * CHANNELS);
    for c in colors {
        data.push(c.r);
        data.push(c.g);
        data.push(c.b);
        data.push(c.a);
    }
    data
}

/// Flattens premultiplied [`PremulRgba`] colours into the row-major,
/// 4-per-colour `f32` layout the device storage buffers expect.
fn flatten_premul(colors: &[PremulRgba]) -> Vec<f32> {
    let mut data = Vec::with_capacity(colors.len() * CHANNELS);
    for c in colors {
        data.push(c.r);
        data.push(c.g);
        data.push(c.b);
        data.push(c.a);
    }
    data
}

/// Allocates an uninitialized storage buffer sized for `count` `RGBA` colours,
/// usable as a kernel output and readable via a staging copy.
fn empty_color_buffer(device: &wgpu::Device, count: usize) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some("prism_volumetric_premultiply_alpha_out"),
        size: (count * CHANNELS * size_of::<f32>()) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
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
