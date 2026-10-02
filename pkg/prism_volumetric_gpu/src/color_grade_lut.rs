//! `wgpu` compute twin of the 3D colour-grading `LUT` sampler
//! ([`color_grade_lut`](prism_render_architecture::particle::color_grade_lut),
//! particle design §16, §30).
//!
//! The `CPU` golden
//! [`color_grade_lut`](prism_render_architecture::particle::color_grade_lut)
//! owns the device-free maths a hardware 3D-`LUT` sampler performs: it maps an
//! input `RGB` colour into cube space (`channel * (size - 1)`), locates the
//! enclosing lattice cell with [`floor`](f32::floor), and blends the surrounding
//! lattice points. This module twins the two interpolation filters the golden
//! exposes —
//! [`ColorGradeLut::sample_trilinear`](prism_render_architecture::particle::color_grade_lut::ColorGradeLut::sample_trilinear)
//! (the 8-corner box filter built from the per-channel
//! [`Rgb::lerp`](prism_render_architecture::particle::color_grade_lut::Rgb::lerp))
//! and
//! [`ColorGradeLut::sample_tetrahedral`](prism_render_architecture::particle::color_grade_lut::ColorGradeLut::sample_tetrahedral)
//! (the 6-tetrahedron filter most hardware samplers implement). [`GpuColorGradeLut`]
//! is the on-device twin: one thread grades one input colour, so a passing
//! real-device parity test is direct evidence the ported kernel chooses the same
//! lattice cell, the same tetrahedron and the same interpolation weights the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For a batch of independent input colours, each thread fetches the shared
//! read-only cube (uploaded once as a flat `red`-fastest `f32` array in the
//! golden texel order `r + size * (g + size * b)`) and returns both the
//! trilinear and the tetrahedral graded `RGB`. The per-channel
//! [`Rgb::lerp`](prism_render_architecture::particle::color_grade_lut::Rgb::lerp)
//! is reproduced as a `vec3` affine blend `a + (b - a) * t`, and the
//! tetrahedral branch selection mirrors the reference ordering of the three
//! in-cell fractions branch for branch.
//!
//! # Correctness model
//!
//! Both filters thread the raw fractions through multiplies and adds (plus one
//! [`floor`](f32::floor) and one [`clamp`](f32::clamp) to place the cell), with
//! no transcendental call, so `CPU` and `GPU` evaluate the identical closed-form
//! algebra. They are not bit-exact only because a `GPU` may fuse a multiply-add
//! the scalar reference leaves separate, perturbing the low mantissa bits by a
//! few `ULP`. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on every
//! graded channel, tight enough to catch a wrong port (a dropped corner, a
//! swapped tetrahedron, a mis-indexed lattice) yet loose enough to admit legal
//! fused multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! A `size` of `1` yields a single-texel cube whose only lattice point is
//! returned for every input; the kernel handles this on device because the
//! clamped cell and the zero fractions collapse all eight corners onto texel
//! zero, matching the reference. A `size` of `0` is the empty cube: a storage
//! buffer cannot be zero-sized, so the host short-circuits it to the reference
//! [`Rgb::BLACK`](prism_render_architecture::particle::color_grade_lut::Rgb::BLACK)
//! guard with no dispatch. An empty query batch likewise returns an empty vector
//! with no dispatch.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — [`floor`](f32::floor),
//! [`min`](Ord::min), [`clamp`](f32::clamp), `+ - * /` and unsigned index
//! arithmetic — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse
//! trigonometry, no `sqrt` and no optional device feature, so it runs unmodified
//! on `Metal`, `Vulkan` and `DX12`. There is no loop: each thread performs a
//! fixed, bounded sequence of arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::color_grade_lut`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` colour-grading `LUT` kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `sample` mirrors
/// the `CPU` golden
/// [`color_grade_lut`](prism_render_architecture::particle::color_grade_lut)
/// branch for branch; see the module documentation for the algorithm.
const COLOR_GRADE_LUT_WGSL: &str = r#"
// Colour-grading LUT twin: one thread per input colour reproduces the trilinear
// 8-corner blend and the tetrahedral 6-cell blend of the CPU golden
// `particle::color_grade_lut`. The cube is a shared read-only flat f32 array in
// the golden texel order `r + size * (g + size * b)`, three lanes per texel. The
// kernel uses only the portable core-WGSL subset (floor/min/clamp and + - * /
// plus unsigned index math), needs no sqrt and no transcendental call and takes
// no optional feature, so it runs unmodified on Metal, Vulkan and DX12. There is
// no loop, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::color_grade_lut；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of input colours; threads past this short-circuit.
    count: u32,
    // Lattice points per axis (`size`); the flat cube holds `size^3` texels.
    size: u32,
    pad0: u32,
    pad1: u32,
}

struct Query {
    // Input RGB lookup key; a pad lane keeps the slot 16-byte aligned.
    r: f32,
    g: f32,
    b: f32,
    pad: f32,
}

struct Result {
    // Trilinear graded RGB, with a trailing pad lane.
    tri: vec3<f32>,
    pad0: f32,
    // Tetrahedral graded RGB, with a trailing pad lane.
    tet: vec3<f32>,
    pad1: f32,
}

// The split of one clamped channel into its base lattice index and in-cell
// fraction.
struct Axis {
    base: u32,
    frac: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> lut: array<f32>;
@group(0) @binding(2) var<storage, read> queries: array<Query>;
@group(0) @binding(3) var<storage, read_write> results: array<Result>;

// Maps one input channel into cube space and splits it into the base lattice
// index and the in-cell fraction; mirrors the reference `axis_coord`. The
// channel is clamped into 0..=1, scaled by `last`, and floored, with the base
// capped at `last - 1` so a query at the far face selects the final cell with a
// fraction of 1.0 instead of reading past the lattice.
fn axis_coord(channel: f32, last: u32) -> Axis {
    let scaled = clamp(channel, 0.0, 1.0) * f32(last);
    let floored = floor(scaled);
    var max_base: u32 = 0u;
    if (last > 0u) {
        max_base = last - 1u;
    }
    let base = min(u32(floored), max_base);
    var out: Axis;
    out.base = base;
    out.frac = scaled - f32(base);
    return out;
}

// Fetches the lattice point at coordinate (r, g, b), clamping each index into
// 0..=last and addressing the flat cube as `r + size * (g + size * b)` with
// three lanes per texel; mirrors the reference `texel`.
fn fetch(r: u32, g: u32, b: u32, size: u32) -> vec3<f32> {
    let last = size - 1u;
    let ri = min(r, last);
    let gi = min(g, last);
    let bi = min(b, last);
    let flat = ri + size * (gi + size * bi);
    let base = flat * 3u;
    return vec3<f32>(lut[base], lut[base + 1u], lut[base + 2u]);
}

// Component-wise linear interpolation `a + (b - a) * t`; mirrors the per-channel
// reference `Rgb::lerp` built on `lerp_scalar`.
fn lerp3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

// Affine combination of four lattice points by four weights; mirrors the
// reference `blend4`.
fn blend4(
    v0: vec3<f32>,
    w0: f32,
    v1: vec3<f32>,
    w1: f32,
    v2: vec3<f32>,
    w2: f32,
    v3: vec3<f32>,
    w3: f32,
) -> vec3<f32> {
    return v0 * w0 + v1 * w1 + v2 * w2 + v3 * w3;
}

@compute @workgroup_size(64)
fn color_grade_lut_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let size = params.size;
    let last = size - 1u;
    let q = queries[idx];

    let ax = axis_coord(q.r, last);
    let ay = axis_coord(q.g, last);
    let az = axis_coord(q.b, last);
    let br = ax.base;
    let fr = ax.frac;
    let bg = ay.base;
    let fg = ay.frac;
    let bb = az.base;
    let fb = az.frac;

    let c000 = fetch(br, bg, bb, size);
    let c100 = fetch(br + 1u, bg, bb, size);
    let c010 = fetch(br, bg + 1u, bb, size);
    let c110 = fetch(br + 1u, bg + 1u, bb, size);
    let c001 = fetch(br, bg, bb + 1u, size);
    let c101 = fetch(br + 1u, bg, bb + 1u, size);
    let c011 = fetch(br, bg + 1u, bb + 1u, size);
    let c111 = fetch(br + 1u, bg + 1u, bb + 1u, size);

    // Trilinear: interpolate along red, then green, then blue.
    let c00 = lerp3(c000, c100, fr);
    let c01 = lerp3(c001, c101, fr);
    let c10 = lerp3(c010, c110, fr);
    let c11 = lerp3(c011, c111, fr);
    let tc0 = lerp3(c00, c10, fg);
    let tc1 = lerp3(c01, c11, fg);
    let tri = lerp3(tc0, tc1, fb);

    // Tetrahedral: the unit cell splits into six tetrahedra sharing the
    // (0,0,0)->(1,1,1) diagonal; the ordering of the three fractions selects the
    // enclosing tetrahedron and its four barycentric weights, mirroring the
    // reference branch for branch.
    var tet: vec3<f32>;
    if (fr > fg) {
        if (fg > fb) {
            // fr >= fg >= fb
            tet = blend4(c000, 1.0 - fr, c100, fr - fg, c110, fg - fb, c111, fb);
        } else if (fr > fb) {
            // fr >= fb >= fg
            tet = blend4(c000, 1.0 - fr, c100, fr - fb, c101, fb - fg, c111, fg);
        } else {
            // fb >= fr >= fg
            tet = blend4(c000, 1.0 - fb, c001, fb - fr, c101, fr - fg, c111, fg);
        }
    } else if (fb > fg) {
        // fb >= fg >= fr
        tet = blend4(c000, 1.0 - fb, c001, fb - fg, c011, fg - fr, c111, fr);
    } else if (fb > fr) {
        // fg >= fb >= fr
        tet = blend4(c000, 1.0 - fg, c010, fg - fb, c011, fb - fr, c111, fr);
    } else {
        // fg >= fr >= fb
        tet = blend4(c000, 1.0 - fg, c010, fg - fr, c110, fr - fb, c111, fb);
    }

    var out: Result;
    out.tri = tri;
    out.pad0 = 0.0;
    out.tet = tet;
    out.pad1 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count, the cube `size`, and
/// two pad words filling a `16`-byte, `std140`-aligned uniform struct matching
/// `Params` in [`COLOR_GRADE_LUT_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Lattice points per axis.
    size: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// the input `RGB` channels followed by a pad word keeping the slot `16`-byte
/// aligned.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Input red channel.
    r: f32,
    /// Input green channel.
    g: f32,
    /// Input blue channel.
    b: f32,
    /// Pad lane after the channels.
    pad: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct.
/// Each graded colour occupies a `vec3` lane with a trailing pad word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Trilinear graded `RGB`.
    tri: [f32; 3],
    /// Pad lane after `tri`.
    pad0: f32,
    /// Tetrahedral graded `RGB`.
    tet: [f32; 3],
    /// Pad lane after `tet`.
    pad1: f32,
}

/// One query for the colour-grading twin: the input `RGB` lookup key graded
/// through both filters.
///
/// The input is clamped into the `0..=1` unit cube before it is mapped into
/// lattice space, exactly as the reference does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorGradeLutQuery {
    /// Input colour as `[r, g, b]` linear channels.
    pub input: [f32; 3],
}

/// One graded answer for a single input, mirroring both filters the reference
/// exposes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorGradeLutResult {
    /// Trilinear graded `RGB`, matching
    /// [`ColorGradeLut::sample_trilinear`](prism_render_architecture::particle::color_grade_lut::ColorGradeLut::sample_trilinear).
    pub trilinear: [f32; 3],
    /// Tetrahedral graded `RGB`, matching
    /// [`ColorGradeLut::sample_tetrahedral`](prism_render_architecture::particle::color_grade_lut::ColorGradeLut::sample_tetrahedral).
    pub tetrahedral: [f32; 3],
}

/// Encodes one [`ColorGradeLutQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &ColorGradeLutQuery) -> GpuQuery {
    GpuQuery {
        r: q.input[0],
        g: q.input[1],
        b: q.input[2],
        pad: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`ColorGradeLutResult`].
fn decode_result(raw: &GpuResult) -> ColorGradeLutResult {
    ColorGradeLutResult {
        trilinear: raw.tri,
        tetrahedral: raw.tet,
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

/// A compiled, reusable colour-grading `LUT` compute pipeline, twinning the
/// `CPU` golden
/// [`color_grade_lut`](prism_render_architecture::particle::color_grade_lut).
pub struct GpuColorGradeLut {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuColorGradeLut {
    /// Compiles the colour-grading `LUT` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuColorGradeLut {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_color_grade_lut"),
            source: ShaderSource::Wgsl(COLOR_GRADE_LUT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_color_grade_lut_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_color_grade_lut_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_color_grade_lut_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("color_grade_lut_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuColorGradeLut {
            module,
            layout,
            pipeline,
        }
    }

    /// Grades every input in `queries` against the cube `lut` and returns one
    /// [`ColorGradeLutResult`] per input, in order.
    ///
    /// The `lut` slice is the cube packed `red`-fastest in the golden texel
    /// order `r + size * (g + size * b)` with three contiguous `f32` channels
    /// per texel, so its length must equal `size * size * size * 3`. The graded
    /// colours match the reference
    /// [`ColorGradeLut::sample_trilinear`](prism_render_architecture::particle::color_grade_lut::ColorGradeLut::sample_trilinear)
    /// and
    /// [`ColorGradeLut::sample_tetrahedral`](prism_render_architecture::particle::color_grade_lut::ColorGradeLut::sample_tetrahedral)
    /// of a cube carrying the same texels, to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch, and a `size` of `0` (the empty cube) returns the reference
    /// black guard for every query with no dispatch, since a storage buffer
    /// cannot be zero-sized.
    ///
    /// # Panics
    ///
    /// Panics if `size` is non-zero and `lut.len()` does not equal
    /// `size * size * size * 3`.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        lut: &[f32],
        size: usize,
        queries: &[ColorGradeLutQuery],
    ) -> Vec<ColorGradeLutResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        if size == 0 {
            // Empty cube: a storage buffer cannot be zero-sized, so mirror the
            // reference black guard on the host without a dispatch.
            return queries
                .iter()
                .map(|_| ColorGradeLutResult {
                    trilinear: [0.0, 0.0, 0.0],
                    tetrahedral: [0.0, 0.0, 0.0],
                })
                .collect();
        }
        assert_eq!(
            lut.len(),
            size.saturating_mul(size)
                .saturating_mul(size)
                .saturating_mul(3),
            "lut length must equal size * size * size * 3 channels"
        );
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            size: size as u32,
            pad0: 0,
            pad1: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_color_grade_lut_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let lut_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_color_grade_lut_lut"),
            contents: bytemuck::cast_slice(lut),
            usage: BufferUsages::STORAGE,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_color_grade_lut_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_color_grade_lut_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_color_grade_lut_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: lut_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_color_grade_lut_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_color_grade_lut_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_color_grade_lut_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per input colour, flattened to a 1-D dispatch.
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
