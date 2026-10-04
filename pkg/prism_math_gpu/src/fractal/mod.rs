//! Host orchestration of the fractal-noise compute kernels (§24.1 twin; the
//! multi-octave terrain / cloud / procedural-texture path — fBm, turbulence,
//! and ridged multifractal — driven straight onto the GPU).
//!
//! [`GpuFractal`] batch-evaluates a run of 2D or 3D coordinates **on a real
//! device**, mirroring the CPU reference [`prism_math::noise::Fractal`]'s
//! [`fbm2`](prism_math::noise::Fractal::fbm2),
//! [`fbm3`](prism_math::noise::Fractal::fbm3),
//! [`turbulence2`](prism_math::noise::Fractal::turbulence2), and
//! [`ridged2`](prism_math::noise::Fractal::ridged2). The octave-summation math
//! is **not** duplicated here: every kernel is composed at runtime from the
//! single-sourced fragment
//! [`WGSL_FRACTAL`](prism_math::shader_mirror::WGSL_FRACTAL), so the device
//! sampler cannot silently drift from the CPU reference.
//!
//! # Base source selection (one fragment, never both)
//!
//! The fractal fragment calls `prism_base_sample2` / `prism_base_sample3`,
//! which it does not define. [`GpuFractal::new`] takes a [`NoiseSource`] and
//! prepends exactly one base-noise fragment —
//! [`WGSL_PERLIN`](prism_math::shader_mirror::WGSL_PERLIN) or
//! [`WGSL_SIMPLEX`](prism_math::shader_mirror::WGSL_SIMPLEX) — plus a two-line
//! alias forwarding `prism_base_sample*` to that source's `get2` / `get3`.
//! Only one base fragment is prepended, because both declare the same
//! `@binding(1)` permutation storage array and prepending both would collide.
//!
//! # Permutation table (uploaded, not rebuilt)
//!
//! As with the Perlin and Simplex twins, each call uploads the exact seeded
//! 512-entry permutation table from the matching CPU generator as a
//! `[u32; 512]` storage buffer, so every integer hash lookup inside the base
//! sampler is **bit-exact**. Only the octave-weighted floating-point sum (plus
//! the base `fade` / `grad` / corner-attenuation math) carries a tolerance.
//!
//! # Parity contract (honest boundary)
//!
//! The integer index path of the base sampler is bit-exact; only the
//! floating-point octave accumulation and the base noise's own float math are
//! approximate. Metal compiles WGSL under fast-math, so parity is a tight
//! absolute+relative tolerance, not bit-exact equality. The forward evaluation
//! has no inverse, so there are only forward kernels.
//!
//! Standard `wgpu` compute orchestration. No neural, learned, or data-driven
//! components. No Unreal Engine or Unity source or derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::noise::{Fractal, Perlin, Simplex};
use prism_math::shader_mirror::{WGSL_FRACTAL, WGSL_PERLIN, WGSL_SIMPLEX};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, CommandEncoder,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

use crate::buffer;
use crate::context::GpuContext;

/// Threads per workgroup; a standard 1D batch tiling.
const WORKGROUP: u32 = 64;

/// Which base gradient-noise field the fractal octaves sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoiseSource {
    /// Classic Perlin gradient noise.
    Perlin,
    /// Simplex gradient noise (simplicial grid, fewer directional artifacts).
    Simplex,
}

/// Two-line alias forwarding the fractal fragment's `prism_base_sample*` calls
/// to the Perlin base sampler.
const ALIAS_PERLIN: &str = "\
fn prism_base_sample2(x: f32, y: f32) -> f32 { return prism_perlin_get2(x, y); }\n\
fn prism_base_sample3(x: f32, y: f32, z: f32) -> f32 { return prism_perlin_get3(x, y, z); }\n";

/// Two-line alias forwarding the fractal fragment's `prism_base_sample*` calls
/// to the Simplex base sampler.
const ALIAS_SIMPLEX: &str = "\
fn prism_base_sample2(x: f32, y: f32) -> f32 { return prism_simplex_get2(x, y); }\n\
fn prism_base_sample3(x: f32, y: f32, z: f32) -> f32 { return prism_simplex_get3(x, y, z); }\n";

/// Compute wrapper for 2D fBm over each `vec2<f32>` input.
const WRAP_FBM2: &str = "\
struct Params { count: u32, octaves: u32, lacunarity: f32, gain: f32, frequency: f32, pad0: u32, pad1: u32, pad2: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(2) var<storage, read> src: array<vec2<f32>>;\n\
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let p = src[i];\n\
    dst[i] = prism_fbm2(p.x, p.y, params.octaves, params.lacunarity, params.gain, params.frequency);\n\
}\n";

/// Compute wrapper for 3D fBm over each `vec4<f32>` input (`.w` ignored).
const WRAP_FBM3: &str = "\
struct Params { count: u32, octaves: u32, lacunarity: f32, gain: f32, frequency: f32, pad0: u32, pad1: u32, pad2: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(2) var<storage, read> src: array<vec4<f32>>;\n\
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let p = src[i];\n\
    dst[i] = prism_fbm3(p.x, p.y, p.z, params.octaves, params.lacunarity, params.gain, params.frequency);\n\
}\n";

/// Compute wrapper for 2D turbulence over each `vec2<f32>` input.
const WRAP_TURBULENCE2: &str = "\
struct Params { count: u32, octaves: u32, lacunarity: f32, gain: f32, frequency: f32, pad0: u32, pad1: u32, pad2: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(2) var<storage, read> src: array<vec2<f32>>;\n\
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let p = src[i];\n\
    dst[i] = prism_turbulence2(p.x, p.y, params.octaves, params.lacunarity, params.gain, params.frequency);\n\
}\n";

/// Compute wrapper for 2D ridged multifractal over each `vec2<f32>` input.
const WRAP_RIDGED2: &str = "\
struct Params { count: u32, octaves: u32, lacunarity: f32, gain: f32, frequency: f32, pad0: u32, pad1: u32, pad2: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(2) var<storage, read> src: array<vec2<f32>>;\n\
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let p = src[i];\n\
    dst[i] = prism_ridged2(p.x, p.y, params.octaves, params.lacunarity, params.gain, params.frequency);\n\
}\n";

/// Uniform block mirroring [`Fractal`] plus the batch bounds count. Explicitly
/// padded to 32 bytes so the field offsets (`count`\@0, `octaves`\@4,
/// `lacunarity`\@8, `gain`\@12, `frequency`\@16) match the WGSL `Params` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    octaves: u32,
    lacunarity: f32,
    gain: f32,
    frequency: f32,
    _pad: [u32; 3],
}

/// Real-device twin of the fractal-noise evaluation path.
///
/// Build it once per device and base source with [`GpuFractal::new`]; the four
/// forward pipelines (fBm 2D/3D, turbulence 2D, ridged 2D) and the shared
/// four-entry bind-group layout are created up front and reused across batches,
/// seeds, and [`Fractal`] parameter sets. The seeded permutation table is
/// uploaded per call, so one `GpuFractal` serves every seed on its device.
pub struct GpuFractal {
    source: NoiseSource,
    fbm2: ComputePipeline,
    fbm3: ComputePipeline,
    turbulence2: ComputePipeline,
    ridged2: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuFractal {
    /// Compiles all four forward pipelines on `ctx`'s device for the chosen
    /// base [`NoiseSource`], embedding the single-sourced [`WGSL_FRACTAL`]
    /// fragment over the matching base fragment.
    #[must_use]
    pub fn new(ctx: &GpuContext, source: NoiseSource) -> Self {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_fractal_layout"),
            entries: &[
                buffer_layout(
                    0,
                    BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                buffer_layout(
                    1,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                buffer_layout(
                    2,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                buffer_layout(
                    3,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_math_fractal_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let fbm2 = compile(device, &pipeline_layout, source, WRAP_FBM2, "prism_math_fractal_fbm2");
        let fbm3 = compile(device, &pipeline_layout, source, WRAP_FBM3, "prism_math_fractal_fbm3");
        let turbulence2 = compile(
            device,
            &pipeline_layout,
            source,
            WRAP_TURBULENCE2,
            "prism_math_fractal_turbulence2",
        );
        let ridged2 = compile(
            device,
            &pipeline_layout,
            source,
            WRAP_RIDGED2,
            "prism_math_fractal_ridged2",
        );
        GpuFractal {
            source,
            fbm2,
            fbm3,
            turbulence2,
            ridged2,
            layout,
        }
    }

    /// Base noise source this twin was built for.
    #[must_use]
    pub fn source(&self) -> NoiseSource {
        self.source
    }

    /// Batch-evaluates 2D fBm at each `[x, y]` on the device for the field
    /// seeded by `seed` with `params`, mirroring [`Fractal::fbm2`].
    #[must_use]
    pub fn fbm2(
        &self,
        ctx: &GpuContext,
        seed: u64,
        params: &Fractal,
        points: &[[f32; 2]],
    ) -> Vec<f32> {
        self.run2(ctx, &self.fbm2, seed, params, points)
    }

    /// Batch-evaluates 2D turbulence at each `[x, y]`, mirroring
    /// [`Fractal::turbulence2`].
    #[must_use]
    pub fn turbulence2(
        &self,
        ctx: &GpuContext,
        seed: u64,
        params: &Fractal,
        points: &[[f32; 2]],
    ) -> Vec<f32> {
        self.run2(ctx, &self.turbulence2, seed, params, points)
    }

    /// Batch-evaluates 2D ridged multifractal at each `[x, y]`, mirroring
    /// [`Fractal::ridged2`].
    #[must_use]
    pub fn ridged2(
        &self,
        ctx: &GpuContext,
        seed: u64,
        params: &Fractal,
        points: &[[f32; 2]],
    ) -> Vec<f32> {
        self.run2(ctx, &self.ridged2, seed, params, points)
    }

    /// Batch-evaluates 3D fBm at each `[x, y, z]`, mirroring [`Fractal::fbm3`].
    #[must_use]
    pub fn fbm3(
        &self,
        ctx: &GpuContext,
        seed: u64,
        params: &Fractal,
        points: &[[f32; 3]],
    ) -> Vec<f32> {
        let n = points.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let perm = self.perm_u32(seed);
        let perm_buf = buffer::storage_read(device, "prism_math_fractal_perm", &perm);
        // Pad to `[f32; 4]` so the upload stride matches `array<vec4<f32>>`.
        let padded: Vec<[f32; 4]> = points.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
        let in_buf = buffer::storage_read(device, "prism_math_fractal_in3", &padded);
        self.run(ctx, &self.fbm3, params, &perm_buf, &in_buf, n)
    }

    /// Shared 2D path: upload the seeded perm table and `vec2` inputs, dispatch.
    fn run2(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        seed: u64,
        params: &Fractal,
        points: &[[f32; 2]],
    ) -> Vec<f32> {
        let n = points.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let perm = self.perm_u32(seed);
        let perm_buf = buffer::storage_read(device, "prism_math_fractal_perm", &perm);
        let in_buf = buffer::storage_read(device, "prism_math_fractal_in2", points);
        self.run(ctx, pipeline, params, &perm_buf, &in_buf, n)
    }

    /// Shared dispatch/readback for any kernel: allocates the `f32` output,
    /// binds the four buffers, dispatches, and reads the result back.
    fn run(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        params: &Fractal,
        perm_buf: &Buffer,
        in_buf: &Buffer,
        n: usize,
    ) -> Vec<f32> {
        let device = ctx.device();
        let out_bytes = (n * size_of::<f32>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_fractal_out", out_bytes);
        let bind_group = self.bind(device, n, params, perm_buf, in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_fractal_encoder"),
        });
        dispatch(&mut enc, pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_fractal_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<f32>(ctx, &stage)
    }

    /// Builds the four-entry bind group (params uniform, perm table, input,
    /// output).
    fn bind(
        &self,
        device: &Device,
        count: usize,
        params: &Fractal,
        perm: &Buffer,
        input: &Buffer,
        output: &Buffer,
    ) -> BindGroup {
        let uniform = buffer::uniform(
            device,
            "prism_math_fractal_params",
            &Params {
                count: count as u32,
                octaves: params.octaves,
                lacunarity: params.lacunarity,
                gain: params.gain,
                frequency: params.frequency,
                _pad: [0; 3],
            },
        );
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_fractal_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: perm.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: input.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: output.as_entire_binding(),
                },
            ],
        })
    }

    /// Widens the seeded 512-entry `u8` permutation table of the configured
    /// base source to the `[u32; 512]` storage layout the kernel binds.
    fn perm_u32(&self, seed: u64) -> [u32; 512] {
        let table = match self.source {
            NoiseSource::Perlin => Perlin::new(seed).permutation_table(),
            NoiseSource::Simplex => Simplex::new(seed).permutation_table(),
        };
        let mut out = [0u32; 512];
        for (dst, &src) in out.iter_mut().zip(table.iter()) {
            *dst = u32::from(src);
        }
        out
    }
}

/// Compiles one forward pipeline from the chosen base fragment, the forwarding
/// alias, the shared `WGSL_FRACTAL` fragment, and a kernel-specific wrapper.
fn compile(
    device: &Device,
    pipeline_layout: &wgpu::PipelineLayout,
    source: NoiseSource,
    wrapper: &str,
    label: &str,
) -> ComputePipeline {
    let (base, alias) = match source {
        NoiseSource::Perlin => (WGSL_PERLIN, ALIAS_PERLIN),
        NoiseSource::Simplex => (WGSL_SIMPLEX, ALIAS_SIMPLEX),
    };
    let mut wgsl = String::new();
    wgsl.push_str(base);
    wgsl.push('\n');
    wgsl.push_str(alias);
    wgsl.push('\n');
    wgsl.push_str(WGSL_FRACTAL);
    wgsl.push('\n');
    wgsl.push_str(wrapper);
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some(label),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// Records a 1D batch dispatch covering `n` elements at [`WORKGROUP`] threads
/// per group.
fn dispatch(enc: &mut CommandEncoder, pipeline: &ComputePipeline, bind_group: &BindGroup, n: usize) {
    let groups = (n as u32).div_ceil(WORKGROUP);
    let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
        label: Some("prism_math_fractal_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(groups, 1, 1);
}

/// One storage/uniform bind-group-layout entry visible to the compute stage.
fn buffer_layout(binding: u32, ty: BindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}
