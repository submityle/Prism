//! `wgpu` compute twin of Prism's melanin-pigment absorption fold
//! ([`melanin_absorption`](prism_render_architecture::hair::melanin::melanin_absorption),
//! batch form
//! [`melanin_absorption_map`](prism_render_architecture::hair::melanin::melanin_absorption_map)).
//!
//! A physically based hair BSDF (`Marschner`/`Chiang` `R`/`TT`/`TRT`) is driven
//! by a spectral absorption coefficient `sigma_a`, not an RGB tint. Real hair
//! colour comes from two pigments — eumelanin (brown-black) and pheomelanin
//! (red-yellow) — so black, brown, blond and red hair are the *same* model at
//! different pigment concentrations. The reference folds each fibre's two
//! non-negative concentrations into an RGB `sigma_a` by a per-pigment
//! absorption spectrum times the concentration, summed:
//! `sigma_a = eu * EUMELANIN_SIGMA_A + pheo * PHEOMELANIN_SIGMA_A` per channel.
//! This is the pigment parameterisation the film-grade `Chiang` 2016 model and
//! `pbrt` expose, and what `UE5` Groom takes as its pigment inputs.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairMelanin::eval`] takes one [`MelaninProfile`] per fibre and returns
//! one RGB `sigma_a` per fibre, preserving input order — the array-in/array-out
//! form used for a per-strand pigment attribute or a root melanin texture. The
//! fibre index is the invocation id (`@compute @workgroup_size(64)`,
//! one-dimensional dispatch over `global_invocation_id.x`); invocations past the
//! fibre count early-return.
//!
//! # Spectra live on the host, not in the shader
//!
//! The two canonical per-unit pigment spectra
//! ([`EUMELANIN_SIGMA_A`](prism_render_architecture::hair::melanin::EUMELANIN_SIGMA_A),
//! [`PHEOMELANIN_SIGMA_A`](prism_render_architecture::hair::melanin::PHEOMELANIN_SIGMA_A))
//! are read from the architecture crate and passed as uniform parameters, so the
//! shader never duplicates the magic constants and cannot drift from the golden.
//!
//! # Portability
//!
//! The kernel is a pure non-negative linear combination — only multiply/add plus
//! the finite/positive concentration guard, no `exp`, `pow`, `sin` or optional
//! device feature — so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The fold is a two-term multiply-add per channel a `GPU` may fuse, so `CPU`
//! and `GPU` agree to within the documented fma tolerance (`abs_diff < 1e-4` or
//! `rel_diff < 1e-3`) rather than bit-for-bit. The concentration guard mirrors
//! the golden's `clamp_concentration` exactly (`v == v && v > 0.0 &&
//! v <= MAX_FINITE_F32` rejects NaN, non-positive and `+inf`), so negative and
//! non-finite concentrations collapse to the same exact `0` the golden emits and
//! the absorption stays finite and non-negative for every input.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: `Chiang` 2016 / `pbrt` melanin pigment parameterisation plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::melanin::{
    melanin_absorption, MelaninProfile, EUMELANIN_SIGMA_A, PHEOMELANIN_SIGMA_A,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Uniform parameters for one absorption-fold dispatch. Layout matches `Params`
/// in `shaders/melanin.wesl`: the two per-unit pigment spectra (xyz, with `w`
/// padding) and the fibre count, in three `16`-byte uniform slots.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    eu_sigma: [f32; 4],
    pheo_sigma: [f32; 4],
    profile_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable per-fibre melanin-absorption pipeline.
pub struct GpuHairMelanin {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairMelanin {
    /// Compiles the per-fibre melanin-absorption kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairMelanin {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_melanin"),
            source: ShaderSource::Wgsl(include_str!("../shaders/melanin.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_melanin_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_melanin_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_melanin_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairMelanin {
            module,
            layout,
            pipeline,
        }
    }

    /// Folds each fibre's pigment profile into an RGB absorption coefficient,
    /// returning one `[r, g, b]` `sigma_a` per fibre in input order.
    ///
    /// The absorption for fibre `i` equals the `CPU` golden
    /// [`melanin_absorption`](prism_render_architecture::hair::melanin::melanin_absorption)
    /// of `profiles[i]` to within the module's documented fma tolerance, with
    /// negative and non-finite concentrations collapsing to the same exact `0`.
    /// An empty batch yields an empty vector without a dispatch — storage
    /// buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, profiles: &[MelaninProfile]) -> Vec<[f32; 3]> {
        let profile_count = profiles.len();
        if profile_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        // The shader sanitizes concentrations itself (bit-faithfully to the
        // golden), so upload the raw authored values unchanged.
        let inputs: Vec<[f32; 2]> = profiles
            .iter()
            .map(|p| [p.eumelanin, p.pheomelanin])
            .collect();

        let uniforms = Params {
            eu_sigma: [
                EUMELANIN_SIGMA_A[0],
                EUMELANIN_SIGMA_A[1],
                EUMELANIN_SIGMA_A[2],
                0.0,
            ],
            pheo_sigma: [
                PHEOMELANIN_SIGMA_A[0],
                PHEOMELANIN_SIGMA_A[1],
                PHEOMELANIN_SIGMA_A[2],
                0.0,
            ],
            profile_count: profile_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Output is one vec4<f32> (16 bytes) per fibre: xyz = sigma_a, w = 0.
        let out_bytes = (profile_count as u64) * (size_of::<[f32; 4]>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_melanin_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let profiles_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_melanin_profiles"),
            contents: bytemuck::cast_slice(&inputs),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_melanin_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_melanin_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_melanin_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: profiles_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_melanin_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_melanin_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (profile_count as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &out_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        out_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = out_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let packed = bytemuck::cast_slice::<u8, [f32; 4]>(&view);
        let out: Vec<[f32; 3]> = packed.iter().map(|v| [v[0], v[1], v[2]]).collect();
        drop(view);
        out_stage.unmap();

        out
    }
}

/// The `CPU` golden absorption for one fibre, re-exported so the parity test can
/// assert the device twin against the identical reference it mirrors.
#[must_use]
pub fn reference_absorption(profile: MelaninProfile) -> [f32; 3] {
    melanin_absorption(profile)
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
