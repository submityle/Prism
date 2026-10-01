//! `wgpu` compute twin of Prism's second-order spherical-harmonic transmittance
//! reconstruction
//! ([`eval_sh`](prism_render_architecture::hair::dual_scatter_sh::eval_sh),
//! which composes
//! [`sh_basis`](prism_render_architecture::hair::dual_scatter_sh::sh_basis) with
//! a coefficient dot).
//!
//! A `Zinke` dual-scattering groom caches its low-frequency directional
//! transmittance in nine real second-order `SH` coefficients; shading then
//! reconstructs the transmittance along an arbitrary light/view direction by
//! evaluating the `SH` basis there and dotting it against the cached
//! coefficients. The second-order band (`1 + 3 + 5 = 9` terms) captures the
//! constant-plus-linear-plus-quadratic directional trend a smooth transmittance
//! field carries while staying tiny.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairEvalSh::eval`] takes one shared nine-coefficient
//! [`ShCoeffs`](prism_render_architecture::hair::dual_scatter_sh::ShCoeffs) set
//! plus a batch of query directions and returns one reconstructed transmittance
//! per direction, preserving input order — the array-in/array-out form used to
//! resample a cached transmittance lobe over a tile of directions. The
//! direction index is the invocation id (`@compute @workgroup_size(64)`,
//! one-dimensional dispatch over `global_invocation_id.x`); invocations past the
//! direction count early-return.
//!
//! # The coefficients live on the host, the directions vary per thread
//!
//! The nine coefficients are uploaded once as a shared read-only storage buffer
//! (band-major, matching
//! [`ShCoeffs::as_array`](prism_render_architecture::hair::dual_scatter_sh::ShCoeffs::as_array));
//! the per-thread variation is the query direction read from a flat `x, y, z`
//! storage array (three floats per element, tightly packed to avoid the `std430`
//! `vec3` stride). Each direction is normalised in-shader exactly like the
//! golden (`normalize_or_zero`: degenerate / non-finite directions collapse to
//! the pole), so the host uploads the raw authored directions unchanged.
//!
//! # Portability
//!
//! The basis uses only multiply/add plus one `sqrt` (vector normalisation), with
//! no `exp`, `pow`, `sin` or optional device feature, so the twin runs
//! unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The reconstruction is a `dot`, a `sqrt` and a divide chain a `GPU` may fuse,
//! so `CPU` and `GPU` agree to within the documented fma tolerance
//! (`abs_diff < 1e-4` or `rel_diff < 1e-3`) rather than bit-for-bit. The
//! sanitiser mirrors the golden exactly (`is_finite` test rejecting `NaN`/±inf,
//! the degenerate-length guard collapsing to zero, the final `sanitize().max(0)`
//! clamping the transmittance non-negative), so non-finite coefficients or
//! directions produce the same bounded, finite, non-negative result.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard orthonormal real spherical-harmonic reconstruction plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::dual_scatter_sh::{eval_sh, ShCoeffs, Vec3, SH_COEFFS};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Uniform parameters for one reconstruction dispatch. Layout matches `Params`
/// in `shaders/eval_sh.wesl`: the direction count in a single `16`-byte uniform
/// slot (one `u32` plus padding).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    element_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable per-direction `SH` reconstruction pipeline.
pub struct GpuHairEvalSh {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairEvalSh {
    /// Compiles the per-direction `SH` reconstruction kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset (multiply/add plus
    /// one `sqrt`), so no optional device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairEvalSh {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_eval_sh"),
            source: ShaderSource::Wgsl(include_str!("../shaders/eval_sh.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_eval_sh_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_eval_sh_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_eval_sh_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairEvalSh {
            module,
            layout,
            pipeline,
        }
    }

    /// Reconstructs the cached transmittance along each direction in `dirs` from
    /// the shared `coeffs`, returning one non-negative value per direction in
    /// input order.
    ///
    /// The value for direction `i` equals the `CPU` golden
    /// [`eval_sh`](prism_render_architecture::hair::dual_scatter_sh::eval_sh) of
    /// `coeffs` at `dirs[i]` to within the module's documented fma tolerance,
    /// with degenerate/non-finite directions collapsing to the same result. An
    /// empty batch yields an empty vector without a dispatch — storage buffers
    /// cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, coeffs: ShCoeffs, dirs: &[Vec3]) -> Vec<f32> {
        let element_count = dirs.len();
        if element_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        let uniforms = Params {
            element_count: element_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Nine band-major coefficients, shared by every thread.
        let coeff_values: [f32; SH_COEFFS] = *coeffs.as_array();
        // Query directions, flattened x,y,z per element (tightly packed).
        let mut dir_values: Vec<f32> = Vec::with_capacity(element_count * 3);
        for dir in dirs {
            dir_values.push(dir.x);
            dir_values.push(dir.y);
            dir_values.push(dir.z);
        }

        // Output is one f32 (4 bytes) per direction.
        let out_bytes = (element_count as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_eval_sh_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let coeffs_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_eval_sh_coeffs"),
            contents: bytemuck::cast_slice(&coeff_values),
            usage: BufferUsages::STORAGE,
        });
        let dirs_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_eval_sh_dirs"),
            contents: bytemuck::cast_slice(&dir_values),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_eval_sh_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_eval_sh_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_eval_sh_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: coeffs_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: dirs_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_eval_sh_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_eval_sh_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (element_count as u32).div_ceil(64);
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
        let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        out
    }
}

/// The `CPU` golden `SH` reconstruction for one direction, re-exported so the
/// parity test can assert the device twin against the identical reference it
/// mirrors.
#[must_use]
pub fn reference_eval_sh(coeffs: ShCoeffs, dir: Vec3) -> f32 {
    eval_sh(coeffs, dir)
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
