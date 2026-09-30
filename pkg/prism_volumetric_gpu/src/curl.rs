//! `wgpu` compute twin of the divergence-free curl noise
//! ([`curl_noise_3d`](prism_render_architecture::volumetric::noise::curl_noise_3d)).
//!
//! The cloud advection field — cirrus streaks, wind disturbance and contrail
//! curl (design section 4) — is the analytic `curl` of a three-component
//! vector potential, each component an independent signed `Perlin` field
//! decorrelated by a spatial offset and a seed. Taking the `curl` with matched
//! second-order central differences yields a field that is divergence-free up
//! to floating-point rounding. The `CPU` golden
//! [`curl_noise_3d`](prism_render_architecture::volumetric::noise::curl_noise_3d)
//! owns that math; [`GpuCurl`] is the on-device twin that runs one thread per
//! sample point and returns the same three-component vector.
//!
//! # Portability
//!
//! The kernel is the same integer `hash` work as the `Perlin` twin plus
//! `floor`, multiply/add and a `clamp` in the portable core-`WGSL` subset — no
//! `exp`, `pow` or optional device feature — so it runs unmodified on Metal,
//! Vulkan and DX12.
//!
//! # Correctness model
//!
//! The lattice `hash` selecting each gradient is pure unsigned-integer work,
//! and `WGSL` unsigned integers wrap on overflow exactly like Rust's
//! `wrapping_mul` / `^` / `>>`, so the `GPU` selects bit-identical gradients to
//! the reference. Only the float central-difference algebra can diverge, and
//! only by a legal multiply-add contraction of a few `ULP` — six potential
//! evaluations feed each output component, so the accumulated slack is a little
//! wider than a single `Perlin` sample yet still far below any real porting
//! error. The parity test asserts each of the three components to within
//! `abs_diff < 1e-6` or `rel_diff < 1e-5`, additionally asserts the field is
//! non-constant, and re-derives the numerical divergence from the `GPU` output
//! stencil to confirm the divergence-free property survived the port, so a
//! degenerate or mis-ported kernel could not pass.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard curl-of-`Perlin`-potential noise (`Bridson` et al.
//! 2007) plus `wgpu` compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::volumetric::Vec3;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One curl-noise query: a sample point and the field seed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurlQuery {
    /// Sample point in noise space.
    pub point: Vec3,
    /// Field seed selecting the deterministic gradient set.
    pub seed: u32,
}

/// One noise query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/curl.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    x: f32,
    y: f32,
    z: f32,
    seed: u32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/curl.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable curl-noise pipeline.
pub struct GpuCurl {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCurl {
    /// Compiles the curl-noise kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCurl {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_curl"),
            source: ShaderSource::Wgsl(include_str!("../shaders/curl.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_curl_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_curl_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_curl_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("curl"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCurl {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the divergence-free curl noise for every query in `queries`,
    /// returning one vector per query in input order.
    ///
    /// The returned vector for query `q` equals
    /// [`curl_noise_3d`](prism_render_architecture::volumetric::noise::curl_noise_3d)`(q.point, q.seed)`
    /// to within the tolerance documented on this module. An empty `queries`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[CurlQuery]) -> Vec<Vec3> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                x: q.point.x,
                y: q.point.y,
                z: q.point.z,
                seed: q.seed,
            })
            .collect();

        let gpu_params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Three f32 per query: the curl vector's x, y and z, tightly packed.
        let out_bytes = (queries.len() as u64) * 3 * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_curl_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_curl_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_curl_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_curl_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_curl_bind_group"),
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
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_curl_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_curl_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (queries.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(flat.len(), queries.len() * 3);
        let values: Vec<Vec3> = flat
            .chunks_exact(3)
            .map(|c| Vec3::new(c[0], c[1], c[2]))
            .collect();
        debug_assert_eq!(values.len(), queries.len());
        values
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
