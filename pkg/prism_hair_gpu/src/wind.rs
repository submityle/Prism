//! `wgpu` compute twin of the guide-strand wind-field coupling
//! ([`wind_acceleration`](prism_render_architecture::hair::wind::wind_acceleration)).
//!
//! Ambient wind is what sells a groom as *alive*: a steady breeze plus
//! turbulent gusts push each free guide particle before the XPBD constraint
//! solve so the hair drifts and flutters coherently with the scene (design
//! 6.3). The `CPU` golden for that per-particle acceleration field is
//! [`wind_acceleration`](prism_render_architecture::hair::wind::wind_acceleration);
//! this crate is the on-device twin that evaluates the same field, one thread
//! per query, so a passing real-device parity test is direct evidence the
//! ported kernel computes the same accelerations as the reference — not merely
//! that its shader compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuWindField::eval`] evaluates a batch of wind samples. Each
//! [`WindQuery`] pairs a world-space sample point and time with one
//! [`WindField`], built with [`WindQuery::new`] or [`query_for_wind`] from an
//! architecture-side [`WindField`]. The kernel reproduces the reference's three
//! additive terms — the steady push along the wind direction, the along-wind
//! gust pulse, and the small cross-wind turbulent flutter — including the
//! hand-written range-reduced Taylor sine (`sin_turns`) the reference uses in
//! place of a hardware `sin` for bit-reproducibility.
//!
//! # Portability
//!
//! The kernel uses only `sqrt`, `min`/`max`, `dot`, `round`, `fma` and
//! multiply/add in the portable core-`WGSL` subset — no `exp`, `pow`, hardware
//! `sin` or optional device feature — so it runs unmodified on Metal, Vulkan
//! and DX12.
//!
//! # Correctness model
//!
//! The field is a closed-form polynomial (the reference deliberately avoids
//! `f32::sin`), so `CPU` and `GPU` evaluate the same expression and diverge only
//! through legal fused-multiply-add contraction: the reference's `f32::mul_add`
//! Horner chain maps to WGSL `fma`, and the surrounding products may still be
//! contracted by the driver. The parity test therefore asserts a tolerance
//! (`abs_diff < 1e-4` or `rel_diff < 1e-3`) per component rather than exact
//! equality — tight enough that a genuinely wrong port (a dropped term, a
//! swapped axis, a missing range reduction) fails, loose enough that legal fma
//! contraction passes.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard steady+gust+turbulence wind acceleration with a
//! hand-written Taylor sine plus `wgpu` compute dispatch; no Unreal Engine
//! source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::dynamics::Vec3;
use prism_render_architecture::hair::wind::WindField;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One wind-evaluation query: a world-space sample point and time paired with
/// the [`WindField`] to sample.
///
/// The fields encode exactly the arguments the `CPU` golden
/// [`wind_acceleration`](prism_render_architecture::hair::wind::wind_acceleration)
/// consumes: the sample `position`, the `time` in seconds, and the field's
/// direction, speed, gust amplitude, gust frequency and turbulence.
///
/// The struct is `48`-byte `repr(C)` matching `Query` in `shaders/wind.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct WindQuery {
    /// Sample point x.
    px: f32,
    /// Sample point y.
    py: f32,
    /// Sample point z.
    pz: f32,
    /// Sample time in seconds.
    time: f32,
    /// Wind heading x (normalized in the kernel).
    dx: f32,
    /// Wind heading y.
    dy: f32,
    /// Wind heading z.
    dz: f32,
    /// Steady acceleration magnitude along the heading.
    speed: f32,
    /// Peak extra along-wind acceleration of the gust pulse.
    gust_amplitude: f32,
    /// Gust cycles per (world unit + second); clamped non-negative in-kernel.
    gust_frequency: f32,
    /// Cross-wind flutter magnitude.
    turbulence: f32,
    /// Padding to a `48`-byte stride.
    pad: f32,
}

impl WindQuery {
    /// Builds a query that samples `field` at `position` and `time` (seconds).
    #[must_use]
    pub fn new(field: WindField, position: Vec3, time: f32) -> WindQuery {
        WindQuery {
            px: position.x,
            py: position.y,
            pz: position.z,
            time,
            dx: field.direction.x,
            dy: field.direction.y,
            dz: field.direction.z,
            speed: field.speed,
            gust_amplitude: field.gust_amplitude,
            gust_frequency: field.gust_frequency,
            turbulence: field.turbulence,
            pad: 0.0,
        }
    }
}

/// Builds the [`WindQuery`] that samples `field` at `position` and `time`.
///
/// A thin free-function alias for [`WindQuery::new`], mirroring
/// [`query_for`](crate::collision::query_for) so callers can build queries the
/// same way across both kernels.
#[must_use]
pub fn query_for_wind(field: WindField, position: Vec3, time: f32) -> WindQuery {
    WindQuery::new(field, position, time)
}

/// Uniform parameters for one wind dispatch. Layout matches `Params` in
/// `shaders/wind.wesl`: the query count then three pad words for the `16`-byte
/// uniform alignment.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable wind-field pipeline.
pub struct GpuWindField {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWindField {
    /// Compiles the wind-field kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWindField {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_wind"),
            source: ShaderSource::Wgsl(include_str!("../shaders/wind.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_wind_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_wind_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_wind_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWindField {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the wind acceleration for every query, returning one
    /// `[x, y, z]` acceleration per query in input order.
    ///
    /// The returned acceleration for query `q` equals
    /// [`wind_acceleration`](prism_render_architecture::hair::wind::wind_acceleration)
    /// applied to the `(field, position, time)` the query carries, to within
    /// the fused-multiply-add tolerance documented on this module. An empty
    /// `queries` slice yields an empty result — storage buffers cannot be
    /// zero-sized, so it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[WindQuery]) -> Vec<[f32; 3]> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Three f32 (x, y, z) per query.
        let out_bytes = (queries.len() as u64) * 3 * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_wind_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_wind_queries"),
            contents: bytemuck::cast_slice(queries),
            usage: BufferUsages::STORAGE,
        });
        let values_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_wind_values"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let values_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_wind_values_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_wind_bind_group"),
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
                    resource: values_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_wind_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_wind_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (queries.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&values_buf, 0, &values_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        values_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = values_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        values_stage.unmap();
        debug_assert_eq!(flat.len(), queries.len() * 3);
        flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
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
