//! `wgpu` compute twin of the multi-scatter probe-grid sampler
//! ([`ProbeGrid::sample`](prism_render_architecture::volumetric::multiscatter::ProbeGrid::sample)).
//!
//! The probe grid caches directional multi-scatter irradiance in six
//! signed-axis bands ([`PROBE_BANDS`](prism_render_architecture::volumetric::multiscatter::PROBE_BANDS))
//! at each lattice corner over an axis-aligned box (design section 7b). A query
//! maps its world position to per-axis fractions (clamped into the box),
//! resolves the two bracketing probes per axis, and blends the eight
//! surrounding probes per band with trilinear weights. The eight corner weights
//! sum to one, so each band is a true convex combination of the stored
//! irradiance; points outside the box clamp to the boundary probes without a
//! `panic`. The `CPU` golden
//! [`ProbeGrid::sample`](prism_render_architecture::volumetric::multiscatter::ProbeGrid::sample)
//! owns that math; [`GpuProbeGridSample`] is the on-device twin that runs one
//! thread per query.
//!
//! # Correctness model
//!
//! The sampler is only multiply/add on the raw probe data plus integer address
//! arithmetic — no transcendental, no lattice `hash`, no optional device
//! feature — so `CPU` and `GPU` evaluate the identical closed-form algebra. The
//! per-band accumulation multiplies each corner weight by the band value in
//! corner order, matching the `CPU` `weights[k] * irradiance[band]` order, so
//! they differ at most by a legal multiply-add contraction of a few `ULP`. The
//! parity test asserts each of the six bands to within `abs_diff < 1e-6` (or a
//! matching relative tolerance) and checks clamp-to-edge behaviour, so a
//! swapped axis, a dropped corner, or a mis-ordered accumulation could not pass.
//!
//! # Portability
//!
//! The kernel is integer address math plus multiply/add in the portable
//! core-`WGSL` subset, so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard trilinear grid sampling with clamp-to-edge plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use prism_render_architecture::volumetric::Vec3;

use crate::context::GpuContext;

/// Number of directional irradiance bands stored per probe, mirroring the `CPU`
/// [`PROBE_BANDS`](prism_render_architecture::volumetric::multiscatter::PROBE_BANDS).
pub const PROBE_BANDS: usize = 6;

/// One probe-grid query: a world-space sample position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeSampleQuery {
    /// World-space position to sample.
    pub pos: Vec3,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/probe_grid_sample.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    x: f32,
    y: f32,
    z: f32,
    pad0: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/probe_grid_sample.wesl`: the query count, the three per-axis probe
/// counts, and the box min/max corners (each `vec3` padded to 16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    dim0: u32,
    dim1: u32,
    dim2: u32,
    min0: f32,
    min1: f32,
    min2: f32,
    pad0: f32,
    max0: f32,
    max1: f32,
    max2: f32,
    pad1: f32,
}

/// A compiled, reusable probe-grid sampler pipeline.
pub struct GpuProbeGridSample {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuProbeGridSample {
    /// Compiles the probe-grid sampler kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuProbeGridSample {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_probe_grid_sample"),
            source: ShaderSource::Wgsl(include_str!("../shaders/probe_grid_sample.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_probe_grid_sample_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_probe_grid_sample_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_probe_grid_sample_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("probe_grid_sample_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuProbeGridSample {
            module,
            layout,
            pipeline,
        }
    }

    /// Samples the row-major probe grid `probes` (probe counts `dims`, box
    /// corners `min_corner`/`max_corner`) at every query in `queries`,
    /// returning one `[f32; PROBE_BANDS]` per query in input order.
    ///
    /// The `probes` slice is band-inner: probe `(x * dims[1] + y) * dims[2] + z`
    /// occupies `[p * PROBE_BANDS .. p * PROBE_BANDS + PROBE_BANDS)`, so it must
    /// have length `dims[0] * dims[1] * dims[2] * PROBE_BANDS`. The returned
    /// bands for query `q` equal the `CPU`
    /// [`ProbeGrid::sample`](prism_render_architecture::volumetric::multiscatter::ProbeGrid::sample)
    /// of a grid carrying the same probes and box, to within the tolerance
    /// documented on this module. An empty `queries` slice yields an empty
    /// result — storage buffers cannot be zero-sized, so it is handled by an
    /// early return.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        probes: &[f32],
        dims: [usize; 3],
        min_corner: Vec3,
        max_corner: Vec3,
        queries: &[ProbeSampleQuery],
    ) -> Vec<[f32; PROBE_BANDS]> {
        if queries.is_empty() {
            return Vec::new();
        }
        assert_eq!(
            probes.len(),
            dims[0] * dims[1] * dims[2] * PROBE_BANDS,
            "probes length must equal dims[0] * dims[1] * dims[2] * PROBE_BANDS"
        );
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                x: q.pos.x,
                y: q.pos.y,
                z: q.pos.z,
                pad0: 0.0,
            })
            .collect();

        let gpu_params = Params {
            count: queries.len() as u32,
            dim0: dims[0] as u32,
            dim1: dims[1] as u32,
            dim2: dims[2] as u32,
            min0: min_corner.x,
            min1: min_corner.y,
            min2: min_corner.z,
            pad0: 0.0,
            max0: max_corner.x,
            max1: max_corner.y,
            max2: max_corner.z,
            pad1: 0.0,
        };

        let out_bytes = (queries.len() as u64) * (PROBE_BANDS as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_probe_grid_sample_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let probes_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_probe_grid_sample_probes"),
            contents: bytemuck::cast_slice(probes),
            usage: BufferUsages::STORAGE,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_probe_grid_sample_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_probe_grid_sample_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_probe_grid_sample_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_probe_grid_sample_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: probes_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_probe_grid_sample_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_probe_grid_sample_pass"),
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
        debug_assert_eq!(flat.len(), queries.len() * PROBE_BANDS);
        let values: Vec<[f32; PROBE_BANDS]> = flat
            .chunks_exact(PROBE_BANDS)
            .map(|c| {
                let mut band = [0.0f32; PROBE_BANDS];
                band.copy_from_slice(c);
                band
            })
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
