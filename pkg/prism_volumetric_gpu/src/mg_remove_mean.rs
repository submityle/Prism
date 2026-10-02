//! `wgpu` compute twin of the multigrid null-space pin — the in-place
//! mean-removal step extracted from the `CPU` golden pressure-projection solver
//! ([`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure),
//! design §10 coarsest solve, §37).
//!
//! The golden module runs a full `V`-cycle; this twin reproduces only the
//! private `remove_mean` helper, which subtracts the field mean from every
//! element to pin the constant null space of a pure-`Neumann` operator. No
//! smoothing, restriction, prolongation, coarse solve, or multi-level recursion
//! is ported — just one mean removal of a one-dimensional field.
//!
//! # Order-sensitive reduction
//!
//! The golden `remove_mean` forms the mean numerator as a single-threaded
//! ascending accumulation `for &v in field.iter() { sum += v; }`, then divides
//! by the length and subtracts the mean from each element. Floating-point
//! addition is non-associative, so the *order* of that accumulation is part of
//! the contract: a tree reduction or a reordered sum would land on a different
//! low-mantissa `mean`.
//!
//! A portable `GPU` reduction cannot bit-reproduce that strict ascending order
//! (there is no `f32` atomic in the core-`WGSL` subset, and a workgroup tree sum
//! reorders the adds). The twin therefore mirrors the host two-level aggregation
//! pattern of the sibling
//! [`gpu_compact`](crate::gpu_compact) twin: the device emits one per-element
//! partial, the host performs the strict ascending sum to form the identical
//! `mean`, and the device then subtracts that `mean` from every element in a
//! second, order-free pass.
//!
//! # Algorithm
//!
//! 1. The `collect` kernel runs one thread per element and copies `field[idx]`
//!    into a scratch buffer; the copy is a bit-exact `f32` move, so the readback
//!    reproduces the input exactly.
//! 2. The host reads the scratch back and sums it in strict ascending linear
//!    index order (identical to the golden `for &v in field.iter()`), divides by
//!    the length, and obtains the bit-identical `mean`.
//! 3. The `subtract` kernel runs one thread per element and writes
//!    `field[idx] - mean`. Each element's subtraction is independent of the
//!    others, so the parallel pass has no order sensitivity — only the `mean`
//!    did, and it was fixed on the host.
//!
//! # Degenerate inputs
//!
//! An empty field returns an empty field with no dispatch, the exact guard the
//! reference takes (`if field.is_empty() { return; }`). A length-one field
//! removes its only value and returns a single zero.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — unsigned index
//! arithmetic plus a single `f32` subtraction, no transcendental call, no `u64`
//! and no optional device feature — so it runs unmodified on `Metal`, `Vulkan`
//! and `DX12`. Both kernels are branch-free past the bounds guard and contain no
//! loop, so they provably terminate.
//!
//! # Correctness model
//!
//! The `mean` is formed on the host in the golden ascending order, so it is
//! bit-identical to the reference. The only device float op is the per-element
//! subtraction `field[idx] - mean`, a single rounding with no fused multiply, so
//! `CPU` and `GPU` evaluate the same algebra. The parity test asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) rather than bit equality
//! to stay robust across backends, tight enough to catch a wrong port (a dropped
//! subtraction, a reordered sum, a wrong divisor).
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::multigrid_pressure` 的私有 `remove_mean`；无第三方引擎源码或衍生代码。
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

/// One-dimensional dispatch width; one invocation per field element. `64` is
/// the portable, warp-friendly default used across this crate's
/// one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The mean-removal kernels, mirroring the `CPU` golden private `remove_mean`.
///
/// Two entry points share one bind-group layout. `collect` copies each element
/// into the scratch buffer so the host can sum it in the golden ascending order;
/// `subtract` writes `field[idx] - mean` once the host has uploaded the mean.
///
/// Provenance: `WGSL` transcription of the golden `multigrid_pressure::remove_mean`;
/// 无第三方引擎源码或衍生代码。
const MG_REMOVE_MEAN_WGSL: &str = r#"
// mg_remove_mean twin: one thread per element reproduces the CPU golden private
// `multigrid_pressure::remove_mean`. The device emits one partial per element
// (the value itself); the host sums those in strict ascending linear-index order
// to form the identical mean; the device then subtracts that mean from every
// element in an order-free second pass.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::multigrid_pressure
// 的私有 remove_mean；无第三方引擎源码或衍生代码。

// Dispatch parameters. 16-byte uniform block: the element count, the host-formed
// mean, and two pad words, matching the host `Params`.
struct Params {
    count: u32,
    mean: f32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// The input field, one f32 per element, in row-major linear-index order.
@group(0) @binding(1) var<storage, read> field: array<f32>;
// The scratch output: the collected value on pass one, the de-meaned value on
// pass two.
@group(0) @binding(2) var<storage, read_write> scratch: array<f32>;

// Pass one: copy each element into scratch so the host can read it back and sum
// it in the golden ascending order. The copy is a bit-exact f32 move.
@compute @workgroup_size(64)
fn collect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    scratch[idx] = field[idx];
}

// Pass two: subtract the host-formed mean from each element. The subtraction is
// a single rounding with no fused multiply, independent per element, so the pass
// has no order sensitivity.
@compute @workgroup_size(64)
fn subtract(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    scratch[idx] = field[idx] - params.mean;
}
"#;

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`MG_REMOVE_MEAN_WGSL`]: the element `count`, the host-formed
/// `mean` and two pad words — `16` bytes with no interior padding.
///
/// Provenance: host mirror of the `WGSL` `Params`; 无第三方引擎源码或衍生代码。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of elements (valid threads).
    count: u32,
    /// The host-formed field mean (zero on the `collect` pass).
    mean: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One mean-removal query: the one-dimensional `field` to pin.
///
/// The field is consumed in full. An empty field returns an empty field with no
/// dispatch, mirroring the reference's `if field.is_empty()` guard.
///
/// Provenance: input mirror of the golden `remove_mean` signature;
/// 无第三方引擎源码或衍生代码。
#[derive(Clone, Debug, PartialEq)]
pub struct GpuMgRemoveMeanQuery {
    /// The field to de-mean, one `f32` per element, in linear-index order.
    pub field: Vec<f32>,
}

/// The outcome of a [`GpuMgRemoveMean::remove_mean`] pass: the de-meaned field.
///
/// Provenance: output mirror of the golden `remove_mean` effect;
/// 无第三方引擎源码或衍生代码。
#[derive(Clone, Debug, PartialEq)]
pub struct GpuMgRemoveMeanResult {
    /// The de-meaned field, one `f32` per element, in linear-index order.
    pub field: Vec<f32>,
}

/// A compiled, reusable mean-removal pipeline pair.
///
/// Provenance: `wgpu` pipeline wrapper around [`MG_REMOVE_MEAN_WGSL`];
/// 无第三方引擎源码或衍生代码。
pub struct GpuMgRemoveMean {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    collect_pipeline: ComputePipeline,
    subtract_pipeline: ComputePipeline,
}

impl GpuMgRemoveMean {
    /// Compiles the mean-removal kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: standard `wgpu` compute-pipeline creation;
    /// 无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMgRemoveMean {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mg_remove_mean"),
            source: ShaderSource::Wgsl(MG_REMOVE_MEAN_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mg_remove_mean_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mg_remove_mean_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let collect_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mg_remove_mean_collect_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("collect"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let subtract_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mg_remove_mean_subtract_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("subtract"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMgRemoveMean {
            module,
            layout,
            collect_pipeline,
            subtract_pipeline,
        }
    }

    /// Removes the mean of `query.field` and returns the de-meaned field.
    ///
    /// The returned field equals the golden `remove_mean` to within the
    /// tolerance documented on this module. An empty field returns an empty
    /// field with no dispatch, matching the reference's early-return guard; a
    /// length-one field returns a single zero.
    ///
    /// Provenance: dispatch-and-readback around the golden `remove_mean`;
    /// 无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn remove_mean(
        &self,
        ctx: &GpuContext,
        query: &GpuMgRemoveMeanQuery,
    ) -> GpuMgRemoveMeanResult {
        let count = query.field.len();
        if count == 0 {
            return GpuMgRemoveMeanResult { field: Vec::new() };
        }
        let device = ctx.device();

        let field_bytes = (count as u64) * (size_of::<f32>() as u64);

        let field_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mg_remove_mean_field"),
            contents: bytemuck::cast_slice(&query.field),
            usage: BufferUsages::STORAGE,
        });
        let scratch_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mg_remove_mean_scratch"),
            size: field_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scratch_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mg_remove_mean_scratch_stage"),
            size: field_bytes,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Pass one: collect each element into the scratch buffer. The mean is not
        // yet known, so a zero is uploaded for it.
        let collect_params = Params {
            count: count as u32,
            mean: 0.0,
            pad0: 0,
            pad1: 0,
        };
        let collect_params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mg_remove_mean_collect_params"),
            contents: bytemuck::bytes_of(&collect_params),
            usage: BufferUsages::UNIFORM,
        });
        let collect_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mg_remove_mean_collect_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: collect_params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: field_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: scratch_buf.as_entire_binding(),
                },
            ],
        });

        let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
        let mut collect_encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mg_remove_mean_collect_encoder"),
        });
        {
            let mut pass = collect_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mg_remove_mean_collect_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.collect_pipeline);
            pass.set_bind_group(0, &collect_bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        collect_encoder.copy_buffer_to_buffer(&scratch_buf, 0, &scratch_stage, 0, field_bytes);
        ctx.queue().submit([collect_encoder.finish()]);

        scratch_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let partials = {
            let view = scratch_stage
                .slice(..)
                .get_mapped_range()
                .expect("mapped scratch readback range should be available after poll");
            let partials = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
            drop(view);
            partials
        };
        scratch_stage.unmap();

        // Host two-level aggregation: the strict ascending sum reproduces the
        // golden `for &v in field.iter() { sum += v; }` bit for bit, so the mean
        // is bit-identical to the reference.
        let mean = ascending_mean(&partials);

        // Pass two: subtract the host-formed mean from each element.
        let subtract_params = Params {
            count: count as u32,
            mean,
            pad0: 0,
            pad1: 0,
        };
        let subtract_params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mg_remove_mean_subtract_params"),
            contents: bytemuck::bytes_of(&subtract_params),
            usage: BufferUsages::UNIFORM,
        });
        let subtract_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mg_remove_mean_subtract_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: subtract_params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: field_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: scratch_buf.as_entire_binding(),
                },
            ],
        });

        let mut subtract_encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mg_remove_mean_subtract_encoder"),
        });
        {
            let mut pass = subtract_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mg_remove_mean_subtract_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.subtract_pipeline);
            pass.set_bind_group(0, &subtract_bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        subtract_encoder.copy_buffer_to_buffer(&scratch_buf, 0, &scratch_stage, 0, field_bytes);
        ctx.queue().submit([subtract_encoder.finish()]);

        scratch_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let field = {
            let view = scratch_stage
                .slice(..)
                .get_mapped_range()
                .expect("mapped scratch readback range should be available after poll");
            let field = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
            drop(view);
            field
        };
        scratch_stage.unmap();

        debug_assert_eq!(field.len(), count);

        GpuMgRemoveMeanResult { field }
    }
}

/// The field mean, formed as the golden `remove_mean` forms it: a strict
/// single-threaded ascending accumulation of the elements in linear-index order,
/// divided by the length. Non-associative `f32` addition makes this order part
/// of the contract, so the loop mirrors `for &v in field.iter() { sum += v; }`
/// exactly. The caller guarantees a non-empty slice.
///
/// Provenance: host mirror of the golden `remove_mean` mean numerator;
/// 无第三方引擎源码或衍生代码。
#[must_use]
fn ascending_mean(values: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &v in values {
        sum += v;
    }
    sum / values.len() as f32
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
