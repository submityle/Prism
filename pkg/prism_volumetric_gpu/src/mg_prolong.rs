//! `wgpu` compute twin of the geometric-multigrid prolongation — the
//! coarse-to-fine inter-grid transfer extracted from the `CPU` golden
//! pressure-projection solver
//! ([`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure),
//! design §10, §37).
//!
//! The golden module runs a full `V`-cycle; this twin reproduces only the
//! prolongation gather, the embarrassingly parallel coarse-to-fine step. No
//! restriction, smoothing, mean removal, coarse solve, or multi-level recursion
//! is ported — just one trilinear interpolation of a coarse field onto a finer
//! [`GridResolution`].
//!
//! # Algorithm
//!
//! Prolongation is a pure gather: each fine cell is the tensor product of three
//! per-axis cell-centered linear-interpolation stencils applied to the coarse
//! field. For one fine index `i` on an axis of `coarse_n` cells, the stencil
//! puts `three quarters` of the weight on the parent `i / 2` and `one quarter`
//! on an adjacent coarse neighbor (the even child leans to `parent − 1`, the
//! odd child to `parent + 1`); when that neighbor leaves the grid the quarter
//! folds back onto the parent, so a constant coarse field prolongs to the same
//! constant on every fine cell. Each fine cell therefore sums at most eight
//! coarse contributions.
//!
//! One thread owns one fine cell. It decodes its row-major index into
//! `(x, y, z)`, forms the three axis stencils, and accumulates
//! `weight_x · weight_y · weight_z · coarse[...]` in the identical `z`-`y`-`x`
//! nesting (the `z` stencil outermost, the `x` stencil innermost) the golden
//! triple loop uses, with the per-term factor order
//! `weight_x · weight_y · weight_z · coarse` preserved, so the two evaluate the
//! same arithmetic in the same order against the non-associative `f32` add.
//!
//! # Degenerate inputs
//!
//! An empty fine grid returns an empty field with no dispatch. A `coarse` slice
//! shorter than `coarse_res.voxel_count()` returns an all-zero fine field, the
//! exact guard the reference takes. A coarse axis of zero extent yields a
//! zero-length axis stencil, so the gather contributes nothing and the fine
//! field stays zero — matching the reference without any special case.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — integer index
//! arithmetic plus `+ − × ÷` on `f32`, no transcendental call and no optional
//! device feature — so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The
//! per-axis stencil length is at most two, so the triple loop is a statically
//! bounded `2 × 2 × 2` gather.
//!
//! # Correctness model
//!
//! No gather term contains a transcendental call, so `CPU` and `GPU` evaluate
//! the same closed-form algebra. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few `ULP`. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`), tight enough to catch a genuinely
//! wrong port (a swapped stencil weight, a dropped mirror fold, a transposed
//! axis) yet loose enough to admit legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `Briggs` multigrid trilinear prolongation plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::fluid::GridResolution;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One-dimensional dispatch width; one invocation per fine voxel.
const WORKGROUP_SIZE: u32 = 64;

/// The prolongation gather kernel.
///
/// `Params` carries the coarse and fine grid extents (`std430` layout, eight
/// `u32` words, two of them padding). The single entry point `prolong_gather`
/// decodes a fine linear index into `(x, y, z)`, forms the three per-axis
/// stencils with `axis_contributors`, and accumulates the tensor-product gather
/// in `z`-`y`-`x` order.
///
/// Provenance: `WGSL` transcription of the golden `prolong` plus its
/// `axis_contributors` helper; no Unreal Engine source or derived code.
const MG_PROLONG_WGSL: &str = r#"
struct Params {
    cnx: u32,
    cny: u32,
    cnz: u32,
    fnx: u32,
    fny: u32,
    fnz: u32,
    pad0: u32,
    pad1: u32,
}

// Up to two coarse contributors and their interpolation weights for one fine
// index on one axis, mirroring the golden `AxisStencil`.
struct AxisStencil {
    idx0: u32,
    idx1: u32,
    w0: f32,
    w1: f32,
    len: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// The coarse field, row-major, one f32 per coarse voxel.
@group(0) @binding(1) var<storage, read> coarse: array<f32>;
// The prolonged fine field, one f32 per fine voxel.
@group(0) @binding(2) var<storage, read_write> fine_out: array<f32>;

// Row-major coarse linear index, matching `GridResolution::linear_index`:
// (z * cny + y) * cnx + x.
fn lin_coarse(x: u32, y: u32, z: u32) -> u32 {
    return (z * params.cny + y) * params.cnx + x;
}

// Cell-centered linear-interpolation contributors of one fine index on one
// axis; the exact integer/ratio logic of the golden `axis_contributors`. The
// even child leans to parent-1, the odd child to parent+1; a far neighbor out
// of range folds its quarter back onto the parent.
fn axis_contributors(i: u32, coarse_n: u32) -> AxisStencil {
    var s: AxisStencil;
    if (coarse_n == 0u) {
        s.idx0 = 0u;
        s.idx1 = 0u;
        s.w0 = 0.0;
        s.w1 = 0.0;
        s.len = 0u;
        return s;
    }
    var parent = i / 2u;
    if (parent >= coarse_n) {
        parent = coarse_n - 1u;
    }
    let own_weight = 0.75;
    let far_weight = 0.25;
    let far_is_lower = (i % 2u) == 0u;
    var far_in_range: bool;
    if (far_is_lower) {
        far_in_range = parent > 0u;
    } else {
        far_in_range = (parent + 1u) < coarse_n;
    }
    if (far_in_range) {
        var far: u32;
        if (far_is_lower) {
            far = parent - 1u;
        } else {
            far = parent + 1u;
        }
        s.idx0 = parent;
        s.idx1 = far;
        s.w0 = own_weight;
        s.w1 = far_weight;
        s.len = 2u;
    } else {
        s.idx0 = parent;
        s.idx1 = 0u;
        s.w0 = own_weight + far_weight;
        s.w1 = 0.0;
        s.len = 1u;
    }
    return s;
}

// Dynamic selection of one stencil slot without an indexable array field.
fn sel_idx(s: AxisStencil, k: u32) -> u32 {
    if (k == 0u) {
        return s.idx0;
    }
    return s.idx1;
}

fn sel_w(s: AxisStencil, k: u32) -> f32 {
    if (k == 0u) {
        return s.w0;
    }
    return s.w1;
}

@compute @workgroup_size(64)
fn prolong_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let count = params.fnx * params.fny * params.fnz;
    if (idx >= count) {
        return;
    }
    // Decode the row-major fine index back into (x, y, z); the inverse of the
    // fine linear_index.
    let x = idx % params.fnx;
    let plane = idx / params.fnx;
    let y = plane % params.fny;
    let z = plane / params.fny;

    let sx = axis_contributors(x, params.cnx);
    let sy = axis_contributors(y, params.cny);
    let sz = axis_contributors(z, params.cnz);

    // Tensor-product gather in the exact z-y-x nesting the reference uses, with
    // the per-term factor order weight_x * weight_y * weight_z * coarse.
    var acc = 0.0;
    for (var iz = 0u; iz < sz.len; iz = iz + 1u) {
        let wz = sel_w(sz, iz);
        let cz = sel_idx(sz, iz);
        for (var iy = 0u; iy < sy.len; iy = iy + 1u) {
            let wy = sel_w(sy, iy);
            let cy = sel_idx(sy, iy);
            for (var ix = 0u; ix < sx.len; ix = ix + 1u) {
                let cidx = lin_coarse(sel_idx(sx, ix), cy, cz);
                acc = acc + sel_w(sx, ix) * wy * wz * coarse[cidx];
            }
        }
    }
    fine_out[idx] = acc;
}
"#;

/// Uniform parameters for one prolongation. `repr(C)` `std430` layout matching
/// `Params` in [`MG_PROLONG_WGSL`]: the three coarse extents, the three fine
/// extents, then two pad words — `32` bytes with no interior padding.
///
/// Provenance: layout mirror of the `WGSL` `Params` block; no Unreal Engine
/// source or derived code.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Coarse grid extent in voxels along `x`.
    cnx: u32,
    /// Coarse grid extent in voxels along `y`.
    cny: u32,
    /// Coarse grid extent in voxels along `z`.
    cnz: u32,
    /// Fine grid extent in voxels along `x`.
    fnx: u32,
    /// Fine grid extent in voxels along `y`.
    fny: u32,
    /// Fine grid extent in voxels along `z`.
    fnz: u32,
    /// Padding to keep the struct a multiple of `16` bytes.
    pad0: u32,
    /// Padding to keep the struct a multiple of `16` bytes.
    pad1: u32,
}

/// One coarse-to-fine prolongation query.
///
/// The field layout is row-major with
/// [`GridResolution::linear_index`](prism_render_architecture::particle::fluid::GridResolution::linear_index);
/// `coarse` must carry at least `coarse_res.voxel_count()` samples, and only
/// that prefix is consumed. A shorter slice yields an all-zero fine field.
///
/// Provenance: input mirror of the golden `prolong` signature; no Unreal Engine
/// source or derived code.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuMgProlongQuery {
    /// The coarse field, one `f32` per coarse voxel.
    pub coarse: Vec<f32>,
    /// The coarse grid the field is discretized on.
    pub coarse_res: GridResolution,
    /// The fine grid the field is prolonged onto.
    pub fine_res: GridResolution,
}

/// The outcome of a [`GpuMgProlong::prolong`] gather.
///
/// Provenance: output mirror of the golden `prolong` return; no Unreal Engine
/// source or derived code.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuMgProlongResult {
    /// The prolonged fine field, row-major, one `f32` per fine voxel.
    pub fine: Vec<f32>,
}

/// A compiled, reusable prolongation gather pipeline.
///
/// Provenance: `wgpu` pipeline wrapper around [`MG_PROLONG_WGSL`]; no Unreal
/// Engine source or derived code.
pub struct GpuMgProlong {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMgProlong {
    /// Compiles the prolongation kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: standard `wgpu` compute-pipeline creation; no Unreal Engine
    /// source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMgProlong {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mg_prolong"),
            source: ShaderSource::Wgsl(MG_PROLONG_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mg_prolong_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mg_prolong_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mg_prolong_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("prolong_gather"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMgProlong {
            module,
            layout,
            pipeline,
        }
    }

    /// Prolongs `query.coarse` from `query.coarse_res` onto `query.fine_res` and
    /// returns the fine field.
    ///
    /// The returned field equals the golden `prolong` to within the tolerance
    /// documented on this module. An empty fine grid returns an empty field (no
    /// dispatch); a `coarse` slice shorter than `coarse_res.voxel_count()`
    /// returns an all-zero fine field (no dispatch), matching the reference's
    /// degenerate-input guards.
    ///
    /// Provenance: dispatch-and-readback around the golden `prolong`; no Unreal
    /// Engine source or derived code.
    #[must_use]
    pub fn prolong(&self, ctx: &GpuContext, query: &GpuMgProlongQuery) -> GpuMgProlongResult {
        let fine_count = query.fine_res.voxel_count() as usize;
        if fine_count == 0 {
            return GpuMgProlongResult { fine: Vec::new() };
        }
        let coarse_count = query.coarse_res.voxel_count() as usize;
        if query.coarse.len() < coarse_count {
            return GpuMgProlongResult {
                fine: vec![0.0f32; fine_count],
            };
        }

        let device = ctx.device();

        let gpu_params = Params {
            cnx: query.coarse_res.nx,
            cny: query.coarse_res.ny,
            cnz: query.coarse_res.nz,
            fnx: query.fine_res.nx,
            fny: query.fine_res.ny,
            fnz: query.fine_res.nz,
            pad0: 0,
            pad1: 0,
        };

        // Pad the coarse upload to at least one element so the storage buffer is
        // never zero-sized; the kernel never reads past the live prefix.
        let mut coarse_data = query.coarse[..coarse_count].to_vec();
        if coarse_data.is_empty() {
            coarse_data.push(0.0);
        }
        let fine_bytes = (fine_count as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mg_prolong_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let coarse_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mg_prolong_coarse"),
            contents: bytemuck::cast_slice(&coarse_data),
            usage: BufferUsages::STORAGE,
        });
        let fine_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mg_prolong_fine"),
            size: fine_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let fine_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mg_prolong_fine_stage"),
            size: fine_bytes,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mg_prolong_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: coarse_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: fine_buf.as_entire_binding(),
                },
            ],
        });

        let groups = (fine_count as u32).div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mg_prolong_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mg_prolong_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&fine_buf, 0, &fine_stage, 0, fine_bytes);
        ctx.queue().submit([encoder.finish()]);

        fine_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let fine_view = fine_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped fine readback range should be available after poll");
        let fine = bytemuck::cast_slice::<u8, f32>(&fine_view).to_vec();
        drop(fine_view);
        fine_stage.unmap();

        debug_assert_eq!(fine.len(), fine_count);

        GpuMgProlongResult { fine }
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
