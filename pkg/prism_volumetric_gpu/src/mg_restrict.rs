//! `wgpu` compute twin of the geometric-multigrid full-weighting restriction —
//! the fine-to-coarse inter-grid transfer extracted from the `CPU` golden
//! pressure-projection solver
//! ([`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure),
//! design §10, §37).
//!
//! The golden module runs a full `V`-cycle; this twin reproduces only the
//! restriction operator `R = (1 / 2^k)·Pᵀ`, the scaled transpose of the
//! prolongation. No prolongation, smoothing, mean removal, coarse solve, or
//! multi-level recursion is ported — just one full-weighting restriction of a
//! fine field onto a coarser [`GridResolution`].
//!
//! # Scatter versus gather
//!
//! The golden `restrict` is a *scatter*: it walks every fine cell and adds its
//! scaled, tensor-product-weighted value into each coarse contributor. A direct
//! port would need an `f32` atomic add, which the portable core-`WGSL` subset
//! does not provide. The twin therefore inverts the loop into a *gather*: one
//! thread owns one coarse cell and sums the contributions of every fine cell
//! that scatters into it. The two express the identical bilinear form, so the
//! coarse field is the same up to floating-point reassociation.
//!
//! # Algorithm
//!
//! For coarse cell `c = (cx, cy, cz)` the thread scans the small, statically
//! bounded window of fine indices whose per-axis stencil can reach `c` (the
//! fine indices `2·c − 1 ..= 2·c + 2` on each axis, widened by one on each side
//! and clamped to the grid for safety). For each candidate fine cell it forms
//! the three per-axis stencils with the exact golden `axis_contributors` logic
//! and keeps the cell only when `c` is produced on all three axes. The surviving
//! term `scale · weight_x · weight_y · weight_z · fine[...]` is accumulated in
//! the identical `z`-`y`-`x` ascending fine-index order (the `z` loop outermost,
//! the `x` loop innermost) the golden scatter visits fine cells in, with the
//! per-term factor order `scale · weight_x · weight_y · weight_z · fine`
//! preserved, so the two evaluate the same arithmetic in the same order against
//! the non-associative `f32` add. The normalization `scale = 1 / 2^k`, with `k`
//! the number of genuinely halved axes, is formed once on the host (exactly as
//! the reference does) and uploaded, so the device consumes the identical
//! `f32`.
//!
//! # Degenerate inputs
//!
//! An empty coarse grid returns an empty field with no dispatch. A `fine` slice
//! shorter than `fine_res.voxel_count()`, or a zero-extent fine grid, returns an
//! all-zero coarse field, the exact guard the reference takes. A coarse axis of
//! zero extent makes the coarse voxel count zero and so also returns empty.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — integer index
//! arithmetic plus `+ − × ÷` on `f32`, no transcendental call and no optional
//! device feature — so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The
//! per-axis scan window is at most six indices, so the triple loop is a
//! statically bounded gather.
//!
//! # Correctness model
//!
//! No gather term contains a transcendental call, so `CPU` and `GPU` evaluate
//! the same closed-form algebra. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few `ULP`. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`), tight enough to catch a genuinely
//! wrong port (a swapped stencil weight, a dropped mirror fold, a missing
//! `1 / 2^k` normalization, a transposed axis) yet loose enough to admit legal
//! fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `Briggs` multigrid full-weighting restriction (the
//! scaled transpose of trilinear prolongation) plus `wgpu` compute dispatch;
//! no Unreal Engine source or derived code.

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

/// One-dimensional dispatch width; one invocation per coarse voxel.
const WORKGROUP_SIZE: u32 = 64;

/// The restriction gather kernel.
///
/// `Params` carries the fine and coarse grid extents and the precomputed
/// `scale = 1 / 2^k` normalization (`std430` layout, eight words, one of them
/// padding). The single entry point `restrict_gather` decodes a coarse linear
/// index into `(x, y, z)`, scans the bounded fine-index window per axis, forms
/// the three per-axis stencils with `axis_contributors`, and accumulates the
/// tensor-product gather in ascending `z`-`y`-`x` fine-index order.
///
/// Provenance: `WGSL` transcription of the golden `restrict` plus its
/// `axis_contributors` helper, loop-inverted scatter-to-gather; no Unreal
/// Engine source or derived code.
const MG_RESTRICT_WGSL: &str = r#"
struct Params {
    fnx: u32,
    fny: u32,
    fnz: u32,
    cnx: u32,
    cny: u32,
    cnz: u32,
    scale: f32,
    pad0: u32,
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
// The fine field, row-major, one f32 per fine voxel.
@group(0) @binding(1) var<storage, read> fine: array<f32>;
// The restricted coarse field, one f32 per coarse voxel.
@group(0) @binding(2) var<storage, read_write> coarse_out: array<f32>;

// Row-major fine linear index, matching `GridResolution::linear_index`:
// (z * fny + y) * fnx + x.
fn lin_fine(x: u32, y: u32, z: u32) -> u32 {
    return (z * params.fny + y) * params.fnx + x;
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

// The weight the stencil `s` places on coarse index `want`, selected by exact
// u32 index comparison. `hit` is a 0/1 integer flag so the caller never has to
// compare an f32 to decide membership; weights are never zero for a live entry.
struct AxisPick {
    hit: u32,
    weight: f32,
}

fn axis_pick(s: AxisStencil, want: u32) -> AxisPick {
    var p: AxisPick;
    p.hit = 0u;
    p.weight = 0.0;
    if (s.len >= 1u) {
        if (s.idx0 == want) {
            p.hit = 1u;
            p.weight = s.w0;
        }
    }
    if (s.len >= 2u) {
        if (s.idx1 == want) {
            p.hit = 1u;
            p.weight = s.w1;
        }
    }
    return p;
}

// Lower bound of the fine-index scan window for coarse index `c`: `2*c` dropped
// by two, saturating at zero.
fn window_lo(c: u32) -> u32 {
    let base = 2u * c;
    if (base >= 2u) {
        return base - 2u;
    }
    return 0u;
}

// Upper bound (inclusive) of the fine-index scan window for coarse index `c` on
// an axis of `fine_n` cells: `2*c + 3`, clamped to the last fine index.
fn window_hi(c: u32, fine_n: u32) -> u32 {
    let top = 2u * c + 3u;
    if (top >= fine_n) {
        return fine_n - 1u;
    }
    return top;
}

@compute @workgroup_size(64)
fn restrict_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let count = params.cnx * params.cny * params.cnz;
    if (idx >= count) {
        return;
    }
    // Decode the row-major coarse index back into (cx, cy, cz); the inverse of
    // the coarse linear_index.
    let cx = idx % params.cnx;
    let cplane = idx / params.cnx;
    let cy = cplane % params.cny;
    let cz = cplane / params.cny;

    let lo_x = window_lo(cx);
    let hi_x = window_hi(cx, params.fnx);
    let lo_y = window_lo(cy);
    let hi_y = window_hi(cy, params.fny);
    let lo_z = window_lo(cz);
    let hi_z = window_hi(cz, params.fnz);

    // Gather every fine cell that scatters into this coarse cell, visited in the
    // same ascending z-y-x fine-index order the golden scatter uses, with the
    // factor order scale * weight_x * weight_y * weight_z * fine preserved.
    var acc = 0.0;
    for (var iz = lo_z; iz <= hi_z; iz = iz + 1u) {
        let sz = axis_contributors(iz, params.cnz);
        let pz = axis_pick(sz, cz);
        if (pz.hit == 1u) {
            for (var iy = lo_y; iy <= hi_y; iy = iy + 1u) {
                let sy = axis_contributors(iy, params.cny);
                let py = axis_pick(sy, cy);
                if (py.hit == 1u) {
                    for (var ix = lo_x; ix <= hi_x; ix = ix + 1u) {
                        let sx = axis_contributors(ix, params.cnx);
                        let px = axis_pick(sx, cx);
                        if (px.hit == 1u) {
                            let fidx = lin_fine(ix, iy, iz);
                            acc = acc
                                + params.scale * px.weight * py.weight * pz.weight * fine[fidx];
                        }
                    }
                }
            }
        }
    }
    coarse_out[idx] = acc;
}
"#;

/// Uniform parameters for one restriction. `repr(C)` `std430` layout matching
/// `Params` in [`MG_RESTRICT_WGSL`]: the three fine extents, the three coarse
/// extents, the precomputed `scale`, then one pad word — `32` bytes with no
/// interior padding.
///
/// Provenance: layout mirror of the `WGSL` `Params` block; no Unreal Engine
/// source or derived code.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Fine grid extent in voxels along `x`.
    fnx: u32,
    /// Fine grid extent in voxels along `y`.
    fny: u32,
    /// Fine grid extent in voxels along `z`.
    fnz: u32,
    /// Coarse grid extent in voxels along `x`.
    cnx: u32,
    /// Coarse grid extent in voxels along `y`.
    cny: u32,
    /// Coarse grid extent in voxels along `z`.
    cnz: u32,
    /// The full-weighting normalization `1 / 2^k`, precomputed on the host.
    scale: f32,
    /// Padding to keep the struct a multiple of `16` bytes.
    pad0: u32,
}

/// One fine-to-coarse restriction query.
///
/// The field layout is row-major with
/// [`GridResolution::linear_index`](prism_render_architecture::particle::fluid::GridResolution::linear_index);
/// `fine` must carry at least `fine_res.voxel_count()` samples, and only that
/// prefix is consumed. A shorter slice yields an all-zero coarse field.
///
/// Provenance: input mirror of the golden `restrict` signature; no Unreal
/// Engine source or derived code.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuMgRestrictQuery {
    /// The fine field, one `f32` per fine voxel.
    pub fine: Vec<f32>,
    /// The fine grid the field is discretized on.
    pub fine_res: GridResolution,
    /// The coarse grid the field is restricted onto.
    pub coarse_res: GridResolution,
}

/// The outcome of a [`GpuMgRestrict::restrict`] gather.
///
/// Provenance: output mirror of the golden `restrict` return; no Unreal Engine
/// source or derived code.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuMgRestrictResult {
    /// The restricted coarse field, row-major, one `f32` per coarse voxel.
    pub coarse: Vec<f32>,
}

/// A compiled, reusable restriction gather pipeline.
///
/// Provenance: `wgpu` pipeline wrapper around [`MG_RESTRICT_WGSL`]; no Unreal
/// Engine source or derived code.
pub struct GpuMgRestrict {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMgRestrict {
    /// Compiles the restriction kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: standard `wgpu` compute-pipeline creation; no Unreal Engine
    /// source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMgRestrict {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mg_restrict"),
            source: ShaderSource::Wgsl(MG_RESTRICT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mg_restrict_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mg_restrict_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mg_restrict_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("restrict_gather"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMgRestrict {
            module,
            layout,
            pipeline,
        }
    }

    /// Restricts `query.fine` from `query.fine_res` onto `query.coarse_res` and
    /// returns the coarse field.
    ///
    /// The returned field equals the golden `restrict` to within the tolerance
    /// documented on this module. An empty coarse grid returns an empty field
    /// (no dispatch); a `fine` slice shorter than `fine_res.voxel_count()`, or a
    /// zero-extent fine grid, returns an all-zero coarse field (no dispatch),
    /// matching the reference's degenerate-input guards.
    ///
    /// Provenance: dispatch-and-readback around the golden `restrict`; no Unreal
    /// Engine source or derived code.
    #[must_use]
    pub fn restrict(&self, ctx: &GpuContext, query: &GpuMgRestrictQuery) -> GpuMgRestrictResult {
        let coarse_count = query.coarse_res.voxel_count() as usize;
        if coarse_count == 0 {
            return GpuMgRestrictResult { coarse: Vec::new() };
        }
        let fine_count = query.fine_res.voxel_count() as usize;
        if query.fine.len() < fine_count || fine_count == 0 {
            return GpuMgRestrictResult {
                coarse: vec![0.0f32; coarse_count],
            };
        }

        // The full-weighting normalization is the inverse of two per genuinely
        // halved axis, formed exactly as the reference does so the device reads
        // the identical f32.
        let k = coarsened_axis_count(query.fine_res, query.coarse_res);
        let scale = 1.0 / pow2_f32(k);

        let device = ctx.device();

        let gpu_params = Params {
            fnx: query.fine_res.nx,
            fny: query.fine_res.ny,
            fnz: query.fine_res.nz,
            cnx: query.coarse_res.nx,
            cny: query.coarse_res.ny,
            cnz: query.coarse_res.nz,
            scale,
            pad0: 0,
        };

        // Upload only the live prefix; the kernel never reads past it.
        let fine_data = query.fine[..fine_count].to_vec();
        let coarse_bytes = (coarse_count as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mg_restrict_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let fine_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mg_restrict_fine"),
            contents: bytemuck::cast_slice(&fine_data),
            usage: BufferUsages::STORAGE,
        });
        let coarse_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mg_restrict_coarse"),
            size: coarse_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let coarse_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mg_restrict_coarse_stage"),
            size: coarse_bytes,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mg_restrict_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: fine_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: coarse_buf.as_entire_binding(),
                },
            ],
        });

        let groups = (coarse_count as u32).div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mg_restrict_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mg_restrict_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&coarse_buf, 0, &coarse_stage, 0, coarse_bytes);
        ctx.queue().submit([encoder.finish()]);

        coarse_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let coarse_view = coarse_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped coarse readback range should be available after poll");
        let coarse = bytemuck::cast_slice::<u8, f32>(&coarse_view).to_vec();
        drop(coarse_view);
        coarse_stage.unmap();

        debug_assert_eq!(coarse.len(), coarse_count);

        GpuMgRestrictResult { coarse }
    }
}

/// The number of axes genuinely halved between `fine` and `coarse`; the exact
/// integer logic of the golden `coarsened_axis_count`. Each coarsened axis
/// contributes a factor of two to the full-weighting normalization.
///
/// Provenance: host mirror of the golden `coarsened_axis_count`; no Unreal
/// Engine source or derived code.
#[must_use]
fn coarsened_axis_count(fine: GridResolution, coarse: GridResolution) -> u32 {
    let mut k = 0u32;
    if coarse.nx < fine.nx {
        k += 1;
    }
    if coarse.ny < fine.ny {
        k += 1;
    }
    if coarse.nz < fine.nz {
        k += 1;
    }
    k
}

/// `2^k` as an `f32`, formed with an integer-doubling loop so no `pow` is used;
/// `k` is at most three here. The exact logic of the golden `pow2_f32`.
///
/// Provenance: host mirror of the golden `pow2_f32`; no Unreal Engine source or
/// derived code.
#[must_use]
fn pow2_f32(k: u32) -> f32 {
    let mut value = 1.0f32;
    let mut remaining = k;
    while remaining > 0 {
        value *= 2.0;
        remaining -= 1;
    }
    value
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
