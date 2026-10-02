//! `wgpu` compute twin of the sparse `VDB` *gradient / surface-normal* sampler
//! ([`sample_vdb_gradient`](prism_render_architecture::particle::vdb_volume_sample::sample_vdb_gradient)
//! / [`VdbTree::sample_gradient`](prism_render_architecture::particle::vdb_volume_sample::VdbTree::sample_gradient)),
//! the complement of the already-landed density twin
//! [`GpuVdbSample`](crate::GpuVdbSample).
//!
//! # Why a gradient-only twin
//!
//! The §8.3 data-interface rule "法线=梯度" (design
//! `docs/prism_particle_engine_design_zh.md` §8.3) reads a sparse `VDB`
//! volume's surface normal as the normalized gradient of its density field.
//! The `CPU` golden takes that gradient by **central differences** of the same
//! trilinear density the density sampler reconstructs: it samples the density
//! at the two midpoints `GRAD_STEP` on either side of the query along each
//! axis, divides the difference by the full `2 * GRAD_STEP` span, then guards
//! the `1/sqrt(len^2)` normalization so a flat region yields
//! [`Vec3::ZERO`](prism_render_architecture::particle::Vec3::ZERO) rather than a
//! `NaN`.
//!
//! The density descent, the `std430` flattening and the trilinear blend are
//! already twinned and validated by [`GpuVdbSample`](crate::GpuVdbSample); this
//! module reuses that exact flattened layout and that exact `WGSL`
//! `voxel_value` / trilinear descent, and adds only the novel surface the
//! density twin does not cover: the six-tap central difference and the guarded
//! normalization. [`GpuVdbVolumeSample`] is the on-device twin: one thread
//! solves one [`VdbVolumeSampleQuery`] against the uploaded tree and writes one
//! [`VdbVolumeSampleResult`], so a passing real-device parity test is direct
//! evidence the ported gradient folds the same central difference and the same
//! degenerate guard the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! * The raw (unnormalized) central-difference gradient
//!   ([`VdbVolumeSampleQuery::RawGradient`]): six trilinear density samples at
//!   `pos ± (GRAD_STEP, 0, 0)`, `pos ± (0, GRAD_STEP, 0)`,
//!   `pos ± (0, 0, GRAD_STEP)`, differenced per axis and scaled by
//!   `1 / (2 * GRAD_STEP)`.
//! * The unit gradient / surface normal
//!   ([`VdbVolumeSampleQuery::UnitGradient`]): the raw gradient guard-normalized
//!   by `1/sqrt(len^2)` when `len^2 > GRAD_EPS_SQ`, else [`Vec3::ZERO`].
//!
//! # What is left on the host
//!
//! Everything stateful or 64-bit stays on the host, exactly as for the density
//! twin: [`VdbTree::new`](prism_render_architecture::particle::vdb_volume_sample::VdbTree::new)
//! / [`VdbTree::set_voxel`](prism_render_architecture::particle::vdb_volume_sample::VdbTree::set_voxel)
//! build the mutable sparse tree; the host flattens it into fixed-length
//! `std430` buffers and uploads them; and the `u64` byte-sizing
//! ([`VdbTree::std430_layout`](prism_render_architecture::particle::vdb_volume_sample::VdbTree::std430_layout))
//! stays on the host (the kernel uses only `u32`/`i32`/`f32`). The density
//! descent, trilinear blend, `voxel_value`, `is_voxel_active` and the `std430`
//! `ABI` itself are not re-twinned here: this module imports the identical
//! flattening and `WGSL` descent the density twin already validated.
//!
//! # No transcendental math
//!
//! The gradient uses `floor` (the trilinear integer voxel location), unsigned
//! shifts and masks (the tree descent), multiply/add (the central difference)
//! and a single `sqrt` (the one allowed transcendental, used only to normalize
//! the unit gradient). It uses no `sin`, `cos`, `exp`, `log`, `pow`, no inverse
//! trigonometry and no `smoothstep`, and only `i32`/`u32`/`f32` (no `u64`): the
//! `u64` byte sizing stays on the host.
//!
//! # Correctness model
//!
//! The op selector is an integer classification, so the kernel runs exactly the
//! branch the host requested. Each corner lookup is a bit-exact integer descent
//! and the central difference is pure multiply/add, so `CPU` and `GPU` evaluate
//! the identical closed-form algebra; they are not asserted bit-exact only
//! because a `GPU` may fuse a multiply-add or round a widen or the `sqrt` a hair
//! differently, perturbing the low mantissa bits by a few units in the last
//! place. The parity test asserts each gradient component to within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`). The degenerate
//! (flat) region returns an exact all-zero vector on both backends, so it is
//! compared bit-for-bit.
//!
//! # Degenerate inputs
//!
//! A flat region (central-difference squared length at or below `GRAD_EPS_SQ`)
//! yields an exact [`Vec3::ZERO`] on both backends rather than a `NaN`. The
//! fixtures place every non-degenerate query well clear of that threshold (the
//! random field makes `len^2` of order one, far above `GRAD_EPS_SQ = 1e-12`),
//! and every degenerate query squarely inside a constant-background region
//! where the gradient is exactly zero, so no fixture lands in the thin
//! classification band around the guard.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::vdb_volume_sample`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::vdb_volume_sample::VdbTree;
use prism_render_architecture::particle::Vec3;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Base-two logarithm of a leaf's per-axis voxel count; a leaf spans
/// `LEAF_DIM = 2^LEAF_LOG2 = 8` voxels per axis (the `OpenVDB` default leaf
/// size, mirrored from the `CPU` reference's fixed topology).
const LEAF_LOG2: u32 = 3;

/// Per-axis voxel count of a leaf (`8`).
const LEAF_DIM: u32 = 1 << LEAF_LOG2;

/// Total voxels in one leaf block, `LEAF_DIM^3 = 512`.
const LEAF_SIZE: usize = 1 << (3 * LEAF_LOG2);

/// Base-two logarithm of an internal node's per-axis child count; an internal
/// node spans `INTERNAL_DIM = 2^INTERNAL_LOG2 = 4` leaves per axis.
const INTERNAL_LOG2: u32 = 2;

/// Per-axis child count of an internal node (`4`).
const INTERNAL_DIM: u32 = 1 << INTERNAL_LOG2;

/// Total child slots in one internal node, `INTERNAL_DIM^3 = 64`.
const INTERNAL_SIZE: usize = 1 << (3 * INTERNAL_LOG2);

/// Number of low bits a voxel coordinate dedicates to the sub-tree below the
/// root; each root slot covers `2^SLOT_LOG2 = 32` voxels per axis.
const SLOT_LOG2: u32 = LEAF_LOG2 + INTERNAL_LOG2;

/// Active-mask words per leaf: one `u32` bit per voxel, so `LEAF_SIZE / 32 = 16`
/// words. The `32` is the bit width of a `u32` mask word.
const MASK_WORDS_PER_LEAF: usize = LEAF_SIZE / 32;

/// Sentinel child index meaning "inactive tile / unallocated child"; the `u32`
/// image of the reference's `-1`, so a slot holding it reads the tree
/// `background`.
const NO_CHILD: u32 = u32::MAX;

/// Compute workgroup size: one thread per query.
const WORKGROUP_SIZE: u32 = 64;

/// Byte stride of a scalar `u32` / `f32` `std430` storage element.
const U32_STRIDE: u64 = size_of::<u32>() as u64;

/// Half-voxel central-difference step, mirroring the reference `GRAD_STEP`.
const GRAD_STEP: f32 = 0.5;

/// `WGSL` op selector: the raw (unnormalized) central-difference gradient.
const OP_RAW_GRADIENT: u32 = 0;

/// `WGSL` op selector: the guard-normalized unit gradient (surface normal).
const OP_UNIT_GRADIENT: u32 = 1;

/// Uniform parameters for one dispatch. Field order matches `Params` in the
/// embedded kernel.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Root slot count along `X`.
    root_x: u32,
    /// Root slot count along `Y`.
    root_y: u32,
    /// Root slot count along `Z`.
    root_z: u32,
    /// Voxel domain extent along `X` (`root_x << SLOT_LOG2`).
    domain_x: u32,
    /// Voxel domain extent along `Y`.
    domain_y: u32,
    /// Voxel domain extent along `Z`.
    domain_z: u32,
    /// Number of valid queries in this dispatch.
    count: u32,
    /// Uniform value read by every inactive tile, inactive voxel or
    /// out-of-domain sample.
    background: f32,
}

/// One query as uploaded: a voxel-space position plus the op selector. `16`-byte
/// `repr(C)` so a tightly packed slice matches the kernel's `array<Query>`
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Voxel-space `X` coordinate.
    x: f32,
    /// Voxel-space `Y` coordinate.
    y: f32,
    /// Voxel-space `Z` coordinate.
    z: f32,
    /// Op selector ([`OP_RAW_GRADIENT`] or [`OP_UNIT_GRADIENT`]).
    op: u32,
}

/// One resolved gradient as read back: a `vec3` plus one pad word to a `16`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Gradient `X` component.
    x: f32,
    /// Gradient `Y` component.
    y: f32,
    /// Gradient `Z` component.
    z: f32,
    /// Padding to a `16`-byte stride.
    pad: f32,
}

/// One query for the sparse-`VDB` gradient twin: a voxel-space position and
/// which gradient to evaluate.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::vdb_volume_sample`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VdbVolumeSampleQuery {
    /// The raw, unnormalized central-difference gradient at `pos`, twinning the
    /// reference `raw_gradient` (density-per-voxel units).
    RawGradient {
        /// Voxel-space query position `(x, y, z)`.
        pos: [f32; 3],
    },
    /// The unit gradient (surface normal) at `pos`, twinning
    /// [`VdbTree::sample_gradient`](prism_render_architecture::particle::vdb_volume_sample::VdbTree::sample_gradient):
    /// the guard-normalized central-difference gradient.
    UnitGradient {
        /// Voxel-space query position `(x, y, z)`.
        pos: [f32; 3],
    },
}

impl VdbVolumeSampleQuery {
    /// Returns the voxel-space query position.
    fn pos(&self) -> [f32; 3] {
        match self {
            VdbVolumeSampleQuery::RawGradient { pos }
            | VdbVolumeSampleQuery::UnitGradient { pos } => *pos,
        }
    }

    /// Returns the `WGSL` op selector for this query.
    fn op(&self) -> u32 {
        match self {
            VdbVolumeSampleQuery::RawGradient { .. } => OP_RAW_GRADIENT,
            VdbVolumeSampleQuery::UnitGradient { .. } => OP_UNIT_GRADIENT,
        }
    }
}

/// One resolved answer: the gradient vector for the corresponding
/// [`VdbVolumeSampleQuery`].
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::vdb_volume_sample`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VdbVolumeSampleResult {
    /// A gradient vector `(x, y, z)` (raw or unit, matching the query op).
    Gradient([f32; 3]),
}

/// The `CPU` golden gradient for one query, dispatching to the reference entry
/// points so callers (and the parity test) share one definition of truth.
///
/// The raw gradient re-derives the reference `raw_gradient` from the public
/// [`VdbTree::sample_density`](prism_render_architecture::particle::vdb_volume_sample::VdbTree::sample_density)
/// (which `raw_gradient` itself calls); the unit gradient calls
/// [`VdbTree::sample_gradient`](prism_render_architecture::particle::vdb_volume_sample::VdbTree::sample_gradient)
/// directly.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::vdb_volume_sample`；无第三方引擎源码或衍生代码。
#[must_use]
pub fn cpu_reference(tree: &VdbTree, query: &VdbVolumeSampleQuery) -> VdbVolumeSampleResult {
    let [px, py, pz] = query.pos();
    let p = Vec3::new(px, py, pz);
    match query {
        VdbVolumeSampleQuery::RawGradient { .. } => {
            let g = raw_gradient(tree, p);
            VdbVolumeSampleResult::Gradient([g.x, g.y, g.z])
        }
        VdbVolumeSampleQuery::UnitGradient { .. } => {
            let g = tree.sample_gradient(p);
            VdbVolumeSampleResult::Gradient([g.x, g.y, g.z])
        }
    }
}

/// Re-derives the reference private `raw_gradient` from the public trilinear
/// density sampler: a six-tap central difference scaled by `1 / (2 * GRAD_STEP)`.
fn raw_gradient(tree: &VdbTree, p: Vec3) -> Vec3 {
    let dx = tree.sample_density(p.add(Vec3::new(GRAD_STEP, 0.0, 0.0)))
        - tree.sample_density(p.sub(Vec3::new(GRAD_STEP, 0.0, 0.0)));
    let dy = tree.sample_density(p.add(Vec3::new(0.0, GRAD_STEP, 0.0)))
        - tree.sample_density(p.sub(Vec3::new(0.0, GRAD_STEP, 0.0)));
    let dz = tree.sample_density(p.add(Vec3::new(0.0, 0.0, GRAD_STEP)))
        - tree.sample_density(p.sub(Vec3::new(0.0, 0.0, GRAD_STEP)));
    Vec3::new(dx, dy, dz).scale(1.0 / (2.0 * GRAD_STEP))
}

/// The flattened `std430` image of a [`VdbTree`] plus the uniform geometry the
/// kernel needs to descend it.
struct PackedTree {
    /// Uniform geometry; `count` is filled per dispatch.
    params: Params,
    /// Dense `root` grid of child indices (`X`-fastest), each an internal-node
    /// index or [`NO_CHILD`].
    root: Vec<u32>,
    /// Internal child-index buffer: `INTERNAL_SIZE` `u32`s per allocated node.
    internals: Vec<u32>,
    /// Leaf value buffer: `LEAF_SIZE` `f32`s per allocated leaf, `X`-fastest.
    leaf_values: Vec<f32>,
    /// Leaf active-mask buffer: `MASK_WORDS_PER_LEAF` `u32`s per allocated leaf.
    leaf_masks: Vec<u32>,
}

/// Flattens `tree` into the `std430` buffers the descent kernel binds, reusing
/// the identical reconstruction the density twin validated: occupancy is
/// recovered through the public
/// [`is_voxel_active`](prism_render_architecture::particle::vdb_volume_sample::VdbTree::is_voxel_active)
/// / [`voxel_value`](prism_render_architecture::particle::vdb_volume_sample::VdbTree::voxel_value)
/// surface and cross-checked against the reference `std430` byte-size report.
fn pack(tree: &VdbTree) -> PackedTree {
    let [root_x, root_y, root_z] = tree.root_dims();
    let [domain_x, domain_y, domain_z] = tree.domain_dims();
    let background = tree.background();

    let root_slots = (root_x as usize) * (root_y as usize) * (root_z as usize);
    let mut root = vec![NO_CHILD; root_slots];
    let mut internals: Vec<u32> = Vec::new();
    let mut leaf_values: Vec<f32> = Vec::new();
    let mut leaf_masks: Vec<u32> = Vec::new();

    for sz in 0..root_z {
        for sy in 0..root_y {
            for sx in 0..root_x {
                let root_idx = (((sz * root_y) + sy) * root_x + sx) as usize;
                let mut children = [NO_CHILD; INTERNAL_SIZE];
                let mut any_leaf = false;

                for iz in 0..INTERNAL_DIM {
                    for iy in 0..INTERNAL_DIM {
                        for ix in 0..INTERNAL_DIM {
                            let child_slot =
                                (ix | (iy << INTERNAL_LOG2) | (iz << (2 * INTERNAL_LOG2))) as usize;
                            let base_x = (sx << SLOT_LOG2) | (ix << LEAF_LOG2);
                            let base_y = (sy << SLOT_LOG2) | (iy << LEAF_LOG2);
                            let base_z = (sz << SLOT_LOG2) | (iz << LEAF_LOG2);

                            let mut values = [0.0_f32; LEAF_SIZE];
                            let mut masks = [0_u32; MASK_WORDS_PER_LEAF];
                            let mut any_active = false;

                            for lz in 0..LEAF_DIM {
                                for ly in 0..LEAF_DIM {
                                    for lx in 0..LEAF_DIM {
                                        let voxel_slot =
                                            (lx | (ly << LEAF_LOG2) | (lz << (2 * LEAF_LOG2)))
                                                as usize;
                                        let coord = [
                                            (base_x + lx) as i32,
                                            (base_y + ly) as i32,
                                            (base_z + lz) as i32,
                                        ];
                                        if tree.is_voxel_active(coord) {
                                            any_active = true;
                                            values[voxel_slot] = tree.voxel_value(coord);
                                            masks[voxel_slot >> 5] |= 1_u32 << (voxel_slot & 31);
                                        }
                                    }
                                }
                            }

                            if any_active {
                                let leaf_idx = (leaf_values.len() / LEAF_SIZE) as u32;
                                leaf_values.extend_from_slice(&values);
                                leaf_masks.extend_from_slice(&masks);
                                children[child_slot] = leaf_idx;
                                any_leaf = true;
                            }
                        }
                    }
                }

                if any_leaf {
                    let internal_idx = (internals.len() / INTERNAL_SIZE) as u32;
                    internals.extend_from_slice(&children);
                    root[root_idx] = internal_idx;
                }
            }
        }
    }

    // Cross-check the reconstructed buffer sizes against the reference's own
    // `std430` byte-size report, so the twin provably reuses the reference
    // layout instead of inventing a second one.
    let layout = tree.std430_layout();
    debug_assert_eq!(root.len() as u64 * U32_STRIDE, layout.root_bytes);
    debug_assert_eq!(internals.len() as u64 * U32_STRIDE, layout.internal_bytes);
    debug_assert_eq!(
        leaf_values.len() as u64 * U32_STRIDE,
        layout.leaf_value_bytes
    );
    debug_assert_eq!(leaf_masks.len() as u64 * U32_STRIDE, layout.leaf_mask_bytes);

    let params = Params {
        root_x,
        root_y,
        root_z,
        domain_x,
        domain_y,
        domain_z,
        count: 0,
        background,
    };
    PackedTree {
        params,
        root,
        internals,
        leaf_values,
        leaf_masks,
    }
}

/// Pads a `u32` buffer up to one element: a `WebGPU` storage binding may not be
/// zero-sized, and an empty sub-tree never dereferences the padding slot.
fn padded_u32(mut buffer: Vec<u32>) -> Vec<u32> {
    if buffer.is_empty() {
        buffer.push(0);
    }
    buffer
}

/// Pads an `f32` buffer up to one element, for the same reason as
/// [`padded_u32`].
fn padded_f32(mut buffer: Vec<f32>) -> Vec<f32> {
    if buffer.is_empty() {
        buffer.push(0.0);
    }
    buffer
}

/// A compiled, reusable sparse-`VDB` gradient-sampling pipeline.
pub struct GpuVdbVolumeSample {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuVdbVolumeSample {
    /// Compiles the sparse-`VDB` gradient kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset (plus the one
    /// allowed `sqrt`), so no optional device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuVdbVolumeSample {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_vdb_volume_sample"),
            source: ShaderSource::Wgsl(VDB_VOLUME_SAMPLE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_vdb_volume_sample_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: true }),
                buffer_entry(6, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_vdb_volume_sample_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_vdb_volume_sample_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("vdb_gradient_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuVdbVolumeSample {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` against `tree` and returns one
    /// [`VdbVolumeSampleResult`] per input, in order.
    ///
    /// Each result equals [`cpu_reference`]`(tree, query)` to within the
    /// tolerance documented on this module. An empty `queries` batch returns an
    /// empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        tree: &VdbTree,
        queries: &[VdbVolumeSampleQuery],
    ) -> Vec<VdbVolumeSampleResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let packed = pack(tree);
        let mut params = packed.params;
        params.count = queries.len() as u32;

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| {
                let [x, y, z] = q.pos();
                GpuQuery {
                    x,
                    y,
                    z,
                    op: q.op(),
                }
            })
            .collect();

        let root = padded_u32(packed.root);
        let internals = padded_u32(packed.internals);
        let leaf_values = padded_f32(packed.leaf_values);
        let leaf_masks = padded_u32(packed.leaf_masks);

        let out_bytes = (queries.len() as u64) * (size_of::<GpuResult>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_vdb_volume_sample_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_vdb_volume_sample_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let root_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_vdb_volume_sample_root"),
            contents: bytemuck::cast_slice(&root),
            usage: BufferUsages::STORAGE,
        });
        let internals_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_vdb_volume_sample_internals"),
            contents: bytemuck::cast_slice(&internals),
            usage: BufferUsages::STORAGE,
        });
        let leaf_values_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_vdb_volume_sample_leaf_values"),
            contents: bytemuck::cast_slice(&leaf_values),
            usage: BufferUsages::STORAGE,
        });
        let leaf_masks_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_vdb_volume_sample_leaf_masks"),
            contents: bytemuck::cast_slice(&leaf_masks),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_vdb_volume_sample_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_vdb_volume_sample_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_vdb_volume_sample_bind_group"),
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
                    resource: root_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: internals_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: leaf_values_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: leaf_masks_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_vdb_volume_sample_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_vdb_volume_sample_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
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
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());

        raw.iter()
            .map(|r| VdbVolumeSampleResult::Gradient([r.x, r.y, r.z]))
            .collect()
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

/// Embedded portable core-`WGSL` kernel. Kept inline (rather than a sibling
/// `shaders/*.wesl` file) because this twin ships as a single source file. It
/// reuses the density twin's exact `voxel_value` / trilinear descent (shifts,
/// masks, comparisons, `floor`, multiply/add) and adds a six-tap central
/// difference plus one guarded `sqrt`, so it needs no optional device feature.
const VDB_VOLUME_SAMPLE_WGSL: &str = r#"
// Sparse VDB gradient twin: descends the root -> internal -> leaf tree per
// corner, reconstructs a trilinear density (identical to the validated density
// twin), then takes a six-tap central-difference gradient and, for the unit op,
// guard-normalizes it. Mirrors the CPU golden `sample_vdb_gradient` /
// `raw_gradient` (in `prism_render_architecture::particle::vdb_volume_sample`).
//
// The descent is pure integer shift/mask/compare, the reconstruction is `floor`
// plus multiply/add, the difference is multiply/add, and the only transcendental
// is the single `sqrt` the unit normalization needs - no exp, pow, sin, cos or
// optional feature - so this twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard OpenVDB/NanoVDB sparse descent plus trilinear
// reconstruction and a central-difference gradient; no Unreal Engine source or
// derived code.

const LEAF_LOG2: u32 = 3u;
const LEAF_MASK: u32 = 7u;
const INTERNAL_LOG2: u32 = 2u;
const INTERNAL_MASK: u32 = 3u;
const INTERNAL_SIZE: u32 = 64u;
const SLOT_LOG2: u32 = 5u;
const LEAF_SIZE: u32 = 512u;
const MASK_WORDS_PER_LEAF: u32 = 16u;
const MASK_WORD_BITS: u32 = 32u;
const NO_CHILD: u32 = 4294967295u;

// Half-voxel central-difference step and its reciprocal span, mirroring the
// reference GRAD_STEP; 1 / (2 * GRAD_STEP) = 1.0 for GRAD_STEP = 0.5.
const GRAD_STEP: f32 = 0.5;
const INV_SPAN: f32 = 1.0;
// Squared-length floor below which the gradient is treated as flat.
const GRAD_EPS_SQ: f32 = 1e-12;

const OP_RAW_GRADIENT: u32 = 0u;
const OP_UNIT_GRADIENT: u32 = 1u;

struct Params {
    root_x: u32,
    root_y: u32,
    root_z: u32,
    domain_x: u32,
    domain_y: u32,
    domain_z: u32,
    count: u32,
    background: f32,
}

// One query: a voxel-space position plus the op selector. 16-byte stride.
struct Query {
    x: f32,
    y: f32,
    z: f32,
    op: u32,
}

// One result: a gradient vec3 plus one pad word. 16-byte stride.
struct Result {
    x: f32,
    y: f32,
    z: f32,
    pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read> root: array<u32>;
@group(0) @binding(3) var<storage, read> internals: array<u32>;
@group(0) @binding(4) var<storage, read> leaf_values: array<f32>;
@group(0) @binding(5) var<storage, read> leaf_masks: array<u32>;
@group(0) @binding(6) var<storage, read_write> results: array<Result>;

// Reads the density at integer voxel (cx, cy, cz), descending
// root -> internal -> leaf exactly like the reference `voxel_value`. Returns
// `background` for any inactive tile, inactive voxel or out-of-domain sample.
fn voxel_value(cx: i32, cy: i32, cz: i32) -> f32 {
    if (cx < 0 || cy < 0 || cz < 0) {
        return params.background;
    }
    let ux = u32(cx);
    let uy = u32(cy);
    let uz = u32(cz);
    if (ux >= params.domain_x || uy >= params.domain_y || uz >= params.domain_z) {
        return params.background;
    }

    let sx = ux >> SLOT_LOG2;
    let sy = uy >> SLOT_LOG2;
    let sz = uz >> SLOT_LOG2;
    let root_idx = ((sz * params.root_y) + sy) * params.root_x + sx;
    let internal = root[root_idx];
    if (internal == NO_CHILD) {
        return params.background;
    }

    let ix = (ux >> LEAF_LOG2) & INTERNAL_MASK;
    let iy = (uy >> LEAF_LOG2) & INTERNAL_MASK;
    let iz = (uz >> LEAF_LOG2) & INTERNAL_MASK;
    let child_slot = ix | (iy << INTERNAL_LOG2) | (iz << (2u * INTERNAL_LOG2));
    let leaf = internals[internal * INTERNAL_SIZE + child_slot];
    if (leaf == NO_CHILD) {
        return params.background;
    }

    let lx = ux & LEAF_MASK;
    let ly = uy & LEAF_MASK;
    let lz = uz & LEAF_MASK;
    let voxel_slot = lx | (ly << LEAF_LOG2) | (lz << (2u * LEAF_LOG2));
    let word = leaf_masks[leaf * MASK_WORDS_PER_LEAF + (voxel_slot / MASK_WORD_BITS)];
    let bit = voxel_slot % MASK_WORD_BITS;
    if (((word >> bit) & 1u) == 0u) {
        return params.background;
    }
    return leaf_values[leaf * LEAF_SIZE + voxel_slot];
}

// Trilinearly reconstructs the density at a continuous voxel-space position,
// identical to the validated density twin.
fn sample_density(px: f32, py: f32, pz: f32) -> f32 {
    let i0f = floor(px);
    let j0f = floor(py);
    let k0f = floor(pz);
    let fx = px - i0f;
    let fy = py - j0f;
    let fz = pz - k0f;
    let i0 = i32(i0f);
    let j0 = i32(j0f);
    let k0 = i32(k0f);
    let i1 = i0 + 1;
    let j1 = j0 + 1;
    let k1 = k0 + 1;

    let c000 = voxel_value(i0, j0, k0);
    let c100 = voxel_value(i1, j0, k0);
    let c010 = voxel_value(i0, j1, k0);
    let c110 = voxel_value(i1, j1, k0);
    let c001 = voxel_value(i0, j0, k1);
    let c101 = voxel_value(i1, j0, k1);
    let c011 = voxel_value(i0, j1, k1);
    let c111 = voxel_value(i1, j1, k1);

    let gx = 1.0 - fx;
    let gy = 1.0 - fy;
    let gz = 1.0 - fz;

    let c00 = c000 * gx + c100 * fx;
    let c10 = c010 * gx + c110 * fx;
    let c01 = c001 * gx + c101 * fx;
    let c11 = c011 * gx + c111 * fx;

    let c0 = c00 * gy + c10 * fy;
    let c1 = c01 * gy + c11 * fy;

    return c0 * gz + c1 * fz;
}

@compute @workgroup_size(64)
fn vdb_gradient_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Six-tap central difference of the trilinear field.
    let dx = sample_density(q.x + GRAD_STEP, q.y, q.z)
        - sample_density(q.x - GRAD_STEP, q.y, q.z);
    let dy = sample_density(q.x, q.y + GRAD_STEP, q.z)
        - sample_density(q.x, q.y - GRAD_STEP, q.z);
    let dz = sample_density(q.x, q.y, q.z + GRAD_STEP)
        - sample_density(q.x, q.y, q.z - GRAD_STEP);
    let gx = dx * INV_SPAN;
    let gy = dy * INV_SPAN;
    let gz = dz * INV_SPAN;

    var ox = gx;
    var oy = gy;
    var oz = gz;
    if (q.op == OP_UNIT_GRADIENT) {
        let len2 = gx * gx + gy * gy + gz * gz;
        if (len2 > GRAD_EPS_SQ) {
            let inv = 1.0 / sqrt(len2);
            ox = gx * inv;
            oy = gy * inv;
            oz = gz * inv;
        } else {
            ox = 0.0;
            oy = 0.0;
            oz = 0.0;
        }
    }

    results[idx].x = ox;
    results[idx].y = oy;
    results[idx].z = oz;
    results[idx].pad = 0.0;
}
"#;
