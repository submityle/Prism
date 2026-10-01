//! `wgpu` compute twin of the sparse `VDB` density sampler
//! ([`sample_vdb_density`](prism_render_architecture::particle::vdb_volume_sample::sample_vdb_density)
//! / [`VdbTree::sample_density`](prism_render_architecture::particle::vdb_volume_sample::VdbTree::sample_density)).
//!
//! A next-generation smoke / fire / explosion volume is mostly empty, so its
//! density lives in a sparse `OpenVDB`-style tree whose empty regions cost
//! nothing (design `docs/prism_particle_engine_design_zh.md` §8.3, "场: ...
//! `SDF`/`VDB`"). The `CPU` golden
//! [`sample_vdb_density`](prism_render_architecture::particle::vdb_volume_sample::sample_vdb_density)
//! descends that three-level (`root` → `internal` → `leaf`) tree per corner and
//! reconstructs a continuous density by trilinear interpolation of the eight
//! surrounding voxels. [`GpuVdbSample`] is the on-device twin: it flattens the
//! tree into `std430` storage buffers, uploads them once per dispatch, runs one
//! thread per query point and returns the same density the reference does, so a
//! passing real-device parity test is direct evidence the ported descent plus
//! trilinear blend computes the same values as the reference, not merely that
//! the shader compiles (design §5, §9 require the dual-backend parity).
//!
//! # Packing
//!
//! The reference keeps its node pools private, so the twin reconstructs an
//! equivalent flat layout through the public `VdbTree` surface
//! ([`is_voxel_active`](prism_render_architecture::particle::vdb_volume_sample::VdbTree::is_voxel_active)
//! and
//! [`voxel_value`](prism_render_architecture::particle::vdb_volume_sample::VdbTree::voxel_value)):
//! a `root` slot buffer of child indices, an `internal` child-index buffer, a
//! `leaf` value buffer and a `leaf` active-mask buffer (one `u32` bit per
//! voxel). The sentinel index `0xFFFF_FFFF` is the `u32` image of the
//! reference's `-1` "inactive tile / unallocated child", so the descent math is
//! identical. The resulting buffer byte sizes are cross-checked against the
//! reference's own
//! [`std430_layout`](prism_render_architecture::particle::vdb_volume_sample::VdbTree::std430_layout)
//! report, so the twin reuses the reference `std430` `ABI` rather than
//! inventing a second one.
//!
//! # Portability
//!
//! The kernel descends the tree with integer shifts, masks and comparisons and
//! reconstructs the density with `floor` plus multiply/add in the portable
//! core-`WGSL` subset — no `exp`, `pow`, `sin`, `cos` or any optional device
//! feature — so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! Each corner lookup is an exact integer descent that reproduces
//! [`voxel_value`](prism_render_architecture::particle::vdb_volume_sample::VdbTree::voxel_value)
//! bit for bit (no floating point at all), and the trilinear blend applies the
//! identical multiply/add sequence as the reference. `CPU` and `GPU` therefore
//! evaluate the same closed-form algebra; they are not asserted bit-exact only
//! because a `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few `ULP`. The parity test asserts
//! each density to within `abs_diff < 1e-4` or `rel_diff < 1e-4`, far tighter
//! than any physically meaningful density difference yet enough to fail a wrong
//! port (a swapped shift, a dropped active-mask test, a misordered corner
//! weight).
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `OpenVDB` / `NanoVDB` sparse-tree descent (Museth 2013)
//! plus a trilinear reconstruction and `wgpu` compute dispatch; no Unreal
//! Engine source or derived code.

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

/// Compute workgroup size: one thread per query point.
const WORKGROUP_SIZE: u32 = 64;

/// Byte stride of a scalar `u32` / `f32` `std430` storage element.
const U32_STRIDE: u64 = size_of::<u32>() as u64;

/// Uniform parameters for one dispatch. Field order matches `Params` in the
/// embedded kernel: the root slot grid, the voxel domain extent, the query
/// count and the tree `background` value.
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
    /// Number of valid query points in this dispatch.
    count: u32,
    /// Uniform value read by every inactive tile, inactive voxel or
    /// out-of-domain sample.
    background: f32,
}

/// One query point as uploaded. `16`-byte `repr(C)` so a tightly packed slice
/// matches the kernel's `array<Point>` stride (the position plus one pad word).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPoint {
    /// Voxel-space `X` coordinate.
    x: f32,
    /// Voxel-space `Y` coordinate.
    y: f32,
    /// Voxel-space `Z` coordinate.
    z: f32,
    /// Padding to a `16`-byte stride.
    pad: f32,
}

/// The flattened `std430` image of a [`VdbTree`] plus the uniform geometry the
/// kernel needs to descend it.
struct PackedTree {
    /// Uniform geometry; `count` is filled per dispatch.
    params: Params,
    /// Dense `root` grid of child indices (`X`-fastest), each an internal-node
    /// index or [`NO_CHILD`].
    root: Vec<u32>,
    /// Internal child-index buffer: `INTERNAL_SIZE` `u32`s per allocated node,
    /// each a leaf index or [`NO_CHILD`].
    internals: Vec<u32>,
    /// Leaf value buffer: `LEAF_SIZE` `f32`s per allocated leaf, `X`-fastest.
    leaf_values: Vec<f32>,
    /// Leaf active-mask buffer: `MASK_WORDS_PER_LEAF` `u32`s per allocated leaf,
    /// one bit per voxel.
    leaf_masks: Vec<u32>,
}

/// Flattens `tree` into the `std430` buffers the descent kernel binds.
///
/// The reference node pools are private, so occupancy is recovered through the
/// public
/// [`is_voxel_active`](prism_render_architecture::particle::vdb_volume_sample::VdbTree::is_voxel_active)
/// / [`voxel_value`](prism_render_architecture::particle::vdb_volume_sample::VdbTree::voxel_value)
/// surface: a leaf is emitted when it holds at least one active voxel and an
/// internal node when it holds at least one such leaf. Because the reference
/// only ever allocates a node when a voxel is written active (and never
/// deactivates), that reconstruction yields exactly the reference's sparse
/// occupancy — asserted against its
/// [`std430_layout`](prism_render_architecture::particle::vdb_volume_sample::VdbTree::std430_layout)
/// report so the twin reuses the reference `ABI`.
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
                                            // One `u32` mask word per `32` voxels.
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

/// A compiled, reusable sparse-`VDB` density-sampling pipeline.
pub struct GpuVdbSample {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuVdbSample {
    /// Compiles the sparse-`VDB` density kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuVdbSample {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_vdb_sample"),
            source: ShaderSource::Wgsl(VDB_SAMPLE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_vdb_sample_layout"),
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
            label: Some("prism_volumetric_vdb_sample_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_vdb_sample_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("vdb_sample_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuVdbSample {
            module,
            layout,
            pipeline,
        }
    }

    /// Trilinearly samples `tree`'s density at every voxel-space position in
    /// `points`, returning one `f32` per point in input order.
    ///
    /// The returned density for point `p` equals
    /// [`sample_vdb_density`](prism_render_architecture::particle::vdb_volume_sample::sample_vdb_density)`(tree, p)`
    /// to within the tolerance documented on this module. An empty `points`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, tree: &VdbTree, points: &[Vec3]) -> Vec<f32> {
        if points.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let packed = pack(tree);
        let mut params = packed.params;
        params.count = points.len() as u32;

        let gpu_points: Vec<GpuPoint> = points
            .iter()
            .map(|p| GpuPoint {
                x: p.x,
                y: p.y,
                z: p.z,
                pad: 0.0,
            })
            .collect();

        let root = padded_u32(packed.root);
        let internals = padded_u32(packed.internals);
        let leaf_values = padded_f32(packed.leaf_values);
        let leaf_masks = padded_u32(packed.leaf_masks);

        // One density `f32` per query point, tightly packed.
        let out_bytes = (points.len() as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_vdb_sample_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let points_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_vdb_sample_points"),
            contents: bytemuck::cast_slice(&gpu_points),
            usage: BufferUsages::STORAGE,
        });
        let root_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_vdb_sample_root"),
            contents: bytemuck::cast_slice(&root),
            usage: BufferUsages::STORAGE,
        });
        let internals_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_vdb_sample_internals"),
            contents: bytemuck::cast_slice(&internals),
            usage: BufferUsages::STORAGE,
        });
        let leaf_values_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_vdb_sample_leaf_values"),
            contents: bytemuck::cast_slice(&leaf_values),
            usage: BufferUsages::STORAGE,
        });
        let leaf_masks_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_vdb_sample_leaf_masks"),
            contents: bytemuck::cast_slice(&leaf_masks),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_vdb_sample_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_vdb_sample_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_vdb_sample_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: points_buf.as_entire_binding(),
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
            label: Some("prism_volumetric_vdb_sample_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_vdb_sample_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (points.len() as u32).div_ceil(WORKGROUP_SIZE);
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
        let densities = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(densities.len(), points.len());
        densities
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
/// uses only shifts, masks, comparisons, `floor` and multiply/add, so it needs
/// no optional device feature.
const VDB_SAMPLE_WGSL: &str = r#"
// Sparse VDB density twin: descends the root -> internal -> leaf tree per
// corner and reconstructs a trilinear density, mirroring the CPU golden
// `sample_vdb_density` (in
// `prism_render_architecture::particle::vdb_volume_sample`).
//
// The descent is pure integer shift/mask/compare and the reconstruction is
// `floor` plus multiply/add - no exp, pow, sin, cos or optional feature - so
// CPU and GPU evaluate the identical closed-form algebra and this twin runs
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard OpenVDB/NanoVDB sparse descent plus trilinear
// reconstruction; no Unreal Engine source or derived code.

// Leaf spans 8 voxels per axis (2^3); internal node spans 4 children per axis
// (2^2); a root slot therefore covers 32 voxels per axis (2^5).
const LEAF_LOG2: u32 = 3u;
const LEAF_MASK: u32 = 7u;
const INTERNAL_LOG2: u32 = 2u;
const INTERNAL_MASK: u32 = 3u;
const INTERNAL_SIZE: u32 = 64u;
const SLOT_LOG2: u32 = 5u;
const LEAF_SIZE: u32 = 512u;
const MASK_WORDS_PER_LEAF: u32 = 16u;
// Bit width of one u32 active-mask word.
const MASK_WORD_BITS: u32 = 32u;
// Sentinel: the u32 image of the reference's -1 "inactive / unallocated".
const NO_CHILD: u32 = 4294967295u;

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

// One query point. 16-byte stride matching the bytemuck upload struct.
struct Point {
    x: f32,
    y: f32,
    z: f32,
    pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> points: array<Point>;
@group(0) @binding(2) var<storage, read> root: array<u32>;
@group(0) @binding(3) var<storage, read> internals: array<u32>;
@group(0) @binding(4) var<storage, read> leaf_values: array<f32>;
@group(0) @binding(5) var<storage, read> leaf_masks: array<u32>;
@group(0) @binding(6) var<storage, read_write> results: array<f32>;

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

    // Level 1: root slot (top bits of each coordinate).
    let sx = ux >> SLOT_LOG2;
    let sy = uy >> SLOT_LOG2;
    let sz = uz >> SLOT_LOG2;
    let root_idx = ((sz * params.root_y) + sy) * params.root_x + sx;
    let internal = root[root_idx];
    if (internal == NO_CHILD) {
        return params.background;
    }

    // Level 2: internal child (middle bits).
    let ix = (ux >> LEAF_LOG2) & INTERNAL_MASK;
    let iy = (uy >> LEAF_LOG2) & INTERNAL_MASK;
    let iz = (uz >> LEAF_LOG2) & INTERNAL_MASK;
    let child_slot = ix | (iy << INTERNAL_LOG2) | (iz << (2u * INTERNAL_LOG2));
    let leaf = internals[internal * INTERNAL_SIZE + child_slot];
    if (leaf == NO_CHILD) {
        return params.background;
    }

    // Level 3: dense leaf voxel (low bits), gated by the active-mask bit.
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

@compute @workgroup_size(64)
fn vdb_sample_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let p = points[idx];

    let i0f = floor(p.x);
    let j0f = floor(p.y);
    let k0f = floor(p.z);
    let fx = p.x - i0f;
    let fy = p.y - j0f;
    let fz = p.z - k0f;
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

    results[idx] = c0 * gz + c1 * fz;
}
"#;
