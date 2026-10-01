//! `wgpu` compute twin of Prism's per-frame follicle root transfer
//! ([`transfer_root_map`](prism_render_architecture::hair::follicle_bind::transfer_root_map)),
//! the barycentric root-skinning step that glues hair roots to an animated
//! scalp.
//!
//! A groom is authored once in a rest pose, but the scalp it grows from is an
//! animated skinned mesh. Each follicle (hair root) is bound to one scalp
//! triangle by its barycentric coordinates plus a small signed offset along the
//! interpolated surface normal (roots sit a hair's breadth above the skin).
//! Once the owning renderer skins the scalp, every bound root is reconstructed
//! from the *deformed* triangle so the hair rides the head without sliding or
//! poking through — the root-skinning transfer `UE5` Groom / `TressFX` apply per
//! frame. This kernel is the on-device twin of exactly that transfer: one
//! shared deformed triangle is broadcast to a whole batch of bindings, one
//! thread per binding.
//!
//! # A distinct sibling of `root_bind` / `root_resolve`
//!
//! The [`root_bind`](crate::root_bind) / [`root_resolve`](crate::root_resolve)
//! pair resolves an indexed `MeshBinding` against a flattened vertex pool and a
//! triangle index list, rebuilds a *geometric* face normal (an edge cross
//! product), and emits a full orthonormal `RootFrame`. This kernel instead
//! takes an explicit per-vertex [`TriangleFrame`] shared by every binding,
//! interpolates the authored *per-vertex* normals (not a cross product),
//! sanitises the barycentric weights (clamp negatives, renormalise, centroid
//! fallback), and emits a position only — a different input contract, a
//! different normal, and a different output.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairFollicleBind::eval`] takes a batch of [`FollicleBinding`]s and one
//! deformed [`TriangleFrame`], and returns the transferred world root position
//! for each binding, preserving input order. The binding index is the
//! invocation id (`@compute @workgroup_size(64)`, one-dimensional dispatch over
//! `global_invocation_id.x`); invocations past the binding count early-return.
//!
//! # Degenerate inputs
//!
//! An all-non-positive weight set falls back to the triangle centroid, a
//! zero-length interpolated normal falls back to the canonical `+Z` axis, and a
//! non-finite floated position falls back to the bare surface point — exactly as
//! the golden does, so a bad binding never panics and never poisons its
//! neighbours. An empty batch yields an empty vector without a dispatch —
//! storage buffers cannot be zero-sized.
//!
//! # Portability
//!
//! The transfer uses only `sqrt`, `dot` and multiply/add in the portable
//! core-`WGSL` subset — no `exp`, `pow`, `sin` or optional device feature — so
//! the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The transfer contains no transcendental call, so `CPU` and `GPU` evaluate the
//! same closed-form geometry. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few `ULP`, and the `sqrt` in the normalise may round
//! differently. The parity test therefore asserts a tolerance (`abs_diff < 1e-4`
//! or `rel_diff < 1e-3`) per component rather than a raw bit compare.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard barycentric mesh-attachment transfer plus `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::follicle_bind::{
    transfer_root_map, FollicleBinding, TriangleFrame,
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

extern crate alloc;

/// Uniform parameters for one transfer dispatch. Layout matches `Params` in
/// `shaders/follicle_bind.wesl`: the one shared deformed triangle (three vertex
/// positions and three per-vertex normals as `16`-byte `vec4` slots, xyz used)
/// followed by the binding count in a final `16`-byte slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    pos0: [f32; 4],
    pos1: [f32; 4],
    pos2: [f32; 4],
    nrm0: [f32; 4],
    nrm1: [f32; 4],
    nrm2: [f32; 4],
    binding_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One follicle attachment uploaded to the kernel. `16`-byte `repr(C)` matching
/// `Binding` in `shaders/follicle_bind.wesl`: the three barycentric weights and
/// the signed normal offset.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuBinding {
    bary0: f32,
    bary1: f32,
    bary2: f32,
    normal_offset: f32,
}

/// A compiled, reusable follicle root-transfer pipeline.
pub struct GpuHairFollicleBind {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairFollicleBind {
    /// Compiles the follicle root-transfer kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset (`sqrt`, `dot` and
    /// multiply/add), so no optional device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairFollicleBind {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_follicle_bind"),
            source: ShaderSource::Wgsl(include_str!("../shaders/follicle_bind.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_follicle_bind_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_follicle_bind_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_follicle_bind_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairFollicleBind {
            module,
            layout,
            pipeline,
        }
    }

    /// Transfers every binding against the one deformed triangle, returning one
    /// world root position per binding in input order.
    ///
    /// The position for binding `i` equals the `CPU` golden
    /// [`transfer_root_map`](prism_render_architecture::hair::follicle_bind::transfer_root_map)
    /// of the same inputs within an fma tolerance (the only departures are a
    /// possibly-fused multiply-add and a possibly-differently-rounded `sqrt`). An
    /// empty batch yields an empty vector without a dispatch — storage buffers
    /// cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        bindings: &[FollicleBinding],
        deformed: TriangleFrame,
    ) -> Vec<[f32; 3]> {
        let binding_count = bindings.len();
        if binding_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        let p = deformed.positions;
        let n = deformed.normals;
        let uniforms = Params {
            pos0: [p[0][0], p[0][1], p[0][2], 0.0],
            pos1: [p[1][0], p[1][1], p[1][2], 0.0],
            pos2: [p[2][0], p[2][1], p[2][2], 0.0],
            nrm0: [n[0][0], n[0][1], n[0][2], 0.0],
            nrm1: [n[1][0], n[1][1], n[1][2], 0.0],
            nrm2: [n[2][0], n[2][1], n[2][2], 0.0],
            binding_count: binding_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let gpu_bindings: Vec<GpuBinding> = bindings
            .iter()
            .map(|b| GpuBinding {
                bary0: b.bary.u,
                bary1: b.bary.v,
                bary2: b.bary.w,
                normal_offset: b.normal_offset,
            })
            .collect();

        // Output is one vec4 (16 bytes) per binding; xyz is the position.
        let out_bytes = (binding_count as u64) * ((size_of::<f32>() * 4) as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_follicle_bind_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let bindings_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_follicle_bind_bindings"),
            contents: bytemuck::cast_slice(&gpu_bindings),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_follicle_bind_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_follicle_bind_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_follicle_bind_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: bindings_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_follicle_bind_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_follicle_bind_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (binding_count as u32).div_ceil(64);
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
        let raw = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        raw.chunks_exact(4).map(|c| [c[0], c[1], c[2]]).collect()
    }
}

/// The `CPU` golden follicle root transfer for a batch, re-exported so the
/// parity test can assert the device twin against the identical reference it
/// mirrors.
#[must_use]
pub fn reference_transfer_root_map(
    bindings: &[FollicleBinding],
    deformed: TriangleFrame,
) -> Vec<[f32; 3]> {
    transfer_root_map(bindings, deformed)
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
