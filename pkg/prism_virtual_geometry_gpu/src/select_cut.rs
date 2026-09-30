//! `wgpu` compute twin of the virtual-geometry cut selector
//! ([`ClusterHierarchy::select_cut`](prism_render_architecture::virtual_geometry::ClusterHierarchy::select_cut)).
//!
//! A GPU-driven virtual-geometry pipeline opens each frame by choosing a *cut*
//! through the cluster LOD hierarchy: one cluster per visible surface region,
//! each the coarsest simplification whose on-screen error still fits the pixel
//! budget. The CPU golden
//! [`select_cut`](prism_render_architecture::virtual_geometry::ClusterHierarchy::select_cut)
//! owns that decision as a top-down stack walk; [`GpuCutSelector`] is the
//! on-device twin that returns the same set of drawn clusters.
//!
//! # Parallel formulation
//!
//! The walk's decision at each node is a *pure local* predicate — is the node
//! frustum-visible, and is it a leaf or within the pixel budget — so the twin
//! splits the walk into two data-parallel stages sharing one bind group:
//!
//! * `classify` runs once (one thread per node) and computes `descend` and
//!   `draw_local` from the frustum and squared-form budget tests, seeding the
//!   `reached` bit on roots. This is the only float-bearing stage.
//! * `relax` runs to a host-driven fixed point (one thread per node) and sets a
//!   node's `reached` bit once its parent is both reached and descended-into.
//!   Reachability flips only `0 -> 1`, so the fixed point — and the drawn set —
//!   is independent of intra-dispatch ordering.
//!
//! A node is drawn exactly when it is `reached` and `draw_local`, which is
//! precisely the set the reference stack walk emits (the cut is a set, one
//! cluster per region, so the emission order is irrelevant).
//!
//! # Correctness model
//!
//! Every float comparison lives in `classify` and copies the golden term order
//! verbatim (frustum `nx*px + ny*py + nz*pz + d` / `|nx|*hx + |ny|*hy + |nz|*hz`,
//! budget `projected*projected <= budget*budget*distance_sq`). Away from the
//! razor-thin budget/plane boundary the discrete predicates — and therefore the
//! drawn set — are identical to the reference regardless of fused-multiply-add
//! contraction, so the parity test asserts set equality without a tolerance.
//!
//! # Portability & safety
//!
//! The kernels use only the portable core-`WGSL` subset (no optional feature),
//! so this twin runs unmodified on Metal, Vulkan and DX12. The crate forbids
//! `unsafe` and relies solely on the safe `wgpu` and `bytemuck` surfaces.
//!
//! Provenance: standard Nanite-style hierarchical screen-space-error cut
//! selection plus `wgpu` compute dispatch; no Unreal Engine source or derived
//! code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::virtual_geometry::{
    ClusterHierarchy, CutCluster, Frustum, LodProjection,
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

/// Sentinel parent index for nodes that have no parent (the shader's `SENTINEL`).
const SENTINEL: u32 = u32::MAX;

/// `state` bit for "visible, interior and too coarse" (mirror of `BIT_DESCEND`).
const BIT_DRAW_LOCAL: u32 = 2;
/// `state` bit for "reachable through descended ancestors" (`BIT_REACHED`).
const BIT_REACHED: u32 = 4;

/// Uniform parameters for one cut-select dispatch. Layout matches `Params` in
/// `shaders/select_cut.wesl`: six planes, then the view origin, projection and
/// counts.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    planes: [[f32; 4]; 6],
    view_origin: [f32; 3],
    focal_length_pixels: f32,
    budget: f32,
    node_count: u32,
    pad0: u32,
    pad1: u32,
}

/// One hierarchy node. `48`-byte stride, matching `Node` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuNode {
    center: [f32; 3],
    self_error: f32,
    half_extents: [f32; 3],
    is_leaf: u32,
    parent: u32,
    is_root: u32,
    pad0: u32,
    pad1: u32,
}

/// A compiled, reusable cut-selection pipeline (both `classify` and `relax`
/// entry points over one shared bind-group layout).
pub struct GpuCutSelector {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    classify: ComputePipeline,
    relax: ComputePipeline,
}

impl GpuCutSelector {
    /// Compiles the cut-select kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so this never
    /// returns [`None`]: it compiles on any backend `ctx` acquired.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCutSelector {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_select_cut"),
            source: ShaderSource::Wgsl(include_str!("../shaders/select_cut.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_select_cut_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_select_cut_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let classify = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_select_cut_classify_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("classify"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let relax = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_select_cut_relax_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("relax"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCutSelector {
            module,
            layout,
            classify,
            relax,
        }
    }

    /// Selects the screen-space-error cut for one view over `hierarchy`,
    /// returning the drawn clusters.
    ///
    /// The returned set equals
    /// [`hierarchy.select_cut(view_origin, frustum, projection, target_error_pixels)`](prism_render_architecture::virtual_geometry::ClusterHierarchy::select_cut)
    /// as a set (the reference emits in stack-walk order; this twin emits in
    /// node-index order — sort both to compare). A malformed hierarchy (see
    /// [`ClusterHierarchy::is_well_formed`]) or an empty hierarchy yields an
    /// empty cut, matching the reference and avoiding a zero-sized storage
    /// buffer.
    #[must_use]
    pub fn select_cut(
        &self,
        ctx: &GpuContext,
        view_origin: [f32; 3],
        frustum: &Frustum,
        projection: LodProjection,
        target_error_pixels: f32,
        hierarchy: &ClusterHierarchy,
    ) -> Vec<CutCluster> {
        let nodes = hierarchy.nodes();
        if nodes.is_empty() || !hierarchy.is_well_formed() {
            return Vec::new();
        }
        let device = ctx.device();

        // Structural inversion: derive each node's parent from the children
        // ranges (a well-formed hierarchy is a forest, so the parent is unique),
        // and flag the roots the walk starts from.
        let mut parents = alloc_sentinels(nodes.len());
        for (index, node) in nodes.iter().enumerate() {
            for offset in 0..node.child_count {
                let child = (node.first_child + offset) as usize;
                parents[child] = index as u32;
            }
        }
        let mut is_root = vec![0u32; nodes.len()];
        for &r in hierarchy.roots() {
            is_root[r as usize] = 1;
        }

        let planes = frustum.planes.map(|p| [p.normal[0], p.normal[1], p.normal[2], p.distance]);
        let params = Params {
            planes,
            view_origin,
            focal_length_pixels: projection.focal_length_pixels,
            budget: target_error_pixels.max(0.0),
            node_count: nodes.len() as u32,
            pad0: 0,
            pad1: 0,
        };

        let gpu_nodes: Vec<GpuNode> = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| GpuNode {
                center: n.bounds.center,
                self_error: n.self_error,
                half_extents: n.bounds.half_extents,
                is_leaf: u32::from(n.is_leaf()),
                parent: parents[i],
                is_root: is_root[i],
                pad0: 0,
                pad1: 0,
            })
            .collect();

        let state_bytes = (nodes.len() as u64) * (size_of::<u32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_select_cut_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let nodes_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_select_cut_nodes"),
            contents: bytemuck::cast_slice(&gpu_nodes),
            usage: BufferUsages::STORAGE,
        });
        let state_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_select_cut_state"),
            size: state_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let state_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_select_cut_state_stage"),
            size: state_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let changed_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_select_cut_changed"),
            size: size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let changed_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_select_cut_changed_stage"),
            size: size_of::<u32>() as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_select_cut_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: nodes_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: state_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: changed_buf.as_entire_binding(),
                },
            ],
        });

        let groups = (nodes.len() as u32).div_ceil(64);

        // Stage 1: classify every node once.
        {
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("prism_select_cut_classify_encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("prism_select_cut_classify_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.classify);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            ctx.queue().submit([encoder.finish()]);
        }

        // Stage 2: relax reachability to a fixed point. A forest of `n` nodes
        // converges in at most `n` passes (one new node reached per pass in the
        // worst case), so `node_count` is a safe iteration cap.
        for _ in 0..nodes.len() {
            ctx.queue()
                .write_buffer(&changed_buf, 0, bytemuck::bytes_of(&0u32));
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("prism_select_cut_relax_encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("prism_select_cut_relax_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.relax);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            encoder.copy_buffer_to_buffer(
                &changed_buf,
                0,
                &changed_stage,
                0,
                size_of::<u32>() as u64,
            );
            ctx.queue().submit([encoder.finish()]);

            changed_stage.slice(..).map_async(MapMode::Read, |_| {});
            ctx.wait();
            let view = changed_stage
                .slice(..)
                .get_mapped_range()
                .expect("mapped changed readback range should be available after poll");
            let changed = bytemuck::cast_slice::<u8, u32>(&view)[0];
            drop(view);
            changed_stage.unmap();
            if changed == 0 {
                break;
            }
        }

        // Stage 3: read the final state and assemble the cut in node-index order.
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_select_cut_readback_encoder"),
        });
        encoder.copy_buffer_to_buffer(&state_buf, 0, &state_stage, 0, state_bytes);
        ctx.queue().submit([encoder.finish()]);

        state_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = state_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped state readback range should be available after poll");
        let state = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        state_stage.unmap();

        let mut cut = Vec::new();
        for (index, &flags) in state.iter().enumerate() {
            if (flags & BIT_REACHED) != 0 && (flags & BIT_DRAW_LOCAL) != 0 {
                cut.push(CutCluster {
                    node: index as u32,
                    page: nodes[index].page,
                });
            }
        }
        cut
    }
}

/// Builds a `len`-long parent array pre-filled with [`SENTINEL`].
fn alloc_sentinels(len: usize) -> Vec<u32> {
    vec![SENTINEL; len]
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
