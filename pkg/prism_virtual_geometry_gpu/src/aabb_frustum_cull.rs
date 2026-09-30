//! `wgpu` compute twin of the virtual-geometry axis-aligned-bounding-box
//! frustum test
//! ([`Frustum::intersects_bounds`](prism_render_architecture::virtual_geometry::Frustum::intersects_bounds)).
//!
//! `AABB` culling is the box sibling of the bounding-sphere test: a cluster
//! that carries an axis-aligned box is rejected as soon as it falls fully
//! outside any one frustum plane, using the box's three projected half-extents
//! (`|nx|*hx + |ny|*hy + |nz|*hz`) as its support radius along that plane. The
//! CPU golden
//! [`Frustum::intersects_bounds`](prism_render_architecture::virtual_geometry::Frustum::intersects_bounds)
//! owns that decision; [`GpuAabbFrustumCull`] is the on-device twin that runs
//! one thread per box and returns the same inside/outside verdict, so the
//! projected-extent frustum primitive that gates GPU-driven candidate lists is
//! validated against the reference in isolation rather than only as a sub-step
//! buried inside the full cluster-cull decision.
//!
//! # Portability
//!
//! The kernel is a fixed six-plane loop of multiply-adds, absolute values and
//! one compare, in the portable core-`WGSL` subset, so it needs no optional
//! device feature and runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The emitted verdict is a discrete decision - inside (`1`) or outside (`0`) -
//! derived from sign comparisons, not a continuous value. The kernel mirrors
//! the reference's term order (`nx*cx + ny*cy + nz*cz + d` for the signed
//! distance, `|nx|*hx + |ny|*hy + |nz|*hz` for the projected extent), so away
//! from the razor-thin plane boundary the verdict is identical to the reference
//! regardless of fused-multiply-add contraction. The parity test picks boxes
//! that clear each boundary by a wide margin, so the integer verdict is stable
//! under any legal float reassociation and is asserted index-for-index rather
//! than with a tolerance.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `AABB` projected-radius frustum culling plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::gpu_scene::SceneBounds;
use prism_render_architecture::virtual_geometry::Frustum;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One axis-aligned bounding box to cull: a world-space centre and per-axis
/// half-extents.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AabbQuery {
    /// Box centre in world space.
    pub center: [f32; 3],
    /// Per-axis half-extents, in world units.
    pub half_extents: [f32; 3],
}

/// Uniform parameters for one cull dispatch. Layout matches `Params` in
/// `shaders/aabb_frustum_cull.wesl`: six `vec4` planes then the box count.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    planes: [[f32; 4]; 6],
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One box upload. `24`-byte stride matching `Aabb` in the shader (six scalar
/// `f32` fields, no vec3 alignment padding).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuAabb {
    center_x: f32,
    center_y: f32,
    center_z: f32,
    half_x: f32,
    half_y: f32,
    half_z: f32,
}

/// A compiled, reusable `AABB`-frustum-cull pipeline.
pub struct GpuAabbFrustumCull {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuAabbFrustumCull {
    /// Compiles the `AABB`-cull kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so it needs no
    /// optional device feature.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_aabb_frustum_cull"),
            source: ShaderSource::Wgsl(include_str!("../shaders/aabb_frustum_cull.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_aabb_frustum_cull_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_aabb_frustum_cull_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_aabb_frustum_cull_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cull"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuAabbFrustumCull {
            module,
            layout,
            pipeline,
        }
    }

    /// Culls each box in `boxes` against `frustum`, returning one verdict per
    /// box in input order: `1` when the box is at least partially inside the
    /// frustum, `0` when it is fully outside.
    ///
    /// Each returned value equals
    /// [`Frustum::intersects_bounds`](prism_render_architecture::virtual_geometry::Frustum::intersects_bounds)
    /// evaluated on a [`SceneBounds`] with the same centre and half-extents,
    /// cast to `u32`. An empty `boxes` slice yields an empty result - storage
    /// buffers cannot be zero-sized, so it is handled by an early return.
    #[must_use]
    pub fn cull(&self, ctx: &GpuContext, frustum: &Frustum, boxes: &[AabbQuery]) -> Vec<u32> {
        if boxes.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let mut planes = [[0.0f32; 4]; 6];
        for (dst, plane) in planes.iter_mut().zip(frustum.planes.iter()) {
            *dst = [
                plane.normal[0],
                plane.normal[1],
                plane.normal[2],
                plane.distance,
            ];
        }
        let params = Params {
            planes,
            count: u32::try_from(boxes.len())
                .expect("box count must fit in u32 for the GPU dispatch"),
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let gpu_boxes: Vec<GpuAabb> = boxes
            .iter()
            .map(|b| GpuAabb {
                center_x: b.center[0],
                center_y: b.center[1],
                center_z: b.center[2],
                half_x: b.half_extents[0],
                half_y: b.half_extents[1],
                half_z: b.half_extents[2],
            })
            .collect();

        let out_bytes = (boxes.len() as u64) * (size_of::<u32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_aabb_frustum_cull_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let boxes_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_aabb_frustum_cull_boxes"),
            contents: bytemuck::cast_slice(&gpu_boxes),
            usage: BufferUsages::STORAGE,
        });
        let verdicts_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_aabb_frustum_cull_verdicts"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let verdicts_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_aabb_frustum_cull_verdicts_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_aabb_frustum_cull_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: boxes_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: verdicts_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_aabb_frustum_cull_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_aabb_frustum_cull_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = params.count.div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&verdicts_buf, 0, &verdicts_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        verdicts_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = verdicts_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let gpu_verdicts = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        verdicts_stage.unmap();
        debug_assert_eq!(gpu_verdicts.len(), boxes.len());
        gpu_verdicts
    }
}

/// Convenience: build the [`SceneBounds`] the golden consumes from a centre and
/// half-extents (radius is irrelevant to the projected-extent test).
#[must_use]
pub fn bounds_of(query: &AabbQuery) -> SceneBounds {
    SceneBounds {
        center: query.center,
        radius: 0.0,
        half_extents: query.half_extents,
        _padding: 0.0,
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
