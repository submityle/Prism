//! `wgpu` compute twin of Prism's signed-distance-field body-collision push-out
//! ([`push_out_of_field`](prism_render_architecture::hair::sdf_collision::push_out_of_field)).
//!
//! Analytic sphere/capsule proxies (see [`crate::collision`]) keep hair off the
//! round parts of a body, but a jaw line, a collarbone or a prop against the
//! scalp needs a tighter fit than a sphere gives. Production strand solvers
//! (AMD `TressFX` 4) add a signed distance field collider: the body is sampled
//! as a field that is negative inside the surface and positive outside, and any
//! particle that lands inside is pushed back along the field gradient to the
//! zero isosurface. The `CPU` golden for that projection is
//! [`push_out_of_field`](prism_render_architecture::hair::sdf_collision::push_out_of_field);
//! this crate is the on-device twin that walks the same union-field projection,
//! one thread per query point, so a passing real-device parity test is direct
//! evidence the ported kernel resolves the same positions as the reference —
//! not merely that its shader compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuSdfCollider::eval`] takes a batch of query points, a shared union of
//! primitives (sphere, capsule, half-space, box) and an iteration cap, and
//! returns the pushed-out position of every point. Each pass steps a still
//! inside point along the central-difference gradient of the union field by the
//! current penetration depth; a point on a zero-gradient spot escapes along
//! `+Y` by the depth, matching the golden's deterministic, `NaN`-free fallback.
//!
//! # Portability
//!
//! The kernel uses only `length` (`sqrt`), `min`, `max`, `clamp`, `dot` and
//! multiply/add in the portable core-`WGSL` subset — no `exp`, `pow` or optional
//! device feature — so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The projection contains no transcendental call, so `CPU` and `GPU` evaluate
//! the same closed-form geometry. They are not bit-exact: a `GPU` may fuse a
//! multiply-add or reassociate the union reduction, perturbing the low mantissa
//! bits. The parity test therefore asserts a per-component tolerance rather than
//! exact equality.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard analytic SDF union (Quilez box/capsule distance) plus
//! gradient push-out, and `wgpu` compute dispatch; no Unreal Engine source or
//! derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use prism_render_architecture::hair::sdf_collision::SdfPrimitive;

use crate::context::GpuContext;

/// Uniform parameters for one push-out dispatch. Layout matches `Params` in
/// `shaders/sdf_collision.wesl`: the point count, primitive count, iteration
/// cap, then one pad word for the `16`-byte uniform alignment.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    point_count: u32,
    prim_count: u32,
    iterations: u32,
    pad0: u32,
}

/// Number of `f32` per primitive in the flat `prims` storage buffer, matching
/// the stride the shader indexes: `v0` (`3`), `s0` (`1`), `v1` (`3`).
const PRIM_STRIDE: usize = 7;

/// A compiled, reusable SDF push-out pipeline.
pub struct GpuSdfCollider {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfCollider {
    /// Compiles the SDF push-out kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfCollider {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_sdf_collision"),
            source: ShaderSource::Wgsl(include_str!("../shaders/sdf_collision.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_sdf_collision_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_sdf_collision_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_sdf_collision_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfCollider {
            module,
            layout,
            pipeline,
        }
    }

    /// Pushes every query point out of the SDF union, returning the resolved
    /// positions in input order (same length as `points`).
    ///
    /// The result for point `p` equals
    /// [`push_out_of_field`](prism_render_architecture::hair::sdf_collision::push_out_of_field)
    /// applied to `p`, to within the fused-multiply-add tolerance documented on
    /// this module. An empty point batch, an empty primitive union, or zero
    /// iterations returns the input positions unchanged without a dispatch —
    /// storage buffers cannot be zero-sized, and every one of those cases is a
    /// no-op in the golden.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        points: &[[f32; 3]],
        primitives: &[SdfPrimitive],
        iterations: u32,
    ) -> Vec<[f32; 3]> {
        // The golden is a no-op when there is nothing to project, no field to
        // project against, or no passes to run; reproduce that without touching
        // the GPU so zero-sized storage buffers never arise.
        if points.is_empty() || primitives.is_empty() || iterations == 0 {
            return points.to_vec();
        }

        let device = ctx.device();
        let params = Params {
            point_count: points.len() as u32,
            prim_count: primitives.len() as u32,
            iterations,
            pad0: 0,
        };

        // Flatten primitives to (kind, 7-f32) pairs matching the shader layout.
        let mut kinds: Vec<u32> = Vec::with_capacity(primitives.len());
        let mut prims: Vec<f32> = Vec::with_capacity(primitives.len() * PRIM_STRIDE);
        for primitive in primitives {
            let (kind, v0, s0, v1) = encode_primitive(*primitive);
            kinds.push(kind);
            prims.extend_from_slice(&v0);
            prims.push(s0);
            prims.extend_from_slice(&v1);
        }

        let mut points_flat: Vec<f32> = Vec::with_capacity(points.len() * 3);
        for p in points {
            points_flat.extend_from_slice(p);
        }
        let out_bytes = (points_flat.len() as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_sdf_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let kinds_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_sdf_kinds"),
            contents: bytemuck::cast_slice(&kinds),
            usage: BufferUsages::STORAGE,
        });
        let prims_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_sdf_prims"),
            contents: bytemuck::cast_slice(&prims),
            usage: BufferUsages::STORAGE,
        });
        let points_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_sdf_points_in"),
            contents: bytemuck::cast_slice(&points_flat),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_sdf_points_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_sdf_points_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_sdf_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: kinds_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: prims_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: points_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_sdf_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_sdf_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (points.len() as u32).div_ceil(64);
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
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
    }
}

/// Encodes one primitive to the flat `(kind, v0, s0, v1)` layout the shader
/// reads. Fields unused by a variant are zero, exactly as the shader ignores
/// them.
fn encode_primitive(primitive: SdfPrimitive) -> (u32, [f32; 3], f32, [f32; 3]) {
    match primitive {
        SdfPrimitive::Sphere { center, radius } => {
            (0, [center.x, center.y, center.z], radius, [0.0, 0.0, 0.0])
        }
        SdfPrimitive::Capsule { a, b, radius } => (1, [a.x, a.y, a.z], radius, [b.x, b.y, b.z]),
        SdfPrimitive::HalfSpace { normal, offset } => {
            (2, [normal.x, normal.y, normal.z], offset, [0.0, 0.0, 0.0])
        }
        SdfPrimitive::Box {
            center,
            half_extents,
        } => (
            3,
            [center.x, center.y, center.z],
            0.0,
            [half_extents.x, half_extents.y, half_extents.z],
        ),
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
