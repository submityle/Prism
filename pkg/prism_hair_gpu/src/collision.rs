//! `wgpu` compute twin of the strand body-collider projection
//! ([`Collider::push_out`](prism_render_architecture::hair::collision::Collider::push_out)).
//!
//! `TressFX`-class strand solvers keep hair off the body by projecting each
//! free particle out of a small set of analytic collider proxies (spheres and
//! capsules fitted to the head, neck and shoulders) after every constraint
//! sweep (design 6.2). The `CPU` golden for that projection is
//! [`Collider::push_out`](prism_render_architecture::hair::collision::Collider::push_out);
//! this crate is the on-device twin that evaluates the same push-out, one
//! thread per query, so a passing real-device parity test is direct evidence
//! the ported kernel computes the same projected positions as the reference —
//! not merely that its shader compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuColliderProjector::eval`] projects a batch of points out of their
//! colliders. Each [`CollisionQuery`] pairs a world-space point with one
//! collider (a sphere or a capsule), built with [`CollisionQuery::sphere`],
//! [`CollisionQuery::capsule`], or [`query_for`] from an
//! architecture-side [`Collider`](prism_render_architecture::hair::collision::Collider).
//! The kernel mirrors the reference's two branches exactly:
//!
//! * a sphere pushes an interior point radially out to the surface;
//! * a capsule first finds the closest point on its segment axis, then applies
//!   the same sphere push-out around that point.
//!
//! Every guard is reproduced bit-for-bit: a non-positive radius is inert, an
//! already-exterior point is untouched, a point coincident with the center
//! escapes along `+Y`, and the segment projection parameter is clamped to
//! `[0, 1]`.
//!
//! # Portability
//!
//! The kernel uses only `sqrt`, `min`, `max`, `clamp`, `dot` and multiply/add
//! in the portable core-`WGSL` subset — no `exp`, `pow` or optional device
//! feature — so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The projection contains no transcendental call (the reference restricts
//! itself to `sqrt` for exactly this reason), so `CPU` and `GPU` evaluate the
//! same closed-form geometry. They are **not** bit-exact, however: a `GPU` may
//! fuse a multiply-add that the scalar `CPU` reference leaves separate,
//! perturbing the low mantissa bits by a few `ULP`. The parity test therefore
//! asserts a tolerance (`abs_diff < 1e-4` or `rel_diff < 1e-3`) per component
//! rather than exact equality — tight enough that a genuinely wrong port (a
//! swapped branch, a missing clamp, a sign error) fails, loose enough that
//! legal fma contraction passes.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard analytic sphere/capsule collider push-out plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::collision::Collider;
use prism_render_architecture::hair::dynamics::Vec3;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Collider-kind discriminant for a sphere, matching `KIND_SPHERE` in
/// `shaders/collision.wesl`.
const KIND_SPHERE: u32 = 0;
/// Collider-kind discriminant for a capsule, matching `KIND_CAPSULE` in
/// `shaders/collision.wesl`.
const KIND_CAPSULE: u32 = 1;

/// One collider-projection query: a point paired with the collider to push it
/// out of.
///
/// The fields encode exactly the arguments the `CPU` golden
/// [`Collider::push_out`](prism_render_architecture::hair::collision::Collider::push_out)
/// consumes. For a [`Collider::Sphere`](prism_render_architecture::hair::collision::Collider::Sphere)
/// the center is stored in `a` and `b` is unused; for a
/// [`Collider::Capsule`](prism_render_architecture::hair::collision::Collider::Capsule)
/// the endpoints are `a` and `b`.
///
/// The struct is `48`-byte `repr(C)` matching `Query` in
/// `shaders/collision.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct CollisionQuery {
    /// Point x.
    px: f32,
    /// Point y.
    py: f32,
    /// Point z.
    pz: f32,
    /// Collider kind: `0` sphere, `1` capsule.
    kind: u32,
    /// Sphere center x, or capsule endpoint `a` x.
    ax: f32,
    /// Sphere center y, or capsule endpoint `a` y.
    ay: f32,
    /// Sphere center z, or capsule endpoint `a` z.
    az: f32,
    /// Collider radius.
    radius: f32,
    /// Capsule endpoint `b` x (unused for a sphere).
    bx: f32,
    /// Capsule endpoint `b` y (unused for a sphere).
    by: f32,
    /// Capsule endpoint `b` z (unused for a sphere).
    bz: f32,
    /// Padding to a `16`-byte-aligned `48`-byte stride.
    pad: f32,
}

impl CollisionQuery {
    /// Builds a query that pushes `point` out of the sphere `(center, radius)`.
    #[must_use]
    pub fn sphere(center: Vec3, radius: f32, point: Vec3) -> CollisionQuery {
        CollisionQuery {
            px: point.x,
            py: point.y,
            pz: point.z,
            kind: KIND_SPHERE,
            ax: center.x,
            ay: center.y,
            az: center.z,
            radius,
            bx: 0.0,
            by: 0.0,
            bz: 0.0,
            pad: 0.0,
        }
    }

    /// Builds a query that pushes `point` out of the capsule with segment
    /// `a`..`b` and the given `radius`.
    #[must_use]
    pub fn capsule(a: Vec3, b: Vec3, radius: f32, point: Vec3) -> CollisionQuery {
        CollisionQuery {
            px: point.x,
            py: point.y,
            pz: point.z,
            kind: KIND_CAPSULE,
            ax: a.x,
            ay: a.y,
            az: a.z,
            radius,
            bx: b.x,
            by: b.y,
            bz: b.z,
            pad: 0.0,
        }
    }
}

/// Builds the [`CollisionQuery`] that projects `point` out of `collider`.
///
/// This is the bridge from the architecture-side
/// [`Collider`](prism_render_architecture::hair::collision::Collider) enum to
/// the flat upload struct the kernel consumes, so a caller can dispatch the
/// exact colliders its `CPU` solver uses.
#[must_use]
pub fn query_for(collider: Collider, point: Vec3) -> CollisionQuery {
    match collider {
        Collider::Sphere { center, radius } => CollisionQuery::sphere(center, radius, point),
        Collider::Capsule { a, b, radius } => CollisionQuery::capsule(a, b, radius, point),
    }
}

/// Uniform parameters for one projection dispatch. Layout matches `Params` in
/// `shaders/collision.wesl`: the query count then three pad words for the
/// `16`-byte uniform alignment.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable collider-projection pipeline.
pub struct GpuColliderProjector {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuColliderProjector {
    /// Compiles the collider-projection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuColliderProjector {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_collision"),
            source: ShaderSource::Wgsl(include_str!("../shaders/collision.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_collision_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_collision_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_collision_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuColliderProjector {
            module,
            layout,
            pipeline,
        }
    }

    /// Projects every point out of its collider, returning one `[x, y, z]`
    /// pushed position per query in input order.
    ///
    /// The returned position for query `q` equals
    /// [`Collider::push_out`](prism_render_architecture::hair::collision::Collider::push_out)
    /// applied to the point `q` carries, to within the fused-multiply-add
    /// tolerance documented on this module. An empty `queries` slice yields an
    /// empty result — storage buffers cannot be zero-sized, so it is handled by
    /// an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[CollisionQuery]) -> Vec<[f32; 3]> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Three f32 (x, y, z) per query.
        let out_bytes = (queries.len() as u64) * 3 * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_collision_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_collision_queries"),
            contents: bytemuck::cast_slice(queries),
            usage: BufferUsages::STORAGE,
        });
        let values_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_collision_values"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let values_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_collision_values_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_collision_bind_group"),
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
                    resource: values_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_collision_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_collision_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (queries.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&values_buf, 0, &values_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        values_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = values_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        values_stage.unmap();
        debug_assert_eq!(flat.len(), queries.len() * 3);
        flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
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
