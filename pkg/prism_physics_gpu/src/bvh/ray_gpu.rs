//! Real-device `wgpu` compute implementation of the `BVH` ray query.
//!
//! [`GpuBvhRaycast`] compiles the ray-traversal kernels once and runs them over a
//! device-resident [`GpuResidentLbvh`], binding the built tree's buffers directly
//! with no host round-trip between build and traversal.
//! [`GpuBvhRaycast::query_closest`] dispatches one invocation per ray, walks the
//! hierarchy pruning by the nearest hit found so far, and reads back the closest
//! primitive each ray enters. The result is the same nearest hit the
//! [`cpu_bvh_raycast_closest`](super::ray::cpu_bvh_raycast_closest) twin computes by
//! brute force.
//!
//! # Resident-tree contract
//!
//! A resident tree with fewer than two leaves owns no device buffers (see
//! [`GpuResidentLbvh`]), so it has no traversable hierarchy: every ray reports a
//! miss. This mirrors the resident overlap query and is **not** a stub. An empty
//! scene genuinely has nothing to hit, and a single-leaf scene is a degenerate
//! tree a production host special-cases before it reaches a resident query; the
//! full `n >= 0` behaviour is exercised by the `CPU` twin, and the resident kernel
//! only mirrors the `n >= 2` resident contract.
//!
//! # Provenance
//!
//! Williams et al. (2005) slab intersection over the stackless parent-pointer
//! traversal of Hapala et al. (2011) on the linear `BVH` of Karras (2012);
//! standard `wgpu` compute dispatch. No Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::buffer;
use crate::context::GpuContext;

use super::layout::{buffer_entry, entry};
use super::ray::{Ray, RayHit};
use super::resident::GpuResidentLbvh;

/// Lanes per workgroup; must match `@workgroup_size` in `bvh_raycast.wgsl`.
const WORKGROUP: u32 = 64;

/// Sentinel primitive index a missed ray carries in the closest-hit output.
const NO_PRIM: u32 = u32::MAX;

/// Uniform parameters for the ray kernels. Layout matches `Params` in
/// `shaders/bvh_raycast.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of internal nodes.
    num_internal: u32,
    /// Number of leaves.
    num_leaves: u32,
    /// Encoded id of the root node.
    root: u32,
    /// Number of rays queued in the ray buffer.
    num_rays: u32,
}

/// A compiled, reusable `GPU` `BVH` ray query pipeline.
pub struct GpuBvhRaycast {
    #[expect(
        dead_code,
        reason = "kept alive so the closest-hit pipeline it produced stays valid"
    )]
    closest_module: ShaderModule,
    layout: BindGroupLayout,
    /// Closest-hit kernel binding a device-resident tree's buffers directly.
    closest: ComputePipeline,
}

impl GpuBvhRaycast {
    /// Compiles the ray kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBvhRaycast {
        let device = ctx.device();
        let closest_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_bvh_raycast"),
            source: ShaderSource::Wgsl(include_str!("../shaders/bvh_raycast.wgsl").into()),
        });
        // The eight resident-tree buffers plus params, rays, and hits. The
        // closest and any-hit kernels share this layout: every binding type is
        // identical, and the hit element type differs only inside the shader.
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_bvh_raycast_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: true }),
                buffer_entry(6, BufferBindingType::Storage { read_only: true }),
                buffer_entry(7, BufferBindingType::Storage { read_only: true }),
                buffer_entry(8, BufferBindingType::Storage { read_only: true }),
                buffer_entry(9, BufferBindingType::Storage { read_only: true }),
                buffer_entry(10, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_bvh_raycast_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let closest = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_bvh_raycast_closest_pipeline"),
            layout: Some(&pipeline_layout),
            module: &closest_module,
            entry_point: Some("raycast_closest"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBvhRaycast {
            closest_module,
            layout,
            closest,
        }
    }

    /// Finds the nearest primitive each ray in `rays` hits in the resident `lbvh`,
    /// the device counterpart of
    /// [`cpu_bvh_raycast_closest`](super::ray::cpu_bvh_raycast_closest).
    ///
    /// Returns one entry per ray, in input order: [`Some`] carrying the hit
    /// primitive and entry distance, or [`None`] for a ray that misses every
    /// primitive. A resident tree with fewer than two leaves holds no hierarchy,
    /// so every ray misses (see the [module docs](self)).
    #[must_use]
    pub fn query_closest(
        &self,
        ctx: &GpuContext,
        lbvh: &GpuResidentLbvh,
        rays: &[Ray],
    ) -> Vec<Option<RayHit>> {
        // A resident tree with fewer than two leaves owns no buffers and has no
        // root to traverse, so every ray misses.
        let Some(inner) = lbvh.buffers() else {
            return vec![None; rays.len()];
        };
        if rays.is_empty() {
            return Vec::new();
        }

        let device = ctx.device();
        let num_rays = u32::try_from(rays.len()).unwrap_or(u32::MAX);

        let params = Params {
            num_internal: u32::try_from(inner.num_internal).unwrap_or(u32::MAX),
            num_leaves: u32::try_from(lbvh.num_leaves()).unwrap_or(u32::MAX),
            root: inner.root,
            num_rays,
        };

        let packed = pack_rays(rays);
        let params_buf = buffer::uniform(device, "prism_bvh_raycast_params", &params);
        let rays_buf = buffer::storage_read(device, "prism_bvh_raycast_rays", &packed);
        let hits_bytes = u64::from(num_rays) * 8;
        let hits_buf = buffer::storage_rw_zeroed(device, "prism_bvh_raycast_hits", hits_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_bvh_raycast_closest_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &inner.left),
                entry(2, &inner.right),
                entry(3, &inner.parent),
                entry(4, &inner.node_min),
                entry(5, &inner.node_max),
                entry(6, &inner.aabb_min),
                entry(7, &inner.aabb_max),
                entry(8, inner.sorted.values()),
                entry(9, &rays_buf),
                entry(10, &hits_buf),
            ],
        });

        let hits_stage = buffer::staging(device, "prism_bvh_raycast_hits_stage", hits_bytes);
        let groups = num_rays.div_ceil(WORKGROUP);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_bvh_raycast_encoder"),
        });
        dispatch(&mut encoder, &self.closest, &bind, groups);
        buffer::copy(&mut encoder, &hits_buf, &hits_stage, hits_bytes);
        ctx.queue().submit([encoder.finish()]);

        let raw = buffer::read_back::<[u32; 2]>(ctx, &hits_stage);
        raw.into_iter()
            .take(rays.len())
            .map(|[t_bits, prim]| {
                if prim == NO_PRIM {
                    None
                } else {
                    Some(RayHit {
                        prim,
                        t: f32::from_bits(t_bits),
                    })
                }
            })
            .collect()
    }
}

/// Packs rays into `vec4` lanes for upload: two lanes per ray, the first
/// `(origin.xyz, t_max)` and the second `(dir.xyz, 0)`.
#[must_use]
fn pack_rays(rays: &[Ray]) -> Vec<[f32; 4]> {
    let mut packed = Vec::with_capacity(rays.len() * 2);
    for r in rays {
        packed.push([r.origin.x, r.origin.y, r.origin.z, r.t_max]);
        packed.push([r.dir.x, r.dir.y, r.dir.z, 0.0]);
    }
    packed
}

/// Records one dispatch of `pipeline` bound to `bind`.
fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &ComputePipeline,
    bind: &BindGroup,
    groups: u32,
) {
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("prism_bvh_raycast_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind, &[]);
    pass.dispatch_workgroups(groups, 1, 1);
}
