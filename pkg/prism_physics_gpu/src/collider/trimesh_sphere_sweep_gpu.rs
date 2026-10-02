//! Real-device `wgpu` compute twin of the triangle-mesh swept-sphere scene
//! query: the device form of
//! [`cpu_trimesh_sphere_sweep`](super::cpu_trimesh_sphere_sweep).
//!
//! [`GpuTrimeshSphereSweep`] compiles
//! `shaders/collider_trimesh_sphere_sweep.wgsl` once and exposes
//! [`GpuTrimeshSphereSweep::sweep`], which casts one moving sphere of fixed
//! radius at a static [`Trimesh`] and returns the earliest triangle contact. One
//! device lane handles one triangle: it runs the shared face-plus-three-edge
//! rounded-triangle sweep (with the initial-overlap short circuit) and writes
//! that triangle's time of impact, contact point, push-out normal, and a hit
//! flag. The host then reduces the per-triangle rows to the earliest contact
//! through the shared [`closer_hit`](super::trimesh_sphere_sweep::closer_hit)
//! rule, so the result is the identical [`TrimeshSweepHit`] the `CPU` brute and
//! `BVH` queries return.
//!
//! # Honest brute reduction
//!
//! An earliest-contact query has no device-side priority queue, so this kernel
//! evaluates every triangle (one lane each) and the host keeps the least-time
//! contact. That is the `GPU` analogue of the `CPU` brute golden, not the `BVH`
//! branch and bound; the `BVH` prune lives on the `CPU` query and the device
//! result is pinned to it by the parity suite.
//!
//! # Buffer layout
//!
//! The mesh is flattened to three vertices per triangle so no shared index pool
//! is needed: `vertices` holds `tri0.a, tri0.b, tri0.c, tri1.a, ...` as
//! `vec4<f32>` rows (`xyz` used, `w` padding) and `indices` row `i` is
//! `(3i, 3i + 1, 3i + 2, 0)` as a `vec4<u32>`. [`Params`] carries the sweep and
//! the triangle count. Results read back as two `vec4<f32>` rows per triangle:
//! `results[2i] = (toi, point.x, point.y, point.z)` and
//! `results[2i + 1] = (hit_flag, normal.x, normal.y, normal.z)` with `hit_flag`
//! `1.0` on a contact.
//!
//! Provenance: moving-sphere / rounded-triangle reduction and the
//! ray-versus-capsule and intersecting-moving-sphere tests per Ericson,
//! "Real-Time Collision Detection" (2005). No Unreal Engine source or derived
//! code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::buffer;
use crate::bvh::layout::{buffer_entry, entry};
use crate::context::GpuContext;

use super::trimesh_sphere_sweep::{closer_hit, SphereSweep, TrimeshSweepHit};
use super::Trimesh;

/// Lanes per workgroup; must match `@workgroup_size` in the kernel.
const WORKGROUP: usize = 64;

/// `hit_flag` value the kernel writes on a contact.
const HIT_FLAG: f32 = 1.0;

/// Uniform parameters shared with `Params` in
/// `shaders/collider_trimesh_sphere_sweep.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// `xyz`: sweep start centre; `w`: max travel distance.
    origin_maxdist: [f32; 4],
    /// `xyz`: sweep direction; `w`: sphere radius.
    dir_radius: [f32; 4],
    /// `x`: triangle count; `y`, `z`, `w`: padding.
    counts: [u32; 4],
}

/// One `vec4<f32>` output row; the kernel writes two rows per triangle.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SweepOut {
    /// Either `(toi, point.xyz)` (even row) or `(hit_flag, normal.xyz)` (odd row).
    data: [f32; 4],
}

/// A reusable triangle-mesh swept-sphere query that runs the per-triangle sweep
/// on the `GPU`.
///
/// Holds the compiled pipeline and bind-group layout so repeated sweeps reuse
/// them; build it once per [`GpuContext`] and call
/// [`sweep`](GpuTrimeshSphereSweep::sweep) per query.
pub struct GpuTrimeshSphereSweep {
    /// Retained shader module (kept alive for the pipeline).
    _module: ShaderModule,
    /// Bind-group layout for the four bindings.
    layout: BindGroupLayout,
    /// Compiled compute pipeline.
    pipeline: ComputePipeline,
}

impl GpuTrimeshSphereSweep {
    /// Builds the pipeline on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTrimeshSphereSweep {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_collider_trimesh_sphere_sweep"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/collider_trimesh_sphere_sweep.wgsl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_collider_trimesh_sphere_sweep_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_collider_trimesh_sphere_sweep_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_collider_trimesh_sphere_sweep_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("collider_trimesh_sphere_sweep"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTrimeshSphereSweep {
            _module: module,
            layout,
            pipeline,
        }
    }

    /// Sweeps `sweep` at `mesh` on the `GPU`, returning the earliest triangle
    /// contact.
    ///
    /// Returns [`None`] when the sphere touches no triangle within range or the
    /// mesh is empty. The result matches
    /// [`cpu_trimesh_sphere_sweep`](super::cpu_trimesh_sphere_sweep) including
    /// the lowest-index rule on a time-of-impact tie.
    #[must_use]
    pub fn sweep(
        &self,
        ctx: &GpuContext,
        mesh: &Trimesh,
        sweep: &SphereSweep,
    ) -> Option<TrimeshSweepHit> {
        let n = mesh.triangle_count();
        if n == 0 {
            return None;
        }
        let device = ctx.device();

        // Flatten to three vertices per triangle; indices address that run.
        let mut vertices: Vec<[f32; 4]> = Vec::with_capacity(n * 3);
        let mut indices: Vec<[u32; 4]> = Vec::with_capacity(n);
        for i in 0..n {
            let tri = mesh.triangle(i);
            let base = u32::try_from(vertices.len()).unwrap_or(u32::MAX);
            vertices.push([tri.a.x, tri.a.y, tri.a.z, 0.0]);
            vertices.push([tri.b.x, tri.b.y, tri.b.z, 0.0]);
            vertices.push([tri.c.x, tri.c.y, tri.c.z, 0.0]);
            indices.push([base, base + 1, base + 2, 0]);
        }

        let params = Params {
            origin_maxdist: [
                sweep.origin.x,
                sweep.origin.y,
                sweep.origin.z,
                sweep.max_distance,
            ],
            dir_radius: [
                sweep.direction.x,
                sweep.direction.y,
                sweep.direction.z,
                sweep.radius,
            ],
            counts: [u32::try_from(n).unwrap_or(u32::MAX), 0, 0, 0],
        };
        let params_buf =
            buffer::uniform(device, "prism_collider_trimesh_sphere_sweep_params", &params);
        let vertices_buf = buffer::storage_read(
            device,
            "prism_collider_trimesh_sphere_sweep_vertices",
            &vertices,
        );
        let indices_buf = buffer::storage_read(
            device,
            "prism_collider_trimesh_sphere_sweep_indices",
            &indices,
        );

        let out_bytes = (size_of::<SweepOut>() * 2 * n) as u64;
        let out_buf = buffer::storage_rw_zeroed(
            device,
            "prism_collider_trimesh_sphere_sweep_out",
            out_bytes,
        );

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_collider_trimesh_sphere_sweep_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &vertices_buf),
                entry(2, &indices_buf),
                entry(3, &out_buf),
            ],
        });

        let out_stage = buffer::staging(
            device,
            "prism_collider_trimesh_sphere_sweep_stage",
            out_bytes,
        );
        let groups = u32::try_from(n.div_ceil(WORKGROUP)).unwrap_or(u32::MAX);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_collider_trimesh_sphere_sweep_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_collider_trimesh_sphere_sweep_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        buffer::copy(&mut encoder, &out_buf, &out_stage, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        let raw = buffer::read_back::<SweepOut>(ctx, &out_stage);

        // Reduce to the earliest contact. The shader already produced the final
        // point and normal, so the host only picks the least-time triangle,
        // keeping the lowest index on a tie through the shared closer_hit rule.
        let mut best: Option<TrimeshSweepHit> = None;
        for (i, pair) in raw.chunks_exact(2).enumerate().take(n) {
            let [row0, row1] = pair else { continue };
            if row1.data[0] != HIT_FLAG {
                continue;
            }
            let hit = TrimeshSweepHit {
                triangle: u32::try_from(i).unwrap_or(u32::MAX),
                toi: row0.data[0],
                point: Vec3::new(row0.data[1], row0.data[2], row0.data[3]),
                normal: Vec3::new(row1.data[1], row1.data[2], row1.data[3]),
            };
            if closer_hit(&hit, &best) {
                best = Some(hit);
            }
        }
        best
    }
}
