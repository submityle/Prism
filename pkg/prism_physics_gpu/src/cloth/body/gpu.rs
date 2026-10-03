//! Real-device `wgpu` compute implementation of body-proxy cloth collision and
//! per-particle backstops.
//!
//! [`GpuClothBodyCollision`] compiles `shaders/cloth_body_collision.wgsl` once
//! and exposes two solves:
//!
//! - [`GpuClothBodyCollision::solve`] runs the body-collider pass (sphere /
//!   capsule / half-space projection plus position-level Coulomb friction), the
//!   real-device twin of [`cpu_cloth_body_collision`](super::cpu::cpu_cloth_body_collision).
//! - [`GpuClothBodyCollision::solve_backstops`] runs the per-particle backstop
//!   pass, the twin of [`cpu_cloth_backstops`](super::cpu::cpu_cloth_backstops).
//!
//! Both passes are *per-particle independent* — one thread owns one particle
//! and (for the body pass) walks every collider in slice order, which is exactly
//! the sequential golden's per-particle write set, so no colouring or atomic
//! accumulation is needed and the result matches the golden within a tight
//! tolerance (`GPU` `inverseSqrt`/division rounding perturbs the low bits).
//!
//! # Provenance
//!
//! Analytic body-proxy projections and the one-sided backstop are standard
//! position-based collision techniques; the tangential-friction projection is
//! Macklin et al. (2014). No Unreal Engine source or derived code.

use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use prism_physics_core::{Backstop, BodyCollider};
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::buffer;
use crate::cloth::layout::{buffer_entry, entry};
use crate::context::GpuContext;

use super::collider::{
    pack_backstops, pack_body_scene, GpuBackstop, GpuBodyCollider, GpuConvexPlane,
};

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// Lanes per workgroup; must match `@workgroup_size(64)` in both kernels.
const WORKGROUP: usize = 64;

/// Uniform parameters shared with `Params` in
/// `shaders/cloth_body_collision.wgsl` (16 bytes / 4 words).
///
/// The field order mirrors the `WGSL` struct exactly; reordering silently
/// corrupts the solve.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of addressable particles (positions length).
    particle_count: u32,
    /// Number of body colliders in the slice.
    collider_count: u32,
    /// Number of per-particle backstop planes.
    backstop_count: u32,
    /// Combined Coulomb friction coefficient (already clamped to `0..=1`).
    mu: f32,
}

/// A compiled, reusable `GPU` cloth body-collision pipeline pair.
pub struct GpuClothBodyCollision {
    /// Kept alive so the pipelines it produced stay valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring every buffer both passes share.
    layout: BindGroupLayout,
    /// Body-collider projection + Coulomb friction, per particle.
    body_pass: ComputePipeline,
    /// Per-particle backstop plane clamp.
    backstop_pass: ComputePipeline,
}

impl GpuClothBodyCollision {
    /// Compiles the cloth body-collision kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothBodyCollision {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_cloth_body_collision"),
            source: ShaderSource::Wgsl(
                include_str!("../../shaders/cloth_body_collision.wgsl").into(),
            ),
        });
        let read = BufferBindingType::Storage { read_only: true };
        let write = BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_cloth_body_collision_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, write),
                buffer_entry(2, read),
                buffer_entry(3, read),
                buffer_entry(4, read),
                buffer_entry(5, read),
                buffer_entry(6, read),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_cloth_body_collision_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make = |entry_point: &str| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("prism_cloth_body_collision_pipeline"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let body_pass = make("body_pass");
        let backstop_pass = make("backstop_pass");
        GpuClothBodyCollision {
            module,
            layout,
            body_pass,
            backstop_pass,
        }
    }

    /// Projects every free particle out of every body collider, in slice order,
    /// rubbing each contact's tangential slide with Coulomb friction, and
    /// returns the applied positions.
    ///
    /// This is the real-device twin of
    /// [`cpu_cloth_body_collision`](super::cpu::cpu_cloth_body_collision): the
    /// single entry covers both the plain and the friction path (`mu <= 0` makes
    /// the friction projection a no-op, so `prev_positions` is then irrelevant).
    /// An empty `colliders` slice, an empty position array, or an
    /// `inverse_masses` slice whose length differs from `positions` returns
    /// `positions` unchanged; pinned particles (`inverse_mass <= 0`) never move,
    /// and a `prev_positions` slice shorter than `positions` degrades to no
    /// tangential slide (hence no friction) for the missing indices.
    #[must_use]
    pub fn solve(
        &self,
        ctx: &GpuContext,
        positions: &[Vec3],
        inverse_masses: &[Real],
        prev_positions: &[Vec3],
        colliders: &[BodyCollider],
        mu: Real,
    ) -> Vec<Vec3> {
        if colliders.is_empty() || positions.is_empty() || inverse_masses.len() != positions.len() {
            return positions.to_vec();
        }

        // Pad `prev` to the position length: the golden reads
        // `prev_positions.get(i).unwrap_or(current)`, and the current start-of-
        // particle position equals the original `positions[i]` here.
        let prev_packed: Vec<[f32; 4]> = positions
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let q = prev_positions.get(i).copied().unwrap_or(*p);
                [q.x, q.y, q.z, 0.0]
            })
            .collect();

        let scene = pack_body_scene(colliders);
        let colliders_packed = scene.records;
        // A zero-length storage buffer is invalid; feed a one-element dummy
        // when no convex hull contributed any face planes.
        let convex_dummy = [GpuConvexPlane::zeroed()];
        let convex_planes: &[GpuConvexPlane] = if scene.planes.is_empty() {
            &convex_dummy
        } else {
            &scene.planes
        };
        let mu_clamped = if mu.is_finite() {
            mu.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let params = Params {
            particle_count: u32::try_from(positions.len()).unwrap_or(u32::MAX),
            collider_count: u32::try_from(colliders_packed.len()).unwrap_or(u32::MAX),
            backstop_count: 0,
            mu: mu_clamped,
        };

        // `backstops` is unused by `body_pass` but the binding must be live; a
        // zero-length storage buffer is invalid, so feed a one-element dummy.
        let backstops_dummy = [GpuBackstop::zeroed()];
        self.dispatch(
            ctx,
            &self.body_pass,
            positions,
            inverse_masses,
            &prev_packed,
            &colliders_packed,
            &backstops_dummy,
            convex_planes,
            &params,
        )
    }

    /// Applies each per-particle backstop plane to its matching particle and
    /// returns the applied positions.
    ///
    /// This is the real-device twin of
    /// [`cpu_cloth_backstops`](super::cpu::cpu_cloth_backstops): the pass runs
    /// over the shorter of the position and backstop lengths; pinned particles
    /// are skipped, and an empty `backstops` slice, an empty position array, or
    /// a mismatched `inverse_masses` length returns `positions` unchanged.
    #[must_use]
    pub fn solve_backstops(
        &self,
        ctx: &GpuContext,
        positions: &[Vec3],
        inverse_masses: &[Real],
        backstops: &[Backstop],
    ) -> Vec<Vec3> {
        if backstops.is_empty() || positions.is_empty() || inverse_masses.len() != positions.len() {
            return positions.to_vec();
        }

        let backstops_packed = pack_backstops(backstops);
        let params = Params {
            particle_count: u32::try_from(positions.len()).unwrap_or(u32::MAX),
            collider_count: 0,
            backstop_count: u32::try_from(backstops_packed.len()).unwrap_or(u32::MAX),
            mu: 0.0,
        };

        // `prev` and `colliders` are unused by `backstop_pass`; bind one-element
        // dummies so neither storage buffer is zero-length.
        let prev_dummy = [[0.0f32; 4]];
        let colliders_dummy = [GpuBodyCollider::zeroed()];
        let convex_dummy = [GpuConvexPlane::zeroed()];
        self.dispatch(
            ctx,
            &self.backstop_pass,
            positions,
            inverse_masses,
            &prev_dummy,
            &colliders_dummy,
            &backstops_packed,
            &convex_dummy,
            &params,
        )
    }

    /// Uploads the shared buffers, dispatches `pipeline` over the particles, and
    /// reads the applied positions back.
    #[expect(
        clippy::too_many_arguments,
        reason = "the shared dispatch binds every buffer both passes need"
    )]
    fn dispatch(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        positions: &[Vec3],
        inverse_masses: &[Real],
        prev_packed: &[[f32; 4]],
        colliders_packed: &[GpuBodyCollider],
        backstops_packed: &[GpuBackstop],
        convex_planes: &[GpuConvexPlane],
        params: &Params,
    ) -> Vec<Vec3> {
        let device = ctx.device();

        let packed: Vec<[f32; 4]> = positions.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
        let pos_bytes = (packed.len() as u64) * 16;

        let params_buf = buffer::uniform(device, "prism_cloth_body_params", params);
        let positions_buf = buffer::storage_rw_init(device, "prism_cloth_body_pos", &packed);
        let inv_mass_buf = buffer::storage_read(device, "prism_cloth_body_invmass", inverse_masses);
        let prev_buf = buffer::storage_read(device, "prism_cloth_body_prev", prev_packed);
        let colliders_buf =
            buffer::storage_read(device, "prism_cloth_body_colliders", colliders_packed);
        let backstops_buf =
            buffer::storage_read(device, "prism_cloth_body_backstops", backstops_packed);
        let convex_buf = buffer::storage_read(device, "prism_cloth_body_convex", convex_planes);
        let pos_stage = buffer::staging(device, "prism_cloth_body_stage", pos_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_cloth_body_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &positions_buf),
                entry(2, &inv_mass_buf),
                entry(3, &prev_buf),
                entry(4, &colliders_buf),
                entry(5, &backstops_buf),
                entry(6, &convex_buf),
            ],
        });

        let groups = u32::try_from(positions.len().div_ceil(WORKGROUP)).unwrap_or(u32::MAX);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_cloth_body_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_cloth_body_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups.max(1), 1, 1);
        }
        buffer::copy(&mut encoder, &positions_buf, &pos_stage, pos_bytes);
        ctx.queue().submit([encoder.finish()]);

        let read = buffer::read_back::<[f32; 4]>(ctx, &pos_stage);
        read.iter().map(|q| Vec3::new(q[0], q[1], q[2])).collect()
    }
}
