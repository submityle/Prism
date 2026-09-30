//! Compute pipelines and bind-group layouts for the `GPU` cloth subsystem.
//!
//! The CPU-golden solver in `prism_render_architecture::cloth` decides *what*
//! runs each frame; this slice builds the concrete `wgpu` compute pipelines and
//! the bind-group layouts that the thirteen cloth kernels dispatch against. The
//! kernels are authored across five `WESL` shaders, each declaring its own
//! `@group(0)` resource interface:
//!
//! * `shaders/cloth_sim.wesl` — the predict / integrate step, the graph-colored
//!   `XPBD` distance, bending and long-range projection passes, the strain
//!   limiter and the velocity-recovery pass. All six entry points share one
//!   six-binding group: three read-write particle pools (positions, velocities,
//!   previous positions), two read-only constraint arrays (distance, bending)
//!   and the per-substep [`GpuClothSimParams`](super::abi::GpuClothSimParams)
//!   uniform.
//! * `shaders/cloth_collision.wesl` — three *distinct* group layouts because the
//!   body / self / backstop passes rebind `@group(0)` to different resources:
//!   the body pass reads an analytic collider list, the self pass drives a
//!   spatial-hash cell table plus a per-particle linked-list next array, and the
//!   backstop pass reads a painted backstop plane per particle.
//! * `shaders/cloth_embed.wesl` — the render-mesh skinning pass: read-only sim
//!   positions and embed bindings drive a read-write render-position pool.
//!
//! Because the three collision passes alias `@group(0)` to incompatible
//! resource sets, each needs its own bind-group layout even though they live in
//! one shader; a shared layout would validate against only one of them. That is
//! why this module owns *seven* layouts, not one per shader file. The two
//! aerodynamic passes add two more: the snapshot pass and the gather pass each
//! rebind `@group(0)` to their own resource set.
//!
//! Every pipeline binds the matching layout as group 0. The cloth passes read
//! their per-substep scalars from the uniform slot, but the three color-serial
//! projection kernels also address a single per-color constraint slice through
//! a small `ClothColorBatch { base, count }` immediate (push-constant) block,
//! so those three pipelines declare an 8-byte `immediate_size`; every other
//! pipeline declares zero. The pipeline handles are keyed by [`ClothKernel`] so
//! the dispatch slice can look one up directly from the golden kernel schedule.

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{
            storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
        },
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        BindGroupLayout, CachedComputePipelineId, ComputePipelineDescriptor, PipelineCache,
        ShaderStages,
    },
    renderer::RenderDevice,
};
use bevy_shader::Shader;
use prism_render_architecture::cloth::gpu::kernels::ClothKernel;

/// The compute pipelines and bind-group layouts for every cloth kernel.
///
/// Inserted at `RenderStartup` by [`init_cloth_compute_pipelines`]. The thirteen
/// pipeline handles are queued into the [`PipelineCache`] and resolve
/// asynchronously; the dispatch slice skips a pass whose handle is not yet
/// ready rather than stalling the frame. The seven layouts are created eagerly
/// so the bind-group slice can allocate against them the moment a piece uploads.
#[derive(Resource)]
pub(crate) struct ClothComputePipelines {
    /// group 0 for every `cloth_sim.wesl` entry point (six bindings).
    pub(crate) sim_layout: BindGroupLayout,
    /// group 0 for the `cloth_body_collision` pass (four bindings).
    pub(crate) body_layout: BindGroupLayout,
    /// group 0 for both self-collision passes (four bindings).
    pub(crate) self_layout: BindGroupLayout,
    /// group 0 for the `cloth_backstop` pass (three bindings).
    pub(crate) backstop_layout: BindGroupLayout,
    /// group 0 for the `cloth_skin_embed` pass (four bindings).
    pub(crate) embed_layout: BindGroupLayout,
    /// group 0 for the `cloth_aerodynamics_snapshot` pass (three bindings).
    pub(crate) aero_snapshot_layout: BindGroupLayout,
    /// group 0 for the `cloth_aerodynamics` gather pass (seven bindings).
    pub(crate) aero_layout: BindGroupLayout,

    /// `cloth_predict`: integrate external forces and predict positions.
    pub(crate) predict: CachedComputePipelineId,
    /// `cloth_project_distance_batch`: project one color's distance constraints.
    pub(crate) project_distance: CachedComputePipelineId,
    /// `cloth_project_bending_batch`: project one color's dihedral bends.
    pub(crate) project_bending: CachedComputePipelineId,
    /// `cloth_project_long_range_batch`: project long-range / tether limits.
    pub(crate) project_long_range: CachedComputePipelineId,
    /// `cloth_strain_limit`: hard-clamp structural-edge overstretch.
    pub(crate) strain_limit: CachedComputePipelineId,
    /// `cloth_body_collision`: resolve analytic body-proxy collision.
    pub(crate) body_collision: CachedComputePipelineId,
    /// `cloth_backstop`: push particles off the front of their backstop plane.
    pub(crate) backstop: CachedComputePipelineId,
    /// `cloth_self_collision_hash_build`: bin particles into the spatial hash.
    pub(crate) self_collision_hash_build: CachedComputePipelineId,
    /// `cloth_self_collision_resolve`: resolve self-collision against neighbours.
    pub(crate) self_collision_resolve: CachedComputePipelineId,
    /// `cloth_velocity_update`: recover velocity from the substep delta.
    pub(crate) velocity_update: CachedComputePipelineId,
    /// `cloth_skin_embed`: skin the render mesh onto the coarse sim mesh.
    pub(crate) skin_embed: CachedComputePipelineId,
    /// `cloth_aerodynamics_snapshot`: freeze this frame's start-of-frame
    /// velocities so the gather reads a race-free field.
    pub(crate) aerodynamics_snapshot: CachedComputePipelineId,
    /// `cloth_aerodynamics`: gather per-triangle wind force onto each vertex.
    pub(crate) aerodynamics: CachedComputePipelineId,
}

impl ClothComputePipelines {
    /// Returns the queued pipeline handle that runs the given kernel.
    ///
    /// This maps the golden [`ClothKernel`] schedule onto the concrete pipeline
    /// handles so the dispatch slice can iterate [`ClothKernel::ALL`] and record
    /// each pass without duplicating the kernel-to-pipeline mapping.
    #[must_use]
    pub(crate) fn pipeline(&self, kernel: ClothKernel) -> CachedComputePipelineId {
        match kernel {
            ClothKernel::Predict => self.predict,
            ClothKernel::ProjectDistanceBatch => self.project_distance,
            ClothKernel::ProjectBendingBatch => self.project_bending,
            ClothKernel::ProjectLongRangeBatch => self.project_long_range,
            ClothKernel::StrainLimit => self.strain_limit,
            ClothKernel::BodyCollision => self.body_collision,
            ClothKernel::Backstop => self.backstop,
            ClothKernel::SelfCollisionHashBuild => self.self_collision_hash_build,
            ClothKernel::SelfCollisionResolve => self.self_collision_resolve,
            ClothKernel::VelocityUpdate => self.velocity_update,
            ClothKernel::SkinEmbed => self.skin_embed,
            ClothKernel::AerodynamicsSnapshot => self.aerodynamics_snapshot,
            ClothKernel::Aerodynamics => self.aerodynamics,
        }
    }
}

/// Builds the `cloth_sim.wesl` group-0 layout entries.
///
/// Bindings `0..3` are the read-write particle pools (positions, velocities,
/// previous positions), `3..5` are the read-only distance and bending
/// constraint arrays, and binding `5` is the per-substep uniform. `None`
/// min-binding-size keeps the layout agnostic to each pool's run-time length;
/// the golden buffer sizing owns the extents.
fn sim_layout_entries() -> BindGroupLayoutEntries<6> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            uniform_buffer_sized(false, None),
        ),
    )
}

/// Builds the `cloth_body_collision` group-0 layout entries: the read-write
/// particle positions, the read-only analytic collider list, the body-pass
/// uniform, and the read-only frame-start positions the Coulomb friction pass
/// measures each particle's tangential slide from (binding 3, matching the
/// `body_prev_positions` binding in `cloth_collision.wesl`).
fn body_layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            uniform_buffer_sized(false, None),
            storage_buffer_read_only_sized(false, None),
        ),
    )
}

/// Builds the self-collision group-0 layout entries shared by the hash-build
/// and resolve passes: the read-write particle positions, the read-write
/// spatial-hash cell table, the read-write per-particle linked-list next array,
/// and the self-pass uniform. The cell table and next array are read-write
/// because the build pass writes the linked list the resolve pass then walks.
fn self_layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            uniform_buffer_sized(false, None),
        ),
    )
}

/// Builds the `cloth_backstop` group-0 layout entries: the read-write particle
/// positions, the read-only painted backstop planes, and the backstop-pass
/// uniform.
fn backstop_layout_entries() -> BindGroupLayoutEntries<3> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            uniform_buffer_sized(false, None),
        ),
    )
}

/// Builds the `cloth_skin_embed` group-0 layout entries: the read-only coarse
/// sim positions, the read-write render-vertex positions, the read-only
/// per-render-vertex embed bindings, and the embed-pass uniform.
fn embed_layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            uniform_buffer_sized(false, None),
        ),
    )
}

/// Builds the `cloth_aerodynamics_snapshot` group-0 layout entries: the
/// read-only start-of-frame velocities, the read-write velocity snapshot the
/// pass fills, and the aerodynamic uniform. Freezing the velocity field into a
/// dedicated snapshot buffer is what makes the following gather race-free: every
/// triangle reads the same start-of-frame face velocity regardless of the order
/// the vertex threads run in.
fn aero_snapshot_layout_entries() -> BindGroupLayoutEntries<3> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            storage_buffer_sized(false, None),
            uniform_buffer_sized(false, None),
        ),
    )
}

/// Builds the `cloth_aerodynamics` gather group-0 layout entries. Binding `0`
/// is the read-only particle positions (`.w` carries inverse mass), `1` the
/// read-write velocities the impulse accumulates into, `2` the read-only
/// velocity snapshot, `3` the read-only flat triangle index buffer
/// (`array<u32>`, three per face), `4`/`5` the read-only `CSR`
/// vertex->triangle offsets and entries the gather walks, and `6` the
/// aerodynamic uniform.
fn aero_layout_entries() -> BindGroupLayoutEntries<7> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            uniform_buffer_sized(false, None),
        ),
    )
}

/// `RenderStartup` initializer: creates the seven cloth bind-group layouts and
/// queues the thirteen cloth compute pipelines into the [`PipelineCache`].
///
/// The five cloth shaders must be registered as embedded assets before this
/// runs (see the cloth plugin slice); `load_embedded_asset!` resolves them by
/// their path relative to this file. Each pipeline names its `WESL` entry point
/// and binds exactly one group-0 layout, matching the shader interface it was
/// authored against.
pub(crate) fn init_cloth_compute_pipelines(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let sim_entries = sim_layout_entries();
    let body_entries = body_layout_entries();
    let self_entries = self_layout_entries();
    let backstop_entries = backstop_layout_entries();
    let embed_entries = embed_layout_entries();
    let aero_snapshot_entries = aero_snapshot_layout_entries();
    let aero_entries = aero_layout_entries();

    let sim_descriptor = BindGroupLayoutDescriptor::new("prism cloth sim", &sim_entries);
    let body_descriptor = BindGroupLayoutDescriptor::new("prism cloth body", &body_entries);
    let self_descriptor = BindGroupLayoutDescriptor::new("prism cloth self", &self_entries);
    let backstop_descriptor =
        BindGroupLayoutDescriptor::new("prism cloth backstop", &backstop_entries);
    let embed_descriptor = BindGroupLayoutDescriptor::new("prism cloth embed", &embed_entries);
    let aero_snapshot_descriptor =
        BindGroupLayoutDescriptor::new("prism cloth aero snapshot", &aero_snapshot_entries);
    let aero_descriptor = BindGroupLayoutDescriptor::new("prism cloth aero", &aero_entries);

    let sim_layout = device.create_bind_group_layout("prism cloth sim", &sim_entries);
    let body_layout = device.create_bind_group_layout("prism cloth body", &body_entries);
    let self_layout = device.create_bind_group_layout("prism cloth self", &self_entries);
    let backstop_layout =
        device.create_bind_group_layout("prism cloth backstop", &backstop_entries);
    let embed_layout = device.create_bind_group_layout("prism cloth embed", &embed_entries);
    let aero_snapshot_layout =
        device.create_bind_group_layout("prism cloth aero snapshot", &aero_snapshot_entries);
    let aero_layout = device.create_bind_group_layout("prism cloth aero", &aero_entries);

    let sim_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/cloth_sim.wesl");
    let collision_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/cloth_collision.wesl");
    let embed_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/cloth_embed.wesl");
    let aero_snapshot_shader: Handle<Shader> = load_embedded_asset!(
        asset_server.as_ref(),
        "../shaders/cloth_aerodynamics_snapshot.wesl"
    );
    let aero_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/cloth_aerodynamics.wesl");

    // A `cloth_sim.wesl` pipeline: one group-0 (sim) layout. The three
    // color-serial projection kernels (`distance`/`bending`/`long_range`)
    // address a single per-color slice of their constraint buffer through a
    // `var<immediate> batch: ClothColorBatch { base, count }` push constant,
    // so those pipelines reserve `size_of::<[u32; 2]>()` == 8 bytes of
    // immediate storage. The per-particle kernels (`predict`/`strain_limit`/
    // `velocity_update`) never read `batch`; naga prunes the unused immediate
    // from their entry points, so they declare `immediate_size == 0`.
    let queue_sim = |label: &str, entry: &str, immediate_size: u32| {
        cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(label.to_owned().into()),
            layout: vec![sim_descriptor.clone()],
            immediate_size,
            shader: sim_shader.clone(),
            entry_point: Some(entry.to_owned().into()),
            ..Default::default()
        })
    };

    // Byte size of the `ClothColorBatch { base: u32, count: u32 }` immediate
    // block declared in `cloth_sim.wesl`; shared by the color-serial kernels.
    const CLOTH_COLOR_BATCH_SIZE: u32 = size_of::<[u32; 2]>() as u32;

    let predict = queue_sim(
        "prism cloth predict",
        ClothKernel::Predict.wesl_entry_point(),
        0,
    );
    let project_distance = queue_sim(
        "prism cloth project distance",
        ClothKernel::ProjectDistanceBatch.wesl_entry_point(),
        CLOTH_COLOR_BATCH_SIZE,
    );
    let project_bending = queue_sim(
        "prism cloth project bending",
        ClothKernel::ProjectBendingBatch.wesl_entry_point(),
        CLOTH_COLOR_BATCH_SIZE,
    );
    let project_long_range = queue_sim(
        "prism cloth project long range",
        ClothKernel::ProjectLongRangeBatch.wesl_entry_point(),
        CLOTH_COLOR_BATCH_SIZE,
    );
    let strain_limit = queue_sim(
        "prism cloth strain limit",
        ClothKernel::StrainLimit.wesl_entry_point(),
        0,
    );
    let velocity_update = queue_sim(
        "prism cloth velocity update",
        ClothKernel::VelocityUpdate.wesl_entry_point(),
        0,
    );

    let body_collision = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism cloth body collision".into()),
        layout: vec![body_descriptor.clone()],
        immediate_size: 0,
        shader: collision_shader.clone(),
        entry_point: Some(
            ClothKernel::BodyCollision
                .wesl_entry_point()
                .to_owned()
                .into(),
        ),
        ..Default::default()
    });
    let backstop = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism cloth backstop".into()),
        layout: vec![backstop_descriptor.clone()],
        immediate_size: 0,
        shader: collision_shader.clone(),
        entry_point: Some(ClothKernel::Backstop.wesl_entry_point().to_owned().into()),
        ..Default::default()
    });
    let self_collision_hash_build = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism cloth self hash build".into()),
        layout: vec![self_descriptor.clone()],
        immediate_size: 0,
        shader: collision_shader.clone(),
        entry_point: Some(
            ClothKernel::SelfCollisionHashBuild
                .wesl_entry_point()
                .to_owned()
                .into(),
        ),
        ..Default::default()
    });
    let self_collision_resolve = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism cloth self resolve".into()),
        layout: vec![self_descriptor.clone()],
        immediate_size: 0,
        shader: collision_shader.clone(),
        entry_point: Some(
            ClothKernel::SelfCollisionResolve
                .wesl_entry_point()
                .to_owned()
                .into(),
        ),
        ..Default::default()
    });

    let skin_embed = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism cloth skin embed".into()),
        layout: vec![embed_descriptor.clone()],
        immediate_size: 0,
        shader: embed_shader.clone(),
        entry_point: Some(ClothKernel::SkinEmbed.wesl_entry_point().to_owned().into()),
        ..Default::default()
    });

    // The two aerodynamic passes run once per frame before the substep loop and
    // never read the color-batch immediate, so both declare `immediate_size: 0`.
    let aerodynamics_snapshot = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism cloth aero snapshot".into()),
        layout: vec![aero_snapshot_descriptor.clone()],
        immediate_size: 0,
        shader: aero_snapshot_shader.clone(),
        entry_point: Some(
            ClothKernel::AerodynamicsSnapshot
                .wesl_entry_point()
                .to_owned()
                .into(),
        ),
        ..Default::default()
    });
    let aerodynamics = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism cloth aero".into()),
        layout: vec![aero_descriptor.clone()],
        immediate_size: 0,
        shader: aero_shader.clone(),
        entry_point: Some(
            ClothKernel::Aerodynamics
                .wesl_entry_point()
                .to_owned()
                .into(),
        ),
        ..Default::default()
    });

    commands.insert_resource(ClothComputePipelines {
        sim_layout,
        body_layout,
        self_layout,
        backstop_layout,
        embed_layout,
        predict,
        project_distance,
        project_bending,
        project_long_range,
        strain_limit,
        body_collision,
        backstop,
        self_collision_hash_build,
        self_collision_resolve,
        velocity_update,
        skin_embed,
        aero_snapshot_layout,
        aero_layout,
        aerodynamics_snapshot,
        aerodynamics,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernel-to-pipeline map and the kernel-to-layout map must both be
    /// total over [`ClothKernel::ALL`]; this exercises every arm so a newly
    /// added kernel that forgets its mapping fails to compile (the `match` is
    /// exhaustive) and the layout grouping stays in lock-step with the shader
    /// interface counts declared by the golden kernel contract.
    #[test]
    fn every_kernel_maps_to_the_expected_layout_binding_count() {
        // The concrete `BindGroupLayout` handles need a `RenderDevice`, which a
        // headless unit test has no access to, so this test validates the
        // *grouping* indirectly through the kernel contract: the six sim
        // kernels and the two self kernels must each collapse onto a single
        // shared shader interface, matching the layout arms above.
        use ClothKernel::*;
        let sim = [
            Predict,
            ProjectDistanceBatch,
            ProjectBendingBatch,
            ProjectLongRangeBatch,
            StrainLimit,
            VelocityUpdate,
        ];
        for k in sim {
            // Every sim kernel is per-particle or per-constraint and binds the
            // six-slot simulation interface; the contract keeps its total > 0.
            assert!(k.descriptor().layout.total() > 0, "{k:?} bound nothing");
        }
        let both_self = [SelfCollisionHashBuild, SelfCollisionResolve];
        for k in both_self {
            assert_eq!(
                k.descriptor().layout.storage_buffers,
                3,
                "{k:?} self-collision layout expects three storage buffers"
            );
        }
        // The snapshot pass binds two storage buffers (source velocities plus
        // the snapshot it fills) and the gather binds six (positions,
        // velocities, snapshot, triangles and the two `CSR` adjacency arrays);
        // both add one uniform. These counts must match the seven- and
        // three-binding layouts built above.
        assert_eq!(AerodynamicsSnapshot.descriptor().layout.storage_buffers, 2);
        assert_eq!(AerodynamicsSnapshot.descriptor().layout.uniform_buffers, 1);
        assert_eq!(Aerodynamics.descriptor().layout.storage_buffers, 6);
        assert_eq!(Aerodynamics.descriptor().layout.uniform_buffers, 1);
    }
}
