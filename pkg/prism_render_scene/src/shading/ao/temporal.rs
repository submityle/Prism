//! GTAO cross-frame temporal accumulation: pipeline, per-view ping-pong
//! history, bind group, and the `Core3d` dispatch node.
//!
//! The spatial denoise ([`super::dispatch::gtao_denoise_pass`]) kills most of
//! the per-pixel grain, but a single-frame GTAO estimate still *boils* under
//! motion: each frame the horizon search lands on slightly different depths, so
//! flat walls shimmer and contact shadows crawl. This stage is the GPU twin of
//! [`prism_render_shading::ao::temporal`]: it reprojects last frame's converged
//! AO into the current pixel, clips it to the 3x3 neighbourhood *variance* band
//! (`mean ± γσ`, the AAA Salvi/Karis clip that rejects stale history without
//! the flicker a raw min/max box suffers), then exponentially blends a little of
//! the fresh estimate in each frame (`XeGTAO`'s temporal filter). The blend
//! weight is *adaptive*: history that matches the neighbourhood keeps the full
//! weight for maximum denoising, while history dragged far outside the band (a
//! disocclusion or a moving surface) decays toward a floor so it sheds the
//! stale occlusion instead of ghosting.
//!
//! GTAO runs *before* the resolve writes its motion-vector G-buffer, so there is
//! no per-object motion here. Instead each pixel's view-space position is
//! reconstructed from its linear view depth and pushed through the *previous*
//! frame's `clip_from_world`. The two matrices collapse into one —
//! `clip_prev_from_view = clip_from_world_prev * world_from_view` — because
//! `world_from_view` is affine (`w = 1`), so the CPU-side merge is exact and the
//! immediate block stays small. A disocclusion, an off-screen reprojection, a
//! background pixel, or a flagged-invalid history falls back to the current
//! frame. This is camera-motion-only reprojection: a fast-moving object leaves a
//! short trail the variance clip immediately sheds, which the GTAO golden and
//! `XeGTAO` both accept as the right trade for avoiding a per-object motion pass
//! this early in the frame.
//!
//! History cannot live in the frame-transient [`bevy_render::texture::TextureCache`]
//! the other GTAO targets use — that pool is recycled every frame — so this
//! module keeps a persistent **ping-pong** pair of `r32float` textures per view
//! in a [`Local`] cache keyed by [`RetainedViewEntity`], mirroring
//! [`crate::shading::ssr::temporal`] and
//! [`crate::visibility::hzb::prepare_hzb_history`]. Each frame reads the slot
//! written last frame and writes the other; the resolve then reads the
//! `ambient_occlusion` target the temporal pass writes in place of the raw
//! denoise output.
//!
//! It reads one bind group (group 0, matching `shaders/gtao_temporal.wesl`):
//!
//! * `0` this frame's spatially denoised AO (`textureLoad`ed, both as the
//!   anchor and for the 3x3 neighbourhood variance band),
//! * `1` the previous frame's accumulated AO (sampled with a filtering sampler
//!   so the reprojected UV bilinearly interpolates),
//! * `2` that filtering sampler,
//! * `3` the linear view-depth target (the reconstruct source + real-surface
//!   gate, `textureLoad`ed),
//! * `4` the write-only `r32float` accumulated-AO output (the resolve reads
//!   this), and
//! * `5` the write-only `r32float` history output (next frame's history).
//!
//! The merged reprojection matrix, reconstruct tangents, golden tunables,
//! framebuffer extent and history-validity flag travel in the
//! [`GpuGtaoTemporalConfig`] immediate block.

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_image::ToExtents;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{sampler, texture_2d, texture_storage_2d},
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_math::{Mat4, UVec2};
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    render_resource::{
        AddressMode, BindGroup, BindGroupEntries, BindGroupLayout, CachedComputePipelineId,
        ComputePassDescriptor, ComputePipelineDescriptor, FilterMode, MipmapFilterMode,
        PipelineCache, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages,
        StorageTextureAccess, Texture, TextureDescriptor, TextureDimension, TextureSampleType,
        TextureUsages, TextureView, TextureViewDescriptor,
    },
    renderer::{RenderContext, RenderDevice, ViewQuery},
    view::{ExtractedView, RetainedViewEntity},
};
use bevy_shader::Shader;

use super::abi::{GpuGtaoTemporalConfig, GTAO_TEMPORAL_WORKGROUP_SIZE};
use super::resources::{ViewGtaoTextures, GTAO_AO_FORMAT};

/// Compute pipeline, its owned group-0 layout, and the filtering sampler the
/// accumulation reads the reprojected history through.
#[derive(Resource)]
pub(crate) struct GtaoTemporalPipeline {
    /// `accumulate_gtao` compute entry point, specialized against the group-0
    /// layout and the 96-byte [`GpuGtaoTemporalConfig`] immediate block.
    accumulate: CachedComputePipelineId,
    /// group 0: denoised-AO + linear-depth reads, the filterable history + its
    /// sampler, and the two write-only accumulated/history outputs.
    layout: BindGroupLayout,
    /// Bilinear clamp sampler bound at binding 2 so the reprojected history UV
    /// interpolates. Pass-owned (single-mip history, so no mip filtering).
    sampler: Sampler,
}

/// group-0 layout mirroring `gtao_temporal.wesl`: the non-filterable denoised-AO
/// read, the *filterable* history + its filtering sampler (sampled at the
/// reprojected UV), the non-filterable linear-depth read (reconstruct + gate),
/// and the two write-only `r32float` outputs (the resolve's AO input and next
/// frame's history).
fn layout_entries() -> BindGroupLayoutEntries<6> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: true }),
            sampler(SamplerBindingType::Filtering),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(GTAO_AO_FORMAT, StorageTextureAccess::WriteOnly),
            texture_storage_2d(GTAO_AO_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `RenderStartup` initializer for [`GtaoTemporalPipeline`].
pub(crate) fn init_gtao_temporal_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism GTAO temporal", &entries);
    let layout = device.create_bind_group_layout("prism GTAO temporal", &entries);

    // Bilinear clamp: linear min/mag so the reprojected history UV interpolates.
    // History is single-mip, so mip filtering never engages.
    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("prism GTAO temporal history sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: MipmapFilterMode::Nearest,
        ..Default::default()
    });

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/gtao_temporal.wesl");

    let accumulate = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism GTAO temporal".into()),
        layout: vec![descriptor],
        immediate_size: size_of::<GpuGtaoTemporalConfig>() as u32,
        shader,
        entry_point: Some("accumulate_gtao".into()),
        ..Default::default()
    });

    commands.insert_resource(GtaoTemporalPipeline {
        accumulate,
        layout,
        sampler,
    });
}

/// Per-view temporal state resolved each frame from the persistent ping-pong
/// cache: the readable previous-frame history, the writable current-frame
/// output, the merged view-space -> previous-clip reprojection matrix, the
/// reconstruct tangents, and whether that history is trustworthy this frame.
#[derive(Component)]
pub(crate) struct ViewGtaoTemporal {
    /// Previous frame's accumulated AO, sampled at the reprojected UV.
    read_view: TextureView,
    /// This frame's accumulated output. Written by the temporal pass (storage)
    /// and read by the resolve in place of the raw denoise output.
    write_view: TextureView,
    /// `false` on the first frame, a resize, or a fresh allocation, so the
    /// shader ignores the (garbage) history and passes the denoise through.
    valid: bool,
    /// `clip_from_world_prev * world_from_view`: view space -> previous clip
    /// space. On the first valid frame this is the *current* transform, which
    /// yields zero motion (identity reprojection).
    clip_prev_from_view: Mat4,
    /// `1 / |proj[0][0]|`; scales NDC x into a view-space slope (reconstruct).
    tan_half_fov_x: f32,
    /// `1 / |proj[1][1]|`; scales NDC y into a view-space slope (reconstruct).
    tan_half_fov_y: f32,
}

impl ViewGtaoTemporal {
    /// Previous-frame accumulated AO bound as the sampled history.
    pub(crate) fn read_view(&self) -> &TextureView {
        &self.read_view
    }

    /// Current-frame accumulated output bound as the storage write target.
    pub(crate) fn write_view(&self) -> &TextureView {
        &self.write_view
    }
}

/// A persistent ping-pong history pair for one view. Kept out of the frame
/// transient [`bevy_render::texture::TextureCache`] so last frame's accumulation
/// survives into this frame; both slots carry `STORAGE_BINDING | TEXTURE_BINDING`
/// because they swap read/write roles every frame. The previous frame's
/// `clip_from_world` is stashed here so the CPU can build the merged
/// reprojection matrix without a second per-view uniform.
struct CachedGtaoTemporal {
    view_a: TextureView,
    view_b: TextureView,
    size: UVec2,
    /// Which slot holds the readable previous-frame output: `false` -> A,
    /// `true` -> B. Flipped every frame after the roles are handed out.
    parity: bool,
    /// Column-major `clip_from_world` recorded last frame; the reprojection's
    /// previous-frame transform. Meaningful only when `has_prev`.
    prev_clip_from_world: [[f32; 4]; 4],
    /// `false` until the first frame records a `clip_from_world`, so the very
    /// first accumulation cannot reproject through garbage.
    has_prev: bool,
}

/// The persistent per-view history cache, surviving across frames in a
/// [`Local`]. Keyed by [`RetainedViewEntity`] so a view keeps its history as its
/// render-world entity churns, exactly like
/// [`crate::visibility::hzb::prepare_hzb_history`].
#[derive(Default)]
pub(crate) struct GtaoTemporalHistoryCache {
    views: HashMap<RetainedViewEntity, CachedGtaoTemporal>,
}

/// Allocates one persistent single-mip `r32float` history slot at `size`, with
/// both storage and texture binding so it can serve as the write target on one
/// frame and the sampled history on the next.
fn create_history(device: &RenderDevice, size: UVec2) -> TextureView {
    let texture: Texture = device.create_texture(&TextureDescriptor {
        label: Some("prism GTAO temporal history"),
        size: size.to_extents(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: GTAO_AO_FORMAT,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    texture.create_view(&TextureViewDescriptor {
        label: Some("prism GTAO temporal history view"),
        ..Default::default()
    })
}

/// `PrepareResources` system resolving [`ViewGtaoTemporal`] for every view with
/// resident [`ViewGtaoTextures`] while GTAO temporal is enabled, (re)allocating
/// the persistent ping-pong history to match the viewport, flipping the
/// read/write slots each frame, and building the merged reprojection matrix from
/// the previous frame's stashed `clip_from_world`.
///
/// Runs after `prepare_gtao_textures` so the viewport size is settled. A cache
/// miss or a size change allocates a fresh pair and marks the history invalid
/// (the shader passes the denoise through); a hit hands out last frame's write
/// slot as the readable history. Views that lost their GTAO textures — or run
/// with temporal disabled — drop their cache entry and component.
pub(crate) fn prepare_gtao_temporal_textures(
    mut commands: Commands,
    settings: Res<super::super::runtime::PrismShadingSettings>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedView, &ViewGtaoTextures)>,
    mut cache: Local<GtaoTemporalHistoryCache>,
) {
    if !settings.enable_gtao_temporal {
        // Temporal off: shed every component and free all history textures.
        for (entity, _, _) in &views {
            commands.entity(entity).remove::<ViewGtaoTemporal>();
        }
        cache.views.clear();
        return;
    }

    let mut retained = HashSet::<RetainedViewEntity>::new();
    for (entity, view, textures) in &views {
        let retained_view = view.retained_view_entity;
        let size = textures.size;
        if size.x == 0 || size.y == 0 {
            commands.entity(entity).remove::<ViewGtaoTemporal>();
            cache.views.remove(&retained_view);
            continue;
        }
        retained.insert(retained_view);

        // Reconstruct tangents from the projection diagonal, exactly as the
        // golden `GtaoCamera::from_projection` does (`tan = 1 / |proj_diag|`).
        let clip_from_view = view.clip_from_view;
        let tan_half_fov_x = clip_from_view.x_axis.x.abs().recip();
        let tan_half_fov_y = clip_from_view.y_axis.y.abs().recip();

        // world_from_view is affine (w = 1), so merging it into the previous
        // clip_from_world is exact. clip_from_world prefers the view's cached
        // value and reconstructs it from the projection otherwise.
        let world_from_view = view.world_from_view.to_matrix();
        let clip_from_world = view
            .clip_from_world
            .unwrap_or_else(|| clip_from_view * world_from_view.inverse());

        // Reuse the persistent pair only when its extent still matches; a resize
        // reallocates and drops the history to the (invalid) current frame.
        let reuse = cache
            .views
            .get(&retained_view)
            .is_some_and(|cached| cached.size == size);
        if !reuse {
            cache.views.insert(
                retained_view,
                CachedGtaoTemporal {
                    view_a: create_history(&device, size),
                    view_b: create_history(&device, size),
                    size,
                    parity: false,
                    prev_clip_from_world: [[0.0; 4]; 4],
                    has_prev: false,
                },
            );
        }

        let cached = cache
            .views
            .get_mut(&retained_view)
            .expect("history cache entry was just inserted when absent");

        // Read the previous frame's transform *before* overwriting it. On the
        // first (invalid) frame there is no previous transform, so reproject
        // through the current one (zero motion) and flag the history invalid.
        let had_prev = cached.has_prev;
        let clip_prev_from_view = if had_prev {
            Mat4::from_cols_array_2d(&cached.prev_clip_from_world) * world_from_view
        } else {
            clip_from_world * world_from_view
        };
        // Valid history requires both a reused pair and a recorded previous
        // transform (a fresh allocation resets `has_prev`).
        let valid = reuse && had_prev;

        // read = slot written last frame (parity); write = the other slot.
        let (read_view, write_view) = if cached.parity {
            (cached.view_a.clone(), cached.view_b.clone())
        } else {
            (cached.view_b.clone(), cached.view_a.clone())
        };

        // Next frame reads what we are about to write this frame, and reprojects
        // through the transform we just used.
        cached.parity = !cached.parity;
        cached.prev_clip_from_world = clip_from_world.to_cols_array_2d();
        cached.has_prev = true;

        commands.entity(entity).insert(ViewGtaoTemporal {
            read_view,
            write_view,
            valid,
            clip_prev_from_view,
            tan_half_fov_x,
            tan_half_fov_y,
        });
    }

    // Drop history for views that no longer run GTAO so their textures free.
    cache.views.retain(|view, _| retained.contains(view));
}

/// The temporal accumulation's group-0 bind group for a single view. Present
/// only when both the GTAO textures (denoised AO + linear depth) and the
/// ping-pong history are resident.
#[derive(Component)]
pub(crate) struct ViewGtaoTemporalBindGroup {
    group: BindGroup,
}

/// `PrepareBindGroups` system building [`ViewGtaoTemporalBindGroup`] for every
/// view with resident [`ViewGtaoTextures`] and a resolved [`ViewGtaoTemporal`].
///
/// Binding 4 is the `ambient_occlusion` target the resolve reads, so the
/// temporal pass writes the blended AO exactly where the resolve expects it;
/// the denoise has been redirected to write `denoised_ambient_occlusion`
/// (binding 0) whenever temporal is enabled.
pub(crate) fn prepare_gtao_temporal_bind_groups(
    mut commands: Commands,
    pipeline: Res<GtaoTemporalPipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewGtaoTextures, &ViewGtaoTemporal)>,
) {
    for (entity, textures, temporal) in &views {
        let group = device.create_bind_group(
            "prism GTAO temporal",
            &pipeline.layout,
            &BindGroupEntries::sequential((
                textures.denoised_ambient_occlusion_view(),
                temporal.read_view(),
                &pipeline.sampler,
                textures.linear_depth_view(),
                textures.ambient_occlusion_view(),
                temporal.write_view(),
            )),
        );
        commands
            .entity(entity)
            .insert(ViewGtaoTemporalBindGroup { group });
    }
}

/// `Core3d` node recording the temporal-accumulation dispatch for every view.
///
/// Gated on both `enable_gtao` and `enable_gtao_temporal`. Runs after the
/// spatial denoise (its denoised input) and before the resolve that now reads
/// the accumulated AO. Dispatches one workgroup per 8x8 pixel tile; the shader
/// bounds-checks every invocation and falls back to the current frame wherever
/// the reprojection or the history is invalid.
pub(crate) fn gtao_temporal_pass(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    view: ViewQuery<(
        &ViewGtaoTextures,
        &ViewGtaoTemporalBindGroup,
        &ViewGtaoTemporal,
    )>,
    pipeline: Res<GtaoTemporalPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_gtao || !settings.enable_gtao_temporal {
        return;
    }
    let (textures, group, temporal) = view.into_inner();

    let Some(accumulate) = cache.get_compute_pipeline(pipeline.accumulate) else {
        return;
    };

    let size = textures.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    let config = GpuGtaoTemporalConfig {
        clip_prev_from_view: temporal.clip_prev_from_view.to_cols_array(),
        tan_half_fov_x: temporal.tan_half_fov_x,
        tan_half_fov_y: temporal.tan_half_fov_y,
        history_weight: settings.gtao_temporal_history_weight,
        min_history_weight: settings.gtao_temporal_min_history_weight,
        variance_gamma: settings.gtao_temporal_variance_gamma,
        width: size.x,
        height: size.y,
        valid_history: u32::from(temporal.valid),
    };

    let workgroups_x = size.x.div_ceil(GTAO_TEMPORAL_WORKGROUP_SIZE);
    let workgroups_y = size.y.div_ceil(GTAO_TEMPORAL_WORKGROUP_SIZE);

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism GTAO temporal"),
            timestamp_writes: None,
        });
    pass.set_pipeline(accumulate);
    pass.set_bind_group(0, &group.group, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&config));
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}
