//! SSR cross-frame temporal accumulation: pipeline, per-view ping-pong history,
//! bind group, and the `Core3d` dispatch node.
//!
//! The spatial resolve collapses most of the multi-ray trace noise, but a
//! handful of rays per pixel still leaves temporal shimmer as the camera moves.
//! This stage is the GPU twin of
//! [`prism_render_shading::screen_space::temporal`]: it reprojects last frame's
//! accumulated reflection into the current pixel, clips it to the 3x3
//! neighbourhood *variance* box (`mean ± γσ`, the AAA Salvi/Karis clip that
//! rejects stale history without the flicker a raw min/max box suffers), then
//! exponentially blends it with the freshly resolved reflection. The blend
//! weight is *adaptive*: history that matches the neighbourhood keeps the full
//! weight for maximum denoising, while history dragged far outside the box (a
//! disocclusion or a moving surface) decays toward a floor so it sheds the
//! stale reflection instead of ghosting — integrating many effective samples
//! over time while staying responsive.
//!
//! The resolve pass writes a per-pixel motion-vector G-buffer (`cur_uv -
//! prev_uv`, folding in both camera *and* per-object motion), so the
//! reprojection is a single add: `prev_uv = cur_uv - motion`. This tracks
//! moving surfaces the old camera-only depth reprojection could not, without a
//! previous-frame matrix. Off-screen (after the motion offset), background and
//! flagged-invalid samples fall back to the current frame, so a camera cut or a
//! resize degrades gracefully to the un-accumulated resolve rather than
//! smearing. The `device_depth > 0` gate still restricts accumulation to real
//! surfaces (the motion buffer is zero on pixels the resolve did not cover).
//!
//! History cannot live in the frame-transient [`TextureCache`] the other SSR
//! targets use — that pool is recycled every frame — so this module keeps a
//! persistent **ping-pong** pair of `rgba16float` textures per view in a
//! [`Local`] cache keyed by [`RetainedViewEntity`], mirroring
//! [`crate::visibility::hzb::prepare_hzb_history`]. Each frame reads the slot
//! written last frame and writes the other; the composite then reads the write
//! slot in place of the raw resolve.
//!
//! It reads one bind group (group 0, matching `shaders/ssr_temporal.wesl`):
//!
//! * `0` this frame's spatially resolved reflection (`textureLoad`ed, both as
//!   the anchor colour and for the 3x3 neighbourhood variance box),
//! * `1` the full-resolution reverse-Z device depth (the real-surface gate),
//! * `2` the previous frame's accumulated reflection (sampled with a filtering
//!   sampler so the reprojected UV bilinearly interpolates),
//! * `3` that filtering sampler,
//! * `4` the write-only `rgba16float` accumulated-reflection output, and
//! * `5` the resolve's `rg16float` motion-vector G-buffer (`textureLoad`ed;
//!   `prev_uv = cur_uv - motion`).
//!
//! The framebuffer extent, golden tunables and the history-validity flag travel
//! in the [`GpuSsrTemporalParams`] immediate block. Runs after the reconstruct
//! (its resolved input) and before the composite that now reads the accumulated
//! buffer.

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
use bevy_math::UVec2;
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

use super::super::resources::ViewVisibilityBuffer;
use super::abi::{GpuSsrTemporalParams, SSR_WORKGROUP_SIZE};
use super::resources::{ViewSsrTextures, SSR_OUT_FORMAT};

/// Compute pipeline, its owned group-0 layout, and the filtering sampler the
/// accumulation reads the reprojected history through.
#[derive(Resource)]
pub(crate) struct SsrTemporalPipeline {
    /// `accumulate_ssr` compute entry point, specialized against the group-0
    /// layout and the 32-byte [`GpuSsrTemporalParams`] immediate block.
    accumulate: CachedComputePipelineId,
    /// group 0: resolved reflection + device depth reads, the filterable
    /// history + its sampler, and the write-only accumulated output.
    layout: BindGroupLayout,
    /// Bilinear clamp sampler bound at binding 3 so the reprojected history UV
    /// interpolates. Pass-owned (single-mip history, so no mip filtering).
    sampler: Sampler,
}

/// group-0 layout mirroring `ssr_temporal.wesl`: two non-filterable float reads
/// (the resolved reflection and device depth, both `textureLoad`ed), the
/// *filterable* history + its filtering sampler (sampled at the reprojected UV),
/// the write-only `rgba16float` accumulated output, and the non-filterable
/// motion-vector G-buffer the reprojection adds back to reach last frame's UV.
fn layout_entries() -> BindGroupLayoutEntries<6> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: true }),
            sampler(SamplerBindingType::Filtering),
            texture_storage_2d(SSR_OUT_FORMAT, StorageTextureAccess::WriteOnly),
            texture_2d(TextureSampleType::Float { filterable: false }),
        ),
    )
}

/// `RenderStartup` initializer for [`SsrTemporalPipeline`].
pub(crate) fn init_ssr_temporal_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism SSR temporal", &entries);
    let layout = device.create_bind_group_layout("prism SSR temporal", &entries);

    // Bilinear clamp: linear min/mag so the reprojected history UV interpolates.
    // History is single-mip, so mip filtering never engages.
    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("prism SSR temporal history sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: MipmapFilterMode::Nearest,
        ..Default::default()
    });

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/ssr_temporal.wesl");

    let accumulate = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism SSR temporal".into()),
        layout: vec![descriptor],
        immediate_size: size_of::<GpuSsrTemporalParams>() as u32,
        shader,
        entry_point: Some("accumulate_ssr".into()),
        ..Default::default()
    });

    commands.insert_resource(SsrTemporalPipeline {
        accumulate,
        layout,
        sampler,
    });
}

/// Per-view temporal state resolved each frame from the persistent ping-pong
/// cache: the readable previous-frame history, the writable current-frame
/// output (which the composite reads), and whether that history is trustworthy
/// this frame. The reprojection reads the motion-vector G-buffer, so no
/// previous-frame matrix is carried here anymore.
#[derive(Component)]
pub(crate) struct ViewSsrTemporal {
    /// Previous frame's accumulated reflection, sampled at the reprojected UV.
    read_view: TextureView,
    /// This frame's accumulated output. Written by the temporal pass (storage)
    /// and read by the composite (sampled) in place of the raw resolve.
    write_view: TextureView,
    /// `false` on the first frame, a resize, or a fresh allocation, so the
    /// shader ignores the (garbage) history and passes the resolve through.
    valid: bool,
}

impl ViewSsrTemporal {
    /// Previous-frame accumulated reflection bound as the sampled history.
    pub(crate) fn read_view(&self) -> &TextureView {
        &self.read_view
    }

    /// Current-frame accumulated output bound as the storage write target (and
    /// read by the composite).
    pub(crate) fn write_view(&self) -> &TextureView {
        &self.write_view
    }
}

/// A persistent ping-pong history pair for one view. Kept out of the frame
/// transient [`TextureCache`] so last frame's accumulation survives into this
/// frame; both slots carry `STORAGE_BINDING | TEXTURE_BINDING` because they swap
/// read/write roles every frame.
struct CachedTemporal {
    view_a: TextureView,
    view_b: TextureView,
    size: UVec2,
    /// Which slot holds the readable previous-frame output: `false` -> A,
    /// `true` -> B. Flipped every frame after the roles are handed out.
    parity: bool,
}

/// The persistent per-view history cache, surviving across frames in a
/// [`Local`]. Keyed by [`RetainedViewEntity`] so a view keeps its history as its
/// render-world entity churns, exactly like
/// [`crate::visibility::hzb::prepare_hzb_history`].
#[derive(Default)]
pub(crate) struct TemporalHistoryCache {
    views: HashMap<RetainedViewEntity, CachedTemporal>,
}

/// Allocates one persistent single-mip `rgba16float` history slot at `size`,
/// with both storage and texture binding so it can serve as the write target on
/// one frame and the sampled history on the next.
fn create_history(device: &RenderDevice, size: UVec2) -> TextureView {
    let texture: Texture = device.create_texture(&TextureDescriptor {
        label: Some("prism SSR temporal history"),
        size: size.to_extents(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: SSR_OUT_FORMAT,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    texture.create_view(&TextureViewDescriptor {
        label: Some("prism SSR temporal history view"),
        ..Default::default()
    })
}

/// `Prepare` system resolving [`ViewSsrTemporal`] for every view with resident
/// [`ViewSsrTextures`], (re)allocating the persistent ping-pong history to match
/// the viewport and flipping the read/write slots each frame.
///
/// Runs after `prepare_ssr_textures` so the viewport size is settled. A cache
/// miss or a size change allocates a fresh pair and marks the history invalid
/// (the shader passes the resolve through); a hit hands out last frame's write
/// slot as the readable history (the GPU reprojects it via the motion buffer).
/// Views that lost their SSR textures drop their cache entry.
pub(crate) fn prepare_ssr_temporal_textures(
    mut commands: Commands,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedView, &ViewSsrTextures)>,
    mut cache: Local<TemporalHistoryCache>,
) {
    let mut retained = HashSet::<RetainedViewEntity>::new();
    for (entity, view, textures) in &views {
        let retained_view = view.retained_view_entity;
        let size = textures.size;
        if size.x == 0 || size.y == 0 {
            commands.entity(entity).remove::<ViewSsrTemporal>();
            cache.views.remove(&retained_view);
            continue;
        }
        retained.insert(retained_view);

        // Reuse the persistent pair only when its extent still matches; a resize
        // reallocates and drops the history to the (invalid) current frame.
        let reuse = cache
            .views
            .get(&retained_view)
            .is_some_and(|cached| cached.size == size);
        if !reuse {
            cache.views.insert(
                retained_view,
                CachedTemporal {
                    view_a: create_history(&device, size),
                    view_b: create_history(&device, size),
                    size,
                    parity: false,
                },
            );
        }

        let cached = cache
            .views
            .get_mut(&retained_view)
            .expect("history cache entry was just inserted when absent");

        // read = slot written last frame (parity); write = the other slot.
        let (read_view, write_view) = if cached.parity {
            (cached.view_a.clone(), cached.view_b.clone())
        } else {
            (cached.view_b.clone(), cached.view_a.clone())
        };

        // Next frame reads what we are about to write this frame.
        cached.parity = !cached.parity;

        commands.entity(entity).insert(ViewSsrTemporal {
            read_view,
            write_view,
            valid: reuse,
        });
    }

    // Drop history for views that no longer run SSR so their textures free.
    cache.views.retain(|view, _| retained.contains(view));
}

/// The temporal accumulation's group-0 bind group for a single view. Present
/// only when both the SSR textures (resolved reflection + depth) and the
/// ping-pong history are resident.
#[derive(Component)]
pub(crate) struct ViewSsrTemporalBindGroup {
    group: BindGroup,
}

/// `PrepareBindGroups` system building [`ViewSsrTemporalBindGroup`] for every
/// view with resident [`ViewSsrTextures`] and a resolved [`ViewSsrTemporal`].
pub(crate) fn prepare_ssr_temporal_bind_groups(
    mut commands: Commands,
    pipeline: Res<SsrTemporalPipeline>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ViewSsrTextures,
        &ViewSsrTemporal,
        &ViewVisibilityBuffer,
    )>,
) {
    for (entity, textures, temporal, visibility) in &views {
        let group = device.create_bind_group(
            "prism SSR temporal",
            &pipeline.layout,
            &BindGroupEntries::sequential((
                textures.ssr_resolved_view(),
                textures.scene_depth_sampled(),
                temporal.read_view(),
                &pipeline.sampler,
                temporal.write_view(),
                visibility.motion_vectors_view(),
            )),
        );
        commands
            .entity(entity)
            .insert(ViewSsrTemporalBindGroup { group });
    }
}

/// `Core3d` node recording the temporal-accumulation dispatch for every view.
///
/// Runs after the reconstruct (its resolved input) and before the composite
/// that now reads the accumulated buffer. Dispatches one workgroup per 8x8 pixel
/// tile; the shader bounds-checks every invocation and falls back to the current
/// frame wherever the reprojection or the history is invalid.
pub(crate) fn ssr_temporal_pass(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    view: ViewQuery<(
        &ViewSsrTextures,
        &ViewSsrTemporalBindGroup,
        &ViewSsrTemporal,
    )>,
    pipeline: Res<SsrTemporalPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_ssr {
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

    // The reprojection reads the resolve's motion-vector G-buffer on the GPU,
    // so the params only carry the framebuffer extent and the history-valid flag.
    let params = GpuSsrTemporalParams::new(size.x, size.y, temporal.valid);

    let workgroups_x = size.x.div_ceil(SSR_WORKGROUP_SIZE);
    let workgroups_y = size.y.div_ceil(SSR_WORKGROUP_SIZE);

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism SSR temporal"),
            timestamp_writes: None,
        });
    pass.set_pipeline(accumulate);
    pass.set_bind_group(0, &group.group, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&params));
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}
