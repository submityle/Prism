use bevy_ecs::prelude::*;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        BindGroup, Buffer, BufferDescriptor, BufferInitDescriptor, BufferUsages, TextureFormat,
    },
    renderer::RenderDevice,
    view::Msaa,
};
use prism_render_driver as rhi;
use prism_render_shading::ShadingWorkItem;

use crate::gpu::rhi::{SceneRhi, SceneRhiTexture, SceneRhiTextureView};

use super::classification_gpu::{GpuShadingDispatchArgs, GpuShadingPixelClass};

pub(crate) const VISIBILITY_ID_FORMAT: TextureFormat = TextureFormat::Rgba32Uint;

/// Linear HDR radiance target written by the shading-resolve compute pass
/// (`shading_resolve.wesl`) and later sampled by the composite/tonemap step.
/// `Rgba16Float` matches the `texture_storage_2d<rgba16float, write>` binding
/// in the shader and keeps enough range/precision for pre-exposure radiance.
pub(crate) const SCENE_COLOR_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// Per-pixel screen-space motion (`cur_uv - prev_uv`) written by the resolve
/// pass and consumed by the SSR temporal accumulation (and, later, TAA).
/// `Rg16Float` matches the `texture_storage_2d<rg16float, write>` binding in
/// `shading_resolve.wesl`; two half-float channels are ample for a UV-space
/// delta that is almost always a small fraction of the frame.
pub(crate) const MOTION_VECTOR_FORMAT: TextureFormat = TextureFormat::Rg16Float;

/// Owns both table entries for the exact texture consumed by native passes.
/// No parallel/shadow allocation is made for the RHI graph. Native command
/// buffers retain their handles after view replacement; table IDs are valid
/// only while this owner remains resident on the view.
struct VisibilityTexture {
    // Drop the view table entry before its texture entry. Neither lease
    // destroys native handles retained by already encoded host commands.
    default_view: SceneRhiTextureView,
    texture: SceneRhiTexture,
}

impl VisibilityTexture {
    fn new(
        rhi: &SceneRhi,
        size: bevy_math::UVec2,
        label: &'static str,
        format: rhi::TextureFormat,
        usage: rhi::TextureUsages,
    ) -> Self {
        let mut descriptor =
            rhi::TextureDescriptor::new_2d(rhi::Extent3d::new_2d(size.x, size.y), format, usage);
        descriptor.label = Some(label.into());
        let texture = rhi
            .create_texture(&descriptor)
            .expect("visibility texture descriptor must be supported by the host renderer");
        let default_view = texture
            .create_view(&rhi::TextureViewDescriptor::default())
            .expect("visibility texture default view must match its allocation");
        Self {
            default_view,
            texture,
        }
    }
}

/// All nine attachments belong to one SceneRhi table on the host device.
/// Replace the complete owner on resize or bridge replacement; never mix
/// table-local IDs from different owners, even on the same native device.
#[derive(Component)]
pub(crate) struct ViewVisibilityBuffer {
    ids: VisibilityTexture,
    metadata: VisibilityTexture,
    /// Linear HDR radiance the shading-resolve pass stores into and the
    /// composite step samples from.
    scene_color: VisibilityTexture,
    /// SSR energy-conservation export: the IBL specular radiance the resolve
    /// folded into `scene_color` per pixel, so the SSR composite can subtract
    /// it before substituting the screen-space reflection.
    ssr_env_specular: VisibilityTexture,
    /// SSR energy-conservation export: the split-sum environment-BRDF weight
    /// `(f0 * dfg.x + dfg.y) * occlusion` the resolve scaled that specular by,
    /// reused by the composite to weight the SSR reflection identically.
    ssr_spec_weight: VisibilityTexture,
    /// Per-pixel screen-space motion vector (`cur_uv - prev_uv`) the resolve
    /// pass writes for every covered pixel; consumed by the SSR temporal
    /// reprojection to follow object + camera motion instead of ghosting.
    motion_vectors: VisibilityTexture,
    /// Screen-space GI export: the pre-albedo diffuse *irradiance* the resolve
    /// evaluated per pixel. The SSGI hemisphere gather blends missed rays
    /// toward this so indirect light degrades to the IBL/SH ambient instead of
    /// to black.
    ssgi_ambient: VisibilityTexture,
    /// Screen-space GI export: the Lambertian albedo (`base_color *
    /// (1 - metallic)`) the SSGI composite multiplies the gathered pre-albedo
    /// radiance by before blending it over the IBL diffuse under confidence.
    ssgi_albedo: VisibilityTexture,
    /// World-space ReSTIR direct export: the raw cosine-weighted punctual
    /// *irradiance* the resolve evaluated per pixel (no 1/pi, Fresnel or
    /// albedo), in the same units as the ReSTIR resolve's `gi_out`. The
    /// world_restir composite subtracts the clustered punctual diffuse this
    /// reconstructs before folding in the reservoir estimate, so the
    /// substitution never double-counts the direct lighting the cache reused.
    world_restir_direct: VisibilityTexture,
    pub(crate) size: bevy_math::UVec2,
}

impl ViewVisibilityBuffer {
    pub(crate) fn attachments(
        &self,
    ) -> (
        &bevy_render::render_resource::TextureView,
        &bevy_render::render_resource::TextureView,
    ) {
        (
            self.ids.default_view.view(),
            self.metadata.default_view.view(),
        )
    }

    /// The actual raster attachments used by the visibility pass.  The
    /// feedback bridge copies these textures after the render graph has
    /// recorded the raster work; it never treats the CPU `VisibilityBuffer`
    /// helper as a producer.
    pub(crate) fn feedback_sources(
        &self,
    ) -> (
        &bevy_render::render_resource::Texture,
        &bevy_render::render_resource::Texture,
        bevy_math::UVec2,
    ) {
        (
            self.ids.texture.texture(),
            self.metadata.texture.texture(),
            self.size,
        )
    }

    /// Storage/sampling view of the linear HDR radiance target.
    pub(crate) fn scene_color_view(&self) -> &bevy_render::render_resource::TextureView {
        self.scene_color.default_view.view()
    }

    /// The `scene_color` GPU texture itself, for a full-surface
    /// `copy_texture_to_texture` (an opt-in ping-pong post-process writes its
    /// result back in place here).
    pub(crate) fn scene_color_texture(&self) -> &bevy_render::render_resource::Texture {
        self.scene_color_rhi().texture()
    }

    /// The owner of the HDR resource exported to the native composition pass.
    /// Imports must use this lease's table and keep the view resident through
    /// recording; its ID is not a native Bevy texture ID.
    pub(crate) fn scene_color_rhi(&self) -> &SceneRhiTexture {
        &self.scene_color.texture
    }

    /// Storage/sampling view of the SSR IBL-specular export.
    pub(crate) fn ssr_env_specular_view(&self) -> &bevy_render::render_resource::TextureView {
        self.ssr_env_specular.default_view.view()
    }

    /// Storage/sampling view of the SSR environment-BRDF weight export.
    pub(crate) fn ssr_spec_weight_view(&self) -> &bevy_render::render_resource::TextureView {
        self.ssr_spec_weight.default_view.view()
    }

    /// Storage/sampling view of the per-pixel motion-vector G-buffer.
    pub(crate) fn motion_vectors_view(&self) -> &bevy_render::render_resource::TextureView {
        self.motion_vectors.default_view.view()
    }

    /// Storage/sampling view of the SSGI pre-albedo ambient-irradiance export.
    pub(crate) fn ssgi_ambient_view(&self) -> &bevy_render::render_resource::TextureView {
        self.ssgi_ambient.default_view.view()
    }

    /// Storage/sampling view of the SSGI Lambertian-albedo export.
    pub(crate) fn ssgi_albedo_view(&self) -> &bevy_render::render_resource::TextureView {
        self.ssgi_albedo.default_view.view()
    }

    /// Storage/sampling view of the world-space ReSTIR direct-light export
    /// (raw cosine-weighted punctual irradiance).
    pub(crate) fn world_restir_direct_view(&self) -> &bevy_render::render_resource::TextureView {
        self.world_restir_direct.default_view.view()
    }
}

#[derive(Component)]
pub(crate) struct ViewShadingBuffers {
    pub(crate) pixel_classes: Buffer,
    pub(crate) work_items: Buffer,
    pub(crate) class_counts: Buffer,
    pub(crate) class_offsets: Buffer,
    pub(crate) class_cursors: Buffer,
    pub(crate) dispatch_args: Buffer,
    pub(crate) diagnostics: Buffer,
    pub(crate) buckets: Buffer,
    pub(crate) input_bind_group: Option<BindGroup>,
    pub(crate) output_bind_group: Option<BindGroup>,
    pub(crate) size: bevy_math::UVec2,
    pub(crate) capacity: u32,
    pub(crate) bucket_count: u32,
    pub(crate) bucket_table_version: u32,
}

impl ViewShadingBuffers {
    fn new(
        device: &RenderDevice,
        size: bevy_math::UVec2,
        bucket_table: &prism_render_shading::ResponseBucketTable,
        bucket_table_version: u32,
    ) -> Self {
        let capacity = size.x.saturating_mul(size.y).max(1);
        let bucket_count = bucket_table.len() as u32;
        Self {
            pixel_classes: storage_buffer::<GpuShadingPixelClass>(
                device,
                "prism pixel classes",
                capacity,
                false,
            ),
            work_items: storage_buffer::<ShadingWorkItem>(
                device,
                "prism shading work",
                capacity,
                false,
            ),
            class_counts: storage_buffer::<u32>(device, "prism class counts", bucket_count, false),
            class_offsets: storage_buffer::<u32>(
                device,
                "prism class offsets",
                bucket_count,
                false,
            ),
            class_cursors: storage_buffer::<u32>(
                device,
                "prism class cursors",
                bucket_count,
                false,
            ),
            dispatch_args: storage_buffer::<GpuShadingDispatchArgs>(
                device,
                "prism shading dispatch",
                bucket_count,
                true,
            ),
            diagnostics: storage_buffer::<u32>(device, "prism classification diagnostics", 4, true),
            buckets: device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("prism response bucket table"),
                contents: bytemuck::cast_slice(
                    &bucket_table
                        .buckets()
                        .iter()
                        .map(|bucket| super::classification_gpu::GpuShadingBucket {
                            execution_family: bucket.execution_family as u32,
                            program_id: bucket.program_id.0,
                            render_phase: bucket.render_phase as u32,
                            reserved: 0,
                        })
                        .collect::<Vec<_>>(),
                ),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            }),
            input_bind_group: None,
            output_bind_group: None,
            size,
            capacity,
            bucket_count,
            bucket_table_version,
        }
    }

    pub(crate) fn clear_transient_state(
        &self,
        encoder: &mut bevy_render::render_resource::CommandEncoder,
    ) {
        encoder.clear_buffer(&self.class_counts, 0, None);
        encoder.clear_buffer(&self.class_offsets, 0, None);
        encoder.clear_buffer(&self.class_cursors, 0, None);
        encoder.clear_buffer(&self.dispatch_args, 0, None);
        encoder.clear_buffer(&self.diagnostics, 0, None);
    }
}

fn storage_buffer<T>(
    device: &RenderDevice,
    label: &'static str,
    elements: u32,
    indirect: bool,
) -> Buffer {
    let mut usage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
    if indirect {
        usage |= BufferUsages::INDIRECT | BufferUsages::COPY_SRC;
    }
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: (elements.max(1) as u64).saturating_mul(size_of::<T>() as u64),
        usage,
        mapped_at_creation: false,
    })
}

pub(crate) fn prepare_visibility_buffers(
    mut commands: Commands,
    settings: Res<super::runtime::PrismShadingSettings>,
    rhi: Res<SceneRhi>,
    views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewVisibilityBuffer>,
    )>,
) {
    for (entity, camera, msaa, existing) in &views {
        let size = camera
            .physical_viewport_size
            .filter(|size| size.x != 0 && size.y != 0);
        if !settings.enable_visibility_buffer
            || msaa.is_some_and(|value| value.samples() != 1)
            || size.is_none()
        {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewVisibilityBuffer>();
            }
            continue;
        }
        let size = size.expect("nonzero viewport checked above");
        if existing.is_some_and(|buffer| {
            buffer.size == size
                && std::sync::Arc::ptr_eq(buffer.scene_color_rhi().device(), rhi.device())
        }) {
            continue;
        }

        // Keep the per-view allocation stable between frames, including views
        // with equal extents. Replacing the host bridge also invalidates this
        // owner, so table-local IDs never cross a device recovery.
        let storage = rhi::TextureUsages::STORAGE_BINDING | rhi::TextureUsages::TEXTURE_BINDING;
        let allocate =
            |label, format, usage| VisibilityTexture::new(&rhi, size, label, format, usage);
        commands.entity(entity).insert(ViewVisibilityBuffer {
            // Preserve the unsigned vec4 ABI, storage sampling and feedback
            // copy usages on the exact RHI allocation exported to Bevy.
            ids: allocate(
                "prism visibility IDs",
                rhi::TextureFormat::Rgba32Uint,
                storage | rhi::TextureUsages::RENDER_ATTACHMENT | rhi::TextureUsages::COPY_SRC,
            ),
            metadata: allocate(
                "prism visibility metadata",
                rhi::TextureFormat::Rgba32Uint,
                storage | rhi::TextureUsages::RENDER_ATTACHMENT | rhi::TextureUsages::COPY_SRC,
            ),
            // COPY_DST receives ping-pong post-processing results. Consumers
            // sample the same allocation; this is not an RHI mirror.
            scene_color: allocate(
                "prism scene color (HDR)",
                rhi::TextureFormat::Rgba16Float,
                storage | rhi::TextureUsages::COPY_DST,
            ),
            // Resolve always writes these exports, even when their optional
            // downstream consumers are disabled.
            ssr_env_specular: allocate(
                "prism SSR env specular export",
                rhi::TextureFormat::Rgba16Float,
                storage,
            ),
            ssr_spec_weight: allocate(
                "prism SSR spec weight export",
                rhi::TextureFormat::Rgba16Float,
                storage,
            ),
            // Transparent water writes its MRT over opaque resolve motion.
            motion_vectors: allocate(
                "prism motion vectors",
                rhi::TextureFormat::Rg16Float,
                storage | rhi::TextureUsages::RENDER_ATTACHMENT,
            ),
            ssgi_ambient: allocate(
                "prism SSGI ambient export",
                rhi::TextureFormat::Rgba16Float,
                storage,
            ),
            ssgi_albedo: allocate(
                "prism SSGI albedo export",
                rhi::TextureFormat::Rgba16Float,
                storage,
            ),
            world_restir_direct: allocate(
                "prism world-space ReSTIR direct export",
                rhi::TextureFormat::Rgba16Float,
                storage,
            ),
            size,
        });
    }
}

pub(crate) fn prepare_shading_buffers(
    mut commands: Commands,
    settings: Res<super::runtime::PrismShadingSettings>,
    materials: Res<super::super::material::runtime::RenderMaterialRegistry>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewShadingBuffers>,
    )>,
) {
    for (entity, camera, msaa, existing) in &views {
        if !settings.enable_visibility_buffer || msaa.is_some_and(|value| value.samples() != 1) {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewShadingBuffers>();
            }
            continue;
        }
        let Some(size) = camera.physical_viewport_size else {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewShadingBuffers>();
            }
            continue;
        };
        let bucket_table_version = materials.registry.execution_table_version();
        if existing.is_some_and(|buffers| {
            buffers.size == size && buffers.bucket_table_version == bucket_table_version
        }) {
            continue;
        }
        let bucket_table = match materials.registry.execution_tables() {
            Some(tables) => {
                match prism_render_shading::ResponseBucketTable::from_execution_tables(tables) {
                    Ok(table) => table,
                    Err(error) => {
                        bevy_log::error!("invalid material response bucket table: {error:?}");
                        commands.entity(entity).remove::<ViewShadingBuffers>();
                        continue;
                    }
                }
            }
            None => prism_render_shading::ResponseBucketTable::builtin(),
        };
        commands.entity(entity).insert(ViewShadingBuffers::new(
            &device,
            size,
            &bucket_table,
            bucket_table_version,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::{uuid::Uuid, AssetId};
    use bevy_camera::{CameraOutputMode, ClearColorConfig, MsaaWriteback};
    use bevy_ecs::{schedule::ScheduleLabel, system::RunSystemOnce};
    use bevy_math::UVec2;
    use bevy_platform::future::block_on;
    use bevy_render::renderer::{RenderAdapter, RenderQueue};
    use bevy_shader::{Shader, ShaderCache, ShaderCacheSource, ShaderDefVal};
    use core::ops::Deref;
    use prism_render_driver_wgpu::{require_hardware_gpu, WgpuDevice, WgpuInstance};
    use prism_render_shading::{VisibilityPixel, VisibilityPixelTargets};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::Arc;

    struct Host {
        _instance: WgpuInstance,
        adapter: RenderAdapter,
        device: RenderDevice,
        queue: RenderQueue,
    }

    impl Host {
        fn bridge(&self) -> SceneRhi {
            SceneRhi::from_bevy(&self.adapter, &self.device, &self.queue)
        }
    }

    fn host_or_skip() -> Option<Host> {
        let skip = |reason: &str| {
            assert!(!require_hardware_gpu(), "PRISM_REQUIRE_GPU=1: {reason}");
            eprintln!("SKIPPED visibility attachment GPU test: {reason}");
            None
        };
        let instance = WgpuInstance::new();
        let Some(adapter) = instance.request_adapter(wgpu::PowerPreference::HighPerformance, false)
        else {
            return skip("no adapter");
        };
        let info = adapter.get_info();
        if !matches!(
            info.device_type,
            wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::DiscreteGpu
        ) {
            return skip(&format!("not a physical GPU: {info:?}"));
        }
        let feature = wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        if !adapter.features().contains(feature) {
            return skip("Rg16Float storage needs TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES");
        }
        let storage = wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
        for (format, usage) in [
            (
                VISIBILITY_ID_FORMAT,
                storage | wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            ),
            (SCENE_COLOR_FORMAT, storage | wgpu::TextureUsages::COPY_DST),
            (
                MOTION_VECTOR_FORMAT,
                storage | wgpu::TextureUsages::RENDER_ATTACHMENT,
            ),
        ] {
            if !adapter
                .get_texture_format_features(format)
                .allowed_usages
                .contains(usage)
            {
                return skip(&format!("{format:?} lacks production usages {usage:?}"));
            }
        }
        eprintln!("visibility attachments REAL HARDWARE GPU: {info:?}");
        // Only the test host requests a device. The production prepare system
        // receives its adopted bridge, never a second native allocator.
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: feature,
            ..Default::default()
        }))
        .expect("physical GPU host device");
        Some(Host {
            _instance: instance,
            adapter: RenderAdapter::new(adapter),
            device: RenderDevice::new(device),
            queue: RenderQueue::new(queue),
        })
    }

    fn camera(size: UVec2) -> ExtractedCamera {
        ExtractedCamera {
            target: None,
            physical_viewport_size: Some(size),
            physical_target_size: Some(size),
            viewport: None,
            schedule: bevy_core_pipeline::Core3d.intern(),
            order: 0,
            output_mode: CameraOutputMode::default(),
            msaa_writeback: MsaaWriteback::default(),
            clear_color: ClearColorConfig::default(),
            sorted_camera_index_for_target: 0,
            exposure: 1.0,
            hdr: true,
        }
    }

    fn render_world(host: &Host) -> World {
        let mut world = World::new();
        world.insert_resource(host.bridge());
        world.insert_resource(super::super::runtime::PrismShadingSettings {
            enable_visibility_buffer: true,
            ..Default::default()
        });
        // Intentionally no RenderDevice resource: all nine allocations must
        // come from the adopted SceneRhi, not a native side path.
        world
    }

    fn prepare(world: &mut World) {
        world.run_system_once(prepare_visibility_buffers).unwrap();
    }

    fn attachments(buffer: &ViewVisibilityBuffer) -> [&VisibilityTexture; 9] {
        [
            &buffer.ids,
            &buffer.metadata,
            &buffer.scene_color,
            &buffer.ssr_env_specular,
            &buffer.ssr_spec_weight,
            &buffer.motion_vectors,
            &buffer.ssgi_ambient,
            &buffer.ssgi_albedo,
            &buffer.world_restir_direct,
        ]
    }

    type OwnerIds = [(rhi::TextureId, rhi::TextureViewId); 9];

    fn owner_ids(buffer: &ViewVisibilityBuffer) -> OwnerIds {
        attachments(buffer).map(|owner| (owner.texture.id(), owner.default_view.id()))
    }

    fn assert_same_table(buffer: &ViewVisibilityBuffer, table: &Arc<WgpuDevice>) {
        let formats = [
            VISIBILITY_ID_FORMAT,
            VISIBILITY_ID_FORMAT,
            SCENE_COLOR_FORMAT,
            SCENE_COLOR_FORMAT,
            SCENE_COLOR_FORMAT,
            MOTION_VECTOR_FORMAT,
            SCENE_COLOR_FORMAT,
            SCENE_COLOR_FORMAT,
            SCENE_COLOR_FORMAT,
        ];
        let ids = owner_ids(buffer);
        for (index, owner) in attachments(buffer).into_iter().enumerate() {
            assert!(Arc::ptr_eq(owner.texture.device(), table));
            assert!(Arc::ptr_eq(owner.default_view.device(), table));
            let native = owner.texture.texture();
            assert_eq!(table.wgpu_texture(owner.texture.id()), *native.deref());
            assert_eq!(
                table.wgpu_texture_view(owner.default_view.id()),
                *owner.default_view.view().deref()
            );
            assert_eq!(native.format(), formats[index]);
            assert_eq!(native.width(), buffer.size.x);
            assert_eq!(native.height(), buffer.size.y);
            for previous in &ids[..index] {
                assert_ne!(ids[index].0, previous.0);
                assert_ne!(ids[index].1, previous.1);
            }
        }
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC;
        for owner in [&buffer.ids, &buffer.metadata] {
            assert_eq!(owner.texture.texture().usage(), usage);
        }
        // The actual raster/feedback accessors must expose these exact objects.
        assert_eq!(
            buffer.attachments().0.id(),
            buffer.ids.default_view.view().id()
        );
        assert_eq!(
            buffer.attachments().1.id(),
            buffer.metadata.default_view.view().id()
        );
        assert_eq!(
            buffer.feedback_sources().0.id(),
            buffer.ids.texture.texture().id()
        );
        assert_eq!(
            buffer.feedback_sources().1.id(),
            buffer.metadata.texture.texture().id()
        );
    }

    fn assert_retired(table: &WgpuDevice, ids: OwnerIds) {
        for (texture, view) in ids {
            assert!(catch_unwind(AssertUnwindSafe(|| table.wgpu_texture(texture))).is_err());
            assert!(catch_unwind(AssertUnwindSafe(|| table.wgpu_texture_view(view))).is_err());
        }
    }

    #[test]
    fn nine_visibility_leases_share_host_table_reuse_resize_and_retire() {
        let Some(host) = host_or_skip() else { return };
        let mut world = render_world(&host);
        let table = Arc::clone(world.resource::<SceneRhi>().device());
        assert_eq!(table.wgpu_device(), host.device.wgpu_device());
        let size = UVec2::new(3, 2);
        let first = world.spawn(camera(size)).id();
        let second = world.spawn(camera(size)).id();
        prepare(&mut world);
        let first_buffer = world.get::<ViewVisibilityBuffer>(first).unwrap();
        assert_same_table(first_buffer, &table);
        let original = owner_ids(first_buffer);
        let second_buffer = world.get::<ViewVisibilityBuffer>(second).unwrap();
        assert_same_table(second_buffer, &table);
        let other = owner_ids(second_buffer);
        for (left, right) in original.into_iter().zip(other) {
            assert_ne!(left.0, right.0, "equal extents must not alias textures");
            assert_ne!(left.1, right.1, "equal extents must not alias views");
        }
        prepare(&mut world);
        assert_eq!(owner_ids(world.get(first).unwrap()), original);
        assert_eq!(owner_ids(world.get(second).unwrap()), other);

        world.entity_mut(first).insert(camera(UVec2::new(5, 3)));
        prepare(&mut world);
        let resized = world.get::<ViewVisibilityBuffer>(first).unwrap();
        assert_same_table(resized, &table);
        let resized_ids = owner_ids(resized);
        for (old, new) in original.into_iter().zip(resized_ids) {
            assert_ne!(old.0, new.0);
            assert_ne!(old.1, new.1);
        }
        assert_retired(&table, original);
        assert_eq!(owner_ids(world.get(second).unwrap()), other);

        // A new table on the SAME native host also invalidates all nine owners.
        // Numeric IDs may coincide across tables; compare native objects too.
        let natives = attachments(resized).map(|owner| owner.texture.texture().clone());
        world.insert_resource(host.bridge());
        let replacement = Arc::clone(world.resource::<SceneRhi>().device());
        assert!(!Arc::ptr_eq(&table, &replacement));
        assert_eq!(table.wgpu_device(), replacement.wgpu_device());
        prepare(&mut world);
        let replaced = world.get::<ViewVisibilityBuffer>(first).unwrap();
        assert_same_table(replaced, &replacement);
        assert_same_table(world.get(second).unwrap(), &replacement);
        for (old, owner) in natives.iter().zip(attachments(replaced)) {
            assert_ne!(*old.deref(), *owner.texture.texture().deref());
            let _surviving_native_view = old.create_view(&Default::default());
        }
        assert_retired(&table, resized_ids);
        assert_retired(&table, other);

        // Every admission failure removes both texture AND view entries.
        for invalid in [None, Some(UVec2::new(0, 3)), Some(UVec2::new(3, 0))] {
            let ids = owner_ids(world.get(first).unwrap());
            world
                .get_mut::<ExtractedCamera>(first)
                .unwrap()
                .physical_viewport_size = invalid;
            prepare(&mut world);
            assert!(world.get::<ViewVisibilityBuffer>(first).is_none());
            assert_retired(&replacement, ids);
            world.entity_mut(first).insert(camera(size));
            prepare(&mut world);
            assert_same_table(world.get(first).unwrap(), &replacement);
        }
        let ids = owner_ids(world.get(first).unwrap());
        world.entity_mut(first).insert(Msaa::Sample4);
        prepare(&mut world);
        assert!(world.get::<ViewVisibilityBuffer>(first).is_none());
        assert_retired(&replacement, ids);
        world.entity_mut(first).insert(Msaa::Off);
        prepare(&mut world);
        let ids = owner_ids(world.get(first).unwrap());
        let other = owner_ids(world.get(second).unwrap());
        world
            .resource_mut::<super::super::runtime::PrismShadingSettings>()
            .enable_visibility_buffer = false;
        prepare(&mut world);
        assert!(world.get::<ViewVisibilityBuffer>(first).is_none());
        assert!(world.get::<ViewVisibilityBuffer>(second).is_none());
        assert_retired(&replacement, ids);
        assert_retired(&replacement, other);

        world
            .resource_mut::<super::super::runtime::PrismShadingSettings>()
            .enable_visibility_buffer = true;
        prepare(&mut world);
        let ids = owner_ids(world.get(first).unwrap());
        world.despawn(first);
        assert_retired(&replacement, ids);
        let other = owner_ids(world.get(second).unwrap());
        // The resident leases, not the SceneRhi resource, retain the table.
        world.remove_resource::<SceneRhi>();
        let weak = Arc::downgrade(&replacement);
        drop(replacement);
        assert_eq!(weak.strong_count(), 18, "nine texture/view lease pairs");
        let table = weak.upgrade().unwrap();
        assert_same_table(world.get(second).unwrap(), &table);
        drop(world);
        assert_retired(&table, other);
        drop(table);
        assert!(weak.upgrade().is_none());
    }

    fn integer_pixel() -> VisibilityPixel {
        VisibilityPixel {
            scene_index: 0x0100_0001,
            scene_generation: 0x8000_0001,
            primitive_id: u32::MAX,
            geometry_lod_or_cluster: 0xdead_beef,
            material_index: 0x8100_0001,
            material_generation: 0xffff_fffe,
            barycentrics_unorm16: 0x1234_abcd,
            coverage_and_flags: 0xc000_00a5,
        }
    }

    #[test]
    fn visibility_integer_targets_preserve_channel_order_and_all_u32_bits() {
        let pixel = integer_pixel();
        let targets = pixel.targets();
        assert_eq!(
            targets.ids,
            [0x0100_0001, 0x8000_0001, u32::MAX, 0xdead_beef]
        );
        // GPU metadata.z is coverage/selector flags, NOT the preceding CPU
        // field barycentrics_unorm16. A raw cast of VisibilityPixel is wrong.
        assert_eq!(
            targets.metadata,
            [0x8100_0001, 0xffff_fffe, 0xc000_00a5, 0x1234_abcd]
        );
        assert_eq!(VisibilityPixel::from_targets(targets), pixel);
        assert_eq!(
            VisibilityPixel::from_targets(VisibilityPixel::INVALID.targets()),
            VisibilityPixel::INVALID
        );
    }

    #[test]
    fn retired_visibility_integer_leases_leave_host_handles_usable_for_exact_readback() {
        let Some(host) = host_or_skip() else { return };
        let mut world = render_world(&host);
        let entity = world.spawn(camera(UVec2::ONE)).id();
        prepare(&mut world);
        let table = Arc::clone(world.resource::<SceneRhi>().device());
        let owner = world.get::<ViewVisibilityBuffer>(entity).unwrap();
        let ids = owner_ids(owner);
        let textures = [
            owner.feedback_sources().0.clone(),
            owner.feedback_sources().1.clone(),
        ];
        let views = [owner.attachments().0.clone(), owner.attachments().1.clone()];
        world.despawn(entity);
        assert_retired(&table, ids);

        // Attachment-boundary evidence only, not the production raster shader:
        // clear exact integers through the surviving native views, then copy
        // and decode both physical targets after the RHI owners have retired.
        let expected = integer_pixel();
        let targets = expected.targets();
        let colors = [targets.ids, targets.metadata].map(|[r, g, b, a]| wgpu::Color {
            r: f64::from(r),
            g: f64::from(g),
            b: f64::from(b),
            a: f64::from(a),
        });
        let color_attachments = std::array::from_fn::<_, 2, _>(|index| {
            Some(wgpu::RenderPassColorAttachment {
                view: views[index].deref(),
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(colors[index]),
                    store: wgpu::StoreOp::Store,
                },
            })
        });
        let staging = host
            .device
            .wgpu_device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("retired visibility u32 readback"),
                size: 512,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
        let mut encoder = host.device.create_command_encoder(&Default::default());
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &color_attachments,
                ..Default::default()
            });
        }
        for (index, texture) in textures.iter().enumerate() {
            encoder.copy_texture_to_buffer(
                texture.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: index as u64 * 256,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
        }
        host.queue.submit([encoder.finish()]);
        let (send, receive) = std::sync::mpsc::sync_channel(1);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                send.send(result).unwrap()
            });
        host.device
            .wgpu_device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        receive.recv().unwrap().unwrap();
        let bytes = staging.slice(..).get_mapped_range().unwrap();
        let channels = |offset| {
            std::array::from_fn(|channel| {
                let start = offset + channel * 4;
                u32::from_le_bytes(bytes[start..start + 4].try_into().unwrap())
            })
        };
        let actual = VisibilityPixelTargets {
            ids: channels(0),
            metadata: channels(256),
        };
        assert_eq!(actual, targets);
        assert_eq!(VisibilityPixel::from_targets(actual), expected);
        drop(bytes);
        staging.unmap();
    }

    fn load_source(
        _: &(),
        source: ShaderCacheSource,
        _: &bevy_shader::ValidateShader,
    ) -> Result<String, bevy_shader::ShaderCacheError> {
        match source {
            ShaderCacheSource::Wgsl(source) => Ok(source),
            ShaderCacheSource::SpirV(_) => unreachable!("visibility raster shader is WESL"),
        }
    }

    #[test]
    fn formats_cover_the_two_target_visibility_abi() {
        assert_eq!(VISIBILITY_ID_FORMAT, TextureFormat::Rgba32Uint);
        assert_eq!(VISIBILITY_ID_FORMAT.block_copy_size(None), Some(16));
        assert_eq!(2 * 16, size_of::<prism_render_shading::VisibilityPixel>());
    }

    #[test]
    fn visibility_raster_wesl_compiles_with_barycentric_contract() {
        let shader_id = AssetId::Uuid {
            uuid: Uuid::from_u128(0x5052_4953_4d56_4953_5241_5354_4552_0001),
        };
        let source = include_str!("../shaders/visibility_raster.wesl");
        let source = source[source.find("struct GpuSceneInstance").unwrap()..]
            .replace(
                "bevy_render::utils::decompress_vertex_position",
                "decompress_vertex_position",
            )
            .replace(
                "@builtin(barycentric) barycentrics: vec3<f32>",
                "@location(6) barycentrics: vec3<f32>",
            );
        let stubs = r#"
fn affine3_to_square(value: mat3x4<f32>) -> mat4x4<f32> {
    return mat4x4<f32>(
        vec4<f32>(value[0]), vec4<f32>(value[1]), vec4<f32>(value[2]),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
}
fn position_world_to_clip(position: vec3<f32>) -> vec4<f32> { return vec4(position, 1.0); }
fn decompress_vertex_position(position: vec4<f32>, center: vec3<f32>, half_extents: vec3<f32>) -> vec3<f32> {
    return position.xyz;
}
"#;
        let mut cache = ShaderCache::new((), load_source);
        cache.set_shader(
            shader_id,
            Shader::from_wesl(
                format!("enable wgpu_binding_array;\n{stubs}{source}"),
                "shaders/prism_visibility_raster.wesl",
            ),
        );
        cache
            .get(0, shader_id, &[])
            .unwrap_or_else(|error| panic!("visibility raster shader failed: {error}"));

        let bindless_id = AssetId::Uuid {
            uuid: Uuid::from_u128(0x5052_4953_4d56_4953_5241_5354_4552_0002),
        };
        cache.set_shader(
            bindless_id,
            Shader::from_wesl(
                format!("enable wgpu_binding_array;\n{stubs}{source}"),
                "shaders/prism_visibility_raster_bindless.wesl",
            ),
        );
        let defs = [
            ShaderDefVal::Bool("MATERIAL_BINDLESS".into(), true),
            ShaderDefVal::Bool("REGION_ILLUMINATION".into(), true),
            ShaderDefVal::Bool("ILLUMINATION_BLEND".into(), true),
            ShaderDefVal::Bool("VERTEX_UVS".into(), true),
        ];
        cache
            .get(1, bindless_id, &defs)
            .unwrap_or_else(|error| panic!("bindless visibility raster shader failed: {error}"));

        assert!(source.contains("visibility_material_texture_record_is_valid(record)"));
        assert!(source.contains(
            "material_texture_generation_table[record.sampler_index] != record.generation"
        ));
        assert!(source.contains("Only a validated row is terminal"));
    }
}
