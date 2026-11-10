//! Creator-device ownership and completion-retained binding leases.

use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

use prism_render_architecture::texture_streaming::{
    AtlasGeometry, AtlasPageLayout, AtlasTileFormat,
};
use prism_render_driver::{
    BufferDescriptor, BufferId, BufferUsages, Extent3d, FilterMode, FrameSync, Instance,
    RenderDevice, SamplerDescriptor, SamplerId, TextureDescriptor, TextureFormat, TextureId,
    TextureState, TextureUsages, TextureViewDescriptor, TextureViewDimension, TextureViewId,
};
use prism_render_graph::PreparedGraph;
use prism_render_material::MaterialVtFormat;
use prism_render_runtime::{FrameSubmission, RenderRuntime, SubmissionError};

use super::NativeVtError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeVtConfig {
    pub page_size: u32,
    pub gutter_px: u32,
    pub tiles_per_row: u32,
    pub atlas_layers: u32,
    pub max_entries: u32,
    pub format: MaterialVtFormat,
}

impl NativeVtConfig {
    /// Bounded startup pool: 64 physical tiles and page-table entries in a
    /// single-layer, linear RGBA8 atlas. Each of the two slices has a 1088px
    /// square atlas. This is a capacity policy, not an enabled publication;
    /// sources must match its page size and format before staging.
    pub const DEFAULT: Self = Self {
        page_size: 128,
        gutter_px: 4,
        tiles_per_row: 8,
        atlas_layers: 1,
        max_entries: 64,
        format: MaterialVtFormat::Rgba8Unorm,
    };

    /// Construct the default policy in const contexts without allocating GPU
    /// resources. `NativeMaterialVt::new` validates against the actual device.
    #[must_use]
    pub const fn new() -> Self {
        Self::DEFAULT
    }

    pub(super) fn geometry(self) -> Result<AtlasGeometry, NativeVtError> {
        let layout = AtlasPageLayout::new(
            self.page_size,
            self.gutter_px,
            AtlasTileFormat {
                block_extent_px: 1,
                bytes_per_block: self.format.bytes_per_texel(),
            },
        )
        .ok_or(NativeVtError::InvalidConfig)?;
        AtlasGeometry::for_page_layout(self.tiles_per_row, layout)
            .ok_or(NativeVtError::InvalidConfig)
    }

    pub(super) fn capacity(self) -> Result<u32, NativeVtError> {
        self.tiles_per_row
            .checked_mul(self.tiles_per_row)
            .and_then(|n| n.checked_mul(self.atlas_layers))
            .filter(|n| *n > 0)
            .ok_or(NativeVtError::InvalidConfig)
    }

    pub(super) fn validate(self, device: &impl RenderDevice) -> Result<(), NativeVtError> {
        let geometry = self.geometry()?;
        self.capacity()?;
        let limits = &device.capabilities().limits;
        let edge = geometry
            .tile_extent_px()
            .checked_mul(self.tiles_per_row)
            .ok_or(NativeVtError::InvalidConfig)?;
        let table_bytes = u64::from(self.max_entries.max(1)) * 16;
        if edge > limits.max_texture_dimension_2d
            || self.atlas_layers > limits.max_texture_array_layers
            || table_bytes > limits.max_buffer_size
            || table_bytes > limits.max_storage_buffer_binding_size
            || 32 > limits.max_buffer_size
            || 32 > limits.max_storage_buffer_binding_size
        {
            return Err(NativeVtError::InvalidConfig);
        }
        texture_format(self.format)?;
        Ok(())
    }
}

impl Default for NativeVtConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Shipping `material_virtual_texture.wesl` storage ABI (eight u32 words).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NativeVtGpuConfig {
    pub enabled: u32,
    pub entry_count: u32,
    pub page_size: u32,
    pub tile_extent_px: u32,
    pub tiles_per_row: u32,
    pub atlas_layers: u32,
    pub gutter_px: u32,
    pub coarsest_mip: u32,
}

pub(super) struct GpuSlice<D: RenderDevice> {
    pub(super) device: Arc<D>,
    pub(super) generation: u64,
    texture: Option<TextureId>,
    view: Option<TextureViewId>,
    table: Option<BufferId>,
    config: Option<BufferId>,
    sampler: Option<SamplerId>,
    page_size: u32,
    gutter_px: u32,
    tiles_per_row: u32,
    atlas_layers: u32,
    format: MaterialVtFormat,
    // Increment *before* touching the queue. Only an invoked completion
    // callback (not dropping it) discharges an obligation. Device-loss callback
    // abandonment therefore forbids both reuse and explicit destruction.
    unconfirmed: AtomicUsize,
    // Device loss is monotonic for this allocation set. A late completion may
    // discharge its fence, but must never make lost-device IDs destroyable.
    abandoned: Arc<AtomicBool>,
}

impl<D: RenderDevice + Send + Sync + 'static> GpuSlice<D> {
    pub(super) fn new(
        device: Arc<D>,
        generation: u64,
        config: NativeVtConfig,
        abandoned: Arc<AtomicBool>,
    ) -> Result<Arc<Self>, NativeVtError> {
        config.validate(&*device)?;
        let geometry = config.geometry()?;
        let edge = geometry.tile_extent_px() * config.tiles_per_row;
        // Options give partial resource creation unwind-safe cleanup.
        let mut slice = Self {
            device,
            generation,
            texture: None,
            view: None,
            table: None,
            config: None,
            sampler: None,
            page_size: config.page_size,
            gutter_px: config.gutter_px,
            tiles_per_row: config.tiles_per_row,
            atlas_layers: config.atlas_layers,
            format: config.format,
            unconfirmed: AtomicUsize::new(0),
            abandoned,
        };
        slice.texture = Some(slice.device.create_texture(&TextureDescriptor::new_2d(
            Extent3d {
                width: edge,
                height: edge,
                depth_or_array_layers: config.atlas_layers,
            },
            texture_format(config.format)?,
            TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        )));
        slice.view = Some(slice.device.create_texture_view(
            slice.texture(),
            &TextureViewDescriptor {
                dimension: TextureViewDimension::D2Array,
                ..Default::default()
            },
        ));
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        slice.table = Some(slice.device.create_buffer(&BufferDescriptor::new(
            u64::from(config.max_entries.max(1)) * 16,
            usage,
        )));
        slice.config = Some(
            slice
                .device
                .create_buffer(&BufferDescriptor::new(32, usage)),
        );
        slice.sampler = Some(slice.device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        }));
        Ok(Arc::new(slice))
    }

    pub(super) fn texture(&self) -> TextureId {
        self.texture.expect("allocated VT texture")
    }
    pub(super) fn table(&self) -> BufferId {
        self.table.expect("allocated VT table")
    }
    pub(super) fn config(&self) -> BufferId {
        self.config.expect("allocated VT config")
    }
    pub(super) fn idle(slice: &Arc<Self>) -> bool {
        !slice.abandoned.load(Ordering::Acquire)
            && Arc::strong_count(slice) == 1
            && slice.unconfirmed.load(Ordering::Acquire) == 0
    }
    pub(super) fn begin_work(&self) {
        self.unconfirmed
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
            .expect("bounded VT completion obligations");
    }
    fn complete_work(&self) {
        self.unconfirmed.fetch_sub(1, Ordering::AcqRel);
    }

    pub(super) fn on_completion(self: &Arc<Self>, queue: &impl FrameSync) -> Arc<AtomicBool> {
        let complete = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&complete);
        let owner = Arc::clone(self);
        queue.on_submitted_work_done(Box::new(move || {
            owner.complete_work();
            flag.store(true, Ordering::Release);
        }));
        complete
    }
}

impl<D: RenderDevice> GpuSlice<D> {
    pub(super) fn abandon(&self) {
        self.abandoned.store(true, Ordering::Release);
    }
}

impl<D: RenderDevice> Drop for GpuSlice<D> {
    fn drop(&mut self) {
        if self.abandoned.load(Ordering::Acquire) || self.unconfirmed.load(Ordering::Acquire) != 0 {
            // Unconfirmed GPU work belongs to the old device's teardown; never
            // send destroys or reuse IDs merely because a callback was dropped.
            return;
        }
        if let Some(id) = self.sampler.take() {
            self.device.destroy_sampler(id);
        }
        if let Some(id) = self.config.take() {
            self.device.destroy_buffer(id);
        }
        if let Some(id) = self.table.take() {
            self.device.destroy_buffer(id);
        }
        if let Some(id) = self.view.take() {
            self.device.destroy_texture_view(id);
        }
        if let Some(id) = self.texture.take() {
            self.device.destroy_texture(id);
        }
    }
}

/// Own this lease beside the bind group. Keeping only the numeric IDs cannot
/// protect an inactive slice against overwrite by the next publication.
pub struct NativeVtBinding<D: RenderDevice> {
    pub(super) owner: Arc<GpuSlice<D>>,
}

/// Erased, immutable VT resource lease that can be carried by a material frame
/// bundle without exposing the concrete backend device type.
#[derive(Clone)]
pub struct MaterialVtLeaseSnapshot {
    owner: Arc<dyn std::any::Any + Send + Sync>,
    device: Arc<dyn RenderDevice + Send + Sync>,
    page_table: BufferId,
    config: BufferId,
    atlas_view: TextureViewId,
    sampler: SamplerId,
    device_generation: u64,
    publication_generation: u64,
}

impl MaterialVtLeaseSnapshot {
    #[cfg(test)]
    pub(crate) fn for_test(
        owner: Arc<dyn std::any::Any + Send + Sync>,
        device: Arc<dyn RenderDevice + Send + Sync>,
        page_table: BufferId,
        config: BufferId,
        atlas_view: TextureViewId,
        sampler: SamplerId,
        device_generation: u64,
        publication_generation: u64,
    ) -> Self {
        Self {
            owner,
            device,
            page_table,
            config,
            atlas_view,
            sampler,
            device_generation,
            publication_generation,
        }
    }

    #[must_use]
    pub fn page_table(&self) -> BufferId {
        self.page_table
    }

    #[must_use]
    pub fn config(&self) -> BufferId {
        self.config
    }

    #[must_use]
    pub fn atlas_view(&self) -> TextureViewId {
        self.atlas_view
    }

    #[must_use]
    pub fn sampler(&self) -> SamplerId {
        self.sampler
    }

    #[must_use]
    pub const fn device_generation(&self) -> u64 {
        self.device_generation
    }

    #[must_use]
    pub const fn publication_generation(&self) -> u64 {
        self.publication_generation
    }

    #[must_use]
    pub fn external_owner(&self) -> Arc<dyn std::any::Any + Send + Sync> {
        Arc::clone(&self.owner)
    }

    pub(crate) fn same_device(&self, material: &crate::material::MaterialBufferSnapshot) -> bool {
        Arc::ptr_eq(&self.device, &material.device_identity())
    }
}

impl<D: RenderDevice> Clone for NativeVtBinding<D> {
    fn clone(&self) -> Self {
        Self {
            owner: Arc::clone(&self.owner),
        }
    }
}

#[derive(Debug)]
pub enum NativeVtSubmitError {
    Vt(NativeVtError),
    Submission(SubmissionError),
}

impl core::fmt::Display for NativeVtSubmitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Vt(error) => error.fmt(f),
            Self::Submission(error) => write!(f, "native material VT consumer: {error:?}"),
        }
    }
}

impl std::error::Error for NativeVtSubmitError {}

impl<D: RenderDevice + Send + Sync + 'static> NativeVtBinding<D> {
    pub fn page_table(&self) -> BufferId {
        self.owner.table()
    }
    pub fn config(&self) -> BufferId {
        self.owner.config()
    }
    pub fn atlas_view(&self) -> TextureViewId {
        self.owner.view.expect("allocated VT view")
    }
    pub fn sampler(&self) -> SamplerId {
        self.owner.sampler.expect("allocated VT sampler")
    }
    pub fn device_generation(&self) -> u64 {
        self.owner.generation
    }

    /// Captures the exact creator-device VT allocation for a material frame.
    #[must_use]
    pub fn frame_lease(&self, publication_generation: u64) -> MaterialVtLeaseSnapshot {
        MaterialVtLeaseSnapshot {
            owner: self.owner.clone(),
            device: self.owner.device.clone(),
            page_table: self.page_table(),
            config: self.config(),
            atlas_view: self.atlas_view(),
            sampler: self.sampler(),
            device_generation: self.device_generation(),
            publication_generation,
        }
    }

    /// Describes the atlas from this same immutable publication snapshot.
    pub fn atlas_import(&self) -> crate::frame::FrameGraphTextureImport {
        let geometry = AtlasPageLayout::new(
            self.owner.page_size,
            self.owner.gutter_px,
            AtlasTileFormat {
                block_extent_px: 1,
                bytes_per_block: self.owner.format.bytes_per_texel(),
            },
        )
        .and_then(|layout| AtlasGeometry::for_page_layout(self.owner.tiles_per_row, layout))
        .expect("published VT atlas geometry");
        let edge = geometry.tile_extent_px() * self.owner.tiles_per_row;
        crate::frame::FrameGraphTextureImport {
            texture: self.owner.texture(),
            view: self.atlas_view(),
            desc: prism_render_graph::TextureDesc {
                size: prism_render_graph::SizeClass::Absolute {
                    width: edge,
                    height: edge,
                    depth_or_array_layers: self.owner.atlas_layers,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: prism_render_driver::TextureDimension::D2,
                format: texture_format(self.owner.format).expect("validated VT format"),
                usage: TextureUsages::TEXTURE_BINDING,
            },
            entry_state: TextureState::initial(),
            exit_state: TextureState::shader_read(
                prism_render_driver::PipelineStages::FRAGMENT_SHADER,
            ),
            mip_levels: 1,
            array_layers: self.owner.atlas_layers,
        }
    }

    /// Type-erased creator-device lease retained by every consuming output graph.
    pub fn external_owner(&self) -> Arc<dyn std::any::Any + Send + Sync>
    where
        D: 'static,
    {
        self.owner.clone()
    }

    /// The submit path for a graph sampling this lease. The callback is attached
    /// to the same runtime that actually submits the consumer; no planned frame
    /// number, caller assertion, or upload fence stands in for consumer completion.
    pub fn execute_prepared<I: Instance<Device = D> + 'static>(
        self,
        runtime: &RenderRuntime<I>,
        prepared: PreparedGraph,
        imports: &[FrameSubmission<D>],
    ) -> Result<FrameSubmission<D>, NativeVtSubmitError>
    where
        I::Queue: FrameSync,
    {
        check_runtime(&self.owner, runtime).map_err(NativeVtSubmitError::Vt)?;
        self.owner.begin_work();
        match runtime.execute_prepared_owned(prepared, imports) {
            Ok(frame) => {
                self.owner.on_completion(runtime.queue());
                Ok(frame)
            }
            Err(error) => {
                // Runtime errors return before realization/submission.
                self.owner.complete_work();
                Err(NativeVtSubmitError::Submission(error))
            }
        }
    }
}

pub(super) fn check_runtime<D: RenderDevice, I: Instance<Device = D> + 'static>(
    slice: &GpuSlice<D>,
    runtime: &RenderRuntime<I>,
) -> Result<(), NativeVtError> {
    if slice.generation != runtime.device_generation()
        || !Arc::ptr_eq(&slice.device, &runtime.device_handle())
    {
        return Err(NativeVtError::DeviceMismatch);
    }
    if slice.abandoned.load(Ordering::Acquire) {
        return Err(NativeVtError::DeviceLost);
    }
    if runtime.device_lost().is_some() {
        slice.abandon();
        return Err(NativeVtError::DeviceLost);
    }
    Ok(())
}

pub(super) fn check_owner_runtime<D: RenderDevice, I: Instance<Device = D> + 'static>(
    slices: &[Arc<GpuSlice<D>>; 2],
    runtime: &RenderRuntime<I>,
) -> Result<(), NativeVtError> {
    match check_runtime(&slices[0], runtime) {
        Err(NativeVtError::DeviceLost) => {
            for slice in slices {
                slice.abandon();
            }
            Err(NativeVtError::DeviceLost)
        }
        result => result,
    }
}

fn texture_format(format: MaterialVtFormat) -> Result<TextureFormat, NativeVtError> {
    Ok(match format {
        MaterialVtFormat::R8Unorm => TextureFormat::R8Unorm,
        MaterialVtFormat::Rg8Unorm => TextureFormat::Rg8Unorm,
        MaterialVtFormat::Rgba8Unorm => TextureFormat::Rgba8Unorm,
        MaterialVtFormat::Rgba8Srgb => TextureFormat::Rgba8UnormSrgb,
        MaterialVtFormat::Rgba16Float => TextureFormat::Rgba16Float,
        MaterialVtFormat::Rgba32Float => return Err(NativeVtError::UnsupportedFormat),
    })
}
