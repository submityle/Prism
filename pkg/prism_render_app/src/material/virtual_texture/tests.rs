//! Deterministic native ownership/publication tests. The delayed queue invokes
//! real RHI completion closures on explicit test progress; submission alone and
//! dropping a closure deliberately do not confirm completion. Not GPU evidence.

#[path = "scheduler_tests.rs"]
mod scheduler_tests;

use std::{
    any::Any,
    collections::{BTreeMap, BTreeSet},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};

use prism_asset::{AssetId, AssetIndex};
use prism_image::Image;
use prism_render_architecture::{
    abi::GenerationalHandle,
    texture_streaming::{GpuPageTable, TexturePageKey},
};
use prism_render_driver as rhi;
use prism_render_driver_null::{NullAdapter, NullDevice, NullInstance, NullSurface};
use prism_render_graph::RenderGraph;
use prism_render_material::{
    build_material_vt_atlas_tile, fallback_material_record, GpuMaterialTexture,
    MaterialVtAtlasConfig, MaterialVtFormat, MaterialVtSourceDescriptor, MaterialVtSourceSnapshot,
    NO_VIRTUAL_TEXTURE, PORTABLE_TEXTURE_ATLAS,
};
use prism_render_runtime::RenderRuntime;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use rhi::{FrameSync, Instance, RenderDevice, RenderQueue};

use super::*;
use crate::{
    image::GpuImage,
    material::{BindlessSlot, RenderMaterials},
    render_asset::ExtractedAssets,
};

struct TestDevice {
    inner: NullDevice,
    lost: AtomicBool,
}

macro_rules! forward_resources {
    ($($create:ident($descriptor:ty) -> $id:ty, $destroy:ident;)*) => { $(
        fn $create(&self, descriptor: &$descriptor) -> $id { self.inner.$create(descriptor) }
        fn $destroy(&self, id: $id) { self.inner.$destroy(id); }
    )* };
}

impl RenderDevice for TestDevice {
    fn capabilities(&self) -> &rhi::DeviceCapabilities {
        self.inner.capabilities()
    }
    fn take_device_lost(&self) -> Option<rhi::DeviceLostReason> {
        self.lost
            .swap(false, Ordering::AcqRel)
            .then_some(rhi::DeviceLostReason::Destroyed)
    }
    fn create_texture_view(
        &self,
        texture: rhi::TextureId,
        descriptor: &rhi::TextureViewDescriptor,
    ) -> rhi::TextureViewId {
        self.inner.create_texture_view(texture, descriptor)
    }
    fn destroy_texture_view(&self, id: rhi::TextureViewId) {
        self.inner.destroy_texture_view(id);
    }
    forward_resources! {
        create_buffer(rhi::BufferDescriptor) -> rhi::BufferId, destroy_buffer;
        create_texture(rhi::TextureDescriptor) -> rhi::TextureId, destroy_texture;
        create_sampler(rhi::SamplerDescriptor) -> rhi::SamplerId, destroy_sampler;
        create_shader_module(rhi::ShaderModuleDescriptor) -> rhi::ShaderModuleId, destroy_shader_module;
        create_bind_group_layout(rhi::BindGroupLayoutDescriptor) -> rhi::BindGroupLayoutId, destroy_bind_group_layout;
        create_bind_group(rhi::BindGroupDescriptor) -> rhi::BindGroupId, destroy_bind_group;
        create_pipeline_layout(rhi::PipelineLayoutDescriptor) -> rhi::PipelineLayoutId, destroy_pipeline_layout;
        create_render_pipeline(rhi::RenderPipelineDescriptor) -> rhi::RenderPipelineId, destroy_render_pipeline;
        create_compute_pipeline(rhi::ComputePipelineDescriptor) -> rhi::ComputePipelineId, destroy_compute_pipeline;
    }
}

type Callback = Box<dyn FnOnce() + Send + 'static>;

#[derive(Default)]
struct Progress {
    submitted: u64,
    callbacks: Vec<(u64, Callback)>,
    buffers: Vec<(rhi::BufferId, u64, Vec<u8>)>,
    textures: Vec<(rhi::TextureWrite, Vec<u8>, rhi::ImageDataLayout)>,
}

#[derive(Default)]
struct DeferredQueue {
    progress: Mutex<Progress>,
    abandoned: AtomicBool,
    panic_after_texture_write: AtomicBool,
}

impl DeferredQueue {
    fn complete(&self) {
        let callbacks = std::mem::take(&mut self.progress.lock().unwrap().callbacks);
        for (_, callback) in callbacks {
            callback();
        }
    }
    fn abandon(&self) {
        self.abandoned.store(true, Ordering::Release);
        let callbacks = std::mem::take(&mut self.progress.lock().unwrap().callbacks);
        drop(callbacks);
    }
    fn writes(&self) -> usize {
        let progress = self.progress.lock().unwrap();
        progress.buffers.len() + progress.textures.len()
    }
    fn buffer_bytes(&self, id: rhi::BufferId) -> Vec<u8> {
        self.progress
            .lock()
            .unwrap()
            .buffers
            .iter()
            .rev()
            .find(|(buffer, _, _)| *buffer == id)
            .unwrap()
            .2
            .clone()
    }
}

impl RenderQueue for DeferredQueue {
    fn write_buffer(&self, id: rhi::BufferId, offset: u64, bytes: &[u8]) {
        self.progress
            .lock()
            .unwrap()
            .buffers
            .push((id, offset, bytes.to_vec()));
    }
    fn write_texture(
        &self,
        destination: rhi::TextureWrite,
        bytes: &[u8],
        layout: rhi::ImageDataLayout,
    ) {
        self.progress
            .lock()
            .unwrap()
            .textures
            .push((destination, bytes.to_vec(), layout));
        if self.panic_after_texture_write.swap(false, Ordering::AcqRel) {
            panic!("injected backend failure after accepting a texture write");
        }
    }
    fn submit(&self, _: &[rhi::CommandBuffer]) {
        self.progress.lock().unwrap().submitted += 1;
    }
}

impl FrameSync for DeferredQueue {
    fn submit_tracked(&self, commands: &[rhi::CommandBuffer]) -> rhi::CompletionToken {
        self.submit(commands);
        Box::new(())
    }
    fn wait_for(&self, _: &(dyn Any + Send + Sync)) -> bool {
        false
    }
    fn wait_idle(&self) -> bool {
        if self.abandoned.load(Ordering::Acquire) {
            return false;
        }
        self.complete();
        true
    }
    fn on_submitted_work_done(&self, callback: Callback) {
        let mut progress = self.progress.lock().unwrap();
        let submitted = progress.submitted;
        progress.callbacks.push((submitted, callback));
    }
}

struct DeferredInstance;

impl Instance for DeferredInstance {
    type Adapter = NullAdapter;
    type Device = TestDevice;
    type Queue = DeferredQueue;
    type Surface = NullSurface;
    fn enumerate_adapters(&self) -> Vec<NullAdapter> {
        NullInstance::new().enumerate_adapters()
    }
    fn request_adapter(&self, options: &rhi::RequestAdapterOptions) -> Option<NullAdapter> {
        NullInstance::new().request_adapter(options)
    }
    fn request_adapter_for_surface(
        &self,
        options: &rhi::RequestAdapterOptions,
        surface: &NullSurface,
    ) -> Option<NullAdapter> {
        NullInstance::new().request_adapter_for_surface(options, surface)
    }
    fn create_device(
        &self,
        adapter: &NullAdapter,
        descriptor: &rhi::DeviceDescriptor,
    ) -> Result<(TestDevice, DeferredQueue), rhi::DeviceError> {
        let (inner, _) = NullInstance::new().create_device(adapter, descriptor)?;
        Ok((
            TestDevice {
                inner,
                lost: AtomicBool::new(false),
            },
            DeferredQueue::default(),
        ))
    }
    fn create_surface<W>(
        &self,
        window: Arc<W>,
        device: &TestDevice,
        _: &DeferredQueue,
    ) -> Result<NullSurface, rhi::SurfaceError>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        NullInstance::new().create_surface(
            window,
            &device.inner,
            &prism_render_driver_null::NullQueue::new(),
        )
    }
}

fn runtime() -> RenderRuntime<DeferredInstance> {
    RenderRuntime::headless(DeferredInstance, rhi::Extent3d::new_2d(4, 4)).unwrap()
}
fn config() -> NativeVtConfig {
    NativeVtConfig {
        page_size: 2,
        gutter_px: 1,
        tiles_per_row: 2,
        atlas_layers: 2,
        max_entries: 8,
        format: MaterialVtFormat::Rgba8Unorm,
    }
}
fn image_id(index: u32) -> AssetId<Image> {
    AssetId::new(AssetIndex::from_parts(index, 1))
}
fn image(value: u8) -> Image {
    Image::new(
        rhi::Extent3d::new_2d(1, 1),
        rhi::TextureFormat::Rgba8Unorm,
        vec![value; 4],
    )
    .unwrap()
}
fn unfilterable_image() -> Image {
    Image::new(
        rhi::Extent3d::new_2d(1, 1),
        rhi::TextureFormat::Rgba32Float,
        vec![0; 16],
    )
    .unwrap()
}
fn request(texture_id: u32) -> NativeVtSourceRequest {
    NativeVtSourceRequest {
        texture_id,
        page_size: 2,
    }
}
fn snapshot(
    texture: u32,
    width: u32,
    layers: u16,
    format: MaterialVtFormat,
) -> MaterialVtSourceSnapshot {
    let mip_count = width.ilog2() + 1;
    let descriptor =
        MaterialVtSourceDescriptor::new(texture, width, width, layers, 2, mip_count as u8, format)
            .unwrap();
    let mut bytes = Vec::new();
    for mip in 0..mip_count {
        let edge = (width >> mip).max(1);
        bytes.extend(std::iter::repeat_n(
            (mip + 1) as u8,
            (edge * edge * u32::from(layers) * format.bytes_per_texel()) as usize,
        ));
    }
    MaterialVtSourceSnapshot::new(descriptor, bytes).unwrap()
}
fn key(texture: u32, mip: u8, layer: u16) -> TexturePageKey {
    TexturePageKey {
        texture,
        mip,
        layer,
        x: 0,
        y: 0,
    }
}

struct Fixture {
    runtime: RenderRuntime<DeferredInstance>,
    vt: NativeMaterialVt<TestDevice>,
    sources: NativeVtSources,
    materials: RenderMaterials,
    handle: GenerationalHandle,
    slot: BindlessSlot,
}

impl Fixture {
    fn new() -> Self {
        let runtime = runtime();
        let vt = NativeMaterialVt::new(&runtime, config()).unwrap();
        let mut sources = NativeVtSources::default();
        sources
            .capture_image(image_id(1), &image(9), request(7))
            .unwrap();
        let mut materials = RenderMaterials::new(8);
        let slot = materials.textures_mut().acquire(image_id(1)).unwrap();
        let handle = materials.registry_mut().allocate().unwrap();
        let mut record = fallback_material_record(handle, 1);
        record.textures = vec![GpuMaterialTexture::non_virtual(
            slot.index,
            slot.generation,
            11,
            13,
        )];
        materials.publish(record).unwrap();
        materials.registry_mut().take_dirty();
        Self {
            runtime,
            vt,
            sources,
            materials,
            handle,
            slot,
        }
    }
    fn residency(&self) -> NativeVtResidency {
        NativeVtResidency {
            pages: vec![(key(7, 0, 0), 5)],
            portable: vec![],
        }
    }
    fn stage(&mut self) -> Result<(), NativeVtError> {
        let residency = self.residency();
        self.vt.stage(
            &self.runtime,
            &self.sources,
            self.materials.textures(),
            &residency,
        )
    }
    fn publish(&mut self) -> Result<NativeVtPublish, NativeVtError> {
        self.vt
            .try_publish(&self.runtime, &self.sources, &mut self.materials)
    }
    fn refresh(&mut self) -> Result<u32, NativeVtError> {
        self.vt
            .refresh_rows(&self.runtime, &self.sources, &mut self.materials)
    }
    fn row(&self) -> GpuMaterialTexture {
        self.materials.registry().get(self.handle).unwrap().textures[0]
    }
    fn binding(&self) -> NativeVtBinding<TestDevice> {
        self.vt.active_binding(&self.runtime).unwrap().unwrap()
    }
}

#[test]
fn native_vt_source_aliases_immutable_tokens_and_identical_extraction() {
    let mut sources = NativeVtSources::default();
    let old = sources
        .capture_image(image_id(1), &image(3), request(7))
        .unwrap();
    let repeated = sources
        .capture_image(image_id(1), &image(3), request(7))
        .unwrap();
    assert!(Arc::ptr_eq(&old, &repeated));
    let changed = sources
        .capture_image(image_id(1), &image(4), request(7))
        .unwrap();
    assert!(changed.revision() > old.revision());
    assert!(!sources.is_current(&old));
    assert_eq!(old.cpu_snapshot().unwrap().bytes(), &[3; 4]);
    assert_eq!(changed.cpu_snapshot().unwrap().bytes(), &[4; 4]);
    let reassigned = sources
        .capture_image(image_id(2), &image(4), request(7))
        .unwrap();
    sources.remove_image(image_id(1));
    assert!(sources.is_current(&reassigned));
    let renamed = sources
        .capture_image(image_id(2), &image(4), request(8))
        .unwrap();
    assert!(sources.get(7).is_none());
    assert!(sources.is_current(&renamed));
    sources.remove_image(image_id(2));
    assert!(!sources.is_current(&renamed));
    let aba = sources
        .capture_image(image_id(2), &image(4), request(8))
        .unwrap();
    assert!(!Arc::ptr_eq(&renamed, &aba));
}

#[test]
fn native_vt_observation_does_not_drain_and_request_edits_invalidate() {
    let mut sources = NativeVtSources::default();
    let mut extracted = ExtractedAssets::<GpuImage>::new();
    extracted.push_extracted(image_id(1), image(2));
    let mut requests = BTreeMap::from([(image_id(1), request(7))]);
    assert!(sources.observe_images(&extracted, &requests).is_empty());
    let source = Arc::clone(sources.get(7).unwrap());
    assert!(sources.observe_images(&extracted, &requests).is_empty());
    assert!(sources.is_current(&source));
    assert_eq!(extracted.extracted().len(), 1);
    requests.insert(image_id(1), request(8));
    sources.observe_images(&ExtractedAssets::new(), &requests);
    assert!(!sources.is_current(&source));
    sources.observe_images(&extracted, &requests);
    extracted.push_removed(image_id(1));
    sources.observe_images(&extracted, &requests);
    assert!(sources.get(8).is_none());
    assert_eq!(extracted.extracted().len(), 1);
    assert_eq!(extracted.removed().len(), 1);
}

#[test]
fn native_vt_identical_source_observation_keeps_pending_upload_publishable() {
    let mut f = Fixture::new();
    let source = Arc::clone(f.sources.get(7).unwrap());
    let mut extracted = ExtractedAssets::<GpuImage>::new();
    extracted.push_extracted(image_id(1), image(9));
    let requests = BTreeMap::from([(image_id(1), request(7))]);
    f.stage().unwrap();
    let writes = f.runtime.queue().writes();
    let before = f.materials.registry().snapshot();
    for _ in 0..3 {
        assert!(f.sources.observe_images(&extracted, &requests).is_empty());
        assert!(Arc::ptr_eq(&source, f.sources.get(7).unwrap()));
        assert_eq!(f.publish().unwrap(), NativeVtPublish::Waiting);
        assert_eq!(f.materials.registry().snapshot(), before);
        assert_eq!(f.runtime.queue().writes(), writes);
    }
    assert_eq!(extracted.extracted().len(), 1);
    assert!(f.materials.registry_mut().take_dirty().is_empty());
    f.runtime.queue().complete();
    assert_eq!(
        f.publish().unwrap(),
        NativeVtPublish::Published { refreshed_rows: 1 }
    );
    assert_eq!(f.row().virtual_texture_id, 7);
    assert_eq!(f.runtime.queue().writes(), writes);
}

#[test]
fn native_vt_duplicate_events_with_identical_final_snapshot_preserve_pending_token() {
    let mut f = Fixture::new();
    let source = Arc::clone(f.sources.get(7).unwrap());
    let mut extracted = ExtractedAssets::<GpuImage>::new();
    extracted.push_extracted(image_id(1), unfilterable_image());
    extracted.push_extracted(image_id(1), image(3));
    extracted.push_extracted(image_id(1), image(9));
    let requests = BTreeMap::from([(image_id(1), request(7))]);
    f.stage().unwrap();
    let writes = f.runtime.queue().writes();
    let before = f.materials.registry().snapshot();
    for _ in 0..3 {
        assert!(f.sources.observe_images(&extracted, &requests).is_empty());
        assert!(Arc::ptr_eq(&source, f.sources.get(7).unwrap()));
        assert_eq!(f.publish().unwrap(), NativeVtPublish::Waiting);
        assert_eq!(f.materials.registry().snapshot(), before);
        assert_eq!(f.runtime.queue().writes(), writes);
    }
    assert_eq!(extracted.extracted().len(), 3);
    assert!(f.materials.registry_mut().take_dirty().is_empty());
    f.runtime.queue().complete();
    assert_eq!(
        f.publish().unwrap(),
        NativeVtPublish::Published { refreshed_rows: 1 }
    );
    assert_eq!(f.row().virtual_texture_id, 7);
    assert_eq!(f.runtime.queue().writes(), writes);
}

#[test]
fn native_vt_duplicate_events_capture_only_final_replacement_once() {
    let mut sources = NativeVtSources::default();
    let original = sources
        .capture_image(image_id(1), &image(9), request(7))
        .unwrap();
    let mut extracted = ExtractedAssets::<GpuImage>::new();
    extracted.push_extracted(image_id(1), image(3));
    extracted.push_extracted(image_id(1), unfilterable_image());
    extracted.push_extracted(image_id(1), image(4));
    let requests = BTreeMap::from([(image_id(1), request(7))]);
    assert!(sources.observe_images(&extracted, &requests).is_empty());
    let replacement = Arc::clone(sources.get(7).unwrap());
    assert_eq!(replacement.revision(), original.revision() + 1);
    assert_eq!(replacement.cpu_snapshot().unwrap().bytes(), &[4; 4]);
    assert_eq!(original.cpu_snapshot().unwrap().bytes(), &[9; 4]);
    assert!(!sources.is_current(&original));
    for _ in 0..3 {
        assert!(sources.observe_images(&extracted, &requests).is_empty());
        assert!(Arc::ptr_eq(&replacement, sources.get(7).unwrap()));
    }
    sources.remove_image(image_id(1));
    assert!(sources.get(7).is_none());
    assert_eq!(extracted.extracted().len(), 3);
}

#[test]
fn native_vt_duplicate_events_with_invalid_final_snapshot_revoke_publication() {
    let mut f = Fixture::new();
    f.stage().unwrap();
    f.runtime.queue().complete();
    f.publish().unwrap();
    let table = f.binding().page_table();
    let row = f.row();
    let source = Arc::clone(f.sources.get(7).unwrap());
    f.materials.registry_mut().take_dirty();
    f.stage().unwrap();
    let writes = f.runtime.queue().writes();
    let before = f.materials.registry().snapshot();
    let mut extracted = ExtractedAssets::<GpuImage>::new();
    extracted.push_extracted(image_id(1), image(4));
    extracted.push_extracted(
        image_id(1),
        Image::new(
            rhi::Extent3d::new_2d(2, 2),
            rhi::TextureFormat::Rgba8Unorm,
            vec![0; 16],
        )
        .unwrap(),
    );
    extracted.push_extracted(image_id(1), unfilterable_image());
    let requests = BTreeMap::from([(image_id(1), request(7))]);
    for _ in 0..3 {
        assert_eq!(
            f.sources.observe_images(&extracted, &requests),
            vec![(image_id(1), NativeVtError::UnsupportedFormat)]
        );
        assert!(f.sources.get(7).is_none());
        assert!(!f.sources.is_current(&source));
        assert_eq!(f.publish().unwrap(), NativeVtPublish::Waiting);
        assert_eq!(f.row(), row);
        assert_eq!(f.materials.registry().snapshot(), before);
    }
    f.runtime.queue().complete();
    assert_eq!(f.publish(), Err(NativeVtError::StaleSource(7)));
    assert!(!f.vt.has_pending_upload());
    assert_eq!(f.binding().page_table(), table);
    assert_eq!(f.row(), row);
    assert!(f.materials.registry_mut().take_dirty().is_empty());
    assert_eq!(f.refresh().unwrap(), 1);
    assert_eq!(f.row().virtual_texture_id, NO_VIRTUAL_TEXTURE);
    assert_eq!(f.refresh().unwrap(), 0);
    assert_eq!(f.stage(), Err(NativeVtError::MissingSource(7)));
    assert_eq!(f.runtime.queue().writes(), writes);
    assert_eq!(extracted.extracted().len(), 3);
}

#[test]
fn native_vt_duplicate_events_recover_invalid_source_without_reviving_old_upload() {
    let mut f = Fixture::new();
    let original = Arc::clone(f.sources.get(7).unwrap());
    let requests = BTreeMap::from([(image_id(1), request(7))]);
    f.stage().unwrap();
    let before = f.materials.registry().snapshot();
    let writes = f.runtime.queue().writes();

    let mut invalid = ExtractedAssets::<GpuImage>::new();
    invalid.push_extracted(image_id(1), image(9));
    invalid.push_extracted(image_id(1), unfilterable_image());
    assert_eq!(
        f.sources.observe_images(&invalid, &requests),
        vec![(image_id(1), NativeVtError::UnsupportedFormat)]
    );
    assert!(f.sources.get(7).is_none());
    assert!(!f.sources.is_current(&original));

    let mut recovered = ExtractedAssets::<GpuImage>::new();
    recovered.push_extracted(image_id(1), unfilterable_image());
    recovered.push_extracted(image_id(1), image(9));
    assert!(f.sources.observe_images(&recovered, &requests).is_empty());
    let replacement = Arc::clone(f.sources.get(7).unwrap());
    assert_eq!(
        replacement.cpu_snapshot().unwrap().bytes(),
        original.cpu_snapshot().unwrap().bytes()
    );
    assert!(!Arc::ptr_eq(&original, &replacement));
    assert_eq!(replacement.revision(), original.revision() + 1);
    for _ in 0..3 {
        assert!(f.sources.observe_images(&recovered, &requests).is_empty());
        assert!(Arc::ptr_eq(&replacement, f.sources.get(7).unwrap()));
        assert_eq!(f.publish().unwrap(), NativeVtPublish::Waiting);
    }
    f.runtime.queue().complete();
    assert_eq!(f.publish(), Err(NativeVtError::StaleSource(7)));
    assert!(!f.vt.has_pending_upload());
    assert!(f.vt.active_binding(&f.runtime).unwrap().is_none());
    assert_eq!(f.materials.registry().snapshot(), before);
    assert!(f.materials.registry_mut().take_dirty().is_empty());
    assert_eq!(f.refresh().unwrap(), 0);
    assert_eq!(f.runtime.queue().writes(), writes);

    f.stage().unwrap();
    assert_eq!(f.publish().unwrap(), NativeVtPublish::Waiting);
    assert_eq!(f.row().virtual_texture_id, NO_VIRTUAL_TEXTURE);
    assert!(f.sources.observe_images(&recovered, &requests).is_empty());
    f.runtime.queue().complete();
    assert_eq!(
        f.publish().unwrap(),
        NativeVtPublish::Published { refreshed_rows: 1 }
    );
    assert_eq!(f.row().virtual_texture_id, 7);
    assert!(f.sources.is_current(&replacement));
    assert_eq!(f.refresh().unwrap(), 0);
    assert_eq!(invalid.extracted().len(), 2);
    assert_eq!(recovered.extracted().len(), 2);
}

#[test]
fn native_vt_duplicate_events_report_only_final_invalid_images() {
    let mut sources = NativeVtSources::default();
    let old = sources
        .capture_image(image_id(2), &image(2), request(8))
        .unwrap();
    let mut extracted = ExtractedAssets::<GpuImage>::new();
    extracted.push_extracted(image_id(1), unfilterable_image());
    extracted.push_extracted(image_id(2), image(3));
    extracted.push_extracted(image_id(1), image(4));
    extracted.push_extracted(image_id(2), unfilterable_image());
    let requests = BTreeMap::from([(image_id(1), request(7)), (image_id(2), request(8))]);
    for _ in 0..3 {
        assert_eq!(
            sources.observe_images(&extracted, &requests),
            vec![(image_id(2), NativeVtError::UnsupportedFormat)]
        );
        assert_eq!(
            sources.get(7).unwrap().cpu_snapshot().unwrap().bytes(),
            &[4; 4]
        );
        assert!(sources.get(8).is_none());
        assert!(!sources.is_current(&old));
    }
    assert_eq!(extracted.extracted().len(), 4);
}

#[test]
fn native_vt_removed_events_suppress_even_final_snapshot_errors_and_captures() {
    let mut sources = NativeVtSources::default();
    let old = sources
        .capture_image(image_id(1), &image(2), request(7))
        .unwrap();
    let mut extracted = ExtractedAssets::<GpuImage>::new();
    extracted.push_extracted(image_id(1), image(4));
    extracted.push_extracted(image_id(2), unfilterable_image());
    extracted.push_extracted(image_id(1), unfilterable_image());
    extracted.push_extracted(image_id(2), image(5));
    extracted.push_removed(image_id(1));
    extracted.push_removed(image_id(1));
    extracted.push_removed(image_id(2));
    let requests = BTreeMap::from([(image_id(1), request(7)), (image_id(2), request(8))]);
    for _ in 0..3 {
        assert!(sources.observe_images(&extracted, &requests).is_empty());
        assert!(sources.get(7).is_none());
        assert!(sources.get(8).is_none());
        assert!(!sources.is_current(&old));
    }
    let replacement = sources
        .capture_image(image_id(1), &image(2), request(7))
        .unwrap();
    assert_eq!(replacement.revision(), old.revision() + 1);
    assert!(!Arc::ptr_eq(&old, &replacement));
    assert_eq!(extracted.extracted().len(), 4);
    assert_eq!(extracted.removed().len(), 3);
}

#[test]
fn native_vt_source_request_removal_and_identical_reinsert_cannot_revive_old_upload() {
    let mut f = Fixture::new();
    f.stage().unwrap();
    f.runtime.queue().complete();
    f.publish().unwrap();
    let table = f.binding().page_table();
    let row = f.row();
    let source = Arc::clone(f.sources.get(7).unwrap());
    f.materials.registry_mut().take_dirty();
    let before = f.materials.registry().snapshot();
    f.stage().unwrap();
    let writes = f.runtime.queue().writes();

    // Losing the complete request set must revoke ownership even when there
    // is no image extraction/removal event in this frame.
    assert!(f
        .sources
        .observe_images(&ExtractedAssets::new(), &BTreeMap::new())
        .is_empty());
    assert!(f.sources.get(7).is_none());
    let replacement = f
        .sources
        .capture_image(image_id(1), &image(9), request(7))
        .unwrap();
    assert_eq!(
        replacement.layout().descriptor(),
        source.layout().descriptor()
    );
    assert_eq!(
        replacement.cpu_snapshot().unwrap().bytes(),
        source.cpu_snapshot().unwrap().bytes()
    );
    assert!(!Arc::ptr_eq(&source, &replacement));
    assert!(replacement.revision() > source.revision());
    f.runtime.queue().complete();
    assert_eq!(f.publish(), Err(NativeVtError::StaleSource(7)));
    assert!(!f.vt.has_pending_upload());
    assert_eq!(f.binding().page_table(), table);
    assert_eq!(f.row(), row);
    assert_eq!(f.materials.registry().snapshot(), before);
    assert!(f.materials.registry_mut().take_dirty().is_empty());
    assert_eq!(f.runtime.queue().writes(), writes);

    // Equal bytes are not evidence that the replacement was uploaded. The
    // stale completion must not authorize either row readoption or a new job.
    assert_eq!(f.refresh().unwrap(), 1);
    assert_eq!(f.row().virtual_texture_id, NO_VIRTUAL_TEXTURE);
    assert_eq!(f.refresh().unwrap(), 0);
    f.stage().unwrap();
    assert_eq!(f.publish().unwrap(), NativeVtPublish::Waiting);
    assert_eq!(f.row().virtual_texture_id, NO_VIRTUAL_TEXTURE);
    assert_eq!(f.binding().page_table(), table);
    f.runtime.queue().complete();
    assert_eq!(
        f.publish().unwrap(),
        NativeVtPublish::Published { refreshed_rows: 1 }
    );
    assert_eq!(f.row(), row);
    assert_ne!(f.binding().page_table(), table);
}

#[test]
fn native_vt_failed_source_replacement_revokes_old_token() {
    let mut sources = NativeVtSources::default();
    let old = sources
        .capture_image(image_id(1), &image(2), request(7))
        .unwrap();
    let truncated = Image::new(
        rhi::Extent3d::new_2d(2, 2),
        rhi::TextureFormat::Rgba8Unorm,
        vec![0; 16],
    )
    .unwrap();
    assert!(matches!(
        sources.capture_image(image_id(1), &truncated, request(7)),
        Err(NativeVtError::Source(_))
    ));
    assert!(!sources.is_current(&old));
    for id in [NO_VIRTUAL_TEXTURE, PORTABLE_TEXTURE_ATLAS] {
        assert_eq!(
            sources
                .capture_image(image_id(1), &image(2), request(id))
                .unwrap_err(),
            NativeVtError::ReservedTextureId
        );
    }
    let unfilterable = Image::new(
        rhi::Extent3d::new_2d(1, 1),
        rhi::TextureFormat::Rgba32Float,
        vec![0; 16],
    )
    .unwrap();
    assert_eq!(
        sources
            .capture_image(image_id(1), &unfilterable, request(7))
            .unwrap_err(),
        NativeVtError::UnsupportedFormat
    );
}

#[test]
fn native_vt_upload_uses_shared_tile_geometry_table_and_config_abi() {
    let mut f = Fixture::new();
    let source = Arc::clone(f.sources.get(7).unwrap());
    let expected = build_material_vt_atlas_tile(
        source.cpu_snapshot().unwrap(),
        key(7, 0, 0),
        5,
        &BTreeSet::from([key(7, 0, 0)]),
        MaterialVtAtlasConfig::new(1, 2).unwrap(),
    )
    .unwrap();
    f.stage().unwrap();
    assert_eq!(f.runtime.queue().progress.lock().unwrap().submitted, 1);
    assert_eq!(f.publish().unwrap(), NativeVtPublish::Waiting);
    assert!(f.vt.active_binding(&f.runtime).unwrap().is_none());
    assert_eq!(f.row().virtual_texture_id, NO_VIRTUAL_TEXTURE);
    assert!(f.materials.registry_mut().take_dirty().is_empty());
    {
        let progress = f.runtime.queue().progress.lock().unwrap();
        assert_eq!(progress.textures.len(), 1);
        let (write, bytes, layout) = &progress.textures[0];
        let placement = config().geometry().unwrap().slot_placement(5);
        assert_eq!(
            write.origin,
            [
                placement.origin_x_px,
                placement.origin_y_px,
                placement.layer
            ]
        );
        assert_eq!(
            write.size,
            rhi::Extent3d::new_2d(
                expected.layout.stored_tile_extent_px(),
                expected.layout.stored_tile_extent_px()
            )
        );
        assert_eq!(write.mip_level, 0);
        assert_eq!(*bytes, expected.staging_bytes);
        assert_eq!(layout.bytes_per_row, Some(expected.bytes_per_row));
        assert_eq!(
            layout.rows_per_image,
            Some(expected.layout.stored_tile_extent_px())
        );
    }
    f.runtime.queue().complete();
    assert_eq!(
        f.publish().unwrap(),
        NativeVtPublish::Published { refreshed_rows: 1 }
    );
    let binding = f.binding();
    let table = GpuPageTable::from_bindings(&f.residency().pages);
    assert_eq!(
        f.runtime.queue().buffer_bytes(binding.page_table()),
        bytemuck::cast_slice::<u32, u8>(table.words())
    );
    let gpu_config: NativeVtGpuConfig =
        bytemuck::pod_read_unaligned(&f.runtime.queue().buffer_bytes(binding.config()));
    assert_eq!(std::mem::size_of::<NativeVtGpuConfig>(), 32);
    assert_eq!(
        gpu_config,
        NativeVtGpuConfig {
            enabled: 1,
            entry_count: 1,
            page_size: 2,
            tile_extent_px: expected.layout.stored_tile_extent_px(),
            tiles_per_row: 2,
            atlas_layers: 2,
            gutter_px: 1,
            coarsest_mip: 0
        }
    );
    assert_eq!(
        f.row(),
        source
            .cpu_snapshot()
            .expect("CPU source snapshot")
            .texture_row(f.slot.index, f.slot.generation, 11, 13)
            .gpu
    );
    assert_eq!(f.publish().unwrap(), NativeVtPublish::Idle);
}

#[test]
fn native_vt_all_upload_validation_precedes_queue_writes() {
    let mut f = Fixture::new();
    let page = key(7, 0, 0);
    let cases = [
        (
            vec![(page, 0), (page, 1)],
            NativeVtError::DuplicatePage(page),
        ),
        (vec![(page, 8)], NativeVtError::SlotOutOfBounds(8)),
        (vec![(key(99, 0, 0), 0)], NativeVtError::MissingSource(99)),
        (vec![(page, 0); 9], NativeVtError::TableCapacity),
    ];
    for (pages, expected) in cases {
        assert_eq!(
            f.vt.stage(
                &f.runtime,
                &f.sources,
                f.materials.textures(),
                &NativeVtResidency {
                    pages,
                    portable: vec![]
                }
            ),
            Err(expected)
        );
        assert_eq!(f.runtime.queue().writes(), 0);
        assert!(!f.vt.has_pending_upload());
    }
    f.sources
        .insert(image_id(1), snapshot(7, 4, 1, MaterialVtFormat::Rgba8Unorm))
        .unwrap();
    let residency = NativeVtResidency {
        pages: vec![(key(7, 0, 0), 0)],
        portable: vec![],
    };
    assert_eq!(
        f.vt.stage(&f.runtime, &f.sources, f.materials.textures(), &residency),
        Err(NativeVtError::MissingCoarsestPage(7))
    );
    let residency = NativeVtResidency {
        pages: vec![(key(7, 0, 0), 0), (key(7, 2, 0), 0)],
        portable: vec![],
    };
    assert_eq!(
        f.vt.stage(&f.runtime, &f.sources, f.materials.textures(), &residency),
        Err(NativeVtError::DuplicateSlot(0))
    );
    f.sources
        .insert(image_id(1), snapshot(7, 1, 1, MaterialVtFormat::R8Unorm))
        .unwrap();
    assert_eq!(f.stage(), Err(NativeVtError::AtlasMismatch(7)));
    assert_eq!(f.runtime.queue().writes(), 0);
}

#[test]
fn native_vt_portable_uses_real_heap_slot_and_empty_set_is_initialized() {
    let mut f = Fixture::new();
    let residency = NativeVtResidency {
        pages: vec![],
        portable: vec![7],
    };
    f.vt.stage(&f.runtime, &f.sources, f.materials.textures(), &residency)
        .unwrap();
    let placement = config().geometry().unwrap().slot_placement(f.slot.index);
    assert!(f.slot.index > 0);
    assert_eq!(
        f.runtime.queue().progress.lock().unwrap().textures[0]
            .0
            .origin,
        [
            placement.origin_x_px,
            placement.origin_y_px,
            placement.layer
        ]
    );
    f.runtime.queue().complete();
    f.publish().unwrap();
    assert_eq!(
        f.row(),
        GpuMaterialTexture::with_portable_atlas(f.slot.index, f.slot.generation, 11, 13, 1, 1, 2)
    );
    f.vt.stage(
        &f.runtime,
        &f.sources,
        f.materials.textures(),
        &NativeVtResidency::default(),
    )
    .unwrap();
    f.runtime.queue().complete();
    assert_eq!(
        f.publish().unwrap(),
        NativeVtPublish::Published { refreshed_rows: 1 }
    );
    let binding = f.binding();
    assert_eq!(
        f.runtime.queue().buffer_bytes(binding.page_table()),
        vec![0; 16]
    );
    let gpu: NativeVtGpuConfig =
        bytemuck::pod_read_unaligned(&f.runtime.queue().buffer_bytes(binding.config()));
    assert_eq!((gpu.enabled, gpu.entry_count), (0, 0));
    assert_eq!(f.row().virtual_texture_id, NO_VIRTUAL_TEXTURE);
}

#[test]
fn native_vt_cancel_waits_for_callback_and_abandoned_callback_never_reuses_slice() {
    let mut f = Fixture::new();
    f.stage().unwrap();
    assert_eq!(f.stage(), Err(NativeVtError::PendingUpload));
    assert!(f.vt.cancel());
    assert!(!f.vt.cancel());
    assert_eq!(f.publish().unwrap(), NativeVtPublish::Idle);
    assert_eq!(f.stage(), Err(NativeVtError::SliceBusy));
    f.runtime.queue().complete();
    f.stage().unwrap();
    f.vt.cancel();
    f.runtime.queue().abandon();
    assert_eq!(f.stage(), Err(NativeVtError::SliceBusy));
    let device = f.runtime.device_handle();
    let before = device.inner.live_resource_count();
    drop(f);
    assert_eq!(
        device.inner.live_resource_count(),
        before - 5,
        "only the never-submitted second slice may be explicitly destroyed"
    );
}

#[test]
fn native_vt_upload_unwind_preserves_publication_and_never_confirms_reuse() {
    let mut f = Fixture::new();
    f.stage().unwrap();
    f.runtime.queue().complete();
    f.publish().unwrap();
    let row = f.row();
    let table = f.binding().page_table();
    f.materials.registry_mut().take_dirty();
    let before = f.materials.registry().snapshot();
    f.runtime
        .queue()
        .panic_after_texture_write
        .store(true, Ordering::Release);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f.stage()));
    assert!(result.is_err());
    assert!(!f.vt.has_pending_upload());
    assert_eq!(f.row(), row);
    assert_eq!(f.binding().page_table(), table);
    assert_eq!(f.materials.registry().snapshot(), before);
    assert!(f.materials.registry_mut().take_dirty().is_empty());
    assert_eq!(f.publish().unwrap(), NativeVtPublish::Idle);
    assert_eq!(f.stage(), Err(NativeVtError::SliceBusy));
    // No completion was registered after the backend unwound. Progressing the
    // queue must not manufacture proof that the partially accepted upload ended.
    f.runtime.queue().complete();
    assert_eq!(f.stage(), Err(NativeVtError::SliceBusy));
    let device = f.runtime.device_handle();
    drop(f);
    assert_eq!(device.inner.live_resource_count(), 5);
}

#[test]
fn native_vt_binding_lease_and_consumer_completion_both_prevent_overwrite() {
    let mut f = Fixture::new();
    f.stage().unwrap();
    f.runtime.queue().complete();
    f.publish().unwrap();
    let old = f.binding();
    let old_table = old.page_table();
    f.stage().unwrap();
    f.runtime.queue().complete();
    f.publish().unwrap();
    assert_ne!(old_table, f.binding().page_table());
    assert_eq!(f.stage(), Err(NativeVtError::SliceBusy));
    // A previously cached group may still submit its retained old lease after
    // the next publication. Only that consumer's callback permits reuse.
    let frame = old
        .execute_prepared(
            &f.runtime,
            RenderGraph::new(rhi::Extent3d::new_2d(4, 4))
                .prepare()
                .unwrap(),
            &[],
        )
        .unwrap();
    assert!(!frame.is_complete());
    drop(frame);
    assert_eq!(f.stage(), Err(NativeVtError::SliceBusy));
    f.runtime.queue().complete();
    f.stage().unwrap();
    f.runtime.queue().complete();
    f.publish().unwrap();
    assert_eq!(f.binding().page_table(), old_table);
    let device = f.runtime.device_handle();
    drop(f);
    assert_eq!(device.inner.live_resource_count(), 0);
}

#[test]
fn native_vt_pending_publication_uses_current_material_generation_and_authored_rows() {
    use prism_render_architecture::gpu_scene::GpuCompletionValue;
    use prism_render_material::DynamicChannels;

    let mut f = Fixture::new();
    f.stage().unwrap();
    let old_handle = f.handle;
    f.materials
        .registry_mut()
        .retire(old_handle, GpuCompletionValue(1))
        .unwrap();
    assert_eq!(
        f.materials
            .registry_mut()
            .reclaim_completed(GpuCompletionValue(1)),
        1
    );
    let replacement = f.materials.registry_mut().allocate().unwrap();
    assert_eq!(replacement.index, old_handle.index);
    assert_ne!(replacement.generation, old_handle.generation);
    let mut record = fallback_material_record(replacement, 1);
    record.textures = vec![GpuMaterialTexture::non_virtual(
        f.slot.index,
        f.slot.generation,
        3,
        17,
    )];
    f.materials.publish(record).unwrap();
    let dynamic = DynamicChannels {
        wetness: 0.5,
        ..DynamicChannels::NONE
    };
    f.materials
        .registry_mut()
        .set_dynamic_channels(replacement, dynamic)
        .unwrap();
    f.materials.registry_mut().take_dirty();
    let before = f.materials.registry().snapshot();
    let writes = f.runtime.queue().writes();

    assert_eq!(f.publish().unwrap(), NativeVtPublish::Waiting);
    assert_eq!(f.materials.registry().snapshot(), before);
    assert!(f.materials.registry().get(old_handle).is_none());
    f.runtime.queue().complete();
    assert_eq!(
        f.publish().unwrap(),
        NativeVtPublish::Published { refreshed_rows: 1 }
    );
    assert!(f.materials.registry().get(old_handle).is_none());
    let record = f.materials.registry().get(replacement).unwrap();
    assert_eq!(record.handle, replacement);
    assert_eq!(record.textures[0].virtual_texture_id, 7);
    assert_eq!(
        (
            record.textures[0].semantic,
            record.textures[0].sampler_index
        ),
        (3, 17)
    );
    assert_eq!(
        f.materials
            .registry()
            .dynamic_channels_at(replacement.index),
        dynamic
    );
    assert_eq!(
        f.materials.registry_mut().take_dirty(),
        vec![replacement.index]
    );
    assert_eq!(f.refresh().unwrap(), 0);
    assert_eq!(f.runtime.queue().writes(), writes);
}

#[test]
fn native_vt_portable_mode_transition_waits_for_completion_and_old_binding_lease() {
    let mut f = Fixture::new();
    f.stage().unwrap();
    f.runtime.queue().complete();
    f.publish().unwrap();
    let virtual_row = f.row();
    let virtual_binding = f.binding();
    let virtual_table = virtual_binding.page_table();
    f.materials.registry_mut().take_dirty();
    let portable = NativeVtResidency {
        pages: vec![],
        portable: vec![7],
    };
    f.vt.stage(&f.runtime, &f.sources, f.materials.textures(), &portable)
        .unwrap();
    assert_eq!(f.publish().unwrap(), NativeVtPublish::Waiting);
    assert_eq!(f.refresh().unwrap(), 0);
    assert_eq!(f.row(), virtual_row);
    assert_eq!(f.binding().page_table(), virtual_table);
    assert!(f.materials.registry_mut().take_dirty().is_empty());
    f.runtime.queue().complete();
    assert_eq!(
        f.publish().unwrap(),
        NativeVtPublish::Published { refreshed_rows: 1 }
    );
    let portable_row = f.row();
    let portable_table = f.binding().page_table();
    assert_eq!(portable_row.virtual_texture_id, PORTABLE_TEXTURE_ATLAS);
    assert_ne!(portable_table, virtual_table);
    let gpu: NativeVtGpuConfig =
        bytemuck::pod_read_unaligned(&f.runtime.queue().buffer_bytes(f.binding().config()));
    assert_eq!((gpu.enabled, gpu.entry_count), (1, 0));
    f.materials.registry_mut().take_dirty();
    let writes = f.runtime.queue().writes();
    assert_eq!(f.stage(), Err(NativeVtError::SliceBusy));
    assert_eq!(f.runtime.queue().writes(), writes);
    assert_eq!(f.row(), portable_row);

    drop(virtual_binding);
    f.stage().unwrap();
    assert_eq!(f.publish().unwrap(), NativeVtPublish::Waiting);
    assert_eq!(f.refresh().unwrap(), 0);
    assert_eq!(f.row(), portable_row);
    assert_eq!(f.binding().page_table(), portable_table);
    f.runtime.queue().complete();
    assert_eq!(
        f.publish().unwrap(),
        NativeVtPublish::Published { refreshed_rows: 1 }
    );
    assert_eq!(f.row(), virtual_row);
    assert_eq!(f.binding().page_table(), virtual_table);
    assert_eq!(
        f.materials.registry_mut().take_dirty(),
        vec![f.handle.index]
    );
}

#[test]
fn native_vt_stale_source_or_slot_never_publishes_or_mutates_rows() {
    for stale_slot in [false, true] {
        let mut f = Fixture::new();
        let row = f.row();
        let registry = f.materials.registry().snapshot();
        f.stage().unwrap();
        if stale_slot {
            f.materials.textures_mut().release(image_id(1));
            let replacement = f.materials.textures_mut().acquire(image_id(1)).unwrap();
            assert_eq!(replacement.index, f.slot.index);
            assert_ne!(replacement.generation, f.slot.generation);
        } else {
            f.sources
                .capture_image(image_id(1), &image(8), request(7))
                .unwrap();
        }
        f.runtime.queue().complete();
        assert_eq!(
            f.publish(),
            Err(if stale_slot {
                NativeVtError::StaleSlot(7)
            } else {
                NativeVtError::StaleSource(7)
            })
        );
        assert!(!f.vt.has_pending_upload());
        assert!(f.vt.active_binding(&f.runtime).unwrap().is_none());
        assert_eq!(f.row(), row);
        assert_eq!(f.materials.registry().snapshot(), registry);
        assert!(f.materials.registry_mut().take_dirty().is_empty());
    }
}

#[test]
fn native_vt_device_identity_generation_and_loss_reject_all_entrypoints() {
    let mut f = Fixture::new();
    let foreign = runtime();
    assert!(matches!(
        f.vt.active_binding(&foreign),
        Err(NativeVtError::DeviceMismatch)
    ));
    let residency = f.residency();
    assert_eq!(
        f.vt.stage(&foreign, &f.sources, f.materials.textures(), &residency),
        Err(NativeVtError::DeviceMismatch)
    );
    let slice = super::gpu::GpuSlice::new(
        f.runtime.device_handle(),
        f.runtime.device_generation() + 1,
        config(),
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    assert_eq!(
        super::gpu::check_runtime(&slice, &f.runtime),
        Err(NativeVtError::DeviceMismatch)
    );
    assert!(super::gpu::GpuSlice::idle(&slice));
    // Dropping an unsubmitted generation-mismatch slice is safe on its creator.
    drop(slice);
    f.stage().unwrap();
    f.runtime.queue().complete();
    f.runtime
        .device_handle()
        .lost
        .store(true, Ordering::Release);
    assert_eq!(f.publish(), Err(NativeVtError::DeviceLost));
    assert_eq!(f.refresh(), Err(NativeVtError::DeviceLost));
    assert_eq!(f.stage(), Err(NativeVtError::DeviceLost));
    assert!(matches!(
        f.vt.active_binding(&f.runtime),
        Err(NativeVtError::DeviceLost)
    ));
    assert!(matches!(
        NativeMaterialVt::new(&f.runtime, config()),
        Err(NativeVtError::DeviceLost)
    ));
    assert_eq!(f.row().virtual_texture_id, NO_VIRTUAL_TEXTURE);
    assert!(f.materials.registry_mut().take_dirty().is_empty());
}

#[test]
fn native_vt_known_device_loss_abandons_both_idle_slices() {
    let runtime = runtime();
    let device = runtime.device_handle();
    let mut vt = NativeMaterialVt::new(&runtime, config()).unwrap();
    assert_eq!(device.inner.live_resource_count(), 10);

    device.lost.store(true, Ordering::Release);
    assert!(matches!(
        vt.active_binding(&runtime),
        Err(NativeVtError::DeviceLost)
    ));
    drop(vt);

    assert_eq!(
        device.inner.live_resource_count(),
        10,
        "lost-device resources must be left to device teardown, not explicitly destroyed"
    );
}

#[test]
fn native_vt_binding_observing_loss_retires_sibling_slice_too() {
    let runtime = runtime();
    let device = runtime.device_handle();
    let sources = NativeVtSources::default();
    let mut materials = RenderMaterials::new(8);
    let mut vt = NativeMaterialVt::new(&runtime, config()).unwrap();
    vt.stage(
        &runtime,
        &sources,
        materials.textures(),
        &NativeVtResidency::default(),
    )
    .unwrap();
    runtime.queue().complete();
    assert_eq!(
        vt.try_publish(&runtime, &sources, &mut materials).unwrap(),
        NativeVtPublish::Published { refreshed_rows: 0 }
    );
    let binding = vt.active_binding(&runtime).unwrap().unwrap();
    vt.stage(
        &runtime,
        &sources,
        materials.textures(),
        &NativeVtResidency::default(),
    )
    .unwrap();

    device.lost.store(true, Ordering::Release);
    assert!(matches!(
        binding.execute_prepared(
            &runtime,
            RenderGraph::new(rhi::Extent3d::new_2d(4, 4))
                .prepare()
                .unwrap(),
            &[],
        ),
        Err(NativeVtSubmitError::Vt(NativeVtError::DeviceLost))
    ));
    drop(vt);
    assert_eq!(device.inner.live_resource_count(), 10);
    // A callback arriving after loss can discharge its obligation, but the
    // shared abandonment token still forbids explicit destruction.
    runtime.queue().complete();
    assert_eq!(
        device.inner.live_resource_count(),
        10,
        "a consumer-detected loss must retire both slices through the shared token"
    );
}

#[test]
fn native_vt_publication_accepts_revision_max_without_reauthoring() {
    let mut f = Fixture::new();
    let other = f.materials.registry_mut().allocate().unwrap();
    let mut record = fallback_material_record(other, u64::from(u32::MAX));
    record.textures = vec![f.row()];
    f.materials.publish(record).unwrap();
    f.materials.registry_mut().take_dirty();
    let before = f.materials.registry().snapshot();
    f.stage().unwrap();
    f.runtime.queue().complete();
    assert_eq!(
        f.publish().unwrap(),
        NativeVtPublish::Published { refreshed_rows: 2 }
    );
    assert!(!f.vt.has_pending_upload());
    assert!(f.vt.active_binding(&f.runtime).unwrap().is_some());
    assert_eq!(f.row().virtual_texture_id, 7);
    assert_eq!(f.materials.registry().get(f.handle).unwrap().revision, 1);
    let other_record = f.materials.registry().get(other).unwrap();
    assert_eq!(other_record.revision, u32::MAX);
    assert_eq!(other_record.textures, vec![f.row()]);
    assert_eq!(f.materials.registry().snapshot().epoch, before.epoch + 1);
    assert_eq!(
        f.materials.registry_mut().take_dirty(),
        vec![f.handle.index, other.index]
    );
}

#[test]
fn native_vt_multiple_texture_updates_count_one_material_and_republication_is_noop() {
    let mut f = Fixture::new();
    let mut authored = f.materials.registry().get(f.handle).unwrap().clone();
    authored.revision = u32::MAX;
    authored.textures.push(GpuMaterialTexture {
        semantic: 17,
        sampler_index: 19,
        ..f.row()
    });
    f.materials.publish(authored.clone()).unwrap();
    f.materials.registry_mut().take_dirty();
    let before = f.materials.registry().snapshot();
    f.stage().unwrap();
    f.runtime.queue().complete();
    assert_eq!(
        f.publish().unwrap(),
        NativeVtPublish::Published { refreshed_rows: 1 }
    );
    let live = f.materials.registry().get(f.handle).unwrap();
    assert_eq!(live.revision, authored.revision);
    assert!(live.textures.iter().all(|row| row.virtual_texture_id == 7));
    for (row, authored_row) in live.textures.iter().zip(&authored.textures) {
        assert_eq!(
            (row.index, row.generation, row.semantic, row.sampler_index),
            (
                authored_row.index,
                authored_row.generation,
                authored_row.semantic,
                authored_row.sampler_index
            )
        );
    }
    assert_eq!(f.materials.registry().snapshot().epoch, before.epoch + 1);
    assert_eq!(
        f.materials.registry_mut().take_dirty(),
        vec![f.handle.index]
    );
    let published = f.materials.registry().get(f.handle).unwrap().clone();
    let before = f.materials.registry().snapshot();
    // Unlike a quiet cached refresh, a completed publication always prepares
    // rows again. Identical residency must still leave all registry state alone.
    f.stage().unwrap();
    f.runtime.queue().complete();
    assert_eq!(
        f.publish().unwrap(),
        NativeVtPublish::Published { refreshed_rows: 0 }
    );
    assert_eq!(f.materials.registry().get(f.handle), Some(&published));
    assert_eq!(f.materials.registry().snapshot(), before);
    assert!(f.materials.registry_mut().take_dirty().is_empty());

    // Force an uncached scan without changing the dependent rows.
    let unrelated = f.materials.registry_mut().allocate().unwrap();
    f.materials
        .publish(fallback_material_record(unrelated, 1))
        .unwrap();
    f.materials.registry_mut().take_dirty();
    let before = f.materials.registry().snapshot();
    assert_eq!(f.refresh().unwrap(), 0);
    assert_eq!(f.materials.registry().snapshot(), before);
    assert_eq!(f.materials.registry().get(f.handle), Some(&published));
    assert!(f.materials.registry_mut().take_dirty().is_empty());
}

#[test]
fn native_vt_refresh_is_quiet_and_invalid_sources_are_not_readopted() {
    let mut f = Fixture::new();
    f.stage().unwrap();
    f.runtime.queue().complete();
    f.publish().unwrap();
    f.materials.registry_mut().take_dirty();
    let before = f.materials.registry().snapshot();
    let writes = f.runtime.queue().writes();
    for _ in 0..4 {
        assert_eq!(f.refresh().unwrap(), 0);
    }
    assert_eq!(f.materials.registry().snapshot(), before);
    assert!(f.materials.registry_mut().take_dirty().is_empty());
    assert_eq!(f.runtime.queue().writes(), writes);
    let published = f.row();
    f.sources
        .capture_image(image_id(1), &image(10), request(7))
        .unwrap();
    assert_eq!(f.refresh().unwrap(), 1);
    assert_eq!(f.row().virtual_texture_id, NO_VIRTUAL_TEXTURE);
    assert_eq!(f.refresh().unwrap(), 0);
    // A late authored row carrying the old metadata must be cleared as well.
    let mut record = f.materials.registry().get(f.handle).unwrap().clone();
    record.revision += 1;
    record.textures[0] = published;
    f.materials.publish(record).unwrap();
    assert_eq!(f.refresh().unwrap(), 1);
    assert_eq!(f.row().virtual_texture_id, NO_VIRTUAL_TEXTURE);
    f.stage().unwrap();
    f.runtime.queue().complete();
    f.publish().unwrap();
    assert_eq!(f.row().virtual_texture_id, 7);
}

#[test]
fn native_vt_source_removal_revokes_revision_max_in_the_same_batch() {
    let mut f = Fixture::new();
    f.stage().unwrap();
    f.runtime.queue().complete();
    f.publish().unwrap();
    let row = f.row();
    let table = f.binding().page_table();
    let other = f.materials.registry_mut().allocate().unwrap();
    let mut record = fallback_material_record(other, u64::from(u32::MAX));
    record.textures = vec![row];
    f.materials.publish(record).unwrap();
    f.materials.registry_mut().take_dirty();
    let before = f.materials.registry().snapshot();
    let writes = f.runtime.queue().writes();
    f.sources.remove_image(image_id(1));

    assert_eq!(f.refresh().unwrap(), 2);
    assert_eq!(f.row().virtual_texture_id, NO_VIRTUAL_TEXTURE);
    let other_record = f.materials.registry().get(other).unwrap();
    assert_eq!(other_record.revision, u32::MAX);
    assert_eq!(other_record.textures, vec![f.row()]);
    assert_eq!(f.materials.registry().get(f.handle).unwrap().revision, 1);
    assert_eq!(f.materials.registry().snapshot().epoch, before.epoch + 1);
    assert_eq!(
        f.materials.registry_mut().take_dirty(),
        vec![f.handle.index, other.index]
    );
    let revoked = f.materials.registry().snapshot();
    assert_eq!(f.refresh().unwrap(), 0);
    assert_eq!(f.materials.registry().snapshot(), revoked);
    assert!(f.materials.registry_mut().take_dirty().is_empty());
    assert_eq!(f.binding().page_table(), table);
    assert_eq!(f.runtime.queue().writes(), writes);
}

#[test]
fn native_vt_source_replacement_refreshes_only_its_dependent_materials() {
    let mut f = Fixture::new();
    let other_source = f
        .sources
        .capture_image(image_id(2), &image(4), request(8))
        .unwrap();
    let other_slot = f.materials.textures_mut().acquire(image_id(2)).unwrap();
    let other = f.materials.registry_mut().allocate().unwrap();
    let mut record = fallback_material_record(other, 1);
    record.textures = vec![GpuMaterialTexture::non_virtual(
        other_slot.index,
        other_slot.generation,
        3,
        17,
    )];
    f.materials.publish(record).unwrap();
    let residency = NativeVtResidency {
        pages: vec![(key(7, 0, 0), 5), (key(8, 0, 0), 6)],
        portable: vec![],
    };
    f.vt.stage(&f.runtime, &f.sources, f.materials.textures(), &residency)
        .unwrap();
    f.runtime.queue().complete();
    assert_eq!(
        f.publish().unwrap(),
        NativeVtPublish::Published { refreshed_rows: 2 }
    );
    let other_record = f.materials.registry().get(other).unwrap().clone();
    let table = f.binding().page_table();
    f.materials.registry_mut().take_dirty();
    let writes = f.runtime.queue().writes();

    f.sources
        .capture_image(image_id(1), &image(10), request(7))
        .unwrap();
    assert!(Arc::ptr_eq(&other_source, f.sources.get(8).unwrap()));
    assert_eq!(f.refresh().unwrap(), 1);
    assert_eq!(f.row().virtual_texture_id, NO_VIRTUAL_TEXTURE);
    assert_eq!(
        f.materials.registry().get(other).unwrap().revision,
        other_record.revision
    );
    assert_eq!(
        f.materials.registry().get(other).unwrap().textures,
        other_record.textures
    );
    assert_eq!(
        f.materials.registry_mut().take_dirty(),
        vec![f.handle.index]
    );
    assert_eq!(f.runtime.queue().writes(), writes);
    assert_eq!(f.binding().page_table(), table);

    f.vt.stage(&f.runtime, &f.sources, f.materials.textures(), &residency)
        .unwrap();
    let writes = f.runtime.queue().writes();
    assert_eq!(f.stage(), Err(NativeVtError::PendingUpload));
    assert_eq!(f.runtime.queue().writes(), writes);
    assert_eq!(f.publish().unwrap(), NativeVtPublish::Waiting);
    assert_eq!(f.refresh().unwrap(), 0);
    assert_eq!(f.row().virtual_texture_id, NO_VIRTUAL_TEXTURE);
    f.runtime.queue().complete();
    assert_eq!(
        f.publish().unwrap(),
        NativeVtPublish::Published { refreshed_rows: 1 }
    );
    assert_eq!(f.row().virtual_texture_id, 7);
    assert_eq!(
        f.materials.registry().get(other).unwrap().revision,
        other_record.revision
    );
    assert_eq!(
        f.materials.registry().get(other).unwrap().textures,
        other_record.textures
    );
    assert_eq!(
        f.materials.registry_mut().take_dirty(),
        vec![f.handle.index]
    );
    assert_eq!(f.runtime.queue().writes(), writes);
}

#[test]
fn native_vt_published_identity_does_not_pin_replaced_source_bytes() {
    let mut f = Fixture::new();
    let old = Arc::downgrade(f.sources.get(7).unwrap());
    f.stage().unwrap();
    f.runtime.queue().complete();
    f.publish().unwrap();
    assert_eq!(Arc::strong_count(f.sources.get(7).unwrap()), 1);
    f.sources
        .capture_image(image_id(1), &image(10), request(7))
        .unwrap();
    assert!(old.upgrade().is_none());
    assert_eq!(f.refresh().unwrap(), 1);

    // In contrast, an in-flight upload must retain the exact captured bytes
    // until its source validity has been checked or it is cancelled.
    let pending = Arc::downgrade(f.sources.get(7).unwrap());
    f.stage().unwrap();
    f.sources
        .capture_image(image_id(1), &image(11), request(7))
        .unwrap();
    assert!(pending.upgrade().is_some());
    f.runtime.queue().complete();
    assert_eq!(f.publish(), Err(NativeVtError::StaleSource(7)));
    assert!(pending.upgrade().is_none());
}

#[test]
fn native_vt_late_rows_from_older_publications_clear_without_touching_foreign_metadata() {
    let mut f = Fixture::new();
    f.stage().unwrap();
    f.runtime.queue().complete();
    f.publish().unwrap();
    let mut old = f.row();
    old.semantic = 19;
    old.sampler_index = 23;

    f.sources
        .capture_image(image_id(1), &image(10), request(8))
        .unwrap();
    f.vt.stage(
        &f.runtime,
        &f.sources,
        f.materials.textures(),
        &NativeVtResidency {
            pages: vec![(key(8, 0, 0), 5)],
            portable: vec![],
        },
    )
    .unwrap();
    f.runtime.queue().complete();
    f.publish().unwrap();
    assert_eq!(f.row().virtual_texture_id, 8);
    f.vt.stage(
        &f.runtime,
        &f.sources,
        f.materials.textures(),
        &NativeVtResidency::default(),
    )
    .unwrap();
    f.runtime.queue().complete();
    f.publish().unwrap();

    let foreign = GpuMaterialTexture {
        virtual_texture_id: 99,
        virtual_texture_width: 3,
        ..old
    };
    let mut record = f.materials.registry().get(f.handle).unwrap().clone();
    record.revision += 1;
    record.textures = vec![old, foreign];
    f.materials.publish(record).unwrap();
    assert_eq!(f.refresh().unwrap(), 1);
    let rows = &f.materials.registry().get(f.handle).unwrap().textures;
    assert_eq!(
        rows[0],
        GpuMaterialTexture::non_virtual(f.slot.index, f.slot.generation, 19, 23)
    );
    assert_eq!(rows[1], foreign);
    assert_eq!(f.refresh().unwrap(), 0);
}

#[test]
fn native_vt_stale_pending_keeps_existing_publication_then_refresh_revokes_rows() {
    for stale_slot in [false, true] {
        let mut f = Fixture::new();
        f.stage().unwrap();
        f.runtime.queue().complete();
        f.publish().unwrap();
        let table = f.binding().page_table();
        let row = f.row();
        f.materials.registry_mut().take_dirty();
        let before = f.materials.registry().snapshot();
        f.stage().unwrap();
        if stale_slot {
            f.materials.textures_mut().release(image_id(1));
            f.materials.textures_mut().acquire(image_id(2)).unwrap();
        } else {
            f.sources.remove_image(image_id(1));
        }
        f.runtime.queue().complete();
        assert_eq!(
            f.publish(),
            Err(if stale_slot {
                NativeVtError::StaleSlot(7)
            } else {
                NativeVtError::StaleSource(7)
            })
        );
        assert_eq!(f.binding().page_table(), table);
        assert_eq!(f.row(), row);
        assert_eq!(f.materials.registry().snapshot(), before);
        assert!(f.materials.registry_mut().take_dirty().is_empty());
        assert_eq!(f.refresh().unwrap(), 1);
        assert_eq!(f.row().virtual_texture_id, NO_VIRTUAL_TEXTURE);
        assert_eq!(f.binding().page_table(), table);
    }
}

#[test]
fn native_vt_recovery_retains_row_ownership_but_requires_new_device_completion() {
    let mut f = Fixture::new();
    f.stage().unwrap();
    f.runtime.queue().complete();
    f.publish().unwrap();
    let old = f.binding();
    let old_device = f.runtime.device_handle();
    let token = Arc::clone(f.sources.get(7).unwrap());
    f.stage().unwrap();
    f.runtime.queue().abandon();
    old_device.lost.store(true, Ordering::Release);
    assert_eq!(f.refresh(), Err(NativeVtError::DeviceLost));
    assert_eq!(
        f.vt.recreate_for_device(&f.runtime),
        Err(NativeVtError::DeviceLost)
    );
    assert!(f.vt.has_pending_upload());
    assert!(matches!(
        old.clone().execute_prepared(
            &f.runtime,
            RenderGraph::new(rhi::Extent3d::new_2d(4, 4))
                .prepare()
                .unwrap(),
            &[],
        ),
        Err(NativeVtSubmitError::Vt(NativeVtError::DeviceLost))
    ));
    f.runtime.recover_device().unwrap();
    assert_ne!(old.device_generation(), f.runtime.device_generation());
    assert_eq!(f.stage(), Err(NativeVtError::DeviceMismatch));
    assert!(matches!(
        old.clone().execute_prepared(
            &f.runtime,
            RenderGraph::new(rhi::Extent3d::new_2d(4, 4))
                .prepare()
                .unwrap(),
            &[],
        ),
        Err(NativeVtSubmitError::Vt(NativeVtError::DeviceMismatch))
    ));
    f.vt.recreate_for_device(&f.runtime).unwrap();
    assert!(!f.vt.has_pending_upload());
    assert!(f.vt.active_binding(&f.runtime).unwrap().is_none());
    assert_eq!(f.refresh().unwrap(), 1);
    assert_eq!(f.row().virtual_texture_id, NO_VIRTUAL_TEXTURE);
    assert!(Arc::ptr_eq(&token, f.sources.get(7).unwrap()));
    drop(old);
    assert_eq!(
        old_device.inner.live_resource_count(),
        10,
        "both abandoned old VT slices are left to their creator device's teardown"
    );
    f.stage().unwrap();
    assert_eq!(f.publish().unwrap(), NativeVtPublish::Waiting);
    assert_eq!(f.refresh().unwrap(), 0);
    f.runtime.queue().complete();
    assert_eq!(
        f.publish().unwrap(),
        NativeVtPublish::Published { refreshed_rows: 1 }
    );
    assert_eq!(
        f.binding().device_generation(),
        f.runtime.device_generation()
    );
    assert_eq!(f.row().virtual_texture_id, 7);
}

#[test]
fn native_vt_abandoned_consumer_callback_does_not_release_inactive_slice() {
    let mut f = Fixture::new();
    f.stage().unwrap();
    f.runtime.queue().complete();
    f.publish().unwrap();
    let old = f.binding();
    f.stage().unwrap();
    f.runtime.queue().complete();
    f.publish().unwrap();
    let frame = old
        .execute_prepared(
            &f.runtime,
            RenderGraph::new(rhi::Extent3d::new_2d(4, 4))
                .prepare()
                .unwrap(),
            &[],
        )
        .unwrap();
    f.runtime.queue().abandon();
    assert!(!frame.is_complete());
    drop(frame);
    assert_eq!(f.stage(), Err(NativeVtError::SliceBusy));
    let device = f.runtime.device_handle();
    drop(f);
    assert_eq!(device.inner.live_resource_count(), 5);
}

#[test]
fn native_vt_rejected_consumer_submission_discharges_only_its_unsubmitted_obligation() {
    let mut f = Fixture::new();
    f.stage().unwrap();
    f.runtime.queue().complete();
    f.publish().unwrap();
    let old = f.binding();
    f.stage().unwrap();
    f.runtime.queue().complete();
    f.publish().unwrap();
    let foreign = runtime();
    let import = foreign
        .execute_prepared_owned(
            RenderGraph::new(rhi::Extent3d::new_2d(4, 4))
                .prepare()
                .unwrap(),
            &[],
        )
        .unwrap();
    let writes = f.runtime.queue().writes();
    let submitted = f.runtime.queue().progress.lock().unwrap().submitted;
    assert!(matches!(
        old.execute_prepared(
            &f.runtime,
            RenderGraph::new(rhi::Extent3d::new_2d(4, 4))
                .prepare()
                .unwrap(),
            &[import],
        ),
        Err(NativeVtSubmitError::Submission(
            prism_render_runtime::SubmissionError::ForeignImportOwner { index: 0 }
        ))
    ));
    assert_eq!(f.runtime.queue().writes(), writes);
    assert_eq!(
        f.runtime.queue().progress.lock().unwrap().submitted,
        submitted
    );
    f.stage().unwrap();
    f.runtime.queue().complete();
    f.publish().unwrap();
    foreign.queue().complete();
}

#[test]
fn native_vt_empty_publication_revokes_revision_max_without_resubmission() {
    let mut f = Fixture::new();
    f.stage().unwrap();
    f.runtime.queue().complete();
    f.publish().unwrap();
    let row = f.row();
    let table = f.binding().page_table();
    let other = f.materials.registry_mut().allocate().unwrap();
    let mut record = fallback_material_record(other, u64::from(u32::MAX));
    record.textures = vec![row];
    f.materials.publish(record).unwrap();
    f.materials.registry_mut().take_dirty();
    let before = f.materials.registry().snapshot();
    f.vt.stage(
        &f.runtime,
        &f.sources,
        f.materials.textures(),
        &NativeVtResidency::default(),
    )
    .unwrap();
    f.runtime.queue().complete();
    let submitted = f.runtime.queue().progress.lock().unwrap().submitted;
    assert_eq!(
        f.publish().unwrap(),
        NativeVtPublish::Published { refreshed_rows: 2 }
    );
    assert!(!f.vt.has_pending_upload());
    assert_ne!(f.binding().page_table(), table);
    assert_eq!(f.row().virtual_texture_id, NO_VIRTUAL_TEXTURE);
    let other_record = f.materials.registry().get(other).unwrap();
    assert_eq!(other_record.revision, u32::MAX);
    assert_eq!(other_record.textures, vec![f.row()]);
    assert_eq!(f.materials.registry().get(f.handle).unwrap().revision, 1);
    assert_eq!(f.materials.registry().snapshot().epoch, before.epoch + 1);
    assert_eq!(
        f.materials.registry_mut().take_dirty(),
        vec![f.handle.index, other.index]
    );
    assert_eq!(
        f.runtime.queue().progress.lock().unwrap().submitted,
        submitted
    );
}

#[test]
fn native_vt_publication_invalidation_and_reauthored_rows_preserve_runtime_state() {
    use prism_render_architecture::{
        abi::{AbiHash, AbiVersion},
        shader_package::{
            ArtifactDigest, ManifestExpectation, ModuleId, PermutationSpace, ShaderPackageId,
            ShaderPackageManifest, ShaderPackageRegistry, ShaderPackageVariantManifest,
        },
    };
    use prism_render_material::{
        BuiltinProgram, ChannelMask, CompiledEntry, DynamicChannels, ExecutionFamily,
        MaterialExecutionTables, ProgramRef, ShadingPlan,
    };

    let mut f = Fixture::new();
    let permutations = PermutationSpace::new([2]);
    let manifest = ShaderPackageManifest {
        id: ShaderPackageId::new("native-vt-runtime-tail-fixture"),
        source_revision: "1".into(),
        entry_point: "main".into(),
        abi_version: AbiVersion::from_parts(1, 0),
        abi_hash: AbiHash::from_bytes(
            prism_render_material::response_source::RESPONSE_FUNCTION_ABI.as_bytes(),
        ),
        variants: ShaderPackageVariantManifest::from_digests(
            &permutations,
            [ArtifactDigest([1; 32]), ArtifactDigest([2; 32])],
        )
        .unwrap(),
    };
    let compiled = CompiledEntry::from_response_wgsl(
        manifest.clone(),
        vec!["var response: RegisteredResponse; return response;".into(); 2],
    )
    .unwrap();
    let mut packages = ShaderPackageRegistry::new();
    packages
        .register_verified(compiled.artifact().clone())
        .unwrap();
    packages
        .validate(
            &manifest.id,
            &ManifestExpectation::new(manifest.abi_version, manifest.abi_hash),
        )
        .unwrap();
    let mut tables = MaterialExecutionTables::new();
    let package = tables.register_package(&compiled, &packages).unwrap();
    let mut binding = None;
    // Unlit is split-only; Lit supports fused execution. Keep both real
    // program/plan registrations rather than bypassing family validation.
    for (index, (builtin, family)) in [
        (BuiltinProgram::Unlit, ExecutionFamily::SplitResponse),
        (BuiltinProgram::Lit, ExecutionFamily::FusedBuiltin),
    ]
    .into_iter()
    .enumerate()
    {
        let program = tables
            .register_program(
                ProgramRef::Builtin(builtin),
                &compiled,
                package,
                ModuleId(index as u64),
                permutations.clone(),
            )
            .unwrap();
        if builtin == BuiltinProgram::Unlit {
            assert!(matches!(
                tables.register_plan(
                    ShadingPlan::single_builtin(builtin),
                    program,
                    None,
                    ExecutionFamily::FusedBuiltin,
                    ChannelMask::default(),
                ),
                Err(prism_render_material::MaterialExecutionError::InvalidExecutionFamily)
            ));
        }
        let plan = tables
            .register_plan(
                ShadingPlan::single_builtin(builtin),
                program,
                None,
                family,
                ChannelMask::default(),
            )
            .unwrap();
        binding = Some(
            tables
                .bind_raw(plan.get(), program.get(), package.get(), 1)
                .unwrap(),
        );
    }
    let binding = binding.unwrap();
    let dynamic = DynamicChannels {
        wetness: 0.25,
        snow: 0.5,
        damage: 0.75,
        dissolve: 1.0,
    };
    f.materials
        .registry_mut()
        .install_execution_tables(&tables)
        .unwrap();
    let handle = f.handle;
    let assert_tail = |materials: &RenderMaterials| {
        assert_eq!(
            materials.registry().dynamic_channels_at(handle.index),
            dynamic
        );
        assert_eq!(
            materials.registry().execution_binding_at(handle.index),
            Some(binding)
        );
        let (headers, _, _) = materials.registry().gpu_tables();
        assert_eq!(
            headers[handle.index as usize].execution_binding_words(),
            binding.to_words()
        );
        let header = &headers[handle.index as usize];
        assert_eq!(
            DynamicChannels::from_words([
                header.dynamic_wetness,
                header.dynamic_snow,
                header.dynamic_damage,
                header.dynamic_dissolve,
            ]),
            dynamic
        );
        assert_eq!(materials.registry().get(handle).unwrap().handle, handle);
    };
    f.materials
        .registry_mut()
        .set_dynamic_channels(handle, dynamic)
        .unwrap();
    f.materials
        .registry_mut()
        .set_execution_binding(handle, binding)
        .unwrap();
    f.stage().unwrap();
    f.runtime.queue().complete();
    f.publish().unwrap();
    assert_tail(&f.materials);
    assert_eq!(f.materials.registry().get(handle).unwrap().revision, 1);
    f.sources
        .capture_image(image_id(1), &image(10), request(7))
        .unwrap();
    assert_eq!(f.refresh().unwrap(), 1);
    assert_tail(&f.materials);
    assert_eq!(f.materials.registry().get(handle).unwrap().revision, 1);
    f.stage().unwrap();
    f.runtime.queue().complete();
    f.publish().unwrap();
    assert_tail(&f.materials);
    assert_eq!(f.materials.registry().get(handle).unwrap().revision, 1);

    let mut authored = f.materials.registry().get(handle).unwrap().clone();
    authored.revision = u32::MAX;
    authored.textures[0] = GpuMaterialTexture::non_virtual(f.slot.index, f.slot.generation, 3, 17);
    f.materials.publish(authored).unwrap();
    // Reauthoring intentionally resets runtime state. Restore the caller's
    // current state, which the subsequent texture refresh must preserve.
    f.materials
        .registry_mut()
        .set_dynamic_channels(handle, dynamic)
        .unwrap();
    f.materials
        .registry_mut()
        .set_execution_binding(handle, binding)
        .unwrap();
    assert_eq!(f.refresh().unwrap(), 1);
    assert_tail(&f.materials);
    assert_eq!(
        f.materials.registry().get(handle).unwrap().revision,
        u32::MAX
    );
    assert_eq!((f.row().semantic, f.row().sampler_index), (3, 17));
    f.materials.registry_mut().take_dirty();
    let before = f.materials.registry().snapshot();
    assert_eq!(f.refresh().unwrap(), 0);
    assert_eq!(f.materials.registry().snapshot(), before);
    assert!(f.materials.registry_mut().take_dirty().is_empty());
    f.sources.remove_image(image_id(1));
    assert_eq!(f.refresh().unwrap(), 1);
    assert_eq!(f.row().virtual_texture_id, NO_VIRTUAL_TEXTURE);
    assert_eq!(
        f.materials.registry().get(handle).unwrap().revision,
        u32::MAX
    );
    assert_tail(&f.materials);
}

#[test]
fn native_vt_layer_tails_and_portable_collisions_are_validated_before_writes() {
    let mut f = Fixture::new();
    f.sources
        .insert(image_id(1), snapshot(7, 4, 2, MaterialVtFormat::Rgba8Unorm))
        .unwrap();
    let mut residency = NativeVtResidency {
        pages: vec![(key(7, 2, 0), 0)],
        portable: vec![],
    };
    assert_eq!(
        f.vt.stage(&f.runtime, &f.sources, f.materials.textures(), &residency),
        Err(NativeVtError::MissingCoarsestPage(7))
    );
    assert_eq!(f.runtime.queue().writes(), 0);
    residency.pages.push((key(7, 2, 1), 1));
    f.vt.stage(&f.runtime, &f.sources, f.materials.textures(), &residency)
        .unwrap();
    f.runtime.queue().complete();
    f.publish().unwrap();
    let gpu: NativeVtGpuConfig =
        bytemuck::pod_read_unaligned(&f.runtime.queue().buffer_bytes(f.binding().config()));
    assert_eq!((gpu.entry_count, gpu.coarsest_mip), (2, 2));

    f.sources
        .capture_image(image_id(2), &image(3), request(8))
        .unwrap();
    let slot = f.materials.textures_mut().acquire(image_id(2)).unwrap();
    let writes = f.runtime.queue().writes();
    let cases = [
        (
            NativeVtResidency {
                pages: vec![],
                portable: vec![7],
            },
            NativeVtError::InvalidPortableSource(7),
        ),
        (
            NativeVtResidency {
                pages: vec![],
                portable: vec![8, 8],
            },
            NativeVtError::InvalidPortableSource(8),
        ),
        (
            NativeVtResidency {
                pages: vec![(key(7, 2, 0), slot.index), (key(7, 2, 1), 1)],
                portable: vec![8],
            },
            NativeVtError::DuplicateSlot(slot.index),
        ),
        (
            NativeVtResidency {
                pages: vec![(key(7, 2, 2), 0)],
                portable: vec![],
            },
            NativeVtError::Source(
                f.sources
                    .get(7)
                    .unwrap()
                    .cpu_snapshot()
                    .expect("CPU source snapshot")
                    .page_metadata(key(7, 2, 2))
                    .unwrap_err(),
            ),
        ),
    ];
    for (residency, expected) in cases {
        assert_eq!(
            f.vt.stage(&f.runtime, &f.sources, f.materials.textures(), &residency),
            Err(expected)
        );
        assert_eq!(f.runtime.queue().writes(), writes);
        assert!(!f.vt.has_pending_upload());
    }
}

#[test]
fn native_vt_default_config_is_const_constructible_and_needs_completed_publication() {
    const CONFIG: NativeVtConfig = NativeVtConfig::new();
    assert_eq!(CONFIG, NativeVtConfig::DEFAULT);
    assert_eq!(CONFIG, NativeVtConfig::default());
    assert_eq!(
        CONFIG,
        NativeVtConfig {
            page_size: 128,
            gutter_px: 4,
            tiles_per_row: 8,
            atlas_layers: 1,
            max_entries: 64,
            format: MaterialVtFormat::Rgba8Unorm,
        }
    );
    assert_eq!(CONFIG.capacity().unwrap(), 64);
    assert_eq!(CONFIG.geometry().unwrap().tile_extent_px(), 136);
    let runtime = runtime();
    let device = runtime.device_handle();
    CONFIG.validate(&*device).unwrap();
    assert_eq!(device.inner.live_resource_count(), 0);
    let mut vt = NativeMaterialVt::new(&runtime, CONFIG).unwrap();
    assert_eq!(vt.config(), CONFIG);
    assert_eq!(device.inner.live_resource_count(), 10);
    assert_eq!(runtime.queue().writes(), 0);
    assert!(vt.active_binding(&runtime).unwrap().is_none());
    let sources = NativeVtSources::default();
    let mut materials = RenderMaterials::new(8);
    vt.stage(
        &runtime,
        &sources,
        materials.textures(),
        &NativeVtResidency::default(),
    )
    .unwrap();
    assert_eq!(
        vt.try_publish(&runtime, &sources, &mut materials).unwrap(),
        NativeVtPublish::Waiting
    );
    runtime.queue().complete();
    assert_eq!(
        vt.try_publish(&runtime, &sources, &mut materials).unwrap(),
        NativeVtPublish::Published { refreshed_rows: 0 }
    );
    let binding = vt.active_binding(&runtime).unwrap().unwrap();
    let gpu: NativeVtGpuConfig =
        bytemuck::pod_read_unaligned(&runtime.queue().buffer_bytes(binding.config()));
    assert_eq!((gpu.enabled, gpu.entry_count), (0, 0));
    assert_eq!((gpu.page_size, gpu.tile_extent_px), (128, 136));
    drop(binding);
    drop(vt);
    assert_eq!(device.inner.live_resource_count(), 0);
}

#[test]
fn native_vt_pending_upload_outlives_owner_until_actual_completion() {
    let runtime = runtime();
    let device = runtime.device_handle();
    let sources = NativeVtSources::default();
    let materials = RenderMaterials::new(8);
    let mut vt = NativeMaterialVt::new(&runtime, config()).unwrap();
    assert_eq!(device.inner.live_resource_count(), 10);
    vt.stage(
        &runtime,
        &sources,
        materials.textures(),
        &NativeVtResidency::default(),
    )
    .unwrap();
    assert!(vt.has_pending_upload());
    assert!(vt.active_binding(&runtime).unwrap().is_none());

    // The unused slice can die immediately; submitted resources belong to
    // the queue callback even after the producer resource is removed.
    drop(vt);
    assert_eq!(device.inner.live_resource_count(), 5);
    runtime.queue().complete();
    assert_eq!(device.inner.live_resource_count(), 0);
}

#[test]
fn native_vt_cached_lease_can_submit_after_owner_drop_and_retains_until_completion() {
    let runtime = runtime();
    let device = runtime.device_handle();
    let sources = NativeVtSources::default();
    let mut materials = RenderMaterials::new(8);
    let mut vt = NativeMaterialVt::new(&runtime, config()).unwrap();
    vt.stage(
        &runtime,
        &sources,
        materials.textures(),
        &NativeVtResidency::default(),
    )
    .unwrap();
    runtime.queue().complete();
    assert_eq!(
        vt.try_publish(&runtime, &sources, &mut materials).unwrap(),
        NativeVtPublish::Published { refreshed_rows: 0 }
    );
    let binding = vt.active_binding(&runtime).unwrap().unwrap();
    let cached_group_lease = binding.clone();
    drop(vt);
    assert_eq!(device.inner.live_resource_count(), 5);

    // A cached consumer may submit after producer teardown. Dropping both
    // the cache lease and frame must not substitute for its RHI callback.
    let frame = binding
        .execute_prepared(
            &runtime,
            RenderGraph::new(rhi::Extent3d::new_2d(4, 4))
                .prepare()
                .unwrap(),
            &[],
        )
        .unwrap();
    assert!(!frame.is_complete());
    drop(cached_group_lease);
    drop(frame);
    assert_eq!(device.inner.live_resource_count(), 5);
    runtime.queue().complete();
    assert_eq!(device.inner.live_resource_count(), 0);
}

#[test]
fn native_vt_bad_config_does_not_allocate_resources_or_write_queue() {
    let runtime = runtime();
    let mut zero_rows = config();
    zero_rows.tiles_per_row = 0;
    let mut zero_layers = config();
    zero_layers.atlas_layers = 0;
    let mut overflow = config();
    overflow.tiles_per_row = u32::MAX;
    let mut oversized = config();
    oversized.max_entries = u32::MAX;
    let mut unfilterable = config();
    unfilterable.format = MaterialVtFormat::Rgba32Float;
    for invalid in [zero_rows, zero_layers, overflow, oversized, unfilterable] {
        assert!(NativeMaterialVt::new(&runtime, invalid).is_err());
        assert_eq!(runtime.device_handle().inner.live_resource_count(), 0);
        assert_eq!(runtime.queue().writes(), 0);
    }
}
