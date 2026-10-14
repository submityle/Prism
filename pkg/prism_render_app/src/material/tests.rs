//! Behavioural tests for the material ABI buffers.

use prism_render_driver::{BufferUsages, DeviceDescriptor, Instance, RequestAdapterOptions};
use prism_render_driver_null::NullInstance;
use prism_render_material::{fallback_material_record, BuiltinProgram, ClosureKind, MaterialRecord, ShadingPlan};

use prism_render_architecture::abi::GenerationalHandle;

use crate::material::MaterialParameterBuffers;

fn null_device() -> (
    prism_render_driver_null::NullDevice,
    impl prism_render_driver::RenderQueue,
) {
    let options = RequestAdapterOptions::default();
    let adapter = NullInstance::new()
        .request_adapter(&options)
        .expect("the null backend always has an adapter");
    NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up")
}

/// A texture reference as the ABI carries it: the image's slot identity plus the
/// semantic it is bound as.
fn texture_entry(index: u32, semantic: u32) -> prism_render_material::GpuMaterialTexture {
    prism_render_material::GpuMaterialTexture::non_virtual(index, 1, semantic, 0)
}

/// A record whose packed block is wider than a plain dielectric's, so a test can
/// make a material change block size.
fn record_with_lobes(index: u32, lobes: u32) -> MaterialRecord {
    let handle = GenerationalHandle::new(index, 1);
    let mut record = fallback_material_record(handle, 1);
    record.closure_mask = lobes;
    record
}

#[test]
fn syncing_materials_places_their_blocks_and_writes_their_rows() {
    let (device, queue) = null_device();
    // Sized generously so the assertions below are about *placement*, not about
    // the growth path (which its own test covers).
    let mut buffers = MaterialParameterBuffers::new(&device, 4, 512, 64, BufferUsages::NONE);
    assert_eq!(buffers.header_capacity(), 4);

    let first = record_with_lobes(0, 1);
    let second = record_with_lobes(1, 1);
    let first_words = first.packed_parameters().pack().len() as u32;
    let second_words = second.packed_parameters().pack().len() as u32;
    let report = buffers.sync(
        &device,
        &queue,
        [(0, first.clone()), (1, second.clone())],
        7,
    );

    assert_eq!(report.headers_written, 2, "both rows are new");
    assert_eq!(report.parameters_written, 2, "both blocks are new");
    assert!(!report.grew);
    assert_eq!(
        buffers.used_words(),
        first_words + second_words,
        "the heap covers both blocks"
    );

    let row = buffers.header(0).expect("row 0");
    assert_eq!(row.is_active, 1);
    assert_eq!(row.material_epoch_low, 7, "the epoch rides the row");
    assert_eq!(
        row.parameter_offset, 0,
        "the first block starts at the heap's origin"
    );
    assert_eq!(
        row.parameter_size as usize,
        first.packed_parameters().packed_size_bytes()
    );
    let words = buffers.parameters_of(0).expect("block 0");
    assert_eq!(
        words,
        first.packed_parameters().pack().as_slice(),
        "the words are the ABI's packed block"
    );

    let row = buffers.header(1).expect("row 1");
    assert_eq!(
        row.parameter_offset, first_words,
        "the second block follows the first"
    );
    assert_eq!(
        buffers.parameters_of(1).expect("block 1"),
        second.packed_parameters().pack().as_slice()
    );
}

/// A quiet frame writes nothing, and a re-sync of unchanged records is a no-op.
#[test]
fn an_unchanged_sync_writes_nothing() {
    let (device, queue) = null_device();
    let mut buffers = MaterialParameterBuffers::new(&device, 2, 32, 16, BufferUsages::NONE);
    let record = record_with_lobes(0, 1);
    buffers.sync(&device, &queue, [(0, record.clone())], 7);

    let report = buffers.sync(&device, &queue, [(0, record)], 7);

    assert_eq!(report.headers_written, 0);
    assert_eq!(report.parameters_written, 0);
    assert_eq!(report.relocated, 0);
}

/// A material that gains a lobe needs a wider block, so its old run goes back to
/// the heap and the new one is placed (and the row follows it).
#[test]
fn a_wider_block_relocates_and_frees_the_old_run() {
    let (device, queue) = null_device();
    let mut buffers = MaterialParameterBuffers::new(&device, 2, 64, 16, BufferUsages::NONE);
    buffers.sync(&device, &queue, [(0, record_with_lobes(0, 1))], 1);
    let narrow_words = buffers.used_words();
    assert_eq!(buffers.header(0).expect("row").parameter_offset, 0);

    // A second closure kind makes the derived lobe set larger, so the packed
    // block grows.
    let wider = record_with_lobes(0, 1 | (1 << ClosureKind::ClearCoat as u32));
    assert!(
        wider.packed_parameters().pack().len()
            > record_with_lobes(0, 1).packed_parameters().pack().len(),
        "the fixture really changes the block width"
    );
    let report = buffers.sync(&device, &queue, [(0, wider.clone())], 2);

    assert_eq!(report.relocated, 1, "the block moved");
    assert!(buffers.used_words() > narrow_words);
    assert_eq!(
        buffers.parameters_of(0).expect("block 0"),
        wider.packed_parameters().pack().as_slice()
    );
}

/// A material index past the current capacity grows the buffers, and the mirrors
/// refill the new allocation.
#[test]
fn a_higher_index_grows_and_refills() {
    let (device, queue) = null_device();
    let mut buffers = MaterialParameterBuffers::new(&device, 2, 8, 16, BufferUsages::NONE);
    buffers.sync(&device, &queue, [(0, record_with_lobes(0, 1))], 1);

    let report = buffers.sync(
        &device,
        &queue,
        [(0, record_with_lobes(0, 1)), (9, record_with_lobes(9, 1))],
        2,
    );

    assert!(report.grew, "index 9 does not fit in two rows");
    assert!(buffers.header_capacity() >= 10);
    assert_eq!(
        buffers.header(0).expect("row 0").is_active,
        1,
        "the mirror refilled the existing material"
    );
    assert_eq!(buffers.header(9).expect("row 9").is_active, 1);
    assert_eq!(
        buffers.header(5).expect("row 5").is_active,
        0,
        "rows nobody claimed stay explicitly inactive"
    );
}

/// Retiring a material frees its run and leaves a generation-tagged inactive
/// row, so a stale handle cannot read freed words.
#[test]
fn removing_a_material_frees_its_run_and_retires_its_row() {
    let (device, queue) = null_device();
    let mut buffers = MaterialParameterBuffers::new(&device, 2, 64, 16, BufferUsages::NONE);
    buffers.sync(
        &device,
        &queue,
        [(0, record_with_lobes(0, 1)), (1, record_with_lobes(1, 1))],
        1,
    );

    buffers.remove(&queue, 0, 1);

    let row = buffers.header(0).expect("row 0");
    assert_eq!(row.is_active, 0, "the row is retired");
    assert_eq!(row.generation, 1, "and tagged with the slot's generation");
    assert!(buffers.parameters_of(0).is_none(), "its run is gone");

    // Its words are reusable: the next material placed gets the freed run, which
    // is what "returned to the heap" means for a first-fit free list (the
    // high-water mark only falls when the *tail* is freed).
    let report = buffers.sync(&device, &queue, [(2, record_with_lobes(2, 1))], 2);
    assert_eq!(report.relocated, 0, "material 2 is new");
    assert_eq!(
        buffers.header(2).expect("row 2").parameter_offset,
        0,
        "the freed run at the origin was reused"
    );
    assert_eq!(report.relocated, 0, "material 1 keeps its own run");
}

/// The buffers are released together, and releasing twice is not an error.
#[test]
fn releasing_frees_both_buffers() {
    let (device, queue) = null_device();
    let mut buffers = MaterialParameterBuffers::new(&device, 2, 32, 16, BufferUsages::NONE);
    buffers.sync(&device, &queue, [(0, record_with_lobes(0, 1))], 1);
    assert_eq!(
        device.live_resource_count(),
        3,
        "headers + parameters + texture rows"
    );

    buffers.release(&device);
    assert_eq!(device.live_resource_count(), 0);
    buffers.release(&device);
    assert_eq!(
        device.live_resource_count(),
        0,
        "releasing twice is a no-op"
    );
}

#[test]
fn authoring_illumination_is_preserved_without_polluting_the_gpu_header() {
    let (device, queue) = null_device();
    let mut buffers = MaterialParameterBuffers::new(&device, 1, 32, 16, BufferUsages::NONE);
    let mut record = fallback_material_record(GenerationalHandle::new(0, 1), 1);
    record.response_plan = ShadingPlan::single_builtin(BuiltinProgram::Npr);

    buffers.sync(&device, &queue, [(0, record.clone())], 3);

    let row = buffers.header(0).expect("row");
    assert_eq!(
        row.execution_binding_words(),
        [u32::MAX; 5],
        "an unbound authoring record must use the ABI8 fallback tuple"
    );
    assert_eq!(
        buffers.parameters_of(0).expect("block"),
        record.packed_parameters().pack().as_slice(),
        "the packed block matches the record"
    );
}

/// The material binding's layout is the ABI's five bindings, and the two arrays
/// carry a `count`: an array entry without one would bind a single resource where
/// the shader indexes dynamically.
#[test]
fn the_material_binding_layout_declares_the_abi_and_its_arrays() {
    use crate::material::MaterialBindingLayout;
    use prism_render_driver::{
        BindingType, BufferBindingType, SamplerBindingType, TextureSampleType,
    };

    let entries = MaterialBindingLayout::entries(64);
    assert_eq!(entries.len(), 5, "three buffers and two arrays");
    for (binding, entry) in entries.iter().enumerate() {
        assert_eq!(entry.binding as usize, binding);
        assert!(matches!(
            entry.ty,
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                ..
            } | BindingType::TextureArray {
                sample_type: TextureSampleType::Float,
                ..
            } | BindingType::SamplerArray(SamplerBindingType::Filtering)
        ));
    }
    assert_eq!(entries[4].count, Some(64), "the sampler array is sized");
    assert_eq!(entries[3].count, Some(64), "the texture array is sized");
    assert!(
        entries[..3].iter().all(|entry| entry.count.is_none()),
        "the buffers are not arrays"
    );
}

/// The binding is accepted by a real device: the layout's array counts and the
/// resources bound must agree, which only the backend can confirm (a mismatch is
/// a validation error at group creation).
#[test]
fn the_material_binding_creates_on_a_gpu() {
    use crate::material::texture_heap::FIRST_DYNAMIC_SLOT;
    use crate::material::{FallbackTextures, MaterialBindingLayout, MaterialTextureSlots};
    use prism_render_driver::{BufferDescriptor, BufferUsages, Extent3d, Instance, RenderDevice};
    use prism_render_driver_wgpu::WgpuInstance;

    let extent = Extent3d::new_2d(4, 4);
    let Ok(runtime) = prism_render_runtime::RenderRuntime::<WgpuInstance>::wgpu(extent) else {
        assert!(
            !crate::tests::require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        return;
    };
    let device = runtime.device();
    let queue = runtime.queue();

    let Some(layout) = MaterialBindingLayout::new(device, FIRST_DYNAMIC_SLOT + 1) else {
        // A backend without bindless has no business building this group; the
        // caller takes its non-bindless path instead.
        return;
    };
    assert_eq!(layout.slots, FIRST_DYNAMIC_SLOT + 1);

    let storage = || {
        device.create_buffer(&BufferDescriptor {
            label: Some("material abi".into()),
            size: 256,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let fallbacks = FallbackTextures::new(device, queue);
    let slots = MaterialTextureSlots::build(
        device,
        &crate::RenderAssets::<crate::GpuImage>::new(),
        &fallbacks,
        &crate::TextureSlotHeap::new(FIRST_DYNAMIC_SLOT + 1),
    );
    assert_eq!(slots.len(), (FIRST_DYNAMIC_SLOT + 1) as usize);

    let group = layout.create_group(device, storage(), storage(), Some(storage()), &slots);
    let _ = group;
    assert!(runtime.wait_idle());
}

/// The bindless arrays are *dense* and slot-addressed: a resident image's view
/// sits at its slot, a reserved slot holds its own fallback texture, and a freed
/// slot goes back to the fallback rather than keeping a stale view.
#[test]
fn the_bindless_arrays_place_one_view_and_sampler_per_slot() {
    use crate::material::texture_heap::{FIRST_DYNAMIC_SLOT, WHITE_SLOT};
    use crate::material::{
        FallbackTextures, MaterialTextureSlots, RenderMaterials, TextureRequest,
    };
    use crate::render_asset::{RenderAsset, RenderAssets};
    use prism_asset::{AssetId, AssetIndex};
    use prism_image::Image;

    let (device, queue) = null_device();
    let fallbacks = FallbackTextures::new(&device, &queue);
    let image_id = |tag: u32| AssetId::<Image>::new(AssetIndex::from_parts(tag, 1));

    // Two prepared images, as the asset pipeline would have produced them.
    let mut images = RenderAssets::<crate::GpuImage>::new();
    for tag in [10_u32, 20] {
        let source = Image::new(
            prism_render_driver::Extent3d::new_2d(2, 2),
            prism_render_driver::TextureFormat::Rgba8Unorm,
            vec![tag as u8; 16],
        )
        .expect("valid");
        images.insert(
            image_id(tag),
            crate::GpuImage::prepare(&source, image_id(tag), &device, &queue, None)
                .expect("prepares"),
        );
    }

    let mut materials = RenderMaterials::new(8);
    materials.resolve_textures(
        1,
        [
            (0, TextureRequest::Image(image_id(10))),
            (
                1,
                TextureRequest::Fallback(crate::TextureFallback::FlatNormal),
            ),
            (2, TextureRequest::Image(image_id(20))),
        ],
    );
    // A small capacity so the arrays are cheap to check; re-clamping is allowed
    // here because the heap holds references but the *capacity* clamp is only
    // forbidden while images are resident — release first, then clamp.
    materials.release_textures(1);
    materials.set_texture_capacity(FIRST_DYNAMIC_SLOT + 4);
    materials.resolve_textures(
        1,
        [
            (0, TextureRequest::Image(image_id(10))),
            (
                1,
                TextureRequest::Fallback(crate::TextureFallback::FlatNormal),
            ),
            (2, TextureRequest::Image(image_id(20))),
        ],
    );

    let slots = MaterialTextureSlots::build(&device, &images, &fallbacks, materials.textures());
    assert_eq!(slots.len(), materials.textures().capacity() as usize);
    assert_eq!(
        slots.len(),
        slots.samplers().len(),
        "every slot gets both a view and a sampler"
    );

    // Reserved slots hold their own fallback textures.
    assert_eq!(slots.views()[WHITE_SLOT as usize], fallbacks.white.view);
    assert_eq!(
        slots.views()[crate::material::texture_heap::FLAT_NORMAL_SLOT as usize],
        fallbacks.flat_normal.view,
        "the flat normal is a *different* texture from white"
    );
    assert_eq!(
        slots.views()[crate::material::texture_heap::BLACK_SLOT as usize],
        fallbacks.black.view
    );

    // Resident images sit at their slots, with their own samplers.
    let first = materials.texture_slot(image_id(10)).expect("resident");
    let second = materials.texture_slot(image_id(20)).expect("resident");
    assert_eq!(
        slots.views()[first.index as usize],
        images.get(image_id(10)).expect("prepared").view
    );
    assert_eq!(
        slots.samplers()[first.index as usize],
        images.get(image_id(10)).expect("prepared").sampler
    );
    assert_eq!(
        slots.views()[second.index as usize],
        images.get(image_id(20)).expect("prepared").view
    );

    // A freed slot falls back rather than keeping the stale view.
    materials.release_textures(1);
    let slots = MaterialTextureSlots::build(&device, &images, &fallbacks, materials.textures());
    assert_eq!(
        slots.views()[first.index as usize],
        fallbacks.white.view,
        "the freed slot is a fallback again"
    );
}

/// Resolving a material's texture requests assigns bindless slots: real images
/// get dynamic slots (reference counted, so two materials share one), and a
/// semantic that has no map names its reserved fallback.
#[test]
fn resolving_textures_assigns_slots_and_counts_references() {
    use crate::material::texture_heap::FIRST_DYNAMIC_SLOT;
    use crate::material::{RenderMaterials, TextureRequest};
    use prism_asset::{AssetId, AssetIndex};
    use prism_image::Image;

    let image = |tag: u32| AssetId::<Image>::new(AssetIndex::from_parts(tag, 1));
    let mut materials = RenderMaterials::new(8);

    let rows = materials.resolve_textures(
        1,
        [
            (0, TextureRequest::Image(image(10))),
            (
                1,
                TextureRequest::Fallback(crate::TextureFallback::FlatNormal),
            ),
            (2, TextureRequest::Image(image(20))),
        ],
    );
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].index, FIRST_DYNAMIC_SLOT, "the first image");
    assert_eq!(
        rows[0].semantic, 0,
        "the ABI's semantic number rides the row"
    );
    assert_eq!(
        rows[0].sampler_index, rows[0].index,
        "sampler shares the slot"
    );
    assert_eq!(
        rows[1].index,
        crate::material::texture_heap::FLAT_NORMAL_SLOT,
        "a missing normal map is the reserved flat normal, not white"
    );
    assert_eq!(rows[2].index, FIRST_DYNAMIC_SLOT + 1, "the second image");
    assert_eq!(materials.textures().refcount(image(10)), 1);
    assert_eq!(materials.textures().stats().live_images, 2);

    // A second material referencing the same image shares its slot.
    let shared = materials.resolve_textures(2, [(0, TextureRequest::Image(image(10)))]);
    assert_eq!(shared[0].index, rows[0].index, "the slot is shared");
    assert_eq!(materials.textures().refcount(image(10)), 2);
    assert_eq!(materials.textures().stats().live_images, 2);
}

/// Republishing a material releases exactly the references it no longer holds,
/// and retiring it releases the rest — so a slot comes back only when the last
/// user is done with the image.
#[test]
fn republish_and_retire_release_exactly_what_was_held() {
    use crate::material::{RenderMaterials, TextureRequest};
    use prism_asset::{AssetId, AssetIndex};
    use prism_image::Image;

    let image = |tag: u32| AssetId::<Image>::new(AssetIndex::from_parts(tag, 1));
    let mut materials = RenderMaterials::new(8);
    materials.resolve_textures(
        1,
        [
            (0, TextureRequest::Image(image(10))),
            (1, TextureRequest::Image(image(20))),
        ],
    );
    let kept_slot = materials.texture_slot(image(20)).expect("resident");

    // Republish keeping only the second image.
    let rows = materials.resolve_textures(1, [(0, TextureRequest::Image(image(20)))]);
    assert_eq!(
        rows[0].index, kept_slot.index,
        "the kept image keeps its slot"
    );
    assert_eq!(
        materials.textures().refcount(image(10)),
        0,
        "the dropped image's reference was released"
    );
    assert_eq!(materials.textures().refcount(image(20)), 1);

    assert_eq!(materials.release_textures(1), 1, "one reference was held");
    assert_eq!(
        materials.textures().refcount(image(20)),
        0,
        "and now released"
    );
    assert_eq!(materials.texture_slot(image(20)), None);
    assert_eq!(
        materials.release_textures(1),
        0,
        "releasing twice is a no-op"
    );
}

/// A device-limit re-clamp is only safe while nothing is resident.
#[test]
fn the_slot_capacity_can_be_clamped_before_anything_is_published() {
    use crate::material::{RenderMaterials, TextureRequest};
    use prism_asset::{AssetId, AssetIndex};
    use prism_image::Image;

    let mut materials = RenderMaterials::new(8);
    materials.set_texture_capacity(64);
    assert_eq!(materials.textures().capacity(), 64);
    materials.resolve_textures(
        0,
        [(
            0,
            TextureRequest::Image(AssetId::<Image>::new(AssetIndex::from_parts(1, 1))),
        )],
    );
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        materials.set_texture_capacity(128);
    }));
    assert!(result.is_err(), "clamping under live rows panics");
}

/// The stage end to end: a material published into the render registry lands in
/// the ABI buffers during a frame, and retiring it leaves an inactive row.
#[test]
fn the_frame_packs_published_materials_into_the_buffers() {
    use crate::material::{MaterialUploadStats, RenderMaterials};
    use crate::RenderAppConfig;
    use prism_render_architecture::gpu_scene::GpuCompletionValue;
    use prism_render_driver::Extent3d;

    let mut app = prism_app::App::new();
    app.add_plugins(crate::PrismRenderPlugin::with_config(
        NullInstance::new(),
        RenderAppConfig::new(Extent3d::new_2d(8, 8)).with_material_capacity(8),
    ));

    let handle = {
        let render = app
            .sub_app_mut(crate::RenderApp)
            .expect("the render sub-app");
        let materials = render.world.resource_mut::<RenderMaterials>();
        let handle = materials
            .registry_mut()
            .allocate()
            .expect("the fixture capacity has room");
        let record = record_with_lobes(handle.index, 1);
        materials.publish(record).expect("the record publishes");
        handle
    };

    app.update();

    {
        let render = app
            .get_sub_app(crate::RenderApp)
            .expect("the render sub-app");
        let stats = render.world.resource::<MaterialUploadStats>();
        assert!(
            stats.last.headers_written >= 1,
            "the frame wrote the material's row (plus the permanent fallback)"
        );
        assert!(stats.last.parameters_written >= 1);
        assert_eq!(
            render
                .world
                .resource::<MaterialParameterBuffers>()
                .header(0)
                .expect("the fallback row")
                .is_active,
            1,
            "slot zero holds a live fallback, never a retired row"
        );
        let buffers = render.world.resource::<MaterialParameterBuffers>();
        let row = buffers.header(handle.index).expect("row");
        assert_eq!(row.is_active, 1);
        assert_eq!(row.generation, handle.generation);
        let fallback_words = render
            .world
            .resource::<MaterialParameterBuffers>()
            .header(0)
            .expect("the fallback row")
            .parameter_size
            / 4;
        assert_eq!(
            row.parameter_offset, fallback_words,
            "the material's block follows the permanent fallback's"
        );
    }

    // Retire it: the next frame's sync must find no record for the dirty index
    // and retire the row.
    {
        let render = app
            .sub_app_mut(crate::RenderApp)
            .expect("the render sub-app");
        render
            .world
            .resource_mut::<RenderMaterials>()
            .registry_mut()
            .retire(handle, GpuCompletionValue(0))
            .expect("the handle retires");
    }
    app.update();

    let render = app
        .get_sub_app(crate::RenderApp)
        .expect("the render sub-app");
    let buffers = render.world.resource::<MaterialParameterBuffers>();
    assert_eq!(
        buffers.header(handle.index).expect("row").is_active,
        0,
        "the retired material's row is inactive"
    );
    assert!(
        buffers.parameters_of(handle.index).is_none(),
        "and its words went back to the heap"
    );
}
/// A material's texture references own their own run in the texture table, and the
/// header's `texture_offset` / `texture_count` name exactly that run — the two
/// have to agree, or a shader reads another material's textures.
#[test]
fn texture_rows_are_placed_and_named_by_the_header() {
    let (device, queue) = null_device();
    let mut buffers = MaterialParameterBuffers::new(&device, 4, 512, 64, BufferUsages::NONE);

    let mut first = record_with_lobes(0, 1);
    first.textures = vec![texture_entry(3, 1), texture_entry(4, 2)];
    let mut second = record_with_lobes(1, 1);
    second.textures = vec![texture_entry(7, 1)];

    let report = buffers.sync(
        &device,
        &queue,
        [(0, first.clone()), (1, second.clone())],
        1,
    );
    assert_eq!(report.textures_written, 2, "both materials' rows");
    assert_eq!(buffers.used_texture_rows(), 3, "2 + 1 rows");

    let first_row = buffers.header(0).expect("row 0");
    assert_eq!(first_row.texture_count, 2);
    let first_texture_offset = first_row.texture_offset;
    let first_texture_count = first_row.texture_count;
    assert_eq!(
        buffers.textures_of(0).expect("material 0's rows"),
        first.textures.as_slice()
    );
    let second_row = buffers.header(1).expect("row 1");
    assert_eq!(second_row.texture_count, 1);
    assert_eq!(
        second_row.texture_offset,
        first_texture_offset + first_texture_count,
        "the second material's rows follow the first's"
    );
    assert_eq!(
        buffers.textures_of(1).expect("material 1's rows"),
        second.textures.as_slice()
    );
    assert_eq!(
        second_row.sampler_offset, second_row.texture_offset,
        "the sampler table shares the texture run's addresses today"
    );

    // Retiring the first material frees its rows for the next one.
    buffers.remove(&queue, 0, 1);
    assert!(buffers.textures_of(0).is_none());
    // Two rows, so they fit the hole the retired material left (a three-row
    // request would not, and would bump the table's high-water mark instead).
    let mut third = record_with_lobes(2, 1);
    third.textures = vec![texture_entry(9, 1), texture_entry(10, 2)];
    buffers.sync(&device, &queue, [(2, third.clone())], 2);
    assert_eq!(
        buffers.header(2).expect("row 2").texture_offset,
        first_texture_offset,
        "the freed rows were reused"
    );
    assert_eq!(
        buffers.textures_of(2).expect("material 2's rows"),
        third.textures.as_slice()
    );
    assert_eq!(
        buffers.used_texture_rows(),
        3,
        "the freed rows were reused rather than the table growing"
    );
}

/// A material with no textures names an *empty* run rather than pointing at
/// somebody else's rows.
#[test]
fn a_material_without_textures_gets_an_empty_run() {
    let (device, queue) = null_device();
    let mut buffers = MaterialParameterBuffers::new(&device, 2, 512, 16, BufferUsages::NONE);
    buffers.sync(&device, &queue, [(0, record_with_lobes(0, 1))], 1);

    let row = buffers.header(0).expect("row");
    assert_eq!(row.texture_count, 0);
    assert_eq!(
        buffers.textures_of(0).expect("an empty run"),
        &[][..],
        "a zero-length run, not the fallback's rows"
    );
}

/// The ABI lands byte-exact on hardware: after a frame packs the registry's
/// materials, the header rows read back from the GPU *are* the headers the ABI
/// builds at those offsets, and the word heap read back is their packed surface
/// blocks.
///
/// The buffers are replaced with `COPY_SRC` variants first — a diagnostic
/// readback is the only reason to want that usage — which also exercises the
/// growth path from an empty mirror (`mark_all_dirty`).
#[test]
fn the_abi_lands_byte_exact_on_a_gpu() {
    use crate::material::RenderMaterials;
    use crate::RenderAppConfig;
    use prism_render_architecture::abi::GenerationalHandle;
    use prism_render_driver::{BufferUsages, Extent3d};
    use prism_render_driver_wgpu::WgpuInstance;

    const CAPACITY: u32 = 4;
    let extent = Extent3d::new_2d(8, 8);
    if prism_render_runtime::RenderRuntime::<WgpuInstance>::wgpu(extent).is_err() {
        assert!(
            !crate::tests::require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        return;
    }

    let mut app = prism_app::App::new();
    app.add_plugins(crate::PrismRenderPlugin::with_config(
        WgpuInstance::new(),
        RenderAppConfig::new(extent).with_material_capacity(CAPACITY),
    ));

    // Two materials whose lobe sets differ, so their blocks differ in width and
    // the placement (not just the contents) has to be right.
    let (first, second) = {
        let render = app
            .sub_app_mut(crate::RenderApp)
            .expect("the render sub-app");
        let materials = render.world.resource_mut::<RenderMaterials>();
        let first = materials
            .registry_mut()
            .allocate()
            .expect("capacity for two");
        let second = materials
            .registry_mut()
            .allocate()
            .expect("capacity for two");
        let mut first_record = record_with_lobes(first.index, 1);
        first_record.textures = vec![texture_entry(3, 1), texture_entry(4, 2)];
        materials
            .publish(first_record)
            .expect("the first record publishes");
        let mut second_record =
            record_with_lobes(second.index, 1 | (1 << ClosureKind::ClearCoat as u32));
        second_record.textures = vec![texture_entry(7, 1)];
        materials
            .publish(second_record)
            .expect("the second record publishes");
        (first, second)
    };
    {
        // A fresh resource has an empty mirror, so the frame uploads every row.
        let render = app
            .sub_app_mut(crate::RenderApp)
            .expect("the render sub-app");
        let runtime = render
            .world
            .resource::<crate::RenderRuntimeResource<WgpuInstance>>();
        let device = runtime.runtime().device();
        render.world.insert_resource(MaterialParameterBuffers::new(
            device,
            CAPACITY,
            512,
            64,
            BufferUsages::COPY_SRC,
        ));
        render
            .world
            .resource_mut::<RenderMaterials>()
            .registry_mut()
            .mark_all_dirty();
    }

    app.update();

    let render = app
        .get_sub_app(crate::RenderApp)
        .expect("the render sub-app");
    let (headers, parameters, word_len, epoch) = {
        let buffers = render.world.resource::<MaterialParameterBuffers>();
        (
            buffers.headers().expect("the header buffer"),
            buffers.parameters().expect("the word heap"),
            buffers.used_words(),
            render.world.resource::<RenderMaterials>().snapshot().epoch,
        )
    };
    let (textures, texture_rows) = {
        let buffers = render.world.resource::<MaterialParameterBuffers>();
        (buffers.textures(), buffers.texture_capacity_rows())
    };
    let runtime = render
        .world
        .resource::<crate::RenderRuntimeResource<WgpuInstance>>();
    assert!(runtime.runtime().wait_idle(), "the upload completes");
    let queue = runtime.runtime().queue();

    let header_bytes = queue.read_buffer(
        headers,
        crate::material::header_row_bytes() * u64::from(CAPACITY),
    );
    let row_bytes = crate::material::header_row_bytes() as usize;
    let row = |index: u32| -> prism_render_material::GpuMaterialHeader {
        let start = index as usize * row_bytes;
        // `pod_read_unaligned` wants exactly one value's bytes, so the slice is
        // one row rather than the rest of the buffer.
        bytemuck::pod_read_unaligned(&header_bytes[start..start + row_bytes])
    };
    let words = queue.read_buffer(parameters, u64::from(word_len) * 4);
    let words_at = |offset: u32, len: u32| -> Vec<u32> {
        words[offset as usize * 4..(offset + len) as usize * 4]
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect()
    };

    // Slot zero is the permanent fallback, at the heap's origin.
    let fallback = fallback_material_record(GenerationalHandle::new(0, 0), epoch);
    let fallback_words = fallback.packed_parameters().pack();
    assert_eq!(
        row(0),
        fallback.header(0, 0, epoch),
        "the fallback row read from the GPU is the ABI's header"
    );
    assert_eq!(
        words_at(0, fallback_words.len() as u32),
        fallback_words,
        "the fallback's packed block is at the heap's origin"
    );

    // Each published material: the row names an offset, and the words at that
    // offset *are* its packed block — and the row is what the ABI builds for it.
    let texture_bytes_read = queue.read_buffer(
        textures.expect("the texture table"),
        crate::material::texture_row_bytes() * u64::from(texture_rows),
    );
    let texture_row = |row: u32| -> prism_render_material::GpuMaterialTexture {
        let row_bytes = crate::material::texture_row_bytes() as usize;
        let start = row as usize * row_bytes;
        bytemuck::pod_read_unaligned(&texture_bytes_read[start..start + row_bytes])
    };
    for (handle, lobes, expected_textures) in [
        (first, 1_u32, 2_u32),
        (second, 1 | (1 << ClosureKind::ClearCoat as u32), 1_u32),
    ] {
        let mut record = record_with_lobes(handle.index, lobes);
        record.textures = match handle.index {
            i if i == first.index => vec![texture_entry(3, 1), texture_entry(4, 2)],
            _ => vec![texture_entry(7, 1)],
        };
        assert_eq!(
            row(handle.index).texture_count,
            expected_textures,
            "the header names the material's texture rows"
        );
        let offset = row(handle.index).texture_offset;
        let read: Vec<prism_render_material::GpuMaterialTexture> = (0..expected_textures)
            .map(|i| texture_row(offset + i))
            .collect();
        assert_eq!(
            read, record.textures,
            "the texture rows read back are the ABI's entries"
        );
        let block = record.packed_parameters().pack();
        let gpu_row = row(handle.index);
        assert_eq!(gpu_row.is_active, 1, "material {} is live", handle.index);
        assert_eq!(gpu_row.generation, handle.generation);
        assert_eq!(
            gpu_row.parameter_size as usize,
            block.len() * 4,
            "the row's size is the block's byte length"
        );
        assert_eq!(
            words_at(gpu_row.parameter_offset, block.len() as u32),
            block,
            "the words at the row's offset are the material's packed block"
        );
        assert_eq!(
            gpu_row,
            record.header(gpu_row.parameter_offset, gpu_row.texture_offset, epoch),
            "the row read from the GPU is the ABI's header for that material"
        );
    }
    assert_ne!(
        row(first.index).parameter_offset,
        row(second.index).parameter_offset,
        "the two blocks are placed separately"
    );
}
