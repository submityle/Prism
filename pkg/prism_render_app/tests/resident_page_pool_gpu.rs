//! Physical GPU closure: authorized compressed ranges -> resident pages -> compute consumer.
//! Payload readback exists only in this test. PRISM_REQUIRE_GPU forbids adapter skips.
#![cfg(feature = "resident-page-streaming")]
#![expect(
    clippy::print_stderr,
    reason = "record physical adapter evidence or explicit skips"
)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use prism_app::App;
use prism_asset::StableGuid;
use prism_asset_container::{
    ChunkInput, ChunkPlanner, ChunkPolicy, Codec, ContainerIdentity, ContainerReader,
    ContainerWriter, GpuBlockDecodeBackend, GpuResidentManifestAsset, ManifestRoute, OverlaySet,
    RuntimeManifest,
};
use prism_gdeflate_gpu::{
    buffer, DeflateBatchLimits, DeflateTile, GpuContext, GpuDeflate, ResidentAssemblyError,
    ResidentAssemblyLimits, ResidentDeflateTile, GROUP, LANES, WORD,
};
use prism_render_app::{
    FixedPagePoolDescriptor, FixedPageReclaimEvent, PagePoolError, PrismRenderPlugin, RenderApp,
    RenderAppConfig, RenderRuntimeResource, ResidentPageAllocation, ResidentPageError,
    ResidentPagePool, ResidentRenderDevice, ResidentWgpuInstance,
};
use prism_render_driver::{
    Adapter, BufferUsages, DeviceDescriptor, Extent3d, Instance, PowerPreference,
    RequestAdapterOptions,
};
use prism_render_driver_wgpu::AdapterEvidence;
use prism_render_runtime::{RenderInitError, RenderRuntime};

// Avoid concurrent device/pipeline creation within this hardware test binary.
static GPU_TEST: Mutex<()> = Mutex::new(());
const BLOCK: usize = 65_536;
const LOCAL: StableGuid = StableGuid::from_u128(41);
const LOGICAL: StableGuid = StableGuid::from_u128(97);
const CONSUMER_XOR: u32 = 0xa759_31ed;

fn require_gpu() -> bool {
    std::env::var_os("PRISM_REQUIRE_GPU").is_some() || std::env::var_os("REQUIRE_GPU").is_some()
}

fn physical(info: &prism_render_driver::AdapterInfo) -> bool {
    let evidence = AdapterEvidence::from_info(info);
    if evidence.is_hardware() {
        eprintln!("resident page pool REAL HARDWARE GPU: {evidence}");
        true
    } else {
        assert!(!require_gpu(), "physical GPU required: {evidence}");
        eprintln!("SKIPPED resident page pool: non-hardware adapter: {evidence}");
        false
    }
}

fn runtime() -> Option<RenderRuntime<ResidentWgpuInstance>> {
    match RenderRuntime::headless(ResidentWgpuInstance::new(), Extent3d::new_2d(8, 8)) {
        Ok(runtime) => physical(runtime.adapter_info()).then_some(runtime),
        Err(RenderInitError::NoAdapter) => {
            assert!(
                !require_gpu(),
                "physical GPU required, but no adapter was found"
            );
            eprintln!("SKIPPED resident page pool: no adapter");
            None
        }
        Err(error) => panic!("resident runtime device creation failed: {error}"),
    }
}

fn snapshot(runtime: &RenderRuntime<ResidentWgpuInstance>) -> ResidentRenderDevice {
    let shared = ResidentRenderDevice::from_runtime(runtime)
        .expect("factory snapshot must match the committed runtime tuple");
    assert_eq!(shared.context().device(), runtime.device().wgpu_device());
    assert_eq!(shared.context().queue(), runtime.queue().wgpu_queue());
    shared
}

fn hardware_app() -> Option<App> {
    let instance = ResidentWgpuInstance::new();
    // Probe only the adapter: device creation remains the real plugin's job.
    let Some(adapter) = instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        force_fallback: false,
    }) else {
        assert!(
            !require_gpu(),
            "RenderApp requires a GPU but has no adapter"
        );
        eprintln!("SKIPPED resident RenderApp: no adapter");
        return None;
    };
    if !physical(adapter.info()) {
        return None;
    }
    let config = RenderAppConfig::new(Extent3d::new_2d(8, 8))
        .with_material_capacity(4)
        .with_material_parameter_words(64)
        .with_material_texture_rows(4)
        .with_particle_capacity(0);
    let mut app = App::new();
    app.add_plugins(PrismRenderPlugin::with_config(instance, config));
    let runtime = app
        .get_sub_app(RenderApp)
        .expect("real render sub-app")
        .world
        .resource::<RenderRuntimeResource<ResidentWgpuInstance>>()
        .runtime();
    assert!(
        physical(runtime.adapter_info()),
        "plugin selected a different non-hardware adapter"
    );
    snapshot(runtime);
    Some(app)
}

fn descriptor(page_size: u64, pages: u32) -> FixedPagePoolDescriptor {
    FixedPagePoolDescriptor::new(
        page_size,
        pages,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    )
}

fn pool(shared: ResidentRenderDevice, page_size: u64, pages: u32) -> ResidentPagePool {
    ResidentPagePool::new(
        shared,
        &descriptor(page_size, pages),
        page_size * u64::from(pages),
    )
    .expect("resident backing buffer and assembly pipeline")
}

fn assembly_limits() -> ResidentAssemblyLimits {
    ResidentAssemblyLimits {
        max_slices: 8,
        max_device_bytes: 4 << 20,
    }
}

fn decode_limits() -> DeflateBatchLimits {
    DeflateBatchLimits {
        max_tiles: 8,
        max_device_bytes: 4 << 20,
    }
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|n| (n as u8).wrapping_mul(37).wrapping_add(seed) ^ (n >> 8) as u8)
        .collect()
}

// Independently encode RFC 1951 stored blocks, not the production compressor.
fn decode(ctx: &GpuContext, bytes: &[u8]) -> ResidentDeflateTile {
    assert!(!bytes.is_empty() && bytes.len() <= u16::MAX as usize);
    let len = u16::try_from(bytes.len()).unwrap();
    let mut raw = vec![1]; // BFINAL=1, BTYPE=00 and byte alignment.
    raw.extend_from_slice(&len.to_le_bytes());
    raw.extend_from_slice(&(!len).to_le_bytes());
    raw.extend_from_slice(bytes);
    let mut interleaved = vec![0; raw.len().div_ceil(GROUP) * GROUP];
    let rounds = interleaved.len() / GROUP;
    for (index, &byte) in raw.iter().enumerate() {
        let word = index / WORD;
        interleaved
            [(word % LANES as usize * rounds + word / LANES as usize) * WORD + index % WORD] = byte;
    }
    GpuDeflate::new(ctx)
        .decompress_batch_resident(
            ctx,
            &[DeflateTile {
                interleaved: &interleaved,
                compressed_size: raw.len(),
                expected_output_size: bytes.len(),
            }],
            decode_limits(),
        )
        .expect("independent stored-DEFLATE fixture must decode on GPU")
        .pop()
        .unwrap()
}

fn manifest_asset(shared: &ResidentRenderDevice, raw: &[u8]) -> GpuResidentManifestAsset {
    let mut writer = ContainerWriter::new();
    writer.add(LOCAL, raw.to_vec()).unwrap();
    let plan = ChunkPlanner::new(ChunkPolicy {
        max_chunk_size: raw.len() as u64 + 64,
        chunk_alignment: 64,
        max_chunks: 1,
    })
    .plan([ChunkInput {
        guid: LOCAL,
        size: raw.len() as u64,
        alignment: 1,
        dependencies: Vec::new(),
        priority: 0,
    }])
    .unwrap();
    let bytes = writer
        .finish_with_chunk_compression(&plan, 64, Codec::BlockDeflate)
        .unwrap();
    let reader = ContainerReader::new(bytes).unwrap();
    // Separate CPU decode/oracle check: a framing/compression mistake cannot
    // silently become the expected GPU result.
    assert_eq!(reader.read_all(LOCAL).unwrap(), raw);
    assert_eq!(reader.entry(LOCAL).unwrap().raw_offset, 0);
    let info = reader.info();
    let mut manifest = RuntimeManifest::new();
    manifest
        .insert(ManifestRoute {
            guid: LOGICAL,
            container_name: "pages".into(),
            priority: 7,
            container_guid: LOCAL,
        })
        .unwrap();
    manifest
        .bind_container(
            "pages",
            ContainerIdentity {
                byte_len: info.total_len,
                content_hash: info.content_hash,
                generation: info.generation,
            },
        )
        .unwrap();
    let mut overlays = OverlaySet::new();
    overlays.mount("pages", 7, reader);
    let backend = GpuBlockDecodeBackend::new(Arc::new(shared.context().clone()));
    manifest
        .bind(&overlays)
        .unwrap()
        .bind_gpu_resident_asset(LOGICAL, &backend)
        .unwrap()
}

struct Consumer {
    stage: wgpu::Buffer,
    submission: wgpu::SubmissionIndex,
    logical_len: usize,
}

// A real compute consumer reads the published RHI table's STORAGE buffer at
// the allocation's offset. XOR makes dispatch omission fail even for zeros.
fn submit_consumer(pool: &ResidentPagePool, handle: ResidentPageAllocation) -> Consumer {
    let allocation = handle.allocation();
    assert_eq!(allocation.buffer(), pool.buffer());
    let ctx = pool.device().context();
    let device = ctx.device();
    let words = allocation.byte_len().div_ceil(4);
    let first_word = allocation.offset_bytes(pool.page_size()) / 4;
    let source = pool.device().device().wgpu_buffer(pool.buffer());
    assert!(!source.usage().contains(wgpu::BufferUsages::MAP_READ));
    let output = buffer::storage_rw_zeroed(device, "resident_test_consumer", words * 4);
    let stage = buffer::staging(device, "resident_test_only_readback", words * 4);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("resident_test_consumer"),
        source: wgpu::ShaderSource::Wgsl(format!(
            "@group(0) @binding(0) var<storage, read> pages: array<u32>;\n\
             @group(0) @binding(1) var<storage, read_write> output: array<u32>;\n\
             @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {{\n\
             if (id.x < {words}u) {{ output[id.x] = pages[{first_word}u + id.x] ^ {CONSUMER_XOR}u; }}\n}}"
        ).into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("resident_test_consumer"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("resident_test_consumer"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: source.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(u32::try_from(words.div_ceil(64)).unwrap(), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &stage, 0, words * 4);
    Consumer {
        stage,
        submission: ctx.queue().submit([encoder.finish()]),
        logical_len: allocation.byte_len() as usize,
    }
}

fn assert_consumer(ctx: &GpuContext, consumer: Consumer, expected: &[u8]) {
    assert_eq!(consumer.logical_len, expected.len());
    let bytes: Vec<u8> = buffer::read_back::<u32>(ctx, &consumer.stage, consumer.submission)
        .expect("test-only GPU consumer readback")
        .into_iter()
        .flat_map(|word| (word ^ CONSUMER_XOR).to_le_bytes())
        .collect();
    assert_eq!(&bytes[..expected.len()], expected);
    assert!(
        bytes[expected.len()..].iter().all(|&byte| byte == 0),
        "nonzero word padding"
    );
}

fn assert_payload(pool: &ResidentPagePool, handle: ResidentPageAllocation, expected: &[u8]) {
    assert_consumer(
        pool.device().context(),
        submit_consumer(pool, handle),
        expected,
    );
}

fn wait_and_reclaim(pool: &mut ResidentPagePool) -> Vec<FixedPageReclaimEvent> {
    let ctx = pool.device().context();
    let submission = ctx.queue().submit([]);
    ctx.device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(30)),
        })
        .expect("actual queue completion, never a synthetic frame delay");
    let events = pool
        .reclaim_completed()
        .expect("callback-proven reclamation");
    for event in &events {
        assert!(event.reclaim_after.0 > 0);
        assert!(event.completed.0 >= event.reclaim_after.0);
    }
    events
}

#[test]
fn render_app_manifest_cross_block_odd_boundaries_reach_gpu_consumer() {
    let _guard = GPU_TEST.lock().unwrap();
    let Some(app) = hardware_app() else { return };
    let runtime = app
        .get_sub_app(RenderApp)
        .unwrap()
        .world
        .resource::<RenderRuntimeResource<ResidentWgpuInstance>>()
        .runtime();
    let shared = snapshot(runtime);
    let raw = pattern(2 * BLOCK + 31, 0x73);
    let asset = manifest_asset(&shared, &raw);
    let mut pool = pool(shared, BLOCK as u64, 4);
    // First interval has odd start AND end and spans THREE compressed blocks.
    // Remaining intervals cover odd lengths and all byte-lane seam positions.
    for (revision, (start, len)) in [(BLOCK - 3, BLOCK + 12), (BLOCK - 1, 7), (1, 3), (3, 5)]
        .into_iter()
        .enumerate()
    {
        let range = asset
            .read_range_resident(start as u64, len as u64, decode_limits())
            .unwrap();
        assert_eq!(range.logical_guid(), LOGICAL);
        assert_eq!(range.range(), start as u64..(start + len) as u64);
        if revision == 0 {
            assert_eq!(start % 2, 1);
            assert_eq!((start + len) % 2, 1);
            assert_eq!(range.slices().len(), 3);
            assert_eq!(
                range.slices()[0].buffer_range(),
                (BLOCK - 3) as u64..BLOCK as u64
            );
            assert_eq!(range.slices()[1].output_offset(), 3);
            assert_eq!(range.slices()[2].output_offset(), (BLOCK + 3) as u64);
        }
        let publication = pool
            .upload_range(&range, None, revision as u64 + 1, assembly_limits())
            .unwrap();
        assert_eq!(publication.previous, None);
        assert_eq!(publication.revision, revision as u64 + 1);
        assert_eq!(publication.allocation.allocation().byte_len(), len as u64);
        drop(range); // consumer may not borrow the decoded block outputs.
        assert_payload(&pool, publication.allocation, &raw[start..start + len]);
        pool.release(publication.allocation, 100 + revision as u64)
            .unwrap();
        let reclaimed = wait_and_reclaim(&mut pool);
        assert_eq!(reclaimed.len(), 1);
        assert_eq!(reclaimed[0].allocation, publication.allocation.allocation());
        assert_eq!(pool.free_page_count(), 4);
    }
}

#[test]
fn replacement_quarantines_old_pages_until_queue_completion_and_explicit_reclaim() {
    let _guard = GPU_TEST.lock().unwrap();
    let Some(runtime) = runtime() else { return };
    let shared = snapshot(&runtime);
    let old_bytes = pattern(7, 0x11);
    let new_bytes = pattern(7, 0xe3);
    // Decode both BEFORE submitting the last old consumer, so a decoder's
    // status poll cannot accidentally be used as the retirement proof.
    let old_tile = decode(shared.context(), &old_bytes);
    let new_tile = decode(shared.context(), &new_bytes);
    let mut pool = pool(shared, 8, 2);
    let old = pool.upload(&old_tile, None, 10).unwrap().allocation;
    let old_consumer = submit_consumer(&pool, old);
    let replacement = pool.upload(&new_tile, Some(old), 20).unwrap();
    assert_eq!(replacement.previous, Some(old));
    assert_eq!(replacement.revision, 20);
    assert_ne!(
        replacement.allocation.allocation().first_page(),
        old.allocation().first_page()
    );
    assert_eq!(pool.pending_reclaim_page_count(), 1);
    assert_eq!(pool.free_page_count(), 0);
    // Even a fast GPU does not make quarantine automatically reusable: only
    // reclaim_completed consumes its real callback proof. No timing assertion.
    assert!(matches!(
        pool.upload(&old_tile, None, 21),
        Err(ResidentPageError::Pool(PagePoolError::OutOfPages { .. }))
    ));
    let events = wait_and_reclaim(&mut pool);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].allocation, old.allocation());
    assert_eq!(pool.pending_reclaim_page_count(), 0);
    assert_eq!(pool.free_page_count(), 1);
    let reused = pool.upload(&old_tile, None, 30).unwrap().allocation;
    assert_eq!(
        reused.allocation().first_page(),
        old.allocation().first_page()
    );
    assert_ne!(reused.allocation().token(), old.allocation().token());
    assert_consumer(pool.device().context(), old_consumer, &old_bytes);
    assert_payload(&pool, replacement.allocation, &new_bytes);
    assert_payload(&pool, reused, &old_bytes);
    assert!(pool.reclaim_completed().unwrap().is_empty());
}

#[test]
fn release_rejects_retiring_and_reused_stale_handles_without_damaging_new_pages() {
    let _guard = GPU_TEST.lock().unwrap();
    let Some(runtime) = runtime() else { return };
    let shared = snapshot(&runtime);
    let bytes = pattern(5, 0x9b);
    let tile = decode(shared.context(), &bytes);
    let mut pool = pool(shared, 8, 1);
    let old = pool.upload(&tile, None, 1).unwrap().allocation;
    let consumer = submit_consumer(&pool, old);
    pool.release(old, 2).unwrap();
    assert_eq!(pool.free_page_count(), 0);
    assert_eq!(pool.pending_reclaim_page_count(), 1);
    assert_eq!(
        pool.release(old, 3),
        Err(ResidentPageError::Pool(PagePoolError::InvalidAllocation))
    );
    assert!(matches!(
        pool.upload(&tile, Some(old), 4),
        Err(ResidentPageError::Pool(PagePoolError::InvalidAllocation))
    ));
    assert!(matches!(
        pool.upload(&tile, None, 5),
        Err(ResidentPageError::Pool(PagePoolError::OutOfPages { .. }))
    ));
    let events = wait_and_reclaim(&mut pool);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].allocation, old.allocation());
    assert_consumer(pool.device().context(), consumer, &bytes);
    let new = pool.upload(&tile, None, 6).unwrap().allocation;
    assert_ne!(new, old);
    assert_eq!(new.allocation().first_page(), old.allocation().first_page());
    assert_eq!(
        pool.release(old, 7),
        Err(ResidentPageError::Pool(PagePoolError::InvalidAllocation))
    );
    assert!(matches!(
        pool.upload(&tile, Some(old), 8),
        Err(ResidentPageError::Pool(PagePoolError::InvalidAllocation))
    ));
    assert_eq!(pool.pending_reclaim_page_count(), 0);
    assert_payload(&pool, new, &bytes);
    pool.release(new, 9).unwrap();
    assert_eq!(wait_and_reclaim(&mut pool).len(), 1);
    assert_eq!(pool.free_page_count(), 1);
    assert!(pool.reclaim_completed().unwrap().is_empty());
}

#[test]
fn foreign_instances_devices_and_pool_capabilities_are_rejected_even_when_rhi_ids_alias() {
    let _guard = GPU_TEST.lock().unwrap();
    let Some(runtime) = runtime() else { return };
    let shared = snapshot(&runtime);
    let bytes = pattern(7, 0x49);
    let tile = decode(shared.context(), &bytes);
    let mut local = pool(shared.clone(), 8, 2);
    let old = local.upload(&tile, None, 1).unwrap().allocation;
    let mut same_device = pool(shared.clone(), 8, 2);
    let same_handle = same_device.upload(&tile, None, 1).unwrap().allocation;
    assert_ne!(local.buffer(), same_device.buffer());
    assert_eq!(
        local.release(same_handle, 2),
        Err(ResidentPageError::ForeignAllocation)
    );
    assert_eq!(
        local.upload(&tile, Some(same_handle), 2),
        Err(ResidentPageError::ForeignAllocation)
    );

    // Unlike a duplicate wrapper, this is another actual logical device on
    // the SAME authenticated instance/adapter. Its creation is never skipped.
    runtime
        .instance()
        .create_device(runtime.adapter(), &DeviceDescriptor::default())
        .unwrap();
    let second_device = runtime.instance().resident_device().unwrap();
    assert_eq!(
        shared.context().instance(),
        second_device.context().instance()
    );
    assert_ne!(shared.context().device(), second_device.context().device());
    assert!(
        ResidentRenderDevice::from_runtime(&runtime).is_none(),
        "a factory publication ahead of the committed runtime must not authenticate"
    );
    let second_tile = decode(second_device.context(), &bytes);
    assert_eq!(
        local.upload(&second_tile, Some(old), 3),
        Err(ResidentPageError::DeviceMismatch)
    );
    let mut second_pool = pool(second_device, 8, 2);
    let second_handle = second_pool
        .upload(&second_tile, None, 1)
        .unwrap()
        .allocation;
    assert_eq!(
        local.buffer(),
        second_pool.buffer(),
        "fresh RHI table IDs must really alias"
    );
    assert_eq!(
        old.allocation(),
        second_handle.allocation(),
        "even complete inner handles alias"
    );
    assert_ne!(
        old, second_handle,
        "native pool capability must add non-table identity"
    );
    assert_eq!(
        local.release(second_handle, 4),
        Err(ResidentPageError::ForeignAllocation)
    );
    assert_eq!(
        local.upload(&tile, Some(second_handle), 4),
        Err(ResidentPageError::ForeignAllocation)
    );

    // A second independent native instance is a separate ownership boundary.
    let other_runtime =
        RenderRuntime::headless(ResidentWgpuInstance::new(), Extent3d::new_2d(8, 8))
            .expect("a second real hardware instance/device must be creatable");
    assert!(physical(other_runtime.adapter_info()));
    let foreign = snapshot(&other_runtime);
    assert_ne!(shared.context().instance(), foreign.context().instance());
    let foreign_tile = decode(foreign.context(), &bytes);
    assert_eq!(
        local.upload(&foreign_tile, Some(old), 5),
        Err(ResidentPageError::DeviceMismatch)
    );
    let mut foreign_pool = pool(foreign, 8, 2);
    let foreign_handle = foreign_pool
        .upload(&foreign_tile, None, 1)
        .unwrap()
        .allocation;
    assert_eq!(local.buffer(), foreign_pool.buffer());
    assert_eq!(
        local.release(foreign_handle, 6),
        Err(ResidentPageError::ForeignAllocation)
    );
    assert_eq!(
        local.upload(&tile, Some(foreign_handle), 6),
        Err(ResidentPageError::ForeignAllocation)
    );
    assert_eq!(local.free_page_count(), 1);
    assert_eq!(local.pending_reclaim_page_count(), 0);
    assert_payload(&local, old, &bytes);
    assert_payload(&same_device, same_handle, &bytes);
}

#[test]
fn pool_host_and_assembly_admission_fail_atomically_and_preserve_old_payload() {
    let _guard = GPU_TEST.lock().unwrap();
    let Some(runtime) = runtime() else { return };
    let shared = snapshot(&runtime);
    let bytes = pattern(5, 0x5a);
    let tile = decode(shared.context(), &bytes);
    let large = decode(shared.context(), &pattern(17, 0xb7));
    let too_large = decode(shared.context(), &pattern(25, 0x8d));
    let asset = manifest_asset(&shared, &pattern(31, 0xd3));
    let range = asset.read_range_resident(3, 7, decode_limits()).unwrap();
    let mut descriptor = descriptor(8, 3);
    let required_host = descriptor.required_host_bytes().unwrap();
    descriptor.max_host_bytes = required_host;
    let mut pool = ResidentPagePool::new(shared.clone(), &descriptor, 24).unwrap();
    let old = pool.upload(&tile, None, 1).unwrap().allocation;
    let free = pool.free_page_count();
    assert_eq!(free, 2);
    assert_eq!(
        ResidentPagePool::new(shared.clone(), &descriptor, 23).err(),
        Some(ResidentPageError::DeviceBudget {
            required: 24,
            limit: 23
        })
    );
    descriptor.max_host_bytes = required_host - 1;
    assert_eq!(
        ResidentPagePool::new(shared.clone(), &descriptor, 24).err(),
        Some(ResidentPageError::Pool(PagePoolError::HostBudgetExceeded {
            required: required_host,
            limit: required_host - 1,
        }))
    );
    assert_eq!(pool.free_page_count(), free);
    assert_eq!(pool.pending_reclaim_page_count(), 0);
    assert!(matches!(
        pool.upload(&large, Some(old), 2),
        Err(ResidentPageError::Pool(PagePoolError::OutOfPages { .. }))
    ));
    assert_eq!(pool.free_page_count(), free);
    assert!(matches!(
        pool.upload(&too_large, Some(old), 3),
        Err(ResidentPageError::Pool(
            PagePoolError::UploadTooLarge { .. }
        ))
    ));
    assert_eq!(pool.free_page_count(), free);
    assert_eq!(
        pool.upload_range(
            &range,
            Some(old),
            4,
            ResidentAssemblyLimits {
                max_slices: 0,
                max_device_bytes: 24,
            }
        ),
        Err(ResidentPageError::Assembly(
            ResidentAssemblyError::TooManySlices {
                requested: 1,
                limit: 0,
            }
        ))
    );
    assert_eq!(pool.free_page_count(), free);
    // Seven logical bytes require eight output bytes plus one 16-byte uniform.
    // This path has already reserved pool space: admission failure must roll it back.
    assert_eq!(
        pool.upload_range(
            &range,
            Some(old),
            5,
            ResidentAssemblyLimits {
                max_slices: 1,
                max_device_bytes: 23,
            }
        ),
        Err(ResidentPageError::Assembly(
            ResidentAssemblyError::AllocationBudgetExceeded {
                required: 24,
                limit: 23,
            }
        ))
    );
    assert_eq!(pool.free_page_count(), free);
    assert_eq!(pool.pending_reclaim_page_count(), 0);
    assert_payload(&pool, old, &bytes);
    // Exact-boundary success proves failures neither poisoned the pipeline nor
    // published/retired the previous allocation.
    let publication = pool
        .upload_range(
            &range,
            Some(old),
            6,
            ResidentAssemblyLimits {
                max_slices: 1,
                max_device_bytes: 24,
            },
        )
        .unwrap();
    assert_eq!(publication.previous, Some(old));
    assert_payload(&pool, publication.allocation, &pattern(31, 0xd3)[3..10]);
    assert_eq!(wait_and_reclaim(&mut pool).len(), 1);
    pool.release(publication.allocation, 7).unwrap();
    assert_eq!(wait_and_reclaim(&mut pool).len(), 1);
    assert_eq!(pool.free_page_count(), 3);
}

#[test]
fn odd_lengths_keep_exact_metadata_and_zero_word_padding_across_reuse() {
    let _guard = GPU_TEST.lock().unwrap();
    let Some(runtime) = runtime() else { return };
    let shared = snapshot(&runtime);
    let mut pool = pool(shared.clone(), 8, 3);
    for (revision, len) in [1usize, 2, 3, 5, 7, 9, 15, 17].into_iter().enumerate() {
        let bytes = pattern(len, 0x81u8.wrapping_add(revision as u8));
        let tile = decode(shared.context(), &bytes);
        let publication = pool.upload(&tile, None, revision as u64).unwrap();
        let allocation = publication.allocation.allocation();
        assert_eq!(allocation.byte_len(), len as u64);
        assert_eq!(allocation.page_count(), len.div_ceil(8) as u32);
        assert_eq!(
            allocation.capacity_bytes(pool.page_size()),
            len.div_ceil(8) as u64 * 8
        );
        assert_eq!(pool.free_page_count(), 3 - len.div_ceil(8) as u32);
        drop(tile);
        assert_payload(&pool, publication.allocation, &bytes);
        pool.release(publication.allocation, revision as u64 + 100)
            .unwrap();
        assert_eq!(wait_and_reclaim(&mut pool).len(), 1);
        assert_eq!(pool.free_page_count(), 3);
        assert_eq!(pool.pending_reclaim_page_count(), 0);
    }
}
