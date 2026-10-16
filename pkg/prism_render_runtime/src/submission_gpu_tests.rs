//! Physical-GPU coverage for `execute_owned`, separate from controlled CPU
//! retirement tests. `PRISM_REQUIRE_GPU=1` turns unavailable/non-hardware
//! adapters into failures; a device-creation error is always a failure.

use prism_render_driver as rhi;
use prism_render_driver_wgpu::{require_hardware_gpu, AdapterEvidence, WgpuDevice};
use prism_render_graph::{BufferDesc, ImportedBuffer, Lifetime, PassFlags, RenderGraph};
use rhi::{RenderDevice, RenderQueue};

use crate::{FrameSubmission, RenderInitError, WgpuRuntime};

// Nonzero, nonuniform input makes omitted copies/zero-initialized allocations
// fail. These bytes are uploaded once; every subsequent copy is GPU work.
const PAYLOAD: [u8; 32] = [
    0x91, 0x02, 0xd3, 0x44, 0x15, 0xe6, 0x37, 0x88, 0x79, 0xaa, 0x5b, 0xcc, 0x1d, 0xfe, 0x6f, 0x30,
    0xb1, 0x62, 0x03, 0xf4, 0x85, 0x26, 0xc7, 0x58, 0xe9, 0x7a, 0x4b, 0x9c, 0xdd, 0x0e, 0xaf, 0x70,
];
const BUFFER_SIZE: u64 = PAYLOAD.len() as u64;

fn hardware_runtime_or_skip() -> Option<WgpuRuntime> {
    let runtime = match WgpuRuntime::wgpu(rhi::Extent3d::new_2d(1, 1)) {
        Ok(runtime) => runtime,
        Err(RenderInitError::NoAdapter) => {
            assert!(
                !require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 but execute_owned has no GPU adapter"
            );
            eprintln!("SKIPPED execute_owned physical-GPU test: no adapter");
            return None;
        }
        Err(RenderInitError::Device(error)) => {
            panic!("execute_owned GPU device creation failed: {error}");
        }
    };
    // Classify the adapter that created this runtime's actual device, rather
    // than a separately enumerated adapter or a second test-only device.
    let evidence = AdapterEvidence::from_info(runtime.adapter_info());
    if !evidence.is_hardware() {
        assert!(
            !require_hardware_gpu(),
            "PRISM_REQUIRE_GPU=1 requires physical GPU execution: {evidence}"
        );
        eprintln!("SKIPPED execute_owned physical-GPU test: {evidence}");
        return None;
    }
    eprintln!("execute_owned REAL HARDWARE GPU: {evidence}");
    Some(runtime)
}

struct CopyFrame {
    graph: RenderGraph,
    upload: rhi::BufferId,
    readback: rhi::BufferId,
}

fn transfer_frame(runtime: &WgpuRuntime) -> CopyFrame {
    let usage = rhi::BufferUsages::COPY_SRC | rhi::BufferUsages::COPY_DST;
    let descriptor = |label: &str| rhi::BufferDescriptor {
        label: Some(label.into()),
        size: BUFFER_SIZE,
        usage,
        mapped_at_creation: false,
    };
    let upload = runtime
        .device()
        .create_buffer(&descriptor("owned-submission-upload"));
    let readback = runtime
        .device()
        .create_buffer(&descriptor("owned-submission-readback"));
    runtime.queue().write_buffer(upload, 0, &PAYLOAD);

    let mut graph = runtime.new_graph();
    let source = graph.import_buffer(
        "external upload",
        BufferDesc::new(BUFFER_SIZE, usage),
        ImportedBuffer {
            buffer: upload,
            entry_state: rhi::BufferState::copy_dst(),
        },
    );
    let destination = graph.import_buffer(
        "external readback",
        BufferDesc::new(BUFFER_SIZE, usage),
        ImportedBuffer {
            buffer: readback,
            entry_state: rhi::BufferState::initial(),
        },
    );

    let mut transient = None;
    graph.add_transfer_pass("upload -> owned transient", PassFlags::EMPTY, |builder| {
        let source = builder.copy_src_buffer(source);
        let target = builder.create_buffer("owned transient", BufferDesc::new(BUFFER_SIZE, usage));
        let target = builder.copy_dst_buffer(target);
        transient = Some(target);
        move |context| {
            context.copy_buffer_to_buffer(source, 0, target, 0, BUFFER_SIZE);
        }
    });
    let transient = transient.expect("transfer setup declares the transient");
    let mut history = None;
    graph.add_transfer_pass(
        "owned transient -> owned history",
        PassFlags::EMPTY,
        |builder| {
            let source = builder.copy_src_buffer(transient);
            let target = builder
                .create_persistent_buffer("owned history", BufferDesc::new(BUFFER_SIZE, usage));
            let target = builder.copy_dst_buffer(target);
            history = Some(target);
            move |context| {
                context.copy_buffer_to_buffer(source, 0, target, 0, BUFFER_SIZE);
            }
        },
    );
    let history = history.expect("transfer setup declares persistent history");
    // Only the final pass is a culling root: actual SSA resource dependencies
    // must keep both graph-owned producers in the submitted graph.
    graph.add_transfer_pass(
        "owned history -> external readback",
        PassFlags::SIDE_EFFECT,
        |builder| {
            let source = builder.copy_src_buffer(history);
            let target = builder.copy_dst_buffer(destination);
            move |context| {
                context.copy_buffer_to_buffer(source, 0, target, 0, BUFFER_SIZE);
            }
        },
    );
    CopyFrame {
        graph,
        upload,
        readback,
    }
}

fn owned_buffers(
    owner: &FrameSubmission<WgpuDevice>,
    upload: rhi::BufferId,
    readback: rhi::BufferId,
) -> [rhi::BufferId; 2] {
    let buffers = owner.allocations().buffers();
    assert_eq!(
        buffers.len(),
        2,
        "imports must not enter the submission owner"
    );
    assert!(owner.allocations().textures().is_empty());
    let find = |lifetime| {
        let id = buffers
            .iter()
            .find_map(|(_, class, id)| (*class == lifetime).then_some(*id))
            .expect("both transient and persistent buffers must be realized");
        assert_ne!(id, upload);
        assert_ne!(id, readback);
        id
    };
    let transient = find(Lifetime::Transient);
    let history = find(Lifetime::Persistent);
    assert_ne!(transient, history);
    [transient, history]
}

fn assert_payload(runtime: &WgpuRuntime, buffer: rhi::BufferId) {
    assert_eq!(
        runtime.queue().read_buffer(buffer, BUFFER_SIZE),
        PAYLOAD,
        "GPU transfer/readback payload mismatch for {buffer:?}"
    );
}

fn assert_backend_slots_retired(runtime: &WgpuRuntime, retired: [rhi::BufferId; 2]) {
    // WgpuDevice has no fallible public buffer lookup. Do not catch a stale-id
    // panic or keep native wgpu::Buffer clones alive to test destruction.
    // Its GenerationalSlotMap reuses free slots before growing: allocating two
    // replacements on this otherwise isolated device proves both old entries
    // were removed, not merely marked complete or forgotten by the owner.
    // This checks backend-table retirement, not OS/driver heap reclamation.
    let replacements = retired.map(|_| {
        runtime.device().create_buffer(&rhi::BufferDescriptor {
            label: Some("retired-slot-probe".into()),
            size: BUFFER_SIZE,
            usage: rhi::BufferUsages::COPY_SRC | rhi::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    });
    assert_ne!(replacements[0].index(), replacements[1].index());
    for old in retired {
        let replacement = replacements
            .iter()
            .find(|new| new.index() == old.index())
            .expect("released graph buffer slot must be reusable on its creating device");
        assert_eq!(
            replacement.generation(),
            old.generation()
                .checked_add(1)
                .expect("fresh device generation"),
            "old graph BufferId must no longer identify a live table entry"
        );
    }
    for replacement in replacements {
        runtime.device().destroy_buffer(replacement);
    }
}

fn assert_backend_slot_retired(runtime: &WgpuRuntime, retired: rhi::BufferId) {
    let replacement = runtime.device().create_buffer(&rhi::BufferDescriptor {
        label: Some("retired-history-slot-probe".into()),
        size: BUFFER_SIZE,
        usage: rhi::BufferUsages::COPY_SRC | rhi::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    assert_eq!(replacement.index(), retired.index());
    assert_eq!(
        replacement.generation(),
        retired
            .generation()
            .checked_add(1)
            .expect("fresh device generation")
    );
    runtime.device().destroy_buffer(replacement);
}

fn history_consumer_graph(
    runtime: &WgpuRuntime,
    source: rhi::BufferId,
    readback: rhi::BufferId,
) -> prism_render_graph::PreparedGraph {
    let usage = rhi::BufferUsages::COPY_SRC | rhi::BufferUsages::COPY_DST;
    let mut graph = runtime.new_graph();
    let source = graph.import_buffer(
        "imported persistent history",
        BufferDesc::new(BUFFER_SIZE, usage),
        ImportedBuffer {
            buffer: source,
            entry_state: rhi::BufferState::copy_dst(),
        },
    );
    let destination = graph.import_buffer(
        "history readback",
        BufferDesc::new(BUFFER_SIZE, usage),
        ImportedBuffer {
            buffer: readback,
            entry_state: rhi::BufferState::initial(),
        },
    );
    graph.add_transfer_pass(
        "imported history -> readback",
        PassFlags::SIDE_EFFECT,
        |builder| {
            let source = builder.copy_src_buffer(source);
            let destination = builder.copy_dst_buffer(destination);
            move |context| {
                context.copy_buffer_to_buffer(source, 0, destination, 0, BUFFER_SIZE);
            }
        },
    );
    graph
        .prepare()
        .expect("history consumer graph must prepare")
}

#[test]
fn execute_owned_gpu_clone_preserves_buffers_through_completion_and_readback() {
    let Some(runtime) = hardware_runtime_or_skip() else {
        return;
    };
    let CopyFrame {
        graph,
        upload,
        readback,
    } = transfer_frame(&runtime);
    let expected_value = runtime.next_completion_value();
    let owner = runtime.execute_owned(graph).expect("real GPU submission");
    let value = owner.completion_value();
    assert_eq!(value, expected_value);
    assert_eq!(owner.device_generation(), runtime.device_generation());
    let buffers = owned_buffers(&owner, upload, readback);
    let retained = owner.clone();
    drop(owner);

    assert!(
        runtime.wait_idle(),
        "real queue completion must be confirmed"
    );
    assert!(
        retained.is_complete(),
        "the real completion callback must run"
    );
    assert_eq!(runtime.completed_value(), value);
    assert_eq!(runtime.frames_in_flight(), 0);
    for buffer in buffers {
        assert_payload(&runtime, buffer);
    }
    assert_payload(&runtime, readback);
    // The remaining explicit owner must retain native table entries even
    // after completion and the additional readback submissions have finished.
    drop(retained);
    assert_backend_slots_retired(&runtime, buffers);
    // Graph retirement must not destroy either externally owned import.
    assert_payload(&runtime, upload);
    assert_payload(&runtime, readback);
    runtime.device().destroy_buffer(upload);
    runtime.device().destroy_buffer(readback);
}

#[test]
fn execute_owned_gpu_early_drop_completes_copies_and_retires_callback_resources() {
    let Some(runtime) = hardware_runtime_or_skip() else {
        return;
    };
    let CopyFrame {
        graph,
        upload,
        readback,
    } = transfer_frame(&runtime);
    let expected_value = runtime.next_completion_value();
    let owner = runtime.execute_owned(graph).expect("real GPU submission");
    let value = owner.completion_value();
    assert_eq!(value, expected_value);
    assert_eq!(owner.device_generation(), runtime.device_generation());
    let buffers = owned_buffers(&owner, upload, readback);
    // No polling/readback/idle barrier before dropping the caller's owner.
    // Do not assume GPU work is still pending: fast hardware may finish before
    // execute_owned returns. Controlled pre-completion lifetime assertions
    // belong to retirement_tests; this test checks the actual GPU path.
    drop(owner);
    assert!(
        runtime.wait_idle(),
        "early-drop GPU work must still complete"
    );
    assert_eq!(
        runtime.completed_value(),
        value,
        "completion callback must run"
    );
    assert_eq!(runtime.frames_in_flight(), 0);
    assert_payload(&runtime, readback);
    assert_backend_slots_retired(&runtime, buffers);
    assert_payload(&runtime, upload);
    assert_payload(&runtime, readback);
    runtime.device().destroy_buffer(upload);
    runtime.device().destroy_buffer(readback);
}

#[test]
fn execute_prepared_owned_gpu_imports_completed_history_and_reads_it_back() {
    let Some(runtime) = hardware_runtime_or_skip() else {
        return;
    };
    let CopyFrame {
        graph,
        upload,
        readback,
    } = transfer_frame(&runtime);
    let producer = runtime
        .execute_owned(graph)
        .expect("real GPU producer submission");
    let producer_value = producer.completion_value();
    let history = producer
        .allocations()
        .buffers()
        .iter()
        .find_map(|(_, lifetime, id)| (*lifetime == Lifetime::Persistent).then_some(*id))
        .expect("producer must retain a persistent history buffer");

    // The first frame must be complete before its persistent storage is used as
    // an imported history input. The producer owner is intentionally retained
    // and then passed explicitly to the consumer's completion owner.
    assert!(runtime.wait_idle());
    assert_eq!(runtime.completed_value(), producer_value);
    assert_payload(&runtime, history);

    let prepared = history_consumer_graph(&runtime, history, readback);
    let consumer = runtime
        .execute_prepared_owned(prepared, core::slice::from_ref(&producer))
        .expect("history import must use the exact creating owner");
    assert!(consumer.allocations().buffers().is_empty());
    drop(producer);

    assert!(runtime.wait_idle());
    assert!(consumer.is_complete());
    assert_payload(&runtime, readback);

    // The source owner is retained by the consumer until its actual completion.
    // Once the consumer completes, its retained producer is released and the
    // graph-owned persistent history is retired; only external imports remain.
    drop(consumer);
    assert_backend_slot_retired(&runtime, history);
    assert_payload(&runtime, upload);
    assert_payload(&runtime, readback);
    runtime.device().destroy_buffer(upload);
    runtime.device().destroy_buffer(readback);
}
