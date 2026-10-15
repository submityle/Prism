//! Real-wgpu validation for the persistent page-table and feedback readback slice.
//!
//! These tests skip only when no real headless adapter is available. Set
//! `PRISM_REQUIRE_GPU=1` to turn that environmental skip into a failure.

use prism_render_architecture::texture_streaming::{GpuPageTable, TexturePageKey};
use prism_virtual_texture_gpu::{
    FeedbackReadbackError, GpuContext, GpuFeedbackReadback, GpuPageTableStorage,
    PageTableStorageError,
};

#[expect(
    clippy::print_stderr,
    reason = "test diagnostics: explain a GPU-less skip"
)]
fn with_gpu() -> Option<GpuContext> {
    match GpuContext::try_real_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("[prism_virtual_texture_gpu] no headless GPU adapter; skipping storage/readback test");
            None
        }
    }
}

fn table() -> GpuPageTable {
    GpuPageTable::from_bindings(&[
        (
            TexturePageKey {
                texture: 2,
                mip: 1,
                layer: 0,
                x: 3,
                y: 4,
            },
            17,
        ),
        (
            TexturePageKey {
                texture: 2,
                mip: 2,
                layer: 0,
                x: 1,
                y: 2,
            },
            23,
        ),
    ])
}

#[test]
fn persistent_page_table_upload_is_bounded_and_deterministic() {
    let Some(ctx) = with_gpu() else { return };
    let mut storage = GpuPageTableStorage::new(&ctx, 3).expect("page-table storage allocation");
    let readback = GpuFeedbackReadback::new(&ctx, 12).expect("readback allocation");
    let table = table();

    storage.upload(&ctx, &table).expect("table upload");
    assert_eq!(storage.len(), 2);
    let words = readback
        .read_from(&ctx, storage.buffer(), 12)
        .expect("page-table readback");
    assert_eq!(&words[..8], table.words());
    assert!(words[8..].iter().all(|&word| word == 0));

    storage
        .upload(&ctx, &GpuPageTable::new())
        .expect("empty upload");
    assert!(storage.is_empty());
    let cleared = readback
        .read_from(&ctx, storage.buffer(), 12)
        .expect("cleared page-table readback");
    assert!(cleared.iter().all(|&word| word == 0));

    let too_large = GpuPageTable::from_bindings(&[
        (
            TexturePageKey {
                texture: 9,
                mip: 0,
                layer: 0,
                x: 0,
                y: 0,
            },
            1,
        ),
        (
            TexturePageKey {
                texture: 9,
                mip: 1,
                layer: 0,
                x: 0,
                y: 0,
            },
            2,
        ),
        (
            TexturePageKey {
                texture: 9,
                mip: 2,
                layer: 0,
                x: 0,
                y: 0,
            },
            3,
        ),
        (
            TexturePageKey {
                texture: 9,
                mip: 3,
                layer: 0,
                x: 0,
                y: 0,
            },
            4,
        ),
    ]);
    assert_eq!(
        storage.upload(&ctx, &too_large),
        Err(PageTableStorageError::CapacityExceeded {
            requested: 4,
            capacity: 3,
        })
    );
    assert!(
        storage.is_empty(),
        "rejected upload must not mutate logical length"
    );
}

#[test]
fn external_feedback_storage_buffer_round_trips_through_staging() {
    let Some(ctx) = with_gpu() else { return };
    let words = [3u32, 5, 8, 13, 21];
    let source = prism_virtual_texture_gpu::buffer::storage_rw_init(
        ctx.device(),
        "prism_vt_external_feedback",
        &words,
    );
    let readback = GpuFeedbackReadback::new(&ctx, words.len()).expect("readback allocation");

    assert_eq!(readback.capacity_words(), words.len());
    assert_eq!(
        readback.read_from(&ctx, &source, words.len()).unwrap(),
        words
    );
    assert_eq!(
        readback.read_from(&ctx, &source, 0).unwrap(),
        Vec::<u32>::new()
    );
    assert_eq!(
        readback.read_from(&ctx, &source, words.len() + 1),
        Err(FeedbackReadbackError::CapacityExceeded {
            requested: words.len() + 1,
            capacity: words.len(),
        })
    );

    let too_small = prism_virtual_texture_gpu::buffer::storage_rw_init(
        ctx.device(),
        "prism_vt_short_feedback",
        &[1u32, 2],
    );
    assert_eq!(
        readback.read_from(&ctx, &too_small, 3),
        Err(FeedbackReadbackError::SourceTooSmall {
            requested_bytes: 12,
            source_bytes: 8,
        })
    );

    let not_copyable = prism_virtual_texture_gpu::buffer::storage_read(
        ctx.device(),
        "prism_vt_non_copyable_feedback",
        &words,
    );
    assert_eq!(
        readback.read_from(&ctx, &not_copyable, 1),
        Err(FeedbackReadbackError::SourceMissingCopySourceUsage)
    );
}

#[test]
fn persistent_page_table_expands_without_losing_the_previous_contract() {
    let Some(ctx) = with_gpu() else { return };
    let mut storage = GpuPageTableStorage::new(&ctx, 1).expect("initial storage");
    let readback = GpuFeedbackReadback::new(&ctx, 16).expect("readback");
    let initial = GpuPageTable::from_bindings(&[(
        TexturePageKey {
            texture: 2,
            mip: 1,
            layer: 0,
            x: 3,
            y: 4,
        },
        17,
    )]);
    storage.upload(&ctx, &initial).expect("initial upload");
    let initial_capacity = storage.capacity_entries();
    storage
        .ensure_capacity(&ctx, 4)
        .expect("page-table capacity growth");
    assert_eq!(storage.len(), initial.len());
    let preserved = readback
        .read_from(&ctx, storage.buffer(), storage.capacity_entries() * 4)
        .expect("preserved page-table readback");
    assert_eq!(&preserved[..initial.words().len()], initial.words());
    assert!(preserved[initial.words().len()..]
        .iter()
        .all(|&word| word == 0));

    let expanded = GpuPageTable::from_bindings(&[
        (
            TexturePageKey {
                texture: 4,
                mip: 0,
                layer: 0,
                x: 0,
                y: 0,
            },
            1,
        ),
        (
            TexturePageKey {
                texture: 4,
                mip: 0,
                layer: 0,
                x: 1,
                y: 0,
            },
            2,
        ),
        (
            TexturePageKey {
                texture: 4,
                mip: 0,
                layer: 0,
                x: 2,
                y: 0,
            },
            3,
        ),
    ]);
    storage
        .upload_resizing(&ctx, &expanded)
        .expect("resizing upload");
    assert!(storage.capacity_entries() >= expanded.len());
    assert!(storage.capacity_entries() > initial_capacity);
    let words = readback
        .read_from(&ctx, storage.buffer(), storage.capacity_entries() * 4)
        .expect("expanded table readback");
    assert_eq!(&words[..expanded.words().len()], expanded.words());
}
