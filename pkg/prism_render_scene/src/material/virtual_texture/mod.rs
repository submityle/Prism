//! Independent material virtual-texture GPU resource bundle.
//!
//! The material plugin owns one disabled-safe GPU bundle and binds it into the
//! shared material group at bindings 5–8. Runtime residency is still a
//! separate producer: callers create a validated atlas, build a staging update
//! from material-VT source rows and the existing page-table snapshot, write it
//! through a real Bevy `RenderQueue`, and supply the renderer's completion
//! proof before publishing or reclaiming CPU metadata.

mod abi;
mod atlas;
mod demand;
mod feedback;
mod lifecycle;
mod resources;
mod staging;

pub use abi::{
    page_table_buffer_size, MaterialVirtualTextureGpuConfig, MATERIAL_VIRTUAL_TEXTURE_ENTRY_STRIDE,
    MATERIAL_VIRTUAL_TEXTURE_ENTRY_WORDS,
};
pub use atlas::{MaterialVirtualTextureAtlasConfig, MaterialVirtualTextureAtlasError};
pub use demand::{
    aggregate_material_vt_demands, MaterialVtDemandError, MaterialVtDemandMaterial,
    MaterialVtDemandRegistry, MaterialVtVisibilitySample,
};
pub(crate) use demand::{bootstrap_material_vt_demands, merge_material_vt_demands};
pub(crate) use feedback::{
    collect_material_vt_feedback, map_submitted_material_vt_feedback,
    request_material_vt_feedback_readback, MaterialVtFeedbackReadback,
};
pub use lifecycle::{
    MaterialVirtualTextureLifecycle, MaterialVirtualTextureLifecycleError,
    MaterialVirtualTextureReadyToPublish, MaterialVirtualTextureSubmission,
};
pub use resources::{MaterialVirtualTextureGpuBundle, MaterialVirtualTextureResourceSizes};
pub use staging::{
    MaterialVirtualTextureAtlasAtlasError, MaterialVirtualTextureAtlasUpload,
    MaterialVirtualTextureAtlasUploadRange, MaterialVirtualTextureStaging,
    MaterialVirtualTextureStagingError, MaterialVirtualTextureUploadError,
    MaterialVirtualTextureUploadRanges,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shading::texel_cache::gpu::page_table::{
        MaterialVirtualTexturePageRecord, MaterialVirtualTexturePageTableSnapshot,
    };
    use prism_render_architecture::gpu_scene::GpuCompletionValue;
    use prism_render_architecture::texture_streaming::TexturePageKey;
    use prism_render_material::{
        MaterialVtFormat, MaterialVtSourceDescriptor, MaterialVtSourceSnapshot,
    };

    fn key(x: u16) -> TexturePageKey {
        TexturePageKey {
            texture: 7,
            mip: 0,
            layer: 0,
            x,
            y: 0,
        }
    }

    fn snapshot() -> MaterialVirtualTexturePageTableSnapshot {
        MaterialVirtualTexturePageTableSnapshot::try_from_records(
            3,
            1,
            [MaterialVirtualTexturePageRecord {
                key: key(0),
                slot: 2,
                generation: 1,
                epoch: 3,
                ready: true,
            }],
        )
        .expect("valid material VT snapshot")
    }

    #[test]
    fn abi_and_buffer_layout_are_stable() {
        assert_eq!(core::mem::size_of::<MaterialVirtualTextureGpuConfig>(), 32);
        assert_eq!(core::mem::align_of::<MaterialVirtualTextureGpuConfig>(), 4);
        assert_eq!(MATERIAL_VIRTUAL_TEXTURE_ENTRY_STRIDE, 16);
        assert_eq!(page_table_buffer_size(0), 16);
        assert_eq!(page_table_buffer_size(3), 48);
        let config = MaterialVirtualTextureGpuConfig::new(true, 1, 4, 6, 8, 1, 1, 4);
        assert_eq!(bytemuck::bytes_of(&config).len(), 32);
        assert_eq!(config.enabled, 1);
    }

    #[test]
    fn atlas_geometry_maps_slots_and_validates_tile_layout() {
        let config =
            MaterialVirtualTextureAtlasConfig::new(4, 1, 2, 5, MaterialVtFormat::Rgba8Unorm)
                .expect("valid atlas");
        assert_eq!(config.atlas_layers().unwrap(), 2);
        let placement = config.geometry().unwrap().slot_placement(4);
        assert_eq!(placement.layer, 1);
        assert_eq!(placement.origin_x_px, 0);
        assert_eq!(placement.origin_y_px, 0);
        assert_eq!(config.atlas_extent_px().unwrap(), 12);
    }

    #[test]
    fn source_rows_become_guttered_staging_upload() {
        let descriptor =
            MaterialVtSourceDescriptor::new(7, 4, 4, 1, 4, 3, MaterialVtFormat::Rgba8Unorm)
                .expect("valid source descriptor");
        let mut bytes = Vec::new();
        for mip in 0..3u32 {
            let extent = (4u32 >> mip).max(1);
            bytes.extend((0..extent * extent * 4).map(|value| value as u8));
        }
        let source = MaterialVtSourceSnapshot::new(descriptor, bytes).unwrap();
        let config =
            MaterialVirtualTextureAtlasConfig::new(4, 1, 2, 4, MaterialVtFormat::Rgba8Unorm)
                .unwrap();
        let mut staging = MaterialVirtualTextureStaging::new(
            config,
            &snapshot(),
            MaterialVirtualTextureGpuConfig::new(true, 1, 4, 6, 2, 1, 1, 2),
        )
        .unwrap();
        staging
            .push_source_page(
                config,
                &source,
                key(0),
                2,
                &std::collections::BTreeSet::new(),
            )
            .unwrap();
        let upload = &staging.atlas_uploads()[0];
        assert_eq!(upload.bytes_per_row, 256);
        assert_eq!(upload.rows(), 6);
        assert_eq!(upload.bytes.len(), 256 * 6);
        assert_eq!(staging.page_table_words(), &[7, 0, 0, 2]);
    }

    #[test]
    fn portable_source_page_normalizes_format_and_page_geometry() {
        let descriptor =
            MaterialVtSourceDescriptor::new(7, 4, 4, 1, 4, 3, MaterialVtFormat::Rgba8Srgb)
                .expect("valid non-portable source descriptor");
        let source = MaterialVtSourceSnapshot::new(descriptor, vec![128; (16 + 4 + 1) * 4])
            .expect("valid source payload");
        let config =
            MaterialVirtualTextureAtlasConfig::new(128, 1, 2, 4, MaterialVtFormat::Rgba8Unorm)
                .expect("valid normalized portable atlas");
        let geometry = config.geometry().expect("portable atlas geometry");
        let mut staging = MaterialVirtualTextureStaging::new(
            config,
            &snapshot(),
            MaterialVirtualTextureGpuConfig::new(
                true,
                1,
                128,
                geometry.tile_extent_px(),
                2,
                1,
                1,
                0,
            ),
        )
        .expect("valid portable staging");
        staging
            .push_portable_source_page(config, &source, key(0), 2)
            .expect("source-backed normalization succeeds");
        let upload = &staging.atlas_uploads()[0];
        assert_eq!(upload.key, key(0));
        assert_eq!(upload.extent_px, geometry.tile_extent_px());
        assert_eq!(upload.bytes_per_row, geometry.bytes_per_row());
        assert_eq!(upload.bytes.len(), geometry.tile_staging_bytes() as usize);
    }

    #[test]
    fn completion_gates_publish_and_reclaim() {
        let config =
            MaterialVirtualTextureAtlasConfig::new(4, 0, 2, 4, MaterialVtFormat::Rgba8Unorm)
                .unwrap();
        let staging = MaterialVirtualTextureStaging::new(
            config,
            &snapshot(),
            MaterialVirtualTextureGpuConfig::new(true, 1, 4, 4, 2, 1, 0, 2),
        )
        .unwrap();
        let ranges = MaterialVirtualTextureUploadRanges {
            resource_slot: 1,
            page_table: 0..16,
            config: 0..32,
            atlas: Vec::new(),
        };
        let mut lifecycle = MaterialVirtualTextureLifecycle::new();
        let serial = lifecycle.enqueue(staging, snapshot(), ranges).unwrap();
        let submission = lifecycle
            .mark_submitted(serial, GpuCompletionValue(4))
            .unwrap();
        assert_eq!(
            lifecycle.complete(submission, GpuCompletionValue(3)),
            Err(MaterialVirtualTextureLifecycleError::CompletionNotReached)
        );
        let ready = lifecycle
            .complete(submission, GpuCompletionValue(4))
            .unwrap();
        lifecycle.publish(ready, GpuCompletionValue(4)).unwrap();
        assert!(lifecycle.active().is_some());
        assert_eq!(lifecycle.retired_len(), 0);
        let newer = MaterialVirtualTexturePageTableSnapshot::try_from_records(
            3,
            2,
            [MaterialVirtualTexturePageRecord {
                key: key(0),
                slot: 2,
                generation: 2,
                epoch: 3,
                ready: true,
            }],
        )
        .unwrap();
        let staging = MaterialVirtualTextureStaging::new(
            config,
            &newer,
            MaterialVirtualTextureGpuConfig::new(true, 1, 4, 4, 2, 1, 0, 2),
        )
        .unwrap();
        let serial = lifecycle
            .enqueue(
                staging,
                newer,
                MaterialVirtualTextureUploadRanges {
                    resource_slot: 0,
                    page_table: 0..16,
                    config: 0..32,
                    atlas: Vec::new(),
                },
            )
            .unwrap();
        let submission = lifecycle
            .mark_submitted(serial, GpuCompletionValue(8))
            .unwrap();
        let ready = lifecycle
            .complete(submission, GpuCompletionValue(8))
            .unwrap();
        lifecycle.publish(ready, GpuCompletionValue(8)).unwrap();
        assert_eq!(lifecycle.retired_len(), 1);
        assert_eq!(lifecycle.reclaim(GpuCompletionValue(7)), 0);
        assert_eq!(lifecycle.reclaim(GpuCompletionValue(8)), 1);
    }

    #[test]
    fn lifecycle_rejects_mismatched_staging_and_supports_pre_submission_abort() {
        let config =
            MaterialVirtualTextureAtlasConfig::new(4, 0, 2, 4, MaterialVtFormat::Rgba8Unorm)
                .unwrap();
        let first_snapshot = snapshot();
        let staging = MaterialVirtualTextureStaging::new(
            config,
            &first_snapshot,
            MaterialVirtualTextureGpuConfig::new(false, 1, 4, 4, 2, 1, 0, 2),
        )
        .unwrap();
        let ranges = MaterialVirtualTextureUploadRanges {
            resource_slot: 1,
            page_table: 0..16,
            config: 0..32,
            atlas: Vec::new(),
        };
        let mut lifecycle = MaterialVirtualTextureLifecycle::new();
        assert_eq!(
            lifecycle.enqueue(staging, first_snapshot, ranges.clone()),
            Err(MaterialVirtualTextureLifecycleError::InvalidStaging)
        );

        let second_snapshot = snapshot();
        let staging = MaterialVirtualTextureStaging::new(
            config,
            &second_snapshot,
            MaterialVirtualTextureGpuConfig::new(true, 1, 4, 4, 2, 1, 0, 2),
        )
        .unwrap();
        let serial = lifecycle.enqueue(staging, second_snapshot, ranges).unwrap();
        assert_eq!(lifecycle.abort(serial), Ok(()));
        assert!(lifecycle.uploaded_ranges(serial).is_none());
    }

    #[test]
    fn lifecycle_keeps_submitted_update_until_valid_publication() {
        let config =
            MaterialVirtualTextureAtlasConfig::new(4, 0, 2, 4, MaterialVtFormat::Rgba8Unorm)
                .unwrap();
        let snapshot = snapshot();
        let staging = MaterialVirtualTextureStaging::new(
            config,
            &snapshot,
            MaterialVirtualTextureGpuConfig::new(true, 1, 4, 4, 2, 1, 0, 2),
        )
        .unwrap();
        let mut lifecycle = MaterialVirtualTextureLifecycle::new();
        let serial = lifecycle
            .enqueue(
                staging,
                snapshot,
                MaterialVirtualTextureUploadRanges {
                    resource_slot: 1,
                    page_table: 0..16,
                    config: 0..32,
                    atlas: Vec::new(),
                },
            )
            .unwrap();
        let submission = lifecycle
            .mark_submitted(serial, GpuCompletionValue(4))
            .unwrap();
        assert_eq!(
            lifecycle.abort(serial),
            Err(MaterialVirtualTextureLifecycleError::SubmissionInFlight)
        );
        let ready = lifecycle
            .complete(submission, GpuCompletionValue(4))
            .unwrap();
        assert_eq!(
            lifecycle.publish(ready.clone(), GpuCompletionValue(3)),
            Err(MaterialVirtualTextureLifecycleError::InvalidCompletion)
        );
        assert!(lifecycle.active().is_none());
        lifecycle.publish(ready, GpuCompletionValue(4)).unwrap();
        assert_eq!(
            lifecycle.active().map(|active| active.serial()),
            Some(serial)
        );
    }

    #[test]
    fn lifecycle_publishes_empty_snapshot_as_disabled_source_removal() {
        let config =
            MaterialVirtualTextureAtlasConfig::new(4, 0, 2, 4, MaterialVtFormat::Rgba8Unorm)
                .unwrap();
        let first = snapshot();
        let staging = MaterialVirtualTextureStaging::new(
            config,
            &first,
            MaterialVirtualTextureGpuConfig::new(true, 1, 4, 4, 2, 1, 0, 2),
        )
        .unwrap();
        let mut lifecycle = MaterialVirtualTextureLifecycle::new();
        let serial = lifecycle
            .enqueue(
                staging,
                first,
                MaterialVirtualTextureUploadRanges {
                    resource_slot: 1,
                    page_table: 0..16,
                    config: 0..32,
                    atlas: Vec::new(),
                },
            )
            .unwrap();
        let submission = lifecycle
            .mark_submitted(serial, GpuCompletionValue(1))
            .unwrap();
        let ready = lifecycle
            .complete(submission, GpuCompletionValue(1))
            .unwrap();
        lifecycle.publish(ready, GpuCompletionValue(1)).unwrap();

        let empty = MaterialVirtualTexturePageTableSnapshot::try_from_records(3, 2, [])
            .expect("empty source set is a valid snapshot");
        let staging = MaterialVirtualTextureStaging::new(
            config,
            &empty,
            MaterialVirtualTextureGpuConfig::new(false, 0, 4, 4, 2, 1, 0, 0),
        )
        .unwrap();
        let serial = lifecycle
            .enqueue(
                staging,
                empty,
                MaterialVirtualTextureUploadRanges {
                    resource_slot: 0,
                    page_table: 0..0,
                    config: 0..32,
                    atlas: Vec::new(),
                },
            )
            .unwrap();
        let submission = lifecycle
            .mark_submitted(serial, GpuCompletionValue(2))
            .unwrap();
        let ready = lifecycle
            .complete(submission, GpuCompletionValue(2))
            .unwrap();
        lifecycle.publish(ready, GpuCompletionValue(2)).unwrap();
        let active = lifecycle.active().unwrap();
        assert!(active.snapshot().is_empty());
        assert_eq!(active.config().enabled, 0);
        assert_eq!(active.config().entry_count, 0);
    }

    #[test]
    fn lifecycle_rejects_stale_republication_and_rollbacks_then_accepts_new_epoch_removal() {
        let config =
            MaterialVirtualTextureAtlasConfig::new(4, 0, 2, 4, MaterialVtFormat::Rgba8Unorm)
                .unwrap();
        let initial = snapshot();
        let staging = MaterialVirtualTextureStaging::new(
            config,
            &initial,
            MaterialVirtualTextureGpuConfig::new(true, 1, 4, 4, 2, 1, 0, 2),
        )
        .unwrap();
        let mut lifecycle = MaterialVirtualTextureLifecycle::new();
        let serial = lifecycle
            .enqueue(
                staging,
                initial,
                MaterialVirtualTextureUploadRanges {
                    resource_slot: 1,
                    page_table: 0..16,
                    config: 0..32,
                    atlas: Vec::new(),
                },
            )
            .unwrap();
        let submission = lifecycle
            .mark_submitted(serial, GpuCompletionValue(1))
            .unwrap();
        let ready = lifecycle
            .complete(submission, GpuCompletionValue(1))
            .unwrap();
        lifecycle
            .publish(ready.clone(), GpuCompletionValue(1))
            .unwrap();
        assert_eq!(
            lifecycle.publish(ready, GpuCompletionValue(1)),
            Err(MaterialVirtualTextureLifecycleError::InvalidPublication)
        );

        for (epoch, generation) in [(2, 99), (3, 1)] {
            let stale = MaterialVirtualTexturePageTableSnapshot::try_from_records(
                epoch,
                generation,
                [MaterialVirtualTexturePageRecord {
                    key: key(0),
                    slot: 2,
                    generation,
                    epoch,
                    ready: true,
                }],
            )
            .unwrap();
            let staging = MaterialVirtualTextureStaging::new(
                config,
                &stale,
                MaterialVirtualTextureGpuConfig::new(true, 1, 4, 4, 2, 1, 0, 2),
            )
            .unwrap();
            assert_eq!(
                lifecycle.enqueue(
                    staging,
                    stale,
                    MaterialVirtualTextureUploadRanges {
                        resource_slot: 0,
                        page_table: 0..16,
                        config: 0..32,
                        atlas: Vec::new(),
                    },
                ),
                Err(MaterialVirtualTextureLifecycleError::SnapshotRollback)
            );
        }

        let removed = MaterialVirtualTexturePageTableSnapshot::try_from_records(4, 1, [])
            .expect("a newer empty snapshot revokes the previous publication");
        let staging = MaterialVirtualTextureStaging::new(
            config,
            &removed,
            MaterialVirtualTextureGpuConfig::new(false, 0, 4, 4, 2, 1, 0, 0),
        )
        .unwrap();
        let serial = lifecycle
            .enqueue(
                staging,
                removed,
                MaterialVirtualTextureUploadRanges {
                    resource_slot: 0,
                    page_table: 0..0,
                    config: 0..32,
                    atlas: Vec::new(),
                },
            )
            .unwrap();
        let submission = lifecycle
            .mark_submitted(serial, GpuCompletionValue(2))
            .unwrap();
        let ready = lifecycle
            .complete(submission, GpuCompletionValue(2))
            .unwrap();
        lifecycle.publish(ready, GpuCompletionValue(2)).unwrap();
        let active = lifecycle.active().unwrap();
        assert_eq!(active.snapshot().epoch(), 4);
        assert_eq!(active.snapshot().generation(), 1);
        assert!(active.snapshot().is_empty());
    }
}
