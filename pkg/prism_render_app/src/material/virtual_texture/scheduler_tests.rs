//! Null-backed scheduler state transitions using explicit queue completion.
//! These exercise production CPU staging, not hardware GPU execution.
use super::*;
use crate::material::TextureRequest;

struct Scheduled {
    fixture: Fixture,
    scheduler: NativeVtScheduler,
}

impl Scheduled {
    fn new() -> Self {
        let mut fixture = Fixture::new();
        fixture.materials.textures_mut().release(image_id(1));
        let rows = fixture
            .materials
            .resolve_textures(
                fixture.handle.index,
                [(11, TextureRequest::Image(image_id(1)))],
            )
            .expect("fixture image fits the texture heap");
        fixture.slot = fixture.materials.texture_slot(image_id(1)).unwrap();
        let mut record = fallback_material_record(fixture.handle, 2);
        record.textures = rows;
        fixture.materials.publish(record).unwrap();
        fixture.sources = NativeVtSources::default();
        fixture
            .sources
            .insert(
                image_id(1),
                snapshot(fixture.slot.index, 4, 1, MaterialVtFormat::Rgba8Unorm),
            )
            .unwrap();
        Self {
            fixture,
            scheduler: NativeVtScheduler::default(),
        }
    }

    fn prepare(&mut self) -> NativeVtScheduleReport {
        let f = &mut self.fixture;
        self.scheduler
            .prepare(&f.runtime, &mut f.vt, &f.sources, &mut f.materials)
    }

    fn origin(&self) -> NativeVtFeedbackOrigin {
        let f = &self.fixture;
        self.scheduler
            .feedback_origin(&f.runtime, &f.vt, &f.sources, &f.materials)
            .unwrap()
    }

    fn publish(&mut self) {
        let first = self.prepare();
        assert!(
            first.staged && first.waiting && !first.published && !first.ready,
            "{first:?}"
        );
        assert!(first.errors.is_empty(), "{first:?}");
        self.fixture.runtime.queue().complete();
        let next = self.prepare();
        assert!(
            next.published && next.ready && !next.waiting && !next.staged,
            "{next:?}"
        );
        assert!(next.errors.is_empty(), "{next:?}");
    }
}

#[test]
fn scheduler_quiet_frames_do_not_replan_or_upload() {
    let mut s = Scheduled::new();
    s.publish();
    let writes = s.fixture.runtime.queue().writes();
    let submitted = s.fixture.runtime.queue().progress.lock().unwrap().submitted;
    let revision = s
        .fixture
        .materials
        .registry()
        .get(s.fixture.handle)
        .unwrap()
        .revision;
    for _ in 0..3 {
        let report = s.prepare();
        assert!(report.ready && !report.replanned && !report.staged && !report.published);
        assert_eq!(report.refreshed_rows, 0);
        assert!(report.errors.is_empty());
    }
    assert_eq!(s.fixture.runtime.queue().writes(), writes);
    assert_eq!(
        s.fixture.runtime.queue().progress.lock().unwrap().submitted,
        submitted
    );
    assert_eq!(
        s.fixture
            .materials
            .registry()
            .get(s.fixture.handle)
            .unwrap()
            .revision,
        revision
    );
}

#[test]
fn scheduler_custom_uploader_retries_failure_and_skips_quiet_frames() {
    let mut s = Scheduled::new();
    let mut attempts = 0;
    let f = &mut s.fixture;
    let writes = f.runtime.queue().writes();
    let report = s.scheduler.prepare_with_uploader(
        &f.runtime,
        &mut f.vt,
        &f.sources,
        &mut f.materials,
        |_, _, _, _, _| {
            attempts += 1;
            Err(NativeVtError::AllocationFailed)
        },
    );
    assert!(
        report.replanned && !report.staged && !report.published,
        "{report:?}"
    );
    assert!(!report.ready && !report.waiting, "{report:?}");
    assert_eq!(report.errors, vec![NativeVtError::AllocationFailed]);
    assert_eq!(attempts, 1);
    assert_eq!(f.runtime.queue().writes(), writes);
    assert!(f.vt.active_binding(&f.runtime).unwrap().is_none());

    // A successful plan is not a successful upload: retry the identical plan.
    let report = s.scheduler.prepare_with_uploader(
        &f.runtime,
        &mut f.vt,
        &f.sources,
        &mut f.materials,
        |runtime, owner, sources, heap, residency| {
            attempts += 1;
            owner.stage(runtime, sources, heap, residency)
        },
    );
    assert!(
        !report.replanned && report.staged && report.waiting,
        "{report:?}"
    );
    assert!(!report.published && !report.ready, "{report:?}");
    assert!(report.errors.is_empty(), "{report:?}");
    assert_eq!(attempts, 2);
    let writes = f.runtime.queue().writes();
    let submitted = f.runtime.queue().progress.lock().unwrap().submitted;
    f.runtime.queue().complete();

    for published in [true, false, false] {
        let report = s.scheduler.prepare_with_uploader(
            &f.runtime,
            &mut f.vt,
            &f.sources,
            &mut f.materials,
            |_, _, _, _, _| panic!("unchanged residency must not invoke the uploader"),
        );
        assert_eq!(report.published, published, "{report:?}");
        assert!(
            report.ready && !report.replanned && !report.staged && !report.waiting,
            "{report:?}"
        );
        assert!(report.errors.is_empty(), "{report:?}");
    }
    assert_eq!(f.runtime.queue().writes(), writes);
    assert_eq!(
        f.runtime.queue().progress.lock().unwrap().submitted,
        submitted
    );
}

#[test]
fn scheduler_feedback_replans_but_identical_physical_contents_do_not_upload() {
    let mut s = Scheduled::new();
    s.publish();
    let writes = s.fixture.runtime.queue().writes();
    let tail = key(s.fixture.slot.index, 2, 0);
    s.scheduler.set_feedback(s.origin(), [tail, tail]);
    let report = s.prepare();
    assert!(report.replanned && !report.staged && report.ready);
    assert_eq!(s.fixture.runtime.queue().writes(), writes);
    assert_eq!(s.scheduler.plan().unwrap().stats.unique_demand, 1);
    let fine = key(s.fixture.slot.index, 0, 0);
    s.scheduler.set_feedback(s.origin(), [fine]);
    let report = s.prepare();
    assert!(report.staged && report.waiting && !report.ready);
    s.fixture.runtime.queue().complete();
    assert!(s.prepare().published);
}

#[test]
fn scheduler_source_replacement_rejects_old_feedback_and_reuploads_equal_page_keys() {
    let mut s = Scheduled::new();
    s.publish();
    let origin = s.origin();
    let pages = s.scheduler.plan().unwrap().residency.pages.clone();
    let writes = s.fixture.runtime.queue().writes();
    let f = &mut s.fixture;
    let source = Arc::downgrade(f.sources.get(f.slot.index).unwrap());
    let revision = source.upgrade().unwrap().revision();
    // Reset the registry too: equal revisions, descriptors, bytes and pages
    // must not hide replacement of the immutable source allocation.
    f.sources = NativeVtSources::default();
    let replacement = f
        .sources
        .insert(
            image_id(1),
            snapshot(f.slot.index, 4, 1, MaterialVtFormat::Rgba8Unorm),
        )
        .unwrap();
    assert_eq!(replacement.revision(), revision);
    assert!(
        source.upgrade().is_none(),
        "cache must not retain replaced image bytes"
    );
    s.scheduler
        .set_feedback(origin, [key(s.fixture.slot.index, 0, 0)]);
    let report = s.prepare();
    assert_eq!(report.rejected_feedback, 1);
    assert!(
        report.replanned && report.staged && report.waiting && !report.ready,
        "{report:?}"
    );
    assert!(report.errors.is_empty());
    assert_eq!(s.scheduler.plan().unwrap().residency.pages, pages);
    assert!(s.fixture.runtime.queue().writes() > writes);
    assert_eq!(s.fixture.row().virtual_texture_id, NO_VIRTUAL_TEXTURE);
    s.fixture.runtime.queue().complete();
    assert!(s.prepare().published);
    assert_eq!(s.fixture.row().virtual_texture_id, s.fixture.slot.index);
}

#[test]
fn scheduler_reused_slot_generation_rejects_old_feedback_but_accepts_fresh_origin() {
    let mut s = Scheduled::new();
    s.publish();
    let origin = s.origin();
    let old_slot = s.fixture.slot;
    let source = Arc::clone(s.fixture.sources.get(old_slot.index).unwrap());
    let fine = key(old_slot.index, 0, 0);

    // Resolve the same image again, releasing its last tracked lease first.
    // Only the slot generation changes: owner, heap and source stay identical.
    let f = &mut s.fixture;
    let rows = f
        .materials
        .resolve_textures(f.handle.index, [(11, TextureRequest::Image(image_id(1)))])
        .expect("resident image remains resolvable");
    f.slot = f.materials.texture_slot(image_id(1)).unwrap();
    assert_eq!(f.slot.index, old_slot.index);
    assert_ne!(f.slot.generation, old_slot.generation);
    assert!(Arc::ptr_eq(&source, f.sources.get(f.slot.index).unwrap()));
    let mut record = f.materials.registry().get(f.handle).unwrap().clone();
    record.revision += 1;
    record.textures = rows;
    f.materials.publish(record).unwrap();

    s.scheduler.set_feedback(origin, [fine]);
    let report = s.prepare();
    assert!(
        report.replanned && report.staged && report.waiting,
        "{report:?}"
    );
    assert!(report.errors.is_empty());
    assert_eq!(report.rejected_feedback, 1);
    let plan = s.scheduler.plan().unwrap();
    assert_eq!(plan.stats.demand_count, 0);
    assert_eq!(plan.stats.detail_pages, 0);
    assert_eq!(plan.residency.pages.len(), 1);
    assert_eq!(plan.residency.pages[0].0, key(old_slot.index, 2, 0));
    s.fixture.runtime.queue().complete();
    assert!(s.prepare().published);
    assert_eq!(s.fixture.row().generation, s.fixture.slot.generation);
    assert_eq!(s.fixture.row().virtual_texture_id, old_slot.index);

    s.scheduler.set_feedback(s.origin(), [fine]);
    let report = s.prepare();
    assert_eq!(report.rejected_feedback, 0);
    assert!(
        report.replanned && report.staged && report.waiting,
        "{report:?}"
    );
    assert!(report.errors.is_empty());
    let plan = s.scheduler.plan().unwrap();
    assert_eq!(plan.stats.demand_count, 1);
    assert_eq!(plan.stats.detail_pages, 2);
    assert_eq!(plan.residency.pages.len(), 3);
    assert!(plan.residency.pages.iter().any(|(page, _)| *page == fine));
    s.fixture.runtime.queue().complete();
    let report = s.prepare();
    assert!(report.published && report.ready, "{report:?}");
}

#[test]
fn scheduler_cancels_completed_obsolete_pending_before_publication() {
    let mut s = Scheduled::new();
    assert!(s.prepare().staged);
    s.fixture.runtime.queue().complete();
    s.fixture
        .sources
        .insert(
            image_id(1),
            snapshot(s.fixture.slot.index, 4, 1, MaterialVtFormat::Rgba8Unorm),
        )
        .unwrap();
    let report = s.prepare();
    assert!(!report.published && report.staged && !report.ready);
    assert!(s
        .fixture
        .vt
        .active_binding(&s.fixture.runtime)
        .unwrap()
        .is_none());
    s.fixture.runtime.queue().complete();
    assert!(s.prepare().published);
}

#[test]
fn scheduler_cancellation_waits_for_real_callback_and_retries_slice_busy() {
    let mut s = Scheduled::new();
    assert!(s.prepare().staged);
    s.fixture
        .sources
        .insert(
            image_id(1),
            snapshot(s.fixture.slot.index, 4, 1, MaterialVtFormat::Rgba8Unorm),
        )
        .unwrap();
    let report = s.prepare();
    assert!(!report.staged && report.errors.contains(&NativeVtError::SliceBusy));
    assert!(!report.ready && !report.waiting);
    let writes = s.fixture.runtime.queue().writes();
    assert!(!s.prepare().staged);
    assert_eq!(s.fixture.runtime.queue().writes(), writes);
    s.fixture.runtime.queue().complete();
    assert!(s.prepare().staged);
    s.fixture.runtime.queue().complete();
    assert!(s.prepare().published);
}

#[test]
fn scheduler_removing_last_reference_publishes_real_empty_snapshot() {
    let mut s = Scheduled::new();
    s.publish();
    let old = s.fixture.binding();
    s.fixture.materials.release_textures(s.fixture.handle.index);
    let report = s.prepare();
    assert!(report.staged && report.waiting);
    assert!(s.scheduler.plan().unwrap().residency.pages.is_empty());
    assert!(s.scheduler.plan().unwrap().residency.portable.is_empty());
    s.fixture.runtime.queue().complete();
    let report = s.prepare();
    assert!(report.published && report.ready);
    assert_eq!(s.fixture.row().virtual_texture_id, NO_VIRTUAL_TEXTURE);
    let binding = s.fixture.binding();
    assert_ne!(binding.config(), old.config());
    let bytes = s.fixture.runtime.queue().buffer_bytes(binding.config());
    let config: NativeVtGpuConfig = bytemuck::pod_read_unaligned(&bytes);
    assert_eq!(config.enabled, 0);
    let writes = s.fixture.runtime.queue().writes();
    assert!(!s.prepare().staged);
    assert_eq!(writes, s.fixture.runtime.queue().writes());
}

#[test]
fn scheduler_recovery_rotates_epoch_and_never_publishes_old_feedback() {
    let mut s = Scheduled::new();
    s.publish();
    let origin = s.origin();
    let generation = s.fixture.runtime.device_generation();
    s.fixture
        .runtime
        .device_handle()
        .lost
        .store(true, Ordering::Release);
    let report = s.prepare();
    assert!(!report.ready);
    assert!(report.errors.contains(&NativeVtError::DeviceLost));
    s.fixture.runtime.recover_device().unwrap();
    assert_ne!(s.fixture.runtime.device_generation(), generation);
    s.fixture
        .vt
        .recreate_for_device(&s.fixture.runtime)
        .unwrap();
    assert!(s
        .fixture
        .vt
        .active_binding(&s.fixture.runtime)
        .unwrap()
        .is_none());
    s.scheduler.invalidate();
    assert!(!s.scheduler.is_ready());
    assert!(s.scheduler.plan().is_none());
    s.scheduler
        .set_feedback(origin, [key(s.fixture.slot.index, 0, 0)]);
    let report = s.prepare();
    assert!(
        report.replanned && report.staged && report.waiting && !report.ready,
        "{report:?}"
    );
    assert!(report.errors.is_empty());
    assert_eq!(report.rejected_feedback, 1);
    assert_eq!(s.scheduler.plan().unwrap().stats.demand_count, 0);
    // The same owner's tombstones survive resource recreation, so the stale
    // virtual row is revoked before the replacement upload completes.
    assert_eq!(s.fixture.row().virtual_texture_id, NO_VIRTUAL_TEXTURE);
    s.fixture.runtime.queue().complete();
    let report = s.prepare();
    assert!(report.published && report.ready, "{report:?}");
    assert_eq!(
        s.fixture.binding().device_generation(),
        s.fixture.runtime.device_generation()
    );
    assert_eq!(s.fixture.row().virtual_texture_id, s.fixture.slot.index);
}

#[test]
fn scheduler_feedback_is_bounded_and_invalid_pages_do_not_affect_planning() {
    let mut s = Scheduled::new();
    s.publish();
    let invalid = key(s.fixture.slot.index, 30, 0);
    assert_eq!(
        s.scheduler
            .set_feedback(s.origin(), std::iter::repeat(invalid)),
        MAX_NATIVE_VT_FEEDBACK
    );
    let writes = s.fixture.runtime.queue().writes();
    let report = s.prepare();
    assert_eq!(report.rejected_feedback, MAX_NATIVE_VT_FEEDBACK);
    assert!(!report.replanned && !report.staged);
    assert_eq!(writes, s.fixture.runtime.queue().writes());
}

#[test]
fn scheduler_new_authored_rows_refresh_without_uploading_atlas() {
    let mut s = Scheduled::new();
    s.publish();
    let writes = s.fixture.runtime.queue().writes();
    let mut record = s
        .fixture
        .materials
        .registry()
        .get(s.fixture.handle)
        .unwrap()
        .clone();
    record.revision += 1;
    record.textures[0] =
        GpuMaterialTexture::non_virtual(s.fixture.slot.index, s.fixture.slot.generation, 17, 23);
    s.fixture.materials.publish(record).unwrap();
    let report = s.prepare();
    assert_eq!(report.refreshed_rows, 1);
    assert!(report.ready && !report.staged && !report.replanned);
    let row = s.fixture.row();
    assert_eq!((row.semantic, row.sampler_index), (17, 23));
    assert_eq!(row.virtual_texture_id, s.fixture.slot.index);
    assert_eq!(writes, s.fixture.runtime.queue().writes());
}
