//! Persistent per-camera attachments, shared by window and offscreen paths.
//!
//! Reconcile the complete set of desired camera/output pairs before recording
//! each frame. Never retain a copied target across reconciliation or recovery.
//! After the final submission using these targets, call `seal_submissions`.
//! Retirement first proves completion, then evicts dependent bind groups, and
//! only then destroys attachments through their **creating** device.
//!
//! On recovery, unconfirmed old generations remain quarantined. A completion
//! callback allows `collect_retired` to reclaim them; alternatively call
//! `retire_runtime` while the old runtime is still available. Dropping this
//! cache does not issue GPU destroys: unretired native objects are left in the
//! creating device's resource tables until device teardown. Explicit retirement
//! before dropping the lighting state is the normal shutdown path.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::sync::Arc;
use core::sync::atomic::{AtomicBool, Ordering};

use prism_ecs::prelude::{Entity, Resource};
use prism_render_driver::{FrameSync, Instance, RenderDevice};
use prism_render_runtime::RenderRuntime;
use prism_window::WindowId;

use crate::{RenderTarget, RenderViewport};

/// Convert an output rectangle to the local raster extent requested by a
/// camera. Invalid scales fail open to full resolution; valid scales use
/// ceiling so no output edge is undersampled by integer truncation.
#[must_use]
pub fn surface_view_render_extent(viewport: [u32; 4], resolution_scale: f32) -> RenderViewport {
    let scale = if resolution_scale.is_finite() && (0.0..=1.0).contains(&resolution_scale) {
        resolution_scale
    } else {
        1.0
    };
    let scaled = |extent: u32| ((f64::from(extent) * f64::from(scale)).ceil() as u32).max(1);
    RenderViewport::new(scaled(viewport[2]), scaled(viewport[3]))
}

/// Independent output namespaces, so a camera can feed several resolutions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SurfaceViewOutput {
    /// Application-assigned stable offscreen output identity.
    Offscreen(u64),
    /// Native window output identity.
    Window(WindowId),
}

/// Attachment identity. Viewport origin is intentionally not an allocation key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SurfaceViewTargetKey {
    /// Full scene entity, including its generation.
    pub camera: Entity,
    /// Destination namespace, independent of the camera identity.
    pub output: SurfaceViewOutput,
    /// Local attachment width in physical pixels.
    pub width: u32,
    /// Local attachment height in physical pixels.
    pub height: u32,
}

impl SurfaceViewTargetKey {
    /// Describe a camera's local attachments for one output.
    pub const fn new(camera: Entity, output: SurfaceViewOutput, viewport: RenderViewport) -> Self {
        Self {
            camera,
            output,
            width: viewport.width,
            height: viewport.height,
        }
    }

    /// Attachment dimensions, excluding placement within the destination.
    pub const fn viewport(self) -> RenderViewport {
        RenderViewport::new(self.width, self.height)
    }
}

/// A rejected request leaves all target owners intact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceViewTargetError {
    /// Suspended/zero-sized outputs must be omitted from the desired set.
    EmptyViewport(SurfaceViewTargetKey),
    /// At least one attachment dimension exceeds the creating device's limit.
    ViewportExceedsDeviceLimit(SurfaceViewTargetKey),
    /// A camera/output pair was supplied more than once, possibly at two sizes.
    DuplicateView(SurfaceViewTargetKey),
    /// The creating queue did not confirm completion. Retry without submitting
    /// work that assumes the requested resize/removal has already happened.
    WaitFailed,
    /// A dependent cache refused eviction. Previously evicted groups may be
    /// recreated lazily; no target in this retirement batch has been destroyed.
    EvictionFailed(SurfaceViewTargetKey),
    /// Submission bookkeeping was attempted against another device/generation.
    DeviceMismatch,
}

/// Successful active-generation reconciliation counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SurfaceViewTargetUpdate {
    /// Newly allocated camera/output/extent combinations.
    pub created: usize,
    /// Unchanged combinations that retained exactly the same attachments.
    pub reused: usize,
    /// Attachments explicitly destroyed after completion and eviction.
    pub destroyed: usize,
    /// Old-generation targets still owned by this cache, not exposed for use.
    pub quarantined: usize,
}

struct GenerationTargets<D: RenderDevice> {
    device: Arc<D>,
    generation: u64,
    targets: BTreeMap<SurfaceViewTargetKey, RenderTarget>,
    completion: Option<Arc<AtomicBool>>,
}

impl<D: RenderDevice> GenerationTargets<D> {
    fn belongs_to(&self, device: &Arc<D>, generation: u64) -> bool {
        self.generation == generation && Arc::ptr_eq(&self.device, device)
    }

    fn is_complete(&self) -> bool {
        self.completion
            .as_ref()
            .is_some_and(|complete| complete.load(Ordering::Acquire))
    }
}

/// GPU attachment owners for the complete desired view set (all outputs).
///
/// `D` is the runtime's device type. Device identity is checked in addition to
/// generation, since independent runtimes can both be at generation zero.
/// Callers must serialize reconciliation/retirement with recording/submission.
/// Eviction callbacks must be idempotent and must not submit new GPU work.
/// Never render after a failed reconciliation using the requested new layout.
pub struct SurfaceViewTargets<D: RenderDevice> {
    active: Option<GenerationTargets<D>>,
    retired: Vec<GenerationTargets<D>>,
}

impl<D: RenderDevice> Default for SurfaceViewTargets<D> {
    fn default() -> Self {
        Self {
            active: None,
            retired: Vec::new(),
        }
    }
}

impl<D: RenderDevice + Send + Sync + 'static> Resource for SurfaceViewTargets<D> {}

impl<D: RenderDevice> SurfaceViewTargets<D> {
    /// Active targets, not counting quarantined generations.
    pub fn len(&self) -> usize {
        self.active.as_ref().map_or(0, |owner| owner.targets.len())
    }

    /// Whether the active desired set is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Old-generation target owners awaiting proven completion/eviction.
    pub fn quarantined_len(&self) -> usize {
        self.retired.iter().map(|owner| owner.targets.len()).sum()
    }

    /// Resolve only against the creating runtime. A failed recovery reconcile
    /// can never expose old resource IDs to the replacement device.
    pub fn get<I: Instance<Device = D> + 'static>(
        &self,
        runtime: &RenderRuntime<I>,
        key: SurfaceViewTargetKey,
    ) -> Option<&RenderTarget> {
        let owner = self.active.as_ref()?;
        owner
            .belongs_to(&runtime.device_handle(), runtime.device_generation())
            .then(|| owner.targets.get(&key))
            .flatten()
    }

    /// Deterministic camera/output/extent order, restricted to this runtime.
    pub fn iter<I: Instance<Device = D> + 'static>(
        &self,
        runtime: &RenderRuntime<I>,
    ) -> impl Iterator<Item = (SurfaceViewTargetKey, &RenderTarget)> {
        let device = runtime.device_handle();
        let generation = runtime.device_generation();
        self.active
            .iter()
            .filter(move |owner| owner.belongs_to(&device, generation))
            .flat_map(|owner| owner.targets.iter().map(|(key, target)| (*key, target)))
    }

    /// Reconcile **all** outputs once, before recording the frame. An omitted
    /// camera/output is removed; changing extent replaces its attachments.
    /// Empty requests retire the active set. Stable requests perform no waits,
    /// allocations, eviction, or destruction.
    ///
    /// `evict(target, creating_device, generation)` runs only after completion
    /// has been proved, before any target destruction. Return false to preserve
    /// the entire target retirement batch. The callback must remove just the
    /// matching target's dependent bindings, never a different device's IDs.
    pub fn reconcile<I, F>(
        &mut self,
        runtime: &RenderRuntime<I>,
        desired: &[SurfaceViewTargetKey],
        evict: F,
    ) -> Result<SurfaceViewTargetUpdate, SurfaceViewTargetError>
    where
        I: Instance<Device = D> + 'static,
        I::Queue: FrameSync,
        F: FnMut(&RenderTarget, &Arc<D>, u64) -> bool,
    {
        self.reconcile_with(
            runtime.device_handle(),
            runtime.device_generation(),
            desired,
            || runtime.wait_idle(),
            evict,
        )
    }

    fn reconcile_with<W, F>(
        &mut self,
        device: Arc<D>,
        generation: u64,
        desired: &[SurfaceViewTargetKey],
        mut wait_idle: W,
        mut evict: F,
    ) -> Result<SurfaceViewTargetUpdate, SurfaceViewTargetError>
    where
        W: FnMut() -> bool,
        F: FnMut(&RenderTarget, &Arc<D>, u64) -> bool,
    {
        let mut identities = BTreeSet::new();
        let mut keys = BTreeSet::new();
        let max_dimension = device.capabilities().limits.max_texture_dimension_2d;
        for &key in desired {
            if key.width == 0 || key.height == 0 {
                return Err(SurfaceViewTargetError::EmptyViewport(key));
            }
            if key.width > max_dimension || key.height > max_dimension {
                return Err(SurfaceViewTargetError::ViewportExceedsDeviceLimit(key));
            }
            if !identities.insert((key.camera, key.output)) {
                return Err(SurfaceViewTargetError::DuplicateView(key));
            }
            keys.insert(key);
        }

        let same_device = self
            .active
            .as_ref()
            .is_some_and(|owner| owner.belongs_to(&device, generation));
        let obsolete: Vec<_> = self
            .active
            .iter()
            .filter(|_| same_device)
            .flat_map(|owner| owner.targets.keys())
            .filter(|key| !keys.contains(*key))
            .copied()
            .collect();
        if !obsolete.is_empty() {
            if !wait_idle() {
                return Err(SurfaceViewTargetError::WaitFailed);
            }
            let owner = self.active.as_ref().expect("same-device owner");
            for key in &obsolete {
                if !evict(&owner.targets[key], &owner.device, owner.generation) {
                    return Err(SurfaceViewTargetError::EvictionFailed(*key));
                }
            }
        }

        // Allocate first. A panic during allocation releases unpublished new
        // targets, without removing any existing owners. RenderTarget itself
        // handles partial attachment allocation on the infallible RHI API.
        let mut pending = PendingTargets {
            device: Arc::clone(&device),
            targets: BTreeMap::new(),
        };
        let mut reused = 0;
        for key in keys {
            if same_device && self.active.as_ref().unwrap().targets.contains_key(&key) {
                reused += 1;
            } else {
                let target = RenderTarget::new(&*device, key.viewport().extent());
                pending.targets.insert(key, target);
            }
        }
        let created = pending.targets.len();
        if !same_device {
            if let Some(previous) = self.active.take() {
                if !previous.targets.is_empty() {
                    self.retired.push(previous);
                }
            }
            self.active = Some(GenerationTargets {
                device,
                generation,
                targets: BTreeMap::new(),
                completion: None,
            });
        }
        let owner = self.active.as_mut().expect("active generation installed");
        for key in &obsolete {
            let target = owner.targets.remove(key).expect("obsolete target owned");
            target.destroy(&*owner.device);
        }
        owner.targets.append(&mut pending.targets);
        // A previous frame's completed callback cannot authorize destroying
        // this new recording interval's uses.
        owner.completion = None;
        Ok(SurfaceViewTargetUpdate {
            created,
            reused,
            destroyed: obsolete.len(),
            quarantined: self.quarantined_len(),
        })
    }

    /// Register a real queue-completion callback after the **last** submission
    /// importing active targets in this recording interval, including failed
    /// frame paths that nevertheless submitted work. Do not submit additional
    /// uses until the next `reconcile`; call this again if more work is queued.
    ///
    /// A discarded callback is NOT completion. Its captured creating device
    /// keeps native resource tables alive even if the cache is dropped early.
    pub fn seal_submissions<I>(
        &mut self,
        runtime: &RenderRuntime<I>,
    ) -> Result<(), SurfaceViewTargetError>
    where
        I: Instance<Device = D> + 'static,
        I::Queue: FrameSync,
        D: Send + Sync + 'static,
    {
        let device = runtime.device_handle();
        let Some(owner) = self.active.as_mut() else {
            return Ok(());
        };
        if !owner.belongs_to(&device, runtime.device_generation()) {
            return Err(SurfaceViewTargetError::DeviceMismatch);
        }
        let completion = Arc::new(AtomicBool::new(false));
        owner.completion = Some(Arc::clone(&completion));
        runtime.queue().on_submitted_work_done(Box::new(move || {
            completion.store(true, Ordering::Release);
            drop(device);
        }));
        Ok(())
    }

    /// Reclaim quarantined generations whose final submissions really
    /// completed. Unconfirmed generations are retained without touching their
    /// bindings or textures. Returns the number of targets destroyed.
    pub fn collect_retired<F>(&mut self, mut evict: F) -> Result<usize, SurfaceViewTargetError>
    where
        F: FnMut(&RenderTarget, &Arc<D>, u64) -> bool,
    {
        // Snapshot readiness: a callback completing between preflight and
        // destruction must not bypass dependent-cache eviction.
        let ready: Vec<_> = self
            .retired
            .iter()
            .map(GenerationTargets::is_complete)
            .collect();
        for (owner, ready) in self.retired.iter().zip(&ready) {
            if *ready {
                check_eviction(owner, &mut evict)?;
            }
        }
        let mut destroyed = 0;
        let mut ready = ready.into_iter();
        self.retired.retain(|owner| {
            if !ready.next().expect("readiness for each owner") {
                return true;
            }
            destroyed += destroy_targets(owner);
            false
        });
        Ok(destroyed)
    }

    /// Drain active and quarantined owners belonging to this exact runtime.
    /// Call before recovery/shutdown when possible. A replacement runtime can
    /// never drain an old device, even if numeric generation/IDs collide.
    /// Failed wait/eviction preserves every matching owner for a retry.
    pub fn retire_runtime<I, F>(
        &mut self,
        runtime: &RenderRuntime<I>,
        evict: F,
    ) -> Result<usize, SurfaceViewTargetError>
    where
        I: Instance<Device = D> + 'static,
        I::Queue: FrameSync,
        F: FnMut(&RenderTarget, &Arc<D>, u64) -> bool,
    {
        self.retire_with(
            &runtime.device_handle(),
            runtime.device_generation(),
            || runtime.wait_idle(),
            evict,
        )
    }

    fn retire_with<W, F>(
        &mut self,
        device: &Arc<D>,
        generation: u64,
        mut wait_idle: W,
        mut evict: F,
    ) -> Result<usize, SurfaceViewTargetError>
    where
        W: FnMut() -> bool,
        F: FnMut(&RenderTarget, &Arc<D>, u64) -> bool,
    {
        let matching = |owner: &&GenerationTargets<D>| owner.belongs_to(device, generation);
        let has_targets = self
            .active
            .iter()
            .chain(self.retired.iter())
            .filter(matching)
            .any(|owner| !owner.targets.is_empty());
        if !has_targets {
            return Ok(0);
        }
        if !wait_idle() {
            return Err(SurfaceViewTargetError::WaitFailed);
        }
        for owner in self
            .active
            .iter()
            .chain(self.retired.iter())
            .filter(matching)
        {
            check_eviction(owner, &mut evict)?;
        }
        let mut destroyed = 0;
        if self
            .active
            .as_ref()
            .is_some_and(|owner| owner.belongs_to(device, generation))
        {
            destroyed += destroy_targets(&self.active.take().unwrap());
        }
        self.retired.retain(|owner| {
            if !owner.belongs_to(device, generation) {
                return true;
            }
            destroyed += destroy_targets(owner);
            false
        });
        Ok(destroyed)
    }
}

fn check_eviction<D: RenderDevice, F>(
    owner: &GenerationTargets<D>,
    evict: &mut F,
) -> Result<(), SurfaceViewTargetError>
where
    F: FnMut(&RenderTarget, &Arc<D>, u64) -> bool,
{
    for (key, target) in &owner.targets {
        if !evict(target, &owner.device, owner.generation) {
            return Err(SurfaceViewTargetError::EvictionFailed(*key));
        }
    }
    Ok(())
}

fn destroy_targets<D: RenderDevice>(owner: &GenerationTargets<D>) -> usize {
    for target in owner.targets.values() {
        target.destroy(&*owner.device);
    }
    owner.targets.len()
}

/// Only unpublished resources may be destroyed without a completion barrier.
struct PendingTargets<D: RenderDevice> {
    device: Arc<D>,
    targets: BTreeMap<SurfaceViewTargetKey, RenderTarget>,
}

impl<D: RenderDevice> Drop for PendingTargets<D> {
    fn drop(&mut self) {
        for target in self.targets.values() {
            target.destroy(&*self.device);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::{Cell, RefCell};
    use prism_render_driver::{DeviceDescriptor, Extent3d, RequestAdapterOptions};
    use prism_render_driver_null::{NullDevice, NullInstance};

    const RESOURCES_PER_TARGET: usize = 16;

    #[test]
    fn surface_view_extent_uses_stable_ceil_and_safe_scale_fallback() {
        assert_eq!(
            surface_view_render_extent([7, 5, 101, 55], 1.0),
            RenderViewport::new(101, 55)
        );
        assert_eq!(
            surface_view_render_extent([0, 0, 101, 55], 0.5),
            RenderViewport::new(51, 28)
        );
        assert_eq!(
            surface_view_render_extent([0, 0, 1, 3], 0.01),
            RenderViewport::new(1, 1)
        );
        assert_eq!(
            surface_view_render_extent([0, 0, 101, 55], f32::NAN),
            RenderViewport::new(101, 55)
        );
        assert_eq!(
            surface_view_render_extent([0, 0, 101, 55], 1.5),
            RenderViewport::new(101, 55)
        );
    }

    fn device() -> Arc<NullDevice> {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .unwrap();
        Arc::new(
            instance
                .create_device(&adapter, &DeviceDescriptor::default())
                .unwrap()
                .0,
        )
    }

    fn key(
        index: u32,
        generation: u32,
        output: SurfaceViewOutput,
        width: u32,
    ) -> SurfaceViewTargetKey {
        SurfaceViewTargetKey::new(
            Entity::from_bits((u64::from(generation) << 32) | u64::from(index)).unwrap(),
            output,
            RenderViewport::new(width, 8),
        )
    }

    fn offscreen(index: u32, width: u32) -> SurfaceViewTargetKey {
        key(index, 1, SurfaceViewOutput::Offscreen(0), width)
    }

    fn populate(
        cache: &mut SurfaceViewTargets<NullDevice>,
        device: &Arc<NullDevice>,
        generation: u64,
        desired: &[SurfaceViewTargetKey],
    ) -> SurfaceViewTargetUpdate {
        cache
            .reconcile_with(
                Arc::clone(device),
                generation,
                desired,
                || panic!("new targets need no wait"),
                |_, _, _| panic!("new targets need no eviction"),
            )
            .unwrap()
    }

    #[test]
    fn stable_targets_reuse_all_attachments_in_canonical_order_without_waiting() {
        let device = device();
        let mut cache = SurfaceViewTargets::default();
        let first = offscreen(1, 16);
        let second = offscreen(2, 24);
        assert_eq!(
            populate(&mut cache, &device, 0, &[second, first]).created,
            2
        );
        let before = cache.active.as_ref().unwrap().targets.clone();
        let update = populate(&mut cache, &device, 0, &[first, second]);
        assert_eq!(
            update,
            SurfaceViewTargetUpdate {
                reused: 2,
                ..Default::default()
            }
        );
        assert_eq!(cache.active.as_ref().unwrap().targets, before);
        assert_eq!(
            cache
                .active
                .as_ref()
                .unwrap()
                .targets
                .keys()
                .copied()
                .collect::<Vec<_>>(),
            vec![first, second]
        );
        assert_eq!(device.live_resource_count(), 2 * RESOURCES_PER_TARGET);
    }

    #[test]
    fn full_entity_generation_and_output_namespace_do_not_alias() {
        let device = device();
        let mut cache = SurfaceViewTargets::default();
        let original = offscreen(4, 16);
        let recycled = key(4, 2, original.output, 16);
        let window = key(4, 1, SurfaceViewOutput::Window(WindowId(0)), 32);
        let capture = key(4, 1, SurfaceViewOutput::Offscreen(1), 64);
        populate(
            &mut cache,
            &device,
            0,
            &[capture, recycled, window, original],
        );
        let targets = &cache.active.as_ref().unwrap().targets;
        let views: BTreeSet<_> = targets.values().map(RenderTarget::view).collect();
        assert_eq!(views.len(), 4);
        assert_eq!(targets[&window].viewport(), window.viewport());
        assert_eq!(targets[&capture].viewport(), capture.viewport());
        assert_eq!(device.live_resource_count(), 4 * RESOURCES_PER_TARGET);
    }

    #[test]
    fn resize_and_remove_wait_once_then_evict_before_destroying_attachments() {
        let device = device();
        let mut cache = SurfaceViewTargets::default();
        let original = offscreen(1, 16);
        let removed = offscreen(2, 16);
        let unaffected = key(1, 1, SurfaceViewOutput::Window(WindowId(9)), 48);
        populate(&mut cache, &device, 7, &[removed, original, unaffected]);
        let before = cache.active.as_ref().unwrap().targets.clone();
        let resized = SurfaceViewTargetKey {
            width: 32,
            ..original
        };
        let trace = RefCell::new(Vec::new());
        let update = cache
            .reconcile_with(
                Arc::clone(&device),
                7,
                &[unaffected, resized],
                || {
                    trace.borrow_mut().push("wait");
                    true
                },
                |target, creating_device, generation| {
                    assert_eq!(trace.borrow().first(), Some(&"wait"));
                    assert!(Arc::ptr_eq(creating_device, &device));
                    assert_eq!(generation, 7);
                    assert!(target == &before[&original] || target == &before[&removed]);
                    assert_eq!(device.live_resource_count(), 3 * RESOURCES_PER_TARGET);
                    trace.borrow_mut().push("evict");
                    true
                },
            )
            .unwrap();
        assert_eq!(*trace.borrow(), ["wait", "evict", "evict"]);
        assert_eq!(
            update,
            SurfaceViewTargetUpdate {
                created: 1,
                reused: 1,
                destroyed: 2,
                quarantined: 0
            }
        );
        let targets = &cache.active.as_ref().unwrap().targets;
        assert!(!targets.contains_key(&original));
        assert!(!targets.contains_key(&removed));
        assert_eq!(targets[&unaffected], before[&unaffected]);
        assert_ne!(targets[&resized].view(), before[&original].view());
        assert_ne!(
            targets[&resized].surface_id_view(),
            before[&original].surface_id_view()
        );
        assert_ne!(
            targets[&resized].depth_view(),
            before[&original].depth_view()
        );
        assert_eq!(device.live_resource_count(), 2 * RESOURCES_PER_TARGET);
    }

    #[test]
    fn invalid_or_conflicting_requests_reject_atomically_before_wait_or_allocation() {
        let device = device();
        let mut cache = SurfaceViewTargets::default();
        let original = offscreen(1, 16);
        populate(&mut cache, &device, 0, &[original]);
        let before = cache.active.as_ref().unwrap().targets.clone();
        let empty = SurfaceViewTargetKey {
            width: 0,
            ..original
        };
        let conflict = SurfaceViewTargetKey {
            width: 32,
            ..original
        };
        let oversized = SurfaceViewTargetKey {
            width: u32::MAX,
            ..original
        };
        for (desired, expected) in [
            (
                vec![offscreen(2, 8), empty],
                SurfaceViewTargetError::EmptyViewport(empty),
            ),
            (
                vec![offscreen(2, 8), oversized],
                SurfaceViewTargetError::ViewportExceedsDeviceLimit(oversized),
            ),
            (
                vec![original, conflict],
                SurfaceViewTargetError::DuplicateView(conflict),
            ),
            (
                vec![original, original],
                SurfaceViewTargetError::DuplicateView(original),
            ),
        ] {
            let result = cache.reconcile_with(
                Arc::clone(&device),
                0,
                &desired,
                || panic!("invalid batch must not wait"),
                |_, _, _| panic!("invalid batch must not evict"),
            );
            assert_eq!(result, Err(expected));
            assert_eq!(cache.active.as_ref().unwrap().targets, before);
            assert_eq!(device.live_resource_count(), RESOURCES_PER_TARGET);
        }
    }

    #[test]
    fn failed_wait_preserves_all_owners_and_never_calls_eviction() {
        let device = device();
        let mut cache = SurfaceViewTargets::default();
        let first = offscreen(1, 16);
        let second = offscreen(2, 16);
        populate(&mut cache, &device, 0, &[first, second]);
        let before = cache.active.as_ref().unwrap().targets.clone();
        assert_eq!(
            cache.reconcile_with(
                Arc::clone(&device),
                0,
                &[SurfaceViewTargetKey { width: 32, ..first }],
                || false,
                |_, _, _| panic!("failed wait must not evict"),
            ),
            Err(SurfaceViewTargetError::WaitFailed)
        );
        assert_eq!(
            cache.retire_with(
                &device,
                0,
                || false,
                |_, _, _| panic!("failed wait must not evict"),
            ),
            Err(SurfaceViewTargetError::WaitFailed)
        );
        assert_eq!(cache.active.as_ref().unwrap().targets, before);
        assert_eq!(Arc::strong_count(&device), 2);
        assert_eq!(device.live_resource_count(), 2 * RESOURCES_PER_TARGET);
    }

    #[test]
    fn late_eviction_failure_preserves_entire_target_batch_and_allows_retry() {
        let device = device();
        let mut cache = SurfaceViewTargets::default();
        let first = offscreen(1, 16);
        let second = offscreen(2, 16);
        populate(&mut cache, &device, 0, &[first, second]);
        let before = cache.active.as_ref().unwrap().targets.clone();
        let calls = Cell::new(0);
        assert_eq!(
            cache.reconcile_with(
                Arc::clone(&device),
                0,
                &[],
                || true,
                |_, _, _| {
                    calls.set(calls.get() + 1);
                    calls.get() != 2
                },
            ),
            Err(SurfaceViewTargetError::EvictionFailed(second))
        );
        assert_eq!(calls.get(), 2);
        assert_eq!(cache.active.as_ref().unwrap().targets, before);
        assert_eq!(device.live_resource_count(), 2 * RESOURCES_PER_TARGET);
        assert_eq!(
            cache.retire_with(&device, 0, || true, |_, _, _| true),
            Ok(2)
        );
        assert!(cache.is_empty());
        assert_eq!(device.live_resource_count(), 0);
    }

    #[test]
    fn replacement_device_with_colliding_ids_quarantines_old_owners() {
        let old_device = device();
        let new_device = device();
        let mut cache = SurfaceViewTargets::default();
        let view = offscreen(1, 16);
        populate(&mut cache, &old_device, 0, &[view]);
        let old_target = cache.active.as_ref().unwrap().targets[&view];
        let update = populate(&mut cache, &new_device, 0, &[view]);
        assert_eq!(update.created, 1);
        assert_eq!(update.quarantined, 1);
        assert_eq!(old_target, cache.active.as_ref().unwrap().targets[&view]);
        assert_eq!(
            cache.collect_retired(|_, _, _| panic!("no completion proof")),
            Ok(0)
        );
        assert_eq!(
            cache.retire_with(
                &new_device,
                0,
                || true,
                |_, owner, _| {
                    assert!(Arc::ptr_eq(owner, &new_device));
                    true
                }
            ),
            Ok(1)
        );
        assert_eq!(cache.quarantined_len(), 1);
        assert_eq!(new_device.live_resource_count(), 0);
        assert_eq!(old_device.live_resource_count(), RESOURCES_PER_TARGET);
        assert_eq!(
            cache.retire_with(
                &old_device,
                0,
                || true,
                |_, owner, _| {
                    assert!(Arc::ptr_eq(owner, &old_device));
                    true
                }
            ),
            Ok(1)
        );
        assert_eq!(old_device.live_resource_count(), 0);
    }

    #[test]
    fn generation_is_checked_even_when_device_pointer_is_unchanged() {
        let device = device();
        let mut cache = SurfaceViewTargets::default();
        let view = offscreen(1, 16);
        populate(&mut cache, &device, 4, &[view]);
        let previous = cache.active.as_ref().unwrap().targets[&view];
        populate(&mut cache, &device, 5, &[view]);
        assert_ne!(previous, cache.active.as_ref().unwrap().targets[&view]);
        assert_eq!(
            cache.retire_with(
                &device,
                5,
                || true,
                |_, _, generation| {
                    assert_eq!(generation, 5);
                    true
                }
            ),
            Ok(1)
        );
        assert_eq!(cache.quarantined_len(), 1);
        assert_eq!(device.live_resource_count(), RESOURCES_PER_TARGET);
        assert_eq!(
            cache.retire_with(&device, 4, || true, |_, _, _| true),
            Ok(1)
        );
        assert_eq!(device.live_resource_count(), 0);
    }

    #[test]
    fn quarantined_wait_and_eviction_failure_keep_creating_device_alive() {
        let old_device = device();
        let new_device = device();
        let mut cache = SurfaceViewTargets::default();
        let view = offscreen(1, 16);
        populate(&mut cache, &old_device, 3, &[view]);
        populate(&mut cache, &new_device, 4, &[view]);
        assert_eq!(
            cache.retire_with(&old_device, 3, || false, |_, _, _| false),
            Err(SurfaceViewTargetError::WaitFailed)
        );
        assert_eq!(
            cache.retire_with(&old_device, 3, || true, |_, _, _| false),
            Err(SurfaceViewTargetError::EvictionFailed(view))
        );
        assert_eq!(Arc::strong_count(&old_device), 2);
        assert_eq!(cache.quarantined_len(), 1);
        assert_eq!(old_device.live_resource_count(), RESOURCES_PER_TARGET);
        assert_eq!(new_device.live_resource_count(), RESOURCES_PER_TARGET);
    }

    #[test]
    fn completion_during_eviction_cannot_bypass_preflight_for_another_generation() {
        let first_device = device();
        let second_device = device();
        let current_device = device();
        let mut cache = SurfaceViewTargets::default();
        let view = offscreen(1, 16);
        populate(&mut cache, &first_device, 0, &[view]);
        cache.active.as_mut().unwrap().completion = Some(Arc::new(AtomicBool::new(true)));
        populate(&mut cache, &second_device, 1, &[view]);
        let later = Arc::new(AtomicBool::new(false));
        cache.active.as_mut().unwrap().completion = Some(Arc::clone(&later));
        populate(&mut cache, &current_device, 2, &[view]);
        let evicted = RefCell::new(Vec::new());
        assert_eq!(
            cache.collect_retired(|_, _, generation| {
                evicted.borrow_mut().push(generation);
                later.store(true, Ordering::Release);
                true
            }),
            Ok(1)
        );
        assert_eq!(*evicted.borrow(), [0]);
        assert_eq!(first_device.live_resource_count(), 0);
        assert_eq!(second_device.live_resource_count(), RESOURCES_PER_TARGET);
        assert_eq!(
            cache.collect_retired(|_, owner, generation| {
                assert_eq!(generation, 1);
                assert!(Arc::ptr_eq(owner, &second_device));
                true
            }),
            Ok(1)
        );
        assert_eq!(second_device.live_resource_count(), 0);
        assert_eq!(current_device.live_resource_count(), RESOURCES_PER_TARGET);
    }

    #[test]
    fn completed_generation_still_requires_successful_dependent_eviction() {
        let old_device = device();
        let new_device = device();
        let mut cache = SurfaceViewTargets::default();
        let view = offscreen(1, 16);
        populate(&mut cache, &old_device, 0, &[view]);
        cache.active.as_mut().unwrap().completion = Some(Arc::new(AtomicBool::new(true)));
        populate(&mut cache, &new_device, 1, &[view]);
        assert_eq!(
            cache.collect_retired(|_, _, _| false),
            Err(SurfaceViewTargetError::EvictionFailed(view))
        );
        assert_eq!(cache.quarantined_len(), 1);
        assert_eq!(old_device.live_resource_count(), RESOURCES_PER_TARGET);
        assert_eq!(cache.collect_retired(|_, _, _| true), Ok(1));
        assert_eq!(old_device.live_resource_count(), 0);
        assert_eq!(new_device.live_resource_count(), RESOURCES_PER_TARGET);
    }

    #[test]
    fn runtime_lookup_submission_seal_and_explicit_shutdown_respect_identity() {
        let runtime =
            RenderRuntime::headless(NullInstance::new(), Extent3d::new_2d(16, 8)).unwrap();
        let other = RenderRuntime::headless(NullInstance::new(), Extent3d::new_2d(16, 8)).unwrap();
        let baseline = runtime.device().live_resource_count();
        let mut cache = SurfaceViewTargets::default();
        let view = offscreen(1, 16);
        cache.reconcile(&runtime, &[view], |_, _, _| true).unwrap();
        assert!(cache.get(&runtime, view).is_some());
        assert!(cache.get(&other, view).is_none());
        assert_eq!(cache.iter(&runtime).count(), 1);
        assert_eq!(cache.iter(&other).count(), 0);
        assert_eq!(
            cache.seal_submissions(&other),
            Err(SurfaceViewTargetError::DeviceMismatch)
        );
        cache.seal_submissions(&runtime).unwrap();
        assert!(cache.active.as_ref().unwrap().is_complete());
        // Starting another recording interval invalidates that completed seal.
        cache.reconcile(&runtime, &[view], |_, _, _| true).unwrap();
        assert!(!cache.active.as_ref().unwrap().is_complete());
        cache.seal_submissions(&runtime).unwrap();
        cache.reconcile(&other, &[view], |_, _, _| true).unwrap();
        assert!(cache.get(&runtime, view).is_none());
        assert_eq!(
            cache.collect_retired(|_, owner, generation| {
                assert_eq!(generation, runtime.device_generation());
                assert!(Arc::ptr_eq(owner, &runtime.device_handle()));
                true
            }),
            Ok(1)
        );
        assert_eq!(runtime.device().live_resource_count(), baseline);
        assert_eq!(cache.retire_runtime(&other, |_, _, _| true), Ok(1));
        assert!(cache.is_empty());
        assert_eq!(cache.quarantined_len(), 0);
    }

    #[test]
    fn dropping_unconfirmed_cache_never_issues_early_gpu_destruction() {
        let device = device();
        let mut cache = SurfaceViewTargets::default();
        let view = offscreen(1, 16);
        populate(&mut cache, &device, 0, &[view]);
        let target = cache.active.as_ref().unwrap().targets[&view];
        drop(cache);
        assert_eq!(device.live_resource_count(), RESOURCES_PER_TARGET);
        // No actual submission exists in this null-backend test.
        target.destroy(&*device);
        assert_eq!(device.live_resource_count(), 0);
    }

    #[test]
    fn unpublished_allocation_owner_cleans_up_on_unwind() {
        let device = device();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut pending = PendingTargets {
                device: Arc::clone(&device),
                targets: BTreeMap::new(),
            };
            let view = offscreen(1, 16);
            pending
                .targets
                .insert(view, RenderTarget::new(&*device, view.viewport().extent()));
            panic!("injected allocation failure after one complete target");
        }));
        assert!(result.is_err());
        assert_eq!(device.live_resource_count(), 0);
    }
}
