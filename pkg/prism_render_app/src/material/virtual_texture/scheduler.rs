//! Frame-level residency scheduling. Planning identity and upload identity are
//! deliberately separate: new feedback may yield the same physical contents.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Weak},
};

use prism_asset::AssetId;
use prism_ecs::prelude::Resource;
use prism_image::Image;
use prism_render_architecture::texture_streaming::TexturePageKey;
use prism_render_driver::{FrameSync, Instance, RenderDevice};
use prism_render_runtime::RenderRuntime;

use super::{
    gpu::check_owner_runtime, plan_residency, NativeMaterialVt, NativeVtConfig, NativeVtError,
    NativeVtPlan, NativeVtPlanStats, NativeVtPublish, NativeVtResidency, NativeVtSource,
    NativeVtSources,
};
use crate::material::{BindlessSlot, RenderMaterials, TextureSlotHeap};

/// Maximum requests retained from one completed feedback batch. Input order is
/// priority order; duplicates and stale identities do not consume plan budget.
pub const MAX_NATIVE_VT_FEEDBACK: usize = 4096;

#[derive(Clone, Debug)]
struct Epoch {
    owner: Weak<()>,
    heap: Weak<()>,
    device_generation: u64,
    config: NativeVtConfig,
}

impl PartialEq for Epoch {
    fn eq(&self, other: &Self) -> bool {
        self.owner.ptr_eq(&other.owner)
            && self.heap.ptr_eq(&other.heap)
            && self.device_generation == other.device_generation
            && self.config == other.config
    }
}
impl Eq for Epoch {}

#[derive(Clone, Debug)]
struct Reference {
    image: AssetId<Image>,
    slot: BindlessSlot,
    source: Option<Weak<NativeVtSource>>,
}

impl PartialEq for Reference {
    fn eq(&self, other: &Self) -> bool {
        self.image == other.image
            && self.slot == other.slot
            && match (&self.source, &other.source) {
                (Some(a), Some(b)) => a.ptr_eq(b),
                (None, None) => true,
                _ => false,
            }
    }
}
impl Eq for Reference {}

/// Capture at feedback-producing submission time, never at readback arrival.
/// The immutable token stamps device/owner, heap, full image identity, slot
/// generation and source allocation. Weak ownership does not pin source bytes.
#[derive(Clone, Debug)]
pub struct NativeVtFeedbackOrigin {
    epoch: Epoch,
    references: Vec<Reference>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PlanIdentity {
    epoch: Epoch,
    references: Vec<Reference>,
    demand: Vec<TexturePageKey>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct UploadIdentity {
    epoch: Epoch,
    selected: Vec<Reference>,
    residency: NativeVtResidency,
}

struct CachedPlan {
    identity: PlanIdentity,
    plan: NativeVtPlan,
    upload: UploadIdentity,
}

/// Per-frame evidence. Admission diagnostics remain available through `plan`;
/// errors here describe scheduling/publication/refresh, not a missing fine mip.
#[derive(Clone, Debug, Default)]
pub struct NativeVtScheduleReport {
    pub ready: bool,
    pub replanned: bool,
    pub staged: bool,
    pub published: bool,
    pub waiting: bool,
    pub refreshed_rows: u32,
    pub rejected_feedback: usize,
    pub plan_stats: Option<NativeVtPlanStats>,
    pub errors: Vec<NativeVtError>,
}

/// One scheduler per native VT owner. Quiet frames validate identity and refresh
/// rows but do not rebuild tiles, write buffers, or resubmit unchanged residency.
#[derive(Default, Resource)]
pub struct NativeVtScheduler {
    cached: Option<CachedPlan>,
    pending: Option<UploadIdentity>,
    active: Option<UploadIdentity>,
    feedback: Option<(NativeVtFeedbackOrigin, Vec<TexturePageKey>)>,
    ready: bool,
}

impl NativeVtScheduler {
    /// Recovery invalidates both identities and any feedback produced before
    /// it. The owner must separately recreate its device-owned resources.
    pub fn invalidate(&mut self) {
        *self = Self::default();
    }

    pub fn is_ready(&self) -> bool {
        self.ready
    }

    pub fn plan(&self) -> Option<&NativeVtPlan> {
        self.cached.as_ref().map(|cached| &cached.plan)
    }

    pub fn feedback_origin<I: Instance + 'static>(
        &self,
        runtime: &RenderRuntime<I>,
        owner: &NativeMaterialVt<I::Device>,
        sources: &NativeVtSources,
        materials: &RenderMaterials,
    ) -> Result<NativeVtFeedbackOrigin, NativeVtError>
    where
        I::Device: RenderDevice + Send + Sync + 'static,
    {
        check_owner_runtime(&owner.slices, runtime)?;
        Ok(NativeVtFeedbackOrigin {
            epoch: epoch(runtime, owner, materials),
            references: references(sources, materials),
        })
    }

    /// Replace the previous feedback snapshot, retaining at most the fixed
    /// bound. Call with an empty batch to release old demand. Returns the count
    /// retained, before stale-page filtering during prepare. Reading a bounded
    /// prefix also protects against a corrupt or unbounded producer iterator.
    pub fn set_feedback(
        &mut self,
        origin: NativeVtFeedbackOrigin,
        demand: impl IntoIterator<Item = TexturePageKey>,
    ) -> usize {
        let demand: Vec<_> = demand.into_iter().take(MAX_NATIVE_VT_FEEDBACK).collect();
        let count = demand.len();
        self.feedback = Some((origin, demand));
        count
    }

    pub fn prepare<I: Instance + 'static>(
        &mut self,
        runtime: &RenderRuntime<I>,
        owner: &mut NativeMaterialVt<I::Device>,
        sources: &NativeVtSources,
        materials: &mut RenderMaterials,
    ) -> NativeVtScheduleReport
    where
        I::Device: RenderDevice + Send + Sync + 'static,
        I::Queue: FrameSync,
    {
        self.prepare_with_uploader(
            runtime,
            owner,
            sources,
            materials,
            |runtime, owner, sources, heap, residency| {
                owner.stage(runtime, sources, heap, residency)
            },
        )
    }

    /// Backend staging preserves the same plan, completion and publication
    /// protocol as CPU uploads. The callback is invoked only when physical
    /// residency or its source/device/slot identity changes, not on quiet frames.
    pub fn prepare_with_uploader<I: Instance + 'static>(
        &mut self,
        runtime: &RenderRuntime<I>,
        owner: &mut NativeMaterialVt<I::Device>,
        sources: &NativeVtSources,
        materials: &mut RenderMaterials,
        mut stage: impl FnMut(
            &RenderRuntime<I>,
            &mut NativeMaterialVt<I::Device>,
            &NativeVtSources,
            &TextureSlotHeap,
            &NativeVtResidency,
        ) -> Result<(), NativeVtError>,
    ) -> NativeVtScheduleReport
    where
        I::Device: RenderDevice + Send + Sync + 'static,
        I::Queue: FrameSync,
    {
        self.ready = false;
        let mut report = NativeVtScheduleReport::default();
        // Even failures reach row refresh below. In particular stale rows must
        // be revoked when a replacement source cannot be uploaded this frame.
        let planned = self.prepare_plan(runtime, owner, sources, materials, &mut report);
        if planned {
            let desired = &self.cached.as_ref().expect("prepared plan").upload;
            // The production owner starts with a zeroed, disabled binding so a
            // non-VT material can enter Queue on its first frame. Treat that
            // empty residency as already active; authored VT residency still
            // follows the normal staged/completion-qualified path below.
            if desired.residency.pages.is_empty()
                && desired.residency.portable.is_empty()
                && owner.active_is_empty()
                && owner.active_binding(runtime).ok().flatten().is_some()
            {
                self.active = Some(desired.clone());
            }
            // Check obsolescence BEFORE publication, even if the old upload is
            // already complete. Cancellation never releases GPU obligations.
            if owner.has_pending_upload() && self.pending.as_ref() != Some(desired) {
                owner.cancel();
                self.pending = None;
            }
            if !owner.has_pending_upload() {
                self.pending = None;
            }
            let publish_ok = self.publish(runtime, owner, sources, materials, &mut report);
            let desired = &self.cached.as_ref().expect("prepared plan").upload;
            if publish_ok && self.active.as_ref() != Some(desired) && !owner.has_pending_upload() {
                match stage(
                    runtime,
                    owner,
                    sources,
                    materials.textures(),
                    &desired.residency,
                ) {
                    Ok(()) => {
                        self.pending = Some(desired.clone());
                        report.staged = true;
                        self.publish(runtime, owner, sources, materials, &mut report);
                    }
                    Err(error) => report.errors.push(error),
                }
            }
        } else {
            owner.cancel();
            self.pending = None;
        }
        match owner.refresh_rows(runtime, sources, materials) {
            Ok(count) => {
                report.refreshed_rows = report.refreshed_rows.saturating_add(count);
                // Backpressure may keep the previous valid active set online;
                // rows for replaced sources have just been revoked atomically.
                self.ready = planned
                    && owner.active.is_some()
                    && !owner.has_pending_upload()
                    && !report
                        .errors
                        .iter()
                        .any(|error| matches!(error, NativeVtError::MaterialRegistry(_)));
            }
            Err(error) => report.errors.push(error),
        }
        report.waiting = owner.has_pending_upload();
        report.ready = self.ready;
        report
    }

    fn prepare_plan<I: Instance + 'static>(
        &mut self,
        runtime: &RenderRuntime<I>,
        owner: &NativeMaterialVt<I::Device>,
        sources: &NativeVtSources,
        materials: &RenderMaterials,
        report: &mut NativeVtScheduleReport,
    ) -> bool
    where
        I::Device: RenderDevice + Send + Sync + 'static,
    {
        if let Err(error) = check_owner_runtime(&owner.slices, runtime) {
            report.errors.push(error);
            return false;
        }
        let epoch = epoch(runtime, owner, materials);
        let references = references(sources, materials);
        let mut demand = Vec::new();
        let mut seen = BTreeSet::new();
        if let Some((origin, keys)) = &self.feedback {
            let current: BTreeMap<_, _> = references.iter().map(|r| (r.slot.index, r)).collect();
            let produced: BTreeMap<_, _> = origin
                .references
                .iter()
                .map(|r| (r.slot.index, r))
                .collect();
            for &key in keys {
                let reference = current.get(&key.texture);
                let valid = origin.epoch == epoch
                    && reference.is_some_and(|r| {
                        produced.get(&key.texture) == Some(r)
                            && sources.get(key.texture).is_some_and(|source| {
                                source.image() == r.image
                                    && source.layout().page_metadata(key).is_ok()
                            })
                    });
                if !valid {
                    report.rejected_feedback += 1;
                } else if seen.insert(key) {
                    demand.push(key);
                }
            }
        }
        let identity = PlanIdentity {
            epoch,
            references,
            demand,
        };
        if self
            .cached
            .as_ref()
            .is_none_or(|cached| cached.identity != identity)
        {
            let images = identity
                .references
                .iter()
                .map(|r| (r.image, r.slot))
                .collect();
            match plan_residency(owner.config(), sources, &images, &identity.demand) {
                Ok(plan) => {
                    let selected: BTreeSet<_> = plan
                        .residency
                        .pages
                        .iter()
                        .map(|(key, _)| key.texture)
                        .chain(plan.residency.portable.iter().copied())
                        .collect();
                    let upload = UploadIdentity {
                        epoch: identity.epoch.clone(),
                        selected: identity
                            .references
                            .iter()
                            .filter(|r| selected.contains(&r.slot.index))
                            .cloned()
                            .collect(),
                        residency: plan.residency.clone(),
                    };
                    self.cached = Some(CachedPlan {
                        identity,
                        plan,
                        upload,
                    });
                    report.replanned = true;
                }
                Err(error) => {
                    report.errors.push(error);
                    return false;
                }
            }
        }
        report.plan_stats = Some(self.cached.as_ref().expect("cached plan").plan.stats);
        true
    }

    fn publish<I: Instance + 'static>(
        &mut self,
        runtime: &RenderRuntime<I>,
        owner: &mut NativeMaterialVt<I::Device>,
        sources: &NativeVtSources,
        materials: &mut RenderMaterials,
        report: &mut NativeVtScheduleReport,
    ) -> bool
    where
        I::Device: RenderDevice + Send + Sync + 'static,
    {
        match owner.try_publish(runtime, sources, materials) {
            Ok(NativeVtPublish::Published { refreshed_rows }) => {
                // Zero row changes still publishes real GPU contents.
                self.active = self.pending.take();
                report.published = true;
                report.refreshed_rows = report.refreshed_rows.saturating_add(refreshed_rows);
                true
            }
            Ok(_) => true,
            Err(error) => {
                if !owner.has_pending_upload() {
                    self.pending = None;
                }
                report.errors.push(error);
                false
            }
        }
    }
}

fn epoch<I: Instance + 'static>(
    runtime: &RenderRuntime<I>,
    owner: &NativeMaterialVt<I::Device>,
    materials: &RenderMaterials,
) -> Epoch {
    Epoch {
        owner: Arc::downgrade(&owner.incarnation),
        heap: materials.textures().incarnation(),
        device_generation: runtime.device_generation(),
        config: owner.config,
    }
}

fn references(sources: &NativeVtSources, materials: &RenderMaterials) -> Vec<Reference> {
    materials
        .referenced_texture_images()
        .into_iter()
        .map(|(image, slot)| Reference {
            image,
            slot,
            source: sources.get(slot.index).map(Arc::downgrade),
        })
        .collect()
}
