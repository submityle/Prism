//! Keeping the prepared table in step with the source assets.

use alloc::collections::{BTreeMap, BTreeSet};
use core::sync::atomic::Ordering;

use prism_asset::{Asset, AssetId};
use prism_render_driver::{RenderDevice, RenderQueue};

use super::extract::ExtractedAssets;
use super::{GpuPrepareBudget, PrepareCancellation, PrepareContext, RenderAsset, RenderAssets};

/// Why a prepare could not produce a representation (yet).
///
/// Carries the source asset id back, because the caller's next move is almost
/// always "try this one again" — and the renderer must not confuse "not ready"
/// with "never going to be".
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PrepareAssetError<A: Asset> {
    /// A dependency is not prepared yet: keep whatever representation exists and
    /// retry after the dependency's own prepare has run.
    RetryNextUpdate(AssetId<A>),
    /// The asset itself cannot be prepared — its content is invalid (a mesh whose
    /// indices leave its vertex range, say). Keep whatever representation exists
    /// and do **not** retry: nothing changes until the source does, so retrying
    /// every frame would only burn the frame's budget on a content bug.
    Unpreparable {
        /// The asset that cannot be prepared.
        id: AssetId<A>,
    },
}

/// What one asset sync did.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct AssetSyncReport {
    /// Assets prepared (added or re-prepared) this pass.
    pub prepared: usize,
    /// Assets the prepare asked to retry next update.
    pub retried: usize,
    /// Assets whose content cannot be prepared at all.
    pub unpreparable: usize,
    /// Assets whose source went away, whose representations were released.
    pub released: usize,
}

impl AssetSyncReport {
    /// Whether the pass changed the table at all.
    #[must_use]
    pub const fn changed(self) -> bool {
        self.prepared != 0 || self.released != 0
    }
}

/// Prepares a frame's extracted assets and releases the removed ones.
///
/// This is the render side's asset stage. Every extracted asset is prepared with
/// its previous representation (so an update can reuse GPU objects) and stored;
/// every removed id has its representation [`unload`](RenderAsset::unload)ed and
/// dropped, so a deleted asset cannot leave a GPU object behind.
///
/// Removals run **first**, so an asset removed and re-added within one frame ends
/// up prepared rather than dropped after being prepared.
///
/// A prepare that returns [`PrepareAssetError::RetryNextUpdate`] leaves the
/// previous representation untouched — a renderer must never publish a
/// half-prepared asset — and is counted in the report, so the caller can tell
/// "still loading" from "broken". The copied source is retained in the
/// extracted frame for the next prepare pass; this is important because the
/// source may become ready without producing another asset event.
pub fn prepare_extracted_assets<A: RenderAsset>(
    extracted: &mut ExtractedAssets<A>,
    prepared: &mut RenderAssets<A>,
    device: &dyn RenderDevice,
    queue: &dyn RenderQueue,
) -> AssetSyncReport {
    let mut budget = GpuPrepareBudget::unlimited();
    let cancellation = PrepareCancellation::new(false);
    prepare_extracted_assets_with_context(
        extracted,
        prepared,
        device,
        queue,
        &mut budget,
        &cancellation,
    )
}

/// Context-aware form of [`prepare_extracted_assets`].
///
/// The old entry point remains unchanged and uses an unlimited, never-cancelled
/// context. Callers that need a frame allocation budget or cancellation use
/// this form. A contextual implementation must map [`PrepareContextError`] to
/// [`PrepareAssetError::RetryNextUpdate`]; the transaction then rolls back its
/// tracked resources before the source is queued again.
pub fn prepare_extracted_assets_with_context<A: RenderAsset>(
    extracted: &mut ExtractedAssets<A>,
    prepared: &mut RenderAssets<A>,
    device: &dyn RenderDevice,
    queue: &dyn RenderQueue,
    budget: &mut GpuPrepareBudget,
    cancellation: &PrepareCancellation,
) -> AssetSyncReport {
    let (assets, removed) = extracted.take();
    let mut report = AssetSyncReport::default();
    let removed_ids: BTreeSet<_> = removed.iter().copied().collect();
    for id in removed {
        if let Some(value) = prepared.remove(id) {
            A::unload(&value, device);
            drop(value);
            report.released += 1;
        }
    }

    // Extract can accumulate several events for one id before prepare runs,
    // and a retry from the previous pass is already in this same list. Keep
    // only the last copy: it is the newest source state and prevents one
    // deferred asset from multiplying its retry work every frame.
    let mut latest = BTreeMap::new();
    for (id, source) in assets {
        latest.insert(id, source);
    }

    for (id, source) in latest {
        // Removal wins over a retry (and over stale extracted copies). Never
        // put an asset back into the pending queue after its source is gone.
        if removed_ids.contains(&id) {
            continue;
        }
        // Cancellation is a stage-level gate as well as a cooperative context
        // checkpoint. This prevents a legacy/default contextual implementation
        // that performs no allocation through `PrepareContext` from publishing
        // work after its owner has already cancelled the pass.
        if cancellation.load(Ordering::Acquire) {
            extracted.push_extracted(id, source);
            report.retried += 1;
            continue;
        }
        let mut context = PrepareContext::new(device, queue, budget, cancellation);
        let result = A::prepare_with_context(&source, id, &mut context, prepared.get(id));
        match result {
            Ok(value) => {
                let replaces_previous = context.replaces_previous();
                context.commit();
                let previous = prepared.insert(id, value);
                if replaces_previous {
                    if let Some(previous) = previous {
                        // Publish first: if an implementation's unload hook
                        // fails, the new representation remains authoritative
                        // instead of leaving the table empty.
                        A::unload(&previous, device);
                        drop(previous);
                    }
                }
                report.prepared += 1;
            }
            Err(PrepareAssetError::RetryNextUpdate(_)) => {
                context.rollback();
                extracted.push_extracted(id, source);
                report.retried += 1;
            }
            Err(PrepareAssetError::Unpreparable { .. }) => {
                context.rollback();
                report.unpreparable += 1;
            }
        }
    }
    report
}
