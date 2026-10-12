//! Keeping the prepared table in step with the source assets.

use alloc::vec::Vec;

use prism_asset::{Asset, AssetEvent, AssetId, Assets};
use prism_render_driver::RenderDevice;

use super::{RenderAsset, RenderAssets};

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
}

/// What one asset sync did.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct AssetSyncReport {
    /// Assets prepared (added or re-prepared) this pass.
    pub prepared: usize,
    /// Assets the prepare asked to retry next update.
    pub retried: usize,
    /// Assets whose source went away, whose representations were released.
    pub released: usize,
    /// Events for assets that no longer exist in the source arena.
    pub missing: usize,
}

impl AssetSyncReport {
    /// Whether the pass changed the table at all.
    #[must_use]
    pub const fn changed(self) -> bool {
        self.prepared != 0 || self.released != 0
    }
}

/// Brings `prepared` in step with `sources` for the events in `events`.
///
/// This is the render side's asset stage: for each `Added`/`Modified` event the
/// asset is prepared (with its previous representation, so an update can reuse
/// GPU objects) and stored; for each `Removed` event the representation is
/// [`unload`](RenderAsset::unload)ed and dropped, so a deleted asset cannot leave
/// a GPU object behind; for every other event nothing happens.
///
/// A prepare that returns [`PrepareAssetError::RetryNextUpdate`] leaves the
/// previous representation untouched — a renderer must never publish a
/// half-prepared asset — and is counted in the report so the caller can tell
/// "still loading" from "broken".
pub fn sync_assets<A: RenderAsset>(
    sources: &Assets<A::SourceAsset>,
    prepared: &mut RenderAssets<A>,
    device: &dyn RenderDevice,
    events: impl IntoIterator<Item = AssetEvent<A::SourceAsset>>,
) -> AssetSyncReport {
    let mut report = AssetSyncReport::default();
    for event in events {
        let id = event.id();
        match event {
            AssetEvent::Added { .. } | AssetEvent::Modified { .. } => {
                let Some(source) = sources.get(id) else {
                    // The event outlived the asset (removed later in the same
                    // frame): nothing to prepare, and the removal event that
                    // follows (or already ran) handles the table.
                    report.missing += 1;
                    continue;
                };
                match A::prepare(source, id, device, prepared.get(id)) {
                    Ok(value) => {
                        prepared.insert(id, value);
                        report.prepared += 1;
                    }
                    Err(PrepareAssetError::RetryNextUpdate(_)) => report.retried += 1,
                }
            }
            AssetEvent::Removed { .. } => {
                if let Some(value) = prepared.remove(id) {
                    A::unload(&value, device);
                    drop(value);
                    report.released += 1;
                }
            }
            // `Unused`, `Failed`, and `LoadedWithDependencies` do not change the
            // GPU form: an unused asset may come back (a handle can re-pin it),
            // and the other two are loader-side facts.
            AssetEvent::Unused { .. }
            | AssetEvent::Failed { .. }
            | AssetEvent::LoadedWithDependencies { .. } => {}
        }
    }
    report
}

/// Drains `sources`' queued events and syncs the table from them.
///
/// The convenience the render stage actually calls: the asset arena is the
/// event queue, so nothing else has to carry the change list across the
/// boundary.
pub fn drain_and_sync_assets<A: RenderAsset>(
    sources: &mut Assets<A::SourceAsset>,
    prepared: &mut RenderAssets<A>,
    device: &dyn RenderDevice,
) -> AssetSyncReport {
    let events: Vec<AssetEvent<A::SourceAsset>> = sources.drain_events();
    sync_assets(sources, prepared, device, events)
}

