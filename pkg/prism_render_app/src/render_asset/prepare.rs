//! Keeping the prepared table in step with the source assets.

use prism_asset::{Asset, AssetId};
use prism_render_driver::RenderDevice;

use super::extract::ExtractedAssets;
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
/// "still loading" from "broken". The frame is consumed either way: a retry is
/// driven by the source's next change, not by re-preparing the same copy.
pub fn prepare_extracted_assets<A: RenderAsset>(
    extracted: &mut ExtractedAssets<A>,
    prepared: &mut RenderAssets<A>,
    device: &dyn RenderDevice,
) -> AssetSyncReport {
    let (assets, removed) = extracted.take();
    let mut report = AssetSyncReport::default();
    for id in removed {
        if let Some(value) = prepared.remove(id) {
            A::unload(&value, device);
            drop(value);
            report.released += 1;
        }
    }
    for (id, source) in assets {
        match A::prepare(&source, id, device, prepared.get(id)) {
            Ok(value) => {
                prepared.insert(id, value);
                report.prepared += 1;
            }
            Err(PrepareAssetError::RetryNextUpdate(_)) => report.retried += 1,
        }
    }
    report
}
