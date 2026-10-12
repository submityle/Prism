//! Render assets: the GPU representation of a simulation-side asset.
//!
//! An asset lives once, in the main world's [`Assets`] arena, and its GPU form
//! lives in the render world. Bevy splits that into `RenderAsset` (how to
//! prepare one), `RenderAssets<A>` (the prepared table), and a prepare stage
//! that keeps the table in step with the source. Prism keeps the same three
//! pieces, with two deliberate differences documented on [`RenderAsset`]:
//!
//! - the source asset is **borrowed** while preparing (Prism's asset arena hands
//!   out `&A`; nothing moves data out of the main world), and
//! - releasing GPU objects is an explicit [`RenderAsset::unload`] call, so an
//!   implementation that owns RHI handles has one documented place to destroy
//!   them — the sync reports removals rather than leaking them.
//!
//! The table is keyed by [`AssetId`], so the identity that a render-side phase
//! item or bind group carries is the same one the simulation uses.

mod extract;
mod prepare;
mod stage;
mod table;

#[cfg(test)]
mod tests;

use prism_asset::Asset;
use prism_asset::AssetId;
use prism_render_driver::RenderDevice;

pub use extract::{ExtractedAssets, extract_assets};
pub use prepare::{AssetSyncReport, PrepareAssetError, prepare_extracted_assets};
pub use stage::{AssetSyncStats, prepare_assets, register_render_assets};
pub use table::RenderAssets;

/// An asset the renderer keeps a separate GPU form of.
///
/// The source is borrowed, not consumed: the simulation world stays the single
/// owner of the authored data (design §21's one-way rule), and a prepare copies
/// what the GPU needs — a vertex buffer, a texture upload — out of it. Bevy's
/// `take_gpu_data` exists because its extract *moves* heavy payloads; Prism has
/// no such split to serve, so it is deliberately not ported until a measurement
/// asks for it.
pub trait RenderAsset: Send + Sync + 'static + Sized {
    /// The simulation-side asset this prepares from.
    ///
    /// `Clone` because the render world receives a *copy* of what changed this
    /// frame (the simulation keeps its asset; see [`extract_assets`]).
    type SourceAsset: Asset + Clone;

    /// Prepares (or re-prepares) the GPU form of `source`.
    ///
    /// `previous` is the representation from the last successful prepare, so an
    /// update can grow an existing buffer instead of reallocating, or keep the
    /// parts that did not change.
    ///
    /// Returns [`PrepareAssetError::RetryNextUpdate`] when a dependency is not
    /// ready yet (a mesh waiting on its texture, a material waiting on its
    /// pipeline); the sync then leaves the previous representation in place and
    /// retries next frame rather than recording a half-built asset.
    fn prepare(
        source: &Self::SourceAsset,
        id: AssetId<Self::SourceAsset>,
        device: &dyn RenderDevice,
        previous: Option<&Self>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>>;

    /// Releases the RHI objects `prepared` owns.
    ///
    /// Called by [`sync_assets`] when the source asset is gone (or, eventually,
    /// when the renderer reclaims it), *before* the representation is dropped —
    /// dropping a [`BufferId`](prism_render_driver::BufferId) alone would leak
    /// the GPU buffer, which is exactly what this hook prevents.
    ///
    /// The default does nothing, which is correct for a representation that owns
    /// no RHI objects (one that only borrows cached views, say).
    fn unload(prepared: &Self, device: &dyn RenderDevice) {
        let _ = (prepared, device);
    }
}
