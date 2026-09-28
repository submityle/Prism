//! Binned render phase for the transparent forward (WBOIT) draw pass.
//!
//! Transparent geometry cannot be resolved through the visibility buffer (one
//! opaque surface per pixel), so it is drawn in a dedicated forward pass whose
//! fragments accumulate into the two WBOIT MRT targets. That pass owns its own
//! [`BinnedPhaseItem`] - [`TransparentOit3d`] - rather than reusing the opaque
//! [`super::super::raster::Visibility3d`] phase, because the two passes bind
//! different pipelines (integer visibility IDs vs. floating-point WBOIT MRT) and
//! render at different points in the frame.
//!
//! The item mirrors `Visibility3d` exactly: a batch-set key carrying the cached
//! pipeline, draw function and indexed flag, plus a per-mesh bin key. Because
//! WBOIT is order-independent, binning purely by mesh asset (no depth sort) is
//! correct - the whole point of weighted blending is that draw order does not
//! affect the composited result.

use core::ops::Range;

use bevy_material::labels::DrawFunctionId;
use bevy_render::{
    render_phase::{
        BinnedPhaseItem, CachedRenderPipelinePhaseItem, PhaseItem, PhaseItemBatchSetKey,
        PhaseItemExtraIndex,
    },
    render_resource::CachedRenderPipelineId,
    sync_world::MainEntity,
};
use bevy_ecs::prelude::Entity;

/// Batch-set key for a transparent forward draw: the cached pipeline, the draw
/// function and whether the mesh is indexed. Mirrors
/// [`super::super::raster::VisibilityBatchSetKey`].
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct TransparentOitBatchSetKey {
    pub(crate) pipeline: CachedRenderPipelineId,
    pub(crate) draw_function: DrawFunctionId,
    pub(crate) indexed: bool,
}

impl PhaseItemBatchSetKey for TransparentOitBatchSetKey {
    fn indexed(&self) -> bool {
        self.indexed
    }
}

/// Per-mesh bin key. WBOIT is order-independent so binning by mesh asset alone
/// (no depth ordering) yields the correct composited result.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct TransparentOitBinKey(pub(crate) bevy_asset::UntypedAssetId);

/// One transparent forward draw item. Structurally identical to
/// [`super::super::raster::Visibility3d`]; kept separate so the two passes can
/// carry independent pipelines and draw commands.
pub(crate) struct TransparentOit3d {
    batch_set_key: TransparentOitBatchSetKey,
    _bin_key: TransparentOitBinKey,
    representative_entity: (Entity, MainEntity),
    batch_range: Range<u32>,
    extra_index: PhaseItemExtraIndex,
}

impl PhaseItem for TransparentOit3d {
    fn entity(&self) -> Entity {
        self.representative_entity.0
    }
    fn main_entity(&self) -> MainEntity {
        self.representative_entity.1
    }
    fn draw_function(&self) -> DrawFunctionId {
        self.batch_set_key.draw_function
    }
    fn batch_range(&self) -> &Range<u32> {
        &self.batch_range
    }
    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }
    fn extra_index(&self) -> PhaseItemExtraIndex {
        self.extra_index.clone()
    }
    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.batch_range, &mut self.extra_index)
    }
}

impl BinnedPhaseItem for TransparentOit3d {
    type BatchSetKey = TransparentOitBatchSetKey;
    type BinKey = TransparentOitBinKey;
    fn new(
        batch_set_key: Self::BatchSetKey,
        bin_key: Self::BinKey,
        representative_entity: (Entity, MainEntity),
        batch_range: Range<u32>,
        extra_index: PhaseItemExtraIndex,
    ) -> Self {
        Self {
            batch_set_key,
            _bin_key: bin_key,
            representative_entity,
            batch_range,
            extra_index,
        }
    }
}

impl CachedRenderPipelinePhaseItem for TransparentOit3d {
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.batch_set_key.pipeline
    }
}
