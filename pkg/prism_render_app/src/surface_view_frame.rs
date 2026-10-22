//! Camera-owned native queues. This resource does not route, render, or simulate.
//!
//! Queue replaces the complete snapshot every frame, including on failure.
//! Explicit [`RenderQueues`] and window overrides remain separate producers.

use alloc::collections::BTreeMap;
use prism_ecs::prelude::{Entity, Resource};

use crate::queue::RenderQueues;
use crate::surface_views::SurfaceView;
use crate::visibility_bridge::ExtractedVisibilityView;

/// One camera's sorted draws and immutable surface snapshot.
#[derive(Clone)]
pub struct SurfaceViewQueue {
    /// Camera matrices/history and output-space viewport, not a local raster rect.
    pub surface: SurfaceView,
    /// Dense index into this frame's canonical frame-uniform arena.
    /// Not a persistent camera ID; removal can renumber the next frame's slots.
    pub frame_slot: u32,
    /// Only this camera's work, including its aligned dynamic-binding sidecars.
    pub queues: RenderQueues,
}

impl SurfaceViewQueue {
    /// Full main-world identity, including generation.
    #[must_use]
    pub fn camera(&self) -> Entity {
        self.surface.camera
    }
}

/// A camera-local admission failure. Other cameras remain eligible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceViewQueueRejectionReason {
    /// No validated surface with the same full camera identity exists.
    MissingSurfaceView,
    /// No validated culling result exists; missing visibility is not empty work.
    MissingVisibilityView,
    /// The culling result retained a different surface snapshot/identity.
    SurfaceSnapshotMismatch,
}

/// A rejected camera, ordered by `(index, generation)` in the frame snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceViewQueueRejection {
    /// Complete camera identity; an index-only match is never accepted.
    pub camera: Entity,
    /// Local admission failure.
    pub reason: SurfaceViewQueueRejectionReason,
}

/// A batch-wide failure. No camera queues are published for this batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceViewQueueError {
    /// Too many admitted cameras for dense u32 frame slots.
    FrameSlotOverflow,
    /// Old device work could not be drained before replacing arena resources.
    CompletionUnconfirmed,
    /// The active provider/device layout could not be prepared.
    Layout,
    /// Frame or material arena capacity exceeds the device's addressable limits.
    Capacity,
    /// The canonical material buffers could not be bound.
    MaterialBuffers,
    /// A reserved frame slot failed upload, before any draw binding was emitted.
    FrameUpload {
        /// Dense slot that failed preparation.
        slot: u32,
    },
}

/// Per-frame native surface queues, distinct from explicit global/window queues.
///
/// Published after Queue's deferred commands, before Phase. Consumers must
/// check [`Self::is_ready`] and match `frame_index`/`device_generation` against
/// their execution context. Pending/failed snapshots contain no drawable queues.
/// No targets, GPU ownership, output routing, or simulation state live here.
#[derive(Clone, Default, Resource)]
pub struct SurfaceViewQueues {
    /// Extraction that produced this batch (zero for the default empty value).
    pub frame_index: u64,
    /// Device generation for all bindings in this batch.
    pub device_generation: u64,
    views: Vec<SurfaceViewQueue>,
    rejections: Vec<SurfaceViewQueueRejection>,
    ready: bool,
    error: Option<SurfaceViewQueueError>,
}

impl SurfaceViewQueues {
    /// Accepted cameras in explicit ascending `(index, generation)` order.
    /// A legal camera with no visible work still has an entry.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &SurfaceViewQueue> + DoubleEndedIterator {
        self.views.iter()
    }

    /// Find a camera using its full generational identity.
    #[must_use]
    pub fn get(&self, camera: Entity) -> Option<&SurfaceViewQueue> {
        self.views
            .binary_search_by_key(&(camera.index(), camera.generation()), |view| {
                (view.camera().index(), view.camera().generation())
            })
            .ok()
            .map(|index| &self.views[index])
    }

    /// Number of accepted cameras, including empty-work cameras.
    #[must_use]
    pub fn len(&self) -> usize {
        self.views.len()
    }

    /// Whether no camera queues are published.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }

    /// True only after the complete arena preflight and queue build succeeded.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.ready
    }

    /// Camera-local failures, independent of batch-wide GPU preparation errors.
    #[must_use]
    pub fn rejections(&self) -> &[SurfaceViewQueueRejection] {
        &self.rejections
    }

    /// Batch-wide failure, if any. Never consume bindings from a failed batch.
    #[must_use]
    pub fn error(&self) -> Option<SurfaceViewQueueError> {
        self.error
    }

    /// Discard frame-local queues before recovery or after recording.
    pub fn clear(&mut self) {
        self.views.clear();
        self.rejections.clear();
        self.ready = false;
        self.error = None;
    }

    pub(crate) fn pending(frame_index: u64, device_generation: u64) -> Self {
        Self {
            frame_index,
            device_generation,
            ..Self::default()
        }
    }

    pub(crate) fn fail(&mut self, error: SurfaceViewQueueError) {
        self.views.clear();
        self.ready = false;
        self.error = Some(error);
    }

    pub(crate) fn publish(&mut self, views: Vec<SurfaceViewQueue>) {
        self.views = views;
        self.ready = true;
        self.error = None;
    }

    /// Pair already validated extraction snapshots, retaining camera-local
    /// errors. Returning references avoids copying each camera's work list.
    pub(crate) fn admit<'a>(
        &mut self,
        visibility: impl IntoIterator<Item = &'a ExtractedVisibilityView>,
        surfaces: impl IntoIterator<Item = &'a SurfaceView>,
    ) -> Vec<&'a ExtractedVisibilityView> {
        self.clear();
        let mut pairs = BTreeMap::new();
        for surface in surfaces {
            pairs
                .entry((surface.camera.index(), surface.camera.generation()))
                .or_insert((None, None))
                .1 = Some(surface);
        }
        for view in visibility {
            pairs
                .entry((view.camera.index(), view.camera.generation()))
                .or_insert((None, None))
                .0 = Some(view);
        }
        let mut admitted = Vec::new();
        self.rejections.clear();
        for (_, (visibility, surface)) in pairs {
            let (camera, reason) = match (visibility, surface) {
                (Some(view), Some(surface)) if &view.surface == surface => {
                    admitted.push(view);
                    continue;
                }
                (Some(view), Some(_)) => (
                    view.camera,
                    SurfaceViewQueueRejectionReason::SurfaceSnapshotMismatch,
                ),
                (Some(view), None) => (
                    view.camera,
                    SurfaceViewQueueRejectionReason::MissingSurfaceView,
                ),
                (None, Some(surface)) => (
                    surface.camera,
                    SurfaceViewQueueRejectionReason::MissingVisibilityView,
                ),
                (None, None) => unreachable!("only extraction snapshots create pair entries"),
            };
            self.rejections
                .push(SurfaceViewQueueRejection { camera, reason });
        }
        admitted
    }
}

#[cfg(test)]
#[path = "surface_view_frame/tests.rs"]
mod tests;
