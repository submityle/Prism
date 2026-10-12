//! The extracted per-frame render record: what the render world learns about
//! the main world each frame.

use prism_ecs::prelude::Resource;

use crate::camera::RenderViewport;

/// A snapshot of the render-relevant state pulled from the main world.
///
/// The extract step overwrites this every frame (one-way, main → render), so it
/// is the render world's view of the current frame rather than a growing log.
/// `frame_index` makes the extract observable: it advances once per extracted
/// frame, which is how a test (or a diagnostic) tells that the render world is
/// being fed.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, Resource)]
pub struct RenderRecord {
    /// The frame size requested by the main world's camera this frame.
    pub viewport: RenderViewport,
    /// The number of frames extracted so far, counting this one.
    pub frame_index: u64,
}

impl RenderRecord {
    /// A record for `viewport` at extracted-frame `frame_index`.
    #[must_use]
    pub const fn new(viewport: RenderViewport, frame_index: u64) -> Self {
        Self {
            viewport,
            frame_index,
        }
    }
}
