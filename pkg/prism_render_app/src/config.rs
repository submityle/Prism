//! Configuration for the render sub-app.

use prism_render_driver::Extent3d;
use prism_render_runtime::DEFAULT_FRAMES_IN_FLIGHT;

/// The render sub-app's startup configuration: the initial frame size and the
/// frames-in-flight bound.
///
/// The default is a 1280×720 frame with the runtime's
/// [`DEFAULT_FRAMES_IN_FLIGHT`] bound; a real app overrides the extent to match
/// its window (or backbuffer) size.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RenderAppConfig {
    /// The initial size of the offscreen frame target.
    pub extent: Extent3d,
    /// The maximum number of frames the runtime may keep outstanding at once.
    pub max_frames_in_flight: usize,
}

impl RenderAppConfig {
    /// The default 1280×720 frame with the runtime's default pacing.
    pub const DEFAULT: Self = Self {
        extent: Extent3d::new_2d(1280, 720),
        max_frames_in_flight: DEFAULT_FRAMES_IN_FLIGHT,
    };

    /// A configuration for `extent`, keeping the default frames-in-flight bound.
    #[must_use]
    pub const fn new(extent: Extent3d) -> Self {
        Self {
            extent,
            max_frames_in_flight: DEFAULT_FRAMES_IN_FLIGHT,
        }
    }

    /// Sets the frames-in-flight bound, returning `self` for chaining.
    ///
    /// The runtime clamps the value to at least one.
    #[must_use]
    pub const fn with_max_frames_in_flight(mut self, max_frames_in_flight: usize) -> Self {
        self.max_frames_in_flight = max_frames_in_flight;
        self
    }
}

impl Default for RenderAppConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}
