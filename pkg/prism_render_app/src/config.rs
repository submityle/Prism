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
    /// How many material slots the ABI buffers are created for.
    ///
    /// The header buffer holds one row per slot and a material handle indexes it,
    /// so this is the scene's material budget; the buffers grow past it if a
    /// scene publishes more.
    pub material_capacity: u32,
    /// How many texture rows the material texture table starts with.
    ///
    /// A material owns a run of rows, one per texture it references, so this is
    /// the scene's total texture-reference budget rather than a per-material
    /// bound; the table grows when a scene needs more.
    pub material_texture_rows: u32,
    /// How many words the material parameter heap buffer starts with.
    ///
    /// One material's packed surface block is 12 to ~36 words depending on how
    /// many lobes it carries, so this is a starting budget rather than a bound:
    /// the heap grows the buffer when a scene needs more.
    pub material_parameter_words: u32,
}

impl RenderAppConfig {
    /// The default 1280×720 frame with the runtime's default pacing.
    pub const DEFAULT: Self = Self {
        extent: Extent3d::new_2d(1280, 720),
        max_frames_in_flight: DEFAULT_FRAMES_IN_FLIGHT,
        material_capacity: 4096,
        material_parameter_words: 64 * 1024,
        material_texture_rows: 16 * 1024,
    };

    /// A configuration for `extent`, keeping the default frames-in-flight bound.
    #[must_use]
    pub const fn new(extent: Extent3d) -> Self {
        Self {
            extent,
            max_frames_in_flight: DEFAULT_FRAMES_IN_FLIGHT,
            material_capacity: Self::DEFAULT.material_capacity,
            material_parameter_words: Self::DEFAULT.material_parameter_words,
            material_texture_rows: Self::DEFAULT.material_texture_rows,
        }
    }

    /// Sets the material slot budget, returning `self` for chaining.
    #[must_use]
    pub const fn with_material_capacity(mut self, material_capacity: u32) -> Self {
        self.material_capacity = material_capacity;
        self
    }

    /// Sets the starting parameter word budget, returning `self` for chaining.
    #[must_use]
    pub const fn with_material_parameter_words(mut self, words: u32) -> Self {
        self.material_parameter_words = words;
        self
    }

    /// Sets the starting texture row budget, returning `self` for chaining.
    #[must_use]
    pub const fn with_material_texture_rows(mut self, rows: u32) -> Self {
        self.material_texture_rows = rows;
        self
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
