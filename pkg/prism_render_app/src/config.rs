//! Configuration for the render sub-app.

use prism_render_architecture::lighting::contract::{
    FrameResourceBudget, PlanCaps, SceneStats, Tier,
};
use prism_render_driver::Extent3d;
use prism_render_runtime::DEFAULT_FRAMES_IN_FLIGHT;

/// The render sub-app's startup configuration: the initial frame size and the
/// frames-in-flight bound.
///
/// The default is a 1280×720 frame with the runtime's
/// [`DEFAULT_FRAMES_IN_FLIGHT`] bound; a real app overrides the extent to match
/// its window (or backbuffer) size.
#[derive(Clone, Copy, PartialEq, Debug)]
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
    /// Scene statistics consumed by the lighting/GI budget compiler.
    pub lighting_scene_stats: SceneStats,
    /// Per-frame lighting/GI resource budget.
    pub lighting_budget: FrameResourceBudget,
    /// Hardware and content capability policy for recipe admission.
    pub lighting_caps: PlanCaps,
    /// Initial lighting/GI quality tier.
    pub lighting_tier: Tier,
}

impl RenderAppConfig {
    /// The default 1280×720 frame with the runtime's default pacing.
    pub const DEFAULT: Self = Self {
        extent: Extent3d::new_2d(1280, 720),
        max_frames_in_flight: DEFAULT_FRAMES_IN_FLIGHT,
        material_capacity: 4096,
        material_parameter_words: 64 * 1024,
        material_texture_rows: 16 * 1024,
        lighting_scene_stats: SceneStats {
            lights: 0,
            punctual_max: 64,
            npr_materials: 0,
            proxy_page_ratio: 0.0,
            triangle_density: 0.0,
        },
        lighting_budget: FrameResourceBudget {
            gpu_ms: 16.666,
            bandwidth: 512 * 1024 * 1024,
            resident_bytes: 512 * 1024 * 1024,
            io_ops: 4096,
            shading_rate_budget: 1.0,
        },
        lighting_caps: PlanCaps {
            ray_tracing: false,
            virtual_shadow: false,
            offline: false,
            deterministic_npr: false,
        },
        lighting_tier: Tier::Balanced,
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
            lighting_scene_stats: Self::DEFAULT.lighting_scene_stats,
            lighting_budget: Self::DEFAULT.lighting_budget,
            lighting_caps: Self::DEFAULT.lighting_caps,
            lighting_tier: Self::DEFAULT.lighting_tier,
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

    /// Sets the scene statistics consumed by the lighting planner.
    #[must_use]
    pub const fn with_lighting_scene_stats(mut self, stats: SceneStats) -> Self {
        self.lighting_scene_stats = stats;
        self
    }

    /// Sets the lighting/GI frame budget.
    #[must_use]
    pub const fn with_lighting_budget(mut self, budget: FrameResourceBudget) -> Self {
        self.lighting_budget = budget;
        self
    }

    /// Sets the lighting recipe capability policy.
    #[must_use]
    pub const fn with_lighting_caps(mut self, caps: PlanCaps) -> Self {
        self.lighting_caps = caps;
        self
    }

    /// Sets the initial lighting/GI quality tier.
    #[must_use]
    pub const fn with_lighting_tier(mut self, tier: Tier) -> Self {
        self.lighting_tier = tier;
        self
    }
}

impl Default for RenderAppConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}
