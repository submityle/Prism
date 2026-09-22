//! Temporal reconstruction feature boundary.

use crate::history::InvalidationMask;

#[derive(Clone, Copy, Debug)]
pub struct TemporalUpscaleSettings {
    pub render_scale: f32,
    pub sharpness: f32,
    pub invalidation_dependencies: InvalidationMask,
}

impl Default for TemporalUpscaleSettings {
    fn default() -> Self {
        Self {
            render_scale: 1.0,
            sharpness: 0.0,
            invalidation_dependencies: InvalidationMask(u32::MAX),
        }
    }
}
