//! Frame-time and memory-budget quality controller.

#[derive(Clone, Copy, Debug)]
pub struct QualityTarget {
    pub frame_time_ms: f32,
    pub memory_bytes: u64,
    pub adaptive: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct QualityDecision {
    pub render_scale: f32,
    pub geometry_error_pixels: f32,
    pub shadow_page_budget: u32,
    pub gi_ray_scale: f32,
}

pub trait QualityController {
    fn update(&mut self, target: QualityTarget, measured_frame_ms: f32) -> QualityDecision;
}
