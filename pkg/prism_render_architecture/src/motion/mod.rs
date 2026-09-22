//! Motion, reprojection, and disocclusion contracts.

#[derive(Clone, Copy, Debug, Default)]
pub struct MotionSample {
    pub velocity_pixels: [f32; 2],
    pub reprojection_confidence: f32,
    pub reactive: f32,
    pub transparency: f32,
    pub surface_id: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MotionSource {
    Camera,
    Rigid,
    Skinned,
    Morph,
    VertexAnimation,
    Particle,
    StreamingReveal,
}
