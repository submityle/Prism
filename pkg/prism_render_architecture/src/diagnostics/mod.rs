//! Feature health, breadcrumbs, and performance counters.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeatureHealth {
    Unavailable,
    Initializing,
    Warming,
    Active,
    Degraded,
    Quarantined,
}

#[derive(Clone, Debug)]
pub struct FeatureStatus {
    pub name: &'static str,
    pub health: FeatureHealth,
    pub reason: Option<String>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameCounters {
    pub frame_index: u64,
    pub cpu_frame_ms: f32,
    pub gpu_frame_ms: f32,
    pub gpu_memory_bytes: u64,
}
