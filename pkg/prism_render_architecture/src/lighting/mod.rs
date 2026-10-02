//! Unified lighting, GI, reflection, and stochastic-light contracts.

pub mod culling;
pub mod stochastic;
pub mod restir_di;
pub mod restir_temporal;
pub mod restir_gi;
pub mod restir_gi_resolve;
pub mod regir;

use crate::ray_scene::TraceBackend;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LightHandle(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct GiSettings {
    pub trace_backend: TraceBackend,
    pub diffuse_rays_per_pixel: f32,
    pub reflection_roughness_cutoff: f32,
    pub cache_updates_per_frame: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ReservoirBudget {
    pub initial_candidates: u16,
    pub spatial_neighbors: u16,
    pub temporal_reuse: bool,
}
