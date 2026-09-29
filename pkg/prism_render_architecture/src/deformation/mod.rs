//! Shared skinning and deformation cache.

pub mod schedule;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DeformationHandle(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeformationKind {
    Skinning,
    Morph,
    Cloth,
    VertexAnimation,
    Hair,
    Particle,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DeformationBudget {
    pub vertices_per_frame: u32,
    pub blas_refits_per_frame: u32,
}
