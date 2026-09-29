//! Broad-phase particle input.
//!
//! A [`Particle`] is a bounding sphere: a world-space centre and a radius. The
//! broad phase only ever needs this sphere proxy, so both the `CPU` and `GPU`
//! paths consume a flat slice of them. The `GPU` upload representation packs
//! each particle into a single `vec4<f32>` (`xyz` centre, `w` radius) for
//! natural `std430` alignment.
//!
//! Provenance: standard bounding-sphere proxy; no Unreal Engine source or
//! derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// A bounding-sphere collision proxy fed to the broad phase.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Particle {
    /// World-space centre of the sphere.
    pub position: Vec3,
    /// Sphere radius; must be non-negative.
    pub radius: f32,
}

impl Particle {
    /// Creates a particle from a centre and radius.
    #[must_use]
    pub fn new(position: Vec3, radius: f32) -> Particle {
        Particle { position, radius }
    }

    /// Packs the particle into its `GPU` `vec4<f32>` upload form.
    #[must_use]
    pub(crate) fn to_gpu(self) -> GpuParticle {
        GpuParticle {
            data: [
                self.position.x,
                self.position.y,
                self.position.z,
                self.radius,
            ],
        }
    }
}

/// `std430`-compatible upload form of [`Particle`]: `xyz` centre, `w` radius.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuParticle {
    /// Packed `[x, y, z, radius]`.
    pub data: [f32; 4],
}
