//! Device-resident packing of an [`AcousticScene`] for the acoustics kernels.
//!
//! [`GpuScene`] flattens the triangle mesh and its per-triangle acoustics into
//! one storage buffer of [`GpuTriangle`] records, uploaded once and bound by
//! every kernel dispatch. Each record carries the three world-space vertices,
//! the geometric face normal (the same `(b - a) x (c - a)` unit normal the
//! `CPU` [`AcousticScene::triangle_normal`] computes, or zero for a degenerate
//! face), and the per-band acoustics the kernels need: a three-band
//! transmission spectrum, a three-band reflection spectrum, and the scalar
//! scattering coefficient.
//!
//! # Why the spectra are pre-computed on the host
//!
//! The `CPU` backend stores each surface as a
//! [`BandedAcousticMaterial`](prism_audio_spatial::material_spectrum::BandedAcousticMaterial):
//! a low/mid/high [`BandGains`](prism_audio_spatial::BandGains) transmission
//! spectrum, the matching reflection spectrum, and a scattering coefficient. All
//! three are already clamped to `[0, 1]` on construction, so the host simply
//! copies the three band gains of each spectrum (plus the scalar scattering)
//! into the device record. The kernels then need only multiplies, dot products,
//! and a square root -- the same per-band `combine`/`scaled`/`split_peak`
//! arithmetic the `CPU` path uses -- keeping every transcendental out of the
//! shader so the `GPU` result tracks the `CPU` golden twin without a
//! platform-dependent `exp`/`log` approximation drifting the two apart.
//!
//! # Provenance
//!
//! Original work; standard mesh flattening for a compute buffer; no Unreal
//! Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Dolby, or Google Resonance
//! Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Built from [`prism_audio_geometry::AcousticScene`]; its buffer is bound by
//! [`crate::direct`] and [`crate::reflection`], and its host-side triangle copy
//! drives the `CPU` twins in [`crate::backend`].

use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_audio_geometry::AcousticScene;
use wgpu::Buffer;

use crate::buffer;
use crate::context::GpuContext;

/// One triangle as the kernels see it: geometry plus pre-computed acoustics.
///
/// `vec3` members are padded to 16 bytes so the `std430` storage layout the
/// shader declares matches this record field-for-field.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct GpuTriangle {
    /// First vertex `a` (xyz; w padding).
    pub(crate) a: [f32; 4],
    /// Second vertex `b` (xyz; w padding).
    pub(crate) b: [f32; 4],
    /// Third vertex `c` (xyz; w padding).
    pub(crate) c: [f32; 4],
    /// Unit face normal `(b - a) x (c - a)` normalised, or zero when the face
    /// is degenerate (xyz; w padding).
    pub(crate) normal: [f32; 4],
    /// Per-band linear transmission gains through this surface, each in
    /// `[0, 1]` (xyz = low/mid/high; w padding).
    pub(crate) transmission: [f32; 4],
    /// Per-band linear reflection coefficients off this surface, each in
    /// `[0, 1]` (xyz = low/mid/high; w padding).
    pub(crate) reflection: [f32; 4],
    /// Scattering coefficient in `[0, 1]` diverting energy out of the specular
    /// lobe (the specular share is weighted by `sqrt(1 - scattering)`).
    pub(crate) scattering: f32,
    /// Padding to a 16-byte stride.
    pub(crate) _pad: [f32; 3],
}

/// A triangle-mesh acoustic scene uploaded to the device.
///
/// Holds the storage buffer the kernels bind plus the host-side triangle copy
/// the `CPU` twins trace, so a single build serves both the device dispatch and
/// its golden comparison.
pub struct GpuScene {
    buffer: Buffer,
    triangles: Vec<GpuTriangle>,
}

impl GpuScene {
    /// Packs `scene` into a device storage buffer.
    ///
    /// Every triangle is flattened in index order; a degenerate triangle keeps
    /// a zero normal so the reflection kernel skips it exactly as the `CPU`
    /// backend does.
    #[must_use]
    pub fn upload(ctx: &GpuContext, scene: &AcousticScene) -> GpuScene {
        let count = scene.triangle_count();
        let mut triangles = Vec::with_capacity(count);
        for index in 0..count {
            let [a, b, c] = scene.triangle(index).unwrap_or([
                bevy_math::Vec3::ZERO,
                bevy_math::Vec3::ZERO,
                bevy_math::Vec3::ZERO,
            ]);
            let normal = scene
                .triangle_normal(index)
                .unwrap_or(bevy_math::Vec3::ZERO);
            let material = scene.material(index);
            let transmission = material.transmission().bands();
            let reflection = material.reflection().bands();
            triangles.push(GpuTriangle {
                a: [a.x, a.y, a.z, 0.0],
                b: [b.x, b.y, b.z, 0.0],
                c: [c.x, c.y, c.z, 0.0],
                normal: [normal.x, normal.y, normal.z, 0.0],
                transmission: [transmission[0], transmission[1], transmission[2], 0.0],
                reflection: [reflection[0], reflection[1], reflection[2], 0.0],
                scattering: material.scattering(),
                _pad: [0.0, 0.0, 0.0],
            });
        }
        // A storage buffer with a non-zero length is required for binding even
        // when the scene is empty; a single zeroed sentinel triangle keeps the
        // buffer bindable and is reported by neither kernel because the kernels
        // dispatch over `triangle_count`, not over the padded buffer length.
        let upload = if triangles.is_empty() {
            Vec::from([GpuTriangle::zeroed()])
        } else {
            triangles.clone()
        };
        let buffer = buffer::storage_read(ctx.device(), "prism_acoustics_scene", &upload);
        GpuScene { buffer, triangles }
    }

    /// The storage buffer bound by the kernels.
    #[must_use]
    pub(crate) fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// The host-side packed triangles the `CPU` twins trace.
    #[must_use]
    pub(crate) fn triangles(&self) -> &[GpuTriangle] {
        &self.triangles
    }

    /// Number of real triangles in the scene (excludes the empty-scene
    /// sentinel).
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// Whether the scene has no triangles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }
}
