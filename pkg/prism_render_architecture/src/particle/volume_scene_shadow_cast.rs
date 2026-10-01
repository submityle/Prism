//! Particle-volume → scene **soft shadow casting**: the cast-side dual of the
//! deep-opacity / `froxel` receivers, projecting a participating-media volume's
//! light-ray transmittance onto arbitrary scene receivers (design §20,
//! "接收/投射场景阴影：粒子接收方向光 shadow map / `VSM`,
//! 也可向场景投体积软阴影（高档）", see
//! `docs/prism_particle_engine_design_zh.md` §20).
//!
//! Every sibling here is a *receiver*: [`super::distance_field_shadow`],
//! [`super::contact_shadow`], and [`super::variance_shadow`] all read an
//! occlusion signal, and [`super::shading`] notes a non-volumetric renderer
//! "cast no self-shadow". The *emitter* half — a smoke/fire volume acting as an
//! occluder that darkens the lit scene behind it — had no reference. This module
//! supplies it.
//!
//! ## Algorithm and source
//!
//! Along the "scene receiver → light" direction the volume's extinction is
//! integrated front-to-back as an opacity product. Each march step reads the
//! particle density out of the shared [`super::volumetrics::FroxelDensityField`]
//! and folds a per-step survival factor `1 − clamp(density · sigma · step, 0, 1)`
//! into a running transmittance product. The product of per-step survivals is
//! the fraction of light that still reaches the receiver — the soft-shadow
//! attenuation in `0..=1`. This is the classic `1 − alpha` front-to-back volume
//! compositing (Nelson Max, "Optical Models for Direct Volume Rendering", 1995)
//! used as a first-order, transcendental-free stand-in for the `Beer-Lambert`
//! integral `exp(−τ)`; it is the same light-ray transmittance that `UE`'s
//! Volumetric Fog, `Frostbite`'s volumetric lighting, and Guerrilla's `Decima`
//! volumetric self-shadow all accumulate (design §20 names Guerrilla and
//! `EmberGen` as the baseline). It is reproduced from the public algorithm, not
//! ported from any engine's code.
//!
//! The same march, recorded at increasing depth, bakes a
//! [`super::volumetrics::DeepOpacityLayer`] curve — the "from the light's
//! viewpoint record the transmittance function along the ray" deep opacity map
//! of design §20 — so a receiver can be shadowed by one cheap
//! [`super::volumetrics::sample_deep_transmittance`] lookup at its projected
//! depth instead of re-marching. Those public types are *consumed*, never
//! redefined.
//!
//! ## Determinism
//!
//! Only `f32::sqrt` (through [`super::Vec3::normalize_or_zero`]) is used; there
//! is no `sin` / `cos` / `exp` / `ln` / `powf`, every power is an explicit
//! integer multiply, and float guards compare a magnitude against [`CMP_EPS`]
//! rather than using `==` / `!=`. A future `GPU` kernel that marches the same
//! field in the same order reproduces this `CPU` reference bit for bit.

use alloc::vec::Vec;

use super::volumetrics::{sample_deep_transmittance, DeepOpacityLayer, FroxelDensityField};
use super::{ParticleSystemHandle, Vec3};

/// Absolute tolerance for the `f32` comparison guards in this module.
///
/// The crate bans `==` / `!=` on floating point, so the degenerate step-length
/// and direction guards compare a magnitude against this epsilon instead.
pub const CMP_EPS: f32 = 1e-6;

/// Fraction of a march step at which the volume is sampled: the step midpoint.
///
/// Sampling the center rather than an endpoint keeps the quadrature symmetric,
/// so a volume straddling a cell boundary is weighted the same whether the ray
/// is traced receiver-to-light or light-to-receiver.
const STEP_MIDPOINT: f32 = 0.5;

/// Clamps a scalar into `0..=1` without branching on equality.
#[must_use]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// A scene light whose visibility the particle volume occludes (design §20).
///
/// The variants carry the geometry needed to point a shadow ray from a receiver
/// back toward the light; the "toward light" unit direction is derived by
/// [`SceneLight::toward_light`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SceneLight {
    /// A directional (infinitely distant) light. `travel` is the direction the
    /// light *propagates* (from the light into the scene); the shadow ray walks
    /// the opposite way, toward the light.
    Directional {
        /// Direction the light travels, scene-ward (need not be normalized).
        travel: Vec3,
    },
    /// A positional (point / spot) light at a world location.
    Point {
        /// World-space position of the light.
        position: Vec3,
    },
}

impl SceneLight {
    /// Unit direction from `receiver` toward the light, or [`Vec3::ZERO`] when
    /// the light is degenerate (zero travel direction, or coincident with the
    /// receiver). Normalization uses `sqrt` through
    /// [`Vec3::normalize_or_zero`].
    #[must_use]
    pub fn toward_light(self, receiver: Vec3) -> Vec3 {
        match self {
            // The shadow ray walks against the light's travel direction.
            Self::Directional { travel } => travel.scale(-1.0).normalize_or_zero(),
            Self::Point { position } => position.sub(receiver).normalize_or_zero(),
        }
    }
}

/// A particle volume that casts soft shadows onto the scene (design §20).
///
/// Bundles the volume *handle* (which particle system this occluder belongs to),
/// a borrow of the shared [`FroxelDensityField`] the volume injected its density
/// into, and the march parameters. The lifetime ties the caster to the density
/// field it reads; it owns no density of its own.
#[derive(Clone, Copy, Debug)]
pub struct VolumeShadowCaster<'field> {
    system: ParticleSystemHandle,
    field: &'field FroxelDensityField,
    sigma: f32,
    step_length: f32,
    max_steps: u32,
}

impl<'field> VolumeShadowCaster<'field> {
    /// Builds a caster over a density field.
    ///
    /// `extinction` is the density-to-opacity scale `sigma_t`; a negative value
    /// is clamped to zero so the volume never *adds* light. `step_length` is the
    /// world-space march increment and `max_steps` caps the ray length at
    /// `step_length · max_steps`.
    #[must_use]
    pub fn new(
        system: ParticleSystemHandle,
        field: &'field FroxelDensityField,
        extinction: f32,
        step_length: f32,
        max_steps: u32,
    ) -> Self {
        let sigma = if extinction > 0.0 { extinction } else { 0.0 };
        Self {
            system,
            field,
            sigma,
            step_length,
            max_steps,
        }
    }

    /// The particle system this occluding volume belongs to.
    #[must_use]
    pub const fn handle(&self) -> ParticleSystemHandle {
        self.system
    }

    /// The density field this caster reads.
    #[must_use]
    pub const fn field(&self) -> &FroxelDensityField {
        self.field
    }

    /// Samples the volume density at a world point (`0.0` outside the grid).
    #[must_use]
    fn density_at(&self, point: Vec3) -> f32 {
        match self.field.grid().cell_index(point) {
            Some(cell) => self.field.density_at(cell),
            None => 0.0,
        }
    }

    /// Per-step survival factor `1 − clamp(density · sigma · step, 0, 1)`.
    ///
    /// This is the first-order `1 − alpha` opacity of one march segment; the
    /// product of these over a ray is its transmittance.
    #[must_use]
    fn step_survival(&self, density: f32) -> f32 {
        let opacity = clamp01(density * self.sigma * self.step_length);
        1.0 - opacity
    }

    /// Transmittance along a ray from `start` in a (to-be-normalized)
    /// `direction`, as the front-to-back product of per-step survivals.
    ///
    /// Returns `1.0` (fully lit) when the direction is degenerate or the step
    /// length collapses, so a malformed query never fabricates occlusion. The
    /// sample points are `start + direction · step_length · (i + 0.5)` for
    /// `i` in `0..max_steps`, matching [`VolumeShadowCaster::bake_light_ray`]
    /// step for step.
    #[must_use]
    pub fn transmittance_along(&self, start: Vec3, direction: Vec3) -> f32 {
        let dir = direction.normalize_or_zero();
        if dir.length_squared() <= CMP_EPS || self.step_length <= CMP_EPS {
            return 1.0;
        }
        let mut transmittance = 1.0;
        let mut step = 0u32;
        while step < self.max_steps {
            let offset = self.step_length * (step as f32 + STEP_MIDPOINT);
            let point = start.add(dir.scale(offset));
            let density = self.density_at(point);
            transmittance *= self.step_survival(density);
            step += 1;
        }
        transmittance
    }

    /// Soft-shadow attenuation reaching `receiver` from `light`, in `0..=1`.
    ///
    /// The receiver-facing API: `1.0` is fully lit, `0.0` fully shadowed by the
    /// volume. It points a shadow ray from the receiver toward the light and
    /// returns the volume's transmittance along it.
    #[must_use]
    pub fn cast(&self, receiver: Vec3, light: SceneLight) -> f32 {
        let dir = light.toward_light(receiver);
        self.transmittance_along(receiver, dir)
    }

    /// Bakes a deep opacity map along the light ray from `entry` in the light's
    /// `travel` direction (design §20).
    ///
    /// Records a [`DeepOpacityLayer`] at each step: the first at depth `0.0`
    /// transmittance `1.0`, then one per step at depth `step_length · (i + 1)`
    /// carrying the running transmittance *after* that step. The resulting
    /// curve is ascending in depth and non-increasing in transmittance, exactly
    /// the input [`sample_deep_transmittance`] expects, so a receiver at a known
    /// projected depth is shadowed by one cheap lookup instead of a re-march.
    /// A degenerate direction or step length yields a single fully-lit layer.
    #[must_use]
    pub fn bake_light_ray(&self, entry: Vec3, travel: Vec3) -> Vec<DeepOpacityLayer> {
        let mut layers = Vec::new();
        layers.push(DeepOpacityLayer::new(0.0, 1.0));
        let dir = travel.normalize_or_zero();
        if dir.length_squared() <= CMP_EPS || self.step_length <= CMP_EPS {
            return layers;
        }
        let mut transmittance = 1.0;
        let mut step = 0u32;
        while step < self.max_steps {
            let offset = self.step_length * (step as f32 + STEP_MIDPOINT);
            let point = entry.add(dir.scale(offset));
            let density = self.density_at(point);
            transmittance *= self.step_survival(density);
            let depth = self.step_length * (step as f32 + 1.0);
            layers.push(DeepOpacityLayer::new(depth, transmittance));
            step += 1;
        }
        layers
    }

    /// Projects `receiver` onto the baked light ray and samples the deep opacity
    /// map at that depth, returning the soft-shadow attenuation in `0..=1`.
    ///
    /// `entry` and `travel` are the baking ray's origin and (unit-or-not)
    /// propagation direction; the receiver's depth is its signed distance along
    /// that direction, clamped to `0.0` so a receiver in front of the volume is
    /// fully lit. This is the cheap deep-opacity-map path: it reuses
    /// [`sample_deep_transmittance`] rather than re-marching the field.
    #[must_use]
    pub fn attenuation_from_map(
        layers: &[DeepOpacityLayer],
        entry: Vec3,
        travel: Vec3,
        receiver: Vec3,
    ) -> f32 {
        let dir = travel.normalize_or_zero();
        if dir.length_squared() <= CMP_EPS {
            return 1.0;
        }
        let along = receiver.sub(entry).dot(dir);
        let depth = if along > 0.0 { along } else { 0.0 };
        sample_deep_transmittance(layers, depth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::volumetrics::FroxelGrid;

    const TOL: f32 = 1e-4;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= TOL
    }

    /// A `4 × 1 × 1` grid with a single occluding cell at `x ∈ [1, 2)`.
    ///
    /// The light travels `+X` (its source sits on the `−X` side), so a receiver
    /// at large `+X` is on the volume's backlit side and a receiver at small `x`
    /// (toward the light) marches away from the occluder.
    fn single_slab_field(density: f32) -> FroxelDensityField {
        let grid = FroxelGrid::new([4, 1, 1], Vec3::ZERO, Vec3::splat(1.0));
        let mut field = FroxelDensityField::new(grid);
        // Center of cell [1, 0, 0].
        let _ = field.inject(Vec3::new(1.5, 0.5, 0.5), density);
        field
    }

    fn backlit_receiver() -> Vec3 {
        Vec3::new(3.5, 0.5, 0.5)
    }

    fn lit_receiver() -> Vec3 {
        Vec3::new(0.5, 0.5, 0.5)
    }

    fn travel_plus_x() -> SceneLight {
        SceneLight::Directional {
            travel: Vec3::new(1.0, 0.0, 0.0),
        }
    }

    #[test]
    fn empty_volume_is_fully_lit() {
        let field = single_slab_field(0.0);
        let caster = VolumeShadowCaster::new(ParticleSystemHandle(0), &field, 1.0, 0.25, 16);
        assert!(approx(caster.cast(backlit_receiver(), travel_plus_x()), 1.0));
    }

    #[test]
    fn denser_volume_attenuates_more_strongly() {
        let light = travel_plus_x();
        let receiver = backlit_receiver();
        let sparse = single_slab_field(0.2);
        let dense = single_slab_field(0.8);
        let caster_sparse =
            VolumeShadowCaster::new(ParticleSystemHandle(1), &sparse, 1.0, 0.25, 16);
        let caster_dense = VolumeShadowCaster::new(ParticleSystemHandle(1), &dense, 1.0, 0.25, 16);
        let att_sparse = caster_sparse.cast(receiver, light);
        let att_dense = caster_dense.cast(receiver, light);
        // Monotonic: more density -> less light survives.
        assert!(att_dense < att_sparse);
        assert!(att_sparse < 1.0);
    }

    #[test]
    fn higher_extinction_attenuates_more_strongly() {
        let field = single_slab_field(0.5);
        let light = travel_plus_x();
        let receiver = backlit_receiver();
        let weak = VolumeShadowCaster::new(ParticleSystemHandle(2), &field, 0.5, 0.25, 16);
        let strong = VolumeShadowCaster::new(ParticleSystemHandle(2), &field, 1.5, 0.25, 16);
        assert!(strong.cast(receiver, light) < weak.cast(receiver, light));
    }

    #[test]
    fn backlit_receiver_is_darker_than_lit_receiver() {
        let field = single_slab_field(0.6);
        let light = travel_plus_x();
        let caster = VolumeShadowCaster::new(ParticleSystemHandle(3), &field, 1.0, 0.25, 16);
        let backlit = caster.cast(backlit_receiver(), light);
        let lit = caster.cast(lit_receiver(), light);
        assert!(backlit < lit);
        // The lit side marches away from the slab and stays fully lit.
        assert!(approx(lit, 1.0));
    }

    #[test]
    fn point_light_behind_volume_shadows_receiver() {
        let field = single_slab_field(0.7);
        let caster = VolumeShadowCaster::new(ParticleSystemHandle(4), &field, 1.0, 0.25, 16);
        // Light on the -X side; receiver at +X marches back through the slab.
        let light = SceneLight::Point {
            position: Vec3::new(-2.0, 0.5, 0.5),
        };
        assert!(caster.cast(backlit_receiver(), light) < 1.0);
    }

    #[test]
    fn march_matches_hand_computed_product() {
        // density 0.5, sigma 1.0, step 0.25 -> opacity 0.125, survival 0.875.
        // The slab is sampled four times, so transmittance = 0.875^4.
        let field = single_slab_field(0.5);
        let caster = VolumeShadowCaster::new(ParticleSystemHandle(5), &field, 1.0, 0.25, 16);
        let survival = 0.875;
        let expected = survival * survival * survival * survival;
        assert!(approx(caster.cast(backlit_receiver(), travel_plus_x()), expected));
    }

    #[test]
    fn baked_map_matches_direct_march() {
        let field = single_slab_field(0.5);
        let caster = VolumeShadowCaster::new(ParticleSystemHandle(6), &field, 1.0, 0.25, 16);
        let entry = Vec3::new(0.0, 0.5, 0.5);
        let travel = Vec3::new(1.0, 0.0, 0.0);
        let layers = caster.bake_light_ray(entry, travel);
        let receiver = backlit_receiver();
        let mapped = VolumeShadowCaster::attenuation_from_map(&layers, entry, travel, receiver);
        // The deep-opacity-map lookup agrees with a direct receiver->light march.
        let marched = caster.cast(receiver, travel_plus_x());
        assert!(approx(mapped, marched));
        // The baked curve is monotone non-increasing in depth.
        for pair in layers.windows(2) {
            assert!(pair[1].transmittance <= pair[0].transmittance + TOL);
        }
    }

    #[test]
    fn cast_is_bit_for_bit_deterministic() {
        let field = single_slab_field(0.37);
        let caster = VolumeShadowCaster::new(ParticleSystemHandle(7), &field, 1.1, 0.25, 16);
        let light = travel_plus_x();
        let receiver = backlit_receiver();
        let first = caster.cast(receiver, light);
        let second = caster.cast(receiver, light);
        assert_eq!(first.to_bits(), second.to_bits());
    }

    #[test]
    fn degenerate_direction_is_fully_lit() {
        let field = single_slab_field(0.9);
        let caster = VolumeShadowCaster::new(ParticleSystemHandle(8), &field, 1.0, 0.25, 16);
        let dead = SceneLight::Directional { travel: Vec3::ZERO };
        assert!(approx(caster.cast(backlit_receiver(), dead), 1.0));
    }
}
