//! Effect instancing and template localization contracts (design §26, §30).
//!
//! A single authored *effect template* (an emitter graph with a fixed emitter
//! count and per-emitter particle capacity) is shared by many *effect
//! instances* scattered through the world, each carrying its own local→world
//! transform and a small set of per-instance parameter overrides. This mirrors
//! `Niagara`'s emitter-template plus per-instance override model and Unity `VFX
//! Graph`'s reusable graph assets, without reusing any of their code.
//!
//! The template is authored once and lives in a shared asset store; the
//! instance is the lightweight, pooled runtime object. Localization is the act
//! of turning a template-local quantity (a spawn position, an emitter radius)
//! into world space through the instance transform, so one fireball asset can
//! be placed at a hundred sizes and locations while its `GPU`-resident particle
//! pools and override buffers stay predictable in size.
//!
//! Everything here is `CPU`-verifiable integer / `f32` arithmetic: capacities
//! saturate rather than wrap, an empty instance set localizes to nothing, and
//! nothing panics or divides by zero. The byte-size helpers reuse the shared
//! `std430` strides from [`crate::particle::gpu_layout`] so an override
//! buffer's `VRAM` footprint is derived from the same `ABI` the `GPU` binds
//! against.

use alloc::vec::Vec;

use crate::particle::gpu_layout::VEC4_STRIDE;

/// Absolute tolerance for `f32` comparisons.
///
/// The contract layer never compares floats with `==` / `!=`; two magnitudes
/// are considered equal when their absolute difference is below this epsilon.
const CMP_EPS: f32 = 1e-6;

/// An authored effect template shared by every instance that references it.
///
/// A template fixes how many emitters the effect runs and how many particles
/// each emitter's pool can hold. Instances never resize these; they only place
/// and parameterize a copy, so the template's particle budget is a stable upper
/// bound that per-instance `GPU` pools are sized from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EffectTemplate {
    /// Number of emitters the template's graph runs.
    pub emitter_count: u32,
    /// Fixed particle-pool capacity of each emitter.
    pub particle_capacity_per_emitter: u32,
}

impl EffectTemplate {
    /// Total particle capacity a single instance of this template reserves,
    /// summed across all emitters.
    ///
    /// The product saturates at [`u32::MAX`] so a pathological authored
    /// template can never wrap to a small, under-allocated pool.
    #[must_use]
    pub fn template_particle_capacity(&self) -> u32 {
        self.emitter_count
            .saturating_mul(self.particle_capacity_per_emitter)
    }
}

/// A hand-rolled three-component vector local to this contract.
///
/// `prism_render_architecture` is a dependency-free contracts crate, so effect
/// instancing carries its own lightweight vector rather than importing a math
/// crate or another particle module. Only `sqrt` is used for length; no
/// transcendental functions appear.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
}

impl Vec3 {
    /// The zero vector.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Builds a vector from its components.
    #[must_use]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Euclidean length, computed with `sqrt` (the only radical allowed).
    #[must_use]
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Returns this vector scaled uniformly by `s`.
    #[must_use]
    pub fn scaled(&self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    /// Component-wise sum with `other`.
    #[must_use]
    pub fn add(&self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    /// Whether every component equals `other`'s within [`CMP_EPS`].
    #[must_use]
    pub fn approx_eq(&self, other: Self) -> bool {
        (self.x - other.x).abs() < CMP_EPS
            && (self.y - other.y).abs() < CMP_EPS
            && (self.z - other.z).abs() < CMP_EPS
    }
}

/// A rigid-plus-uniform-scale placement of an effect instance in the world.
///
/// Effect instancing only needs translation and a uniform scale: rotation is
/// deferred to the full transform stack and is intentionally out of scope for
/// this contract, which localizes radii and positions for pool sizing and
/// bounds. A uniform scale keeps emitter-radius localization a single multiply.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InstanceTransform {
    /// World-space translation of the instance origin.
    pub translation: Vec3,
    /// Uniform scale applied to every template-local quantity.
    pub uniform_scale: f32,
}

impl InstanceTransform {
    /// Localizes a template-local radius into world space.
    ///
    /// Because the transform is uniform, a local radius maps to
    /// `local_radius * uniform_scale` with no directional distortion, so
    /// spherical emitter bounds stay spherical.
    #[must_use]
    pub fn local_to_world_scale(&self, local_radius: f32) -> f32 {
        local_radius * self.uniform_scale
    }

    /// Localizes a template-local position into world space.
    ///
    /// The point is first scaled about the instance origin, then translated,
    /// matching the order a `GPU` spawn kernel would apply per instance.
    #[must_use]
    pub fn apply(&self, local: Vec3) -> Vec3 {
        local.scaled(self.uniform_scale).add(self.translation)
    }
}

/// A live placement of an effect: which template it plays, where, and how many
/// per-instance parameter overrides it carries.
///
/// The override count is metadata used to size the shared override buffer; the
/// override *values* live in a separate `GPU` buffer described by
/// [`override_buffer_bytes`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EffectInstance {
    /// Identifier of the shared template this instance plays.
    pub template_id: u32,
    /// World placement of this instance.
    pub transform: InstanceTransform,
    /// Number of per-instance parameter overrides this instance supplies.
    pub param_override_count: u32,
}

/// A fixed-capacity pool of effect-instance slots with an active counter.
///
/// Spawning claims a slot and fails when the pool is full; despawning releases
/// one and saturates at zero so a double-despawn can never underflow the
/// counter. The pool tracks only counts here — slot identity and recycling are
/// owned by the runtime allocator — so the contract stays branch-simple and
/// verifiable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InstancePool {
    capacity: u32,
    active: u32,
}

impl InstancePool {
    /// Builds an empty pool with the given slot `capacity`.
    #[must_use]
    pub fn new(capacity: u32) -> Self {
        Self {
            capacity,
            active: 0,
        }
    }

    /// Claims a slot, returning its index, or `Err(())` when the pool is full.
    ///
    /// # Errors
    ///
    /// Returns `Err(())` when [`InstancePool::is_full`] already holds, leaving
    /// the active count unchanged. The error carries no payload because "pool
    /// full" is the only failure mode.
    #[expect(
        clippy::result_unit_err,
        reason = "the sole failure is a full pool; a unit error keeps the contract minimal"
    )]
    pub fn spawn(&mut self) -> Result<u32, ()> {
        if self.is_full() {
            return Err(());
        }
        let index = self.active;
        self.active += 1;
        Ok(index)
    }

    /// Releases one active slot, saturating at zero.
    ///
    /// Despawning an already-empty pool is a no-op rather than an underflow, so
    /// duplicate release events from the `GPU` readback are harmless.
    pub fn despawn(&mut self) {
        self.active = self.active.saturating_sub(1);
    }

    /// Number of currently active instances.
    #[must_use]
    pub fn active_count(&self) -> u32 {
        self.active
    }

    /// Whether every slot is claimed.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.active >= self.capacity
    }

    /// Number of free slots remaining.
    #[must_use]
    pub fn remaining(&self) -> u32 {
        self.capacity.saturating_sub(self.active)
    }
}

/// Upper bound on the particles a single instance of `template` can hold.
///
/// This is exactly the template's own particle capacity; instancing does not
/// grow the per-instance pool, it only replicates it.
#[must_use]
pub fn instance_particle_upper_bound(template: &EffectTemplate) -> u32 {
    template.template_particle_capacity()
}

/// Upper bound on the particles a whole set of instances can hold, summed
/// across every instance's template capacity.
///
/// Each entry pairs a `template_id` (carried for the caller's bookkeeping) with
/// the resolved template. The sum accumulates in `u64` and saturates, so even a
/// vast instance set with maxed-out templates reports a finite bound rather
/// than wrapping.
#[must_use]
pub fn total_particle_upper_bound(instances: &[(u32, EffectTemplate)]) -> u64 {
    let mut total: u64 = 0;
    for (_template_id, template) in instances {
        total = total.saturating_add(u64::from(template.template_particle_capacity()));
    }
    total
}

/// Byte size of the shared per-instance parameter-override buffer.
///
/// Every override is treated as one `vec4<f32>` (the `std430` [`VEC4_STRIDE`]),
/// and the buffer holds `instance_count * per_instance_overrides` of them. The
/// arithmetic runs in `u64` and saturates so a degenerate request reports a
/// finite `VRAM` footprint instead of wrapping to a small allocation.
#[must_use]
pub fn override_buffer_bytes(instance_count: u32, per_instance_overrides: u32) -> u64 {
    let stride = u64::try_from(VEC4_STRIDE).unwrap_or(u64::MAX);
    let overrides = u64::from(instance_count).saturating_mul(u64::from(per_instance_overrides));
    stride.saturating_mul(overrides)
}

/// Counts how many distinct templates a set of instances references.
///
/// The crate is dependency-free, so this avoids a hash set: it collects the
/// `template_id`s into a [`Vec`], sorts them, and counts the boundaries between
/// runs of equal ids in a single linear scan. An empty set yields zero.
#[must_use]
pub fn distinct_template_count(instances: &[EffectInstance]) -> u32 {
    let mut ids: Vec<u32> = Vec::with_capacity(instances.len());
    for instance in instances {
        ids.push(instance.template_id);
    }
    ids.sort_unstable();

    let mut distinct: u32 = 0;
    let mut previous: Option<u32> = None;
    for id in ids {
        let is_new = match previous {
            Some(prev) => prev != id,
            None => true,
        };
        if is_new {
            distinct = distinct.saturating_add(1);
            previous = Some(id);
        }
    }
    distinct
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_transform() -> InstanceTransform {
        InstanceTransform {
            translation: Vec3::ZERO,
            uniform_scale: 1.0,
        }
    }

    fn instance(template_id: u32) -> EffectInstance {
        EffectInstance {
            template_id,
            transform: identity_transform(),
            param_override_count: 0,
        }
    }

    #[test]
    fn template_capacity_multiplies_emitters_and_pool() {
        let template = EffectTemplate {
            emitter_count: 4,
            particle_capacity_per_emitter: 1000,
        };
        assert_eq!(template.template_particle_capacity(), 4000);
    }

    #[test]
    fn template_capacity_saturates_on_overflow() {
        let template = EffectTemplate {
            emitter_count: u32::MAX,
            particle_capacity_per_emitter: 2,
        };
        assert_eq!(template.template_particle_capacity(), u32::MAX);
    }

    #[test]
    fn vec3_length_uses_sqrt() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!((v.length() - 5.0).abs() < CMP_EPS);
    }

    #[test]
    fn vec3_scaled_and_add_are_componentwise() {
        let v = Vec3::new(1.0, -2.0, 3.0);
        let s = v.scaled(2.0);
        assert!(s.approx_eq(Vec3::new(2.0, -4.0, 6.0)));
        let sum = s.add(Vec3::new(1.0, 1.0, 1.0));
        assert!(sum.approx_eq(Vec3::new(3.0, -3.0, 7.0)));
    }

    #[test]
    fn local_to_world_scale_is_uniform_multiply() {
        let transform = InstanceTransform {
            translation: Vec3::new(10.0, 0.0, 0.0),
            uniform_scale: 2.5,
        };
        assert!((transform.local_to_world_scale(4.0) - 10.0).abs() < CMP_EPS);
    }

    #[test]
    fn apply_scales_then_translates() {
        let transform = InstanceTransform {
            translation: Vec3::new(5.0, -3.0, 2.0),
            uniform_scale: 2.0,
        };
        let world = transform.apply(Vec3::new(1.0, 1.0, 1.0));
        assert!(world.approx_eq(Vec3::new(7.0, -1.0, 4.0)));
    }

    #[test]
    fn apply_with_zero_uniform_scale_collapses_to_translation() {
        let transform = InstanceTransform {
            translation: Vec3::new(1.0, 2.0, 3.0),
            uniform_scale: 0.0,
        };
        let world = transform.apply(Vec3::new(9.0, 9.0, 9.0));
        assert!(world.approx_eq(Vec3::new(1.0, 2.0, 3.0)));
        assert!(transform.local_to_world_scale(100.0).abs() < CMP_EPS);
    }

    #[test]
    fn pool_spawn_hands_out_sequential_indices() {
        let mut pool = InstancePool::new(3);
        assert_eq!(pool.spawn(), Ok(0));
        assert_eq!(pool.spawn(), Ok(1));
        assert_eq!(pool.spawn(), Ok(2));
        assert_eq!(pool.active_count(), 3);
    }

    #[test]
    fn pool_spawn_fails_when_full() {
        let mut pool = InstancePool::new(1);
        assert_eq!(pool.spawn(), Ok(0));
        assert!(pool.is_full());
        assert_eq!(pool.spawn(), Err(()));
        assert_eq!(pool.active_count(), 1);
        assert_eq!(pool.remaining(), 0);
    }

    #[test]
    fn pool_despawn_to_zero_then_again_does_not_underflow() {
        let mut pool = InstancePool::new(2);
        let _ = pool.spawn();
        pool.despawn();
        assert_eq!(pool.active_count(), 0);
        pool.despawn();
        assert_eq!(pool.active_count(), 0);
        assert_eq!(pool.remaining(), 2);
    }

    #[test]
    fn zero_capacity_pool_is_full_immediately() {
        let mut pool = InstancePool::new(0);
        assert!(pool.is_full());
        assert_eq!(pool.spawn(), Err(()));
        assert_eq!(pool.remaining(), 0);
    }

    #[test]
    fn instance_upper_bound_matches_template_capacity() {
        let template = EffectTemplate {
            emitter_count: 3,
            particle_capacity_per_emitter: 64,
        };
        assert_eq!(instance_particle_upper_bound(&template), 192);
    }

    #[test]
    fn total_upper_bound_sums_all_instances() {
        let a = EffectTemplate {
            emitter_count: 2,
            particle_capacity_per_emitter: 100,
        };
        let b = EffectTemplate {
            emitter_count: 1,
            particle_capacity_per_emitter: 50,
        };
        let set = [(0_u32, a), (0_u32, a), (1_u32, b)];
        assert_eq!(total_particle_upper_bound(&set), 200 + 200 + 50);
    }

    #[test]
    fn total_upper_bound_saturates_in_u64() {
        // Each template maxes out u32; summing more than `u32::MAX + 1` of them
        // in a naive `u32` accumulator would wrap, but the `u64` accumulator
        // saturates instead of overflowing.
        let maxed = EffectTemplate {
            emitter_count: u32::MAX,
            particle_capacity_per_emitter: 1,
        };
        let set = [(0_u32, maxed); 8];
        let expected = 8_u64 * u64::from(u32::MAX);
        assert_eq!(total_particle_upper_bound(&set), expected);
    }

    #[test]
    fn total_upper_bound_of_empty_set_is_zero() {
        assert_eq!(total_particle_upper_bound(&[]), 0);
    }

    #[test]
    fn override_buffer_bytes_uses_vec4_stride() {
        assert_eq!(override_buffer_bytes(4, 3), 4 * 3 * 16);
        assert_eq!(override_buffer_bytes(0, 5), 0);
        assert_eq!(override_buffer_bytes(5, 0), 0);
    }

    #[test]
    fn override_buffer_bytes_saturates() {
        assert_eq!(override_buffer_bytes(u32::MAX, u32::MAX), u64::MAX);
    }

    #[test]
    fn distinct_template_count_empty_is_zero() {
        assert_eq!(distinct_template_count(&[]), 0);
    }

    #[test]
    fn distinct_template_count_all_same_is_one() {
        let set = [instance(7), instance(7), instance(7)];
        assert_eq!(distinct_template_count(&set), 1);
    }

    #[test]
    fn distinct_template_count_all_different() {
        let set = [instance(3), instance(1), instance(9), instance(4)];
        assert_eq!(distinct_template_count(&set), 4);
    }

    #[test]
    fn distinct_template_count_mixed_runs() {
        let set = [
            instance(2),
            instance(2),
            instance(5),
            instance(2),
            instance(5),
        ];
        assert_eq!(distinct_template_count(&set), 2);
    }
}
