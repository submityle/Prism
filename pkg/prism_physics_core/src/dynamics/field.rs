//! Deterministic analytic force fields.

use glam::Vec3;

const EPS: f32 = 1.0e-6;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FieldId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldTargetMask(pub u32);

impl FieldTargetMask {
    pub const ALL: Self = Self(u32::MAX);
    pub const NONE: Self = Self(0);
    #[must_use]
    pub const fn matches(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FieldShape {
    Global,
    Sphere {
        center: Vec3,
        radius: f32,
    },
    Aabb {
        min: Vec3,
        max: Vec3,
    },
    Cone {
        apex: Vec3,
        axis: Vec3,
        angle: f32,
        length: f32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FieldFalloff {
    Constant,
    Linear,
    InverseSquare { softening: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FieldEffect {
    ConstantForce(Vec3),
    Radial {
        center: Vec3,
        strength: f32,
    },
    Vortex {
        center: Vec3,
        axis: Vec3,
        strength: f32,
    },
    Drag {
        coefficient: f32,
    },
    Buoyancy {
        fluid_density: f32,
        gravity: Vec3,
        displaced_volume: f32,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FieldSample {
    pub force: Vec3,
    pub torque: Vec3,
}

/// Immutable inputs for one batched sample, in the caller's stable body order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldBody {
    pub position: Vec3,
    pub velocity: Vec3,
    pub mass: f32,
    pub target_mask: FieldTargetMask,
}

/// External compute authority for one complete analytic-field batch.
///
/// Providers must return exactly one world-space wrench per input body, in the
/// same order. The world validates the entire result before committing any
/// force or waking bodies. Implementations must report failure explicitly;
/// silently substituting a different authority is not part of this contract.
pub trait FieldBatchSampler {
    type Error;

    fn sample_batch(
        &mut self,
        registry: &FieldRegistry,
        bodies: &[FieldBody],
    ) -> Result<Vec<FieldSample>, Self::Error>;
}

/// Core CPU authority, useful both as the default and as a backend reference.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuFieldSampler;

impl FieldBatchSampler for CpuFieldSampler {
    type Error = core::convert::Infallible;

    fn sample_batch(
        &mut self,
        registry: &FieldRegistry,
        bodies: &[FieldBody],
    ) -> Result<Vec<FieldSample>, Self::Error> {
        Ok(bodies
            .iter()
            .map(|body| registry.sample(body.position, body.velocity, body.mass, body.target_mask))
            .collect())
    }
}

/// A rejected batch has not changed the world's force queues or body state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FieldBatchError<E> {
    Provider(E),
    SampleCount { expected: usize, actual: usize },
    NonFiniteSample { index: usize },
    AccumulationOverflow { index: usize },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Field {
    pub id: FieldId,
    pub shape: FieldShape,
    pub falloff: FieldFalloff,
    pub effect: FieldEffect,
    pub target_mask: FieldTargetMask,
}

impl Field {
    #[must_use]
    pub fn constant(id: FieldId, force: Vec3) -> Self {
        Self {
            id,
            shape: FieldShape::Global,
            falloff: FieldFalloff::Constant,
            effect: FieldEffect::ConstantForce(force),
            target_mask: FieldTargetMask::ALL,
        }
    }

    #[must_use]
    pub fn sample(
        &self,
        position: Vec3,
        velocity: Vec3,
        mass: f32,
        mask: FieldTargetMask,
    ) -> FieldSample {
        if !self.target_mask.matches(mask)
            || !position.is_finite()
            || !velocity.is_finite()
            || !mass.is_finite()
        {
            return FieldSample::default();
        }
        let Some(weight) = shape_weight(self.shape, position, self.falloff) else {
            return FieldSample::default();
        };
        let force = match self.effect {
            FieldEffect::ConstantForce(force) => force,
            FieldEffect::Radial { center, strength } => {
                let direction = safe_normalize(position - center);
                direction * strength
            }
            FieldEffect::Vortex {
                center,
                axis,
                strength,
            } => {
                let axis = safe_normalize(axis);
                safe_normalize(axis.cross(position - center)) * strength
            }
            FieldEffect::Drag { coefficient } => -velocity * coefficient.max(0.0) * mass.max(0.0),
            FieldEffect::Buoyancy {
                fluid_density,
                gravity,
                displaced_volume,
            } => -gravity * fluid_density.max(0.0) * displaced_volume.max(0.0),
        };
        let force = force * weight;
        if force.is_finite() {
            FieldSample {
                force,
                torque: Vec3::ZERO,
            }
        } else {
            FieldSample::default()
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FieldRegistry {
    fields: Vec<Field>,
    next_id: u64,
}

impl FieldRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, mut field: Field) -> FieldId {
        if field.id == FieldId(0) {
            self.next_id = self.next_id.saturating_add(1).max(1);
            field.id = FieldId(self.next_id);
        } else {
            self.next_id = self.next_id.max(field.id.0);
        }
        self.fields.push(field);
        self.fields.sort_by_key(|field| field.id);
        field.id
    }

    pub fn remove(&mut self, id: FieldId) -> Option<Field> {
        let index = self.fields.iter().position(|field| field.id == id)?;
        Some(self.fields.remove(index))
    }

    #[must_use]
    pub fn fields(&self) -> &[Field] {
        &self.fields
    }

    #[must_use]
    pub fn sample(
        &self,
        position: Vec3,
        velocity: Vec3,
        mass: f32,
        mask: FieldTargetMask,
    ) -> FieldSample {
        self.fields
            .iter()
            .map(|field| field.sample(position, velocity, mass, mask))
            .fold(FieldSample::default(), |a, b| FieldSample {
                force: a.force + b.force,
                torque: a.torque + b.torque,
            })
    }
}

pub trait FieldSampler {
    fn sample_field(
        &self,
        position: Vec3,
        velocity: Vec3,
        mass: f32,
        mask: FieldTargetMask,
    ) -> FieldSample;
}

impl FieldSampler for FieldRegistry {
    fn sample_field(
        &self,
        position: Vec3,
        velocity: Vec3,
        mass: f32,
        mask: FieldTargetMask,
    ) -> FieldSample {
        self.sample(position, velocity, mass, mask)
    }
}

fn shape_weight(shape: FieldShape, position: Vec3, falloff: FieldFalloff) -> Option<f32> {
    if let FieldFalloff::InverseSquare { softening } = falloff {
        if !softening.is_finite() || softening < 0.0 {
            return None;
        }
    }
    let normalized = match shape {
        FieldShape::Global => 1.0,
        FieldShape::Sphere { center, radius } => {
            if !center.is_finite() || !radius.is_finite() || radius <= EPS {
                return None;
            }
            let distance = position.distance(center);
            if distance > radius {
                return None;
            }
            falloff_weight(distance / radius, falloff)
        }
        FieldShape::Aabb { min, max } => {
            if !min.is_finite() || !max.is_finite() || min.cmpge(max).any() {
                return None;
            }
            if position.x < min.x
                || position.y < min.y
                || position.z < min.z
                || position.x > max.x
                || position.y > max.y
                || position.z > max.z
            {
                return None;
            }
            let extent = (max - min).max(Vec3::splat(EPS));
            let edge = ((position - min) / extent)
                .min(Vec3::ONE - (position - min) / extent)
                .min_element()
                .clamp(0.0, 0.5)
                * 2.0;
            falloff_weight(1.0 - edge, falloff)
        }
        FieldShape::Cone {
            apex,
            axis,
            angle,
            length,
        } => {
            if !apex.is_finite()
                || !axis.is_finite()
                || safe_normalize(axis) == Vec3::ZERO
                || !length.is_finite()
                || length <= EPS
                || !(0.0..std::f32::consts::FRAC_PI_2).contains(&angle)
            {
                return None;
            }
            let delta = position - apex;
            let axis = safe_normalize(axis);
            let axial = delta.dot(axis);
            if axial < 0.0 || axial > length {
                return None;
            }
            let radial = (delta - axis * axial).length();
            let radius = axial * angle.tan();
            if radial > radius + EPS {
                return None;
            }
            falloff_weight(axial / length, falloff)
        }
    };
    normalized.is_finite().then(|| normalized.clamp(0.0, 1.0))
}

fn falloff_weight(normalized: f32, falloff: FieldFalloff) -> f32 {
    let x = normalized.clamp(0.0, 1.0);
    match falloff {
        FieldFalloff::Constant => 1.0,
        FieldFalloff::Linear => 1.0 - x,
        FieldFalloff::InverseSquare { softening } => 1.0 / (1.0 + (x * x) / softening.max(EPS)),
    }
}

#[must_use]
fn safe_normalize(value: Vec3) -> Vec3 {
    if !value.is_finite() {
        return Vec3::ZERO;
    }
    let length_squared = value.length_squared();
    if length_squared.is_finite() && length_squared > EPS * EPS {
        value / length_squared.sqrt()
    } else {
        Vec3::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_field_has_deterministic_falloff() {
        let field = Field {
            id: FieldId(1),
            shape: FieldShape::Sphere {
                center: Vec3::ZERO,
                radius: 2.0,
            },
            falloff: FieldFalloff::Linear,
            effect: FieldEffect::ConstantForce(Vec3::Y),
            target_mask: FieldTargetMask::ALL,
        };
        assert_eq!(
            field.sample(
                Vec3::new(0.0, 0.0, 3.0),
                Vec3::ZERO,
                1.0,
                FieldTargetMask::ALL
            ),
            FieldSample::default()
        );
        assert!(
            (field
                .sample(Vec3::X, Vec3::ZERO, 1.0, FieldTargetMask::ALL)
                .force
                .y
                - 0.5)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn registry_orders_fields_by_id_and_sums_samples() {
        let mut registry = FieldRegistry::new();
        registry.insert(Field::constant(FieldId(9), Vec3::X));
        registry.insert(Field::constant(FieldId(3), Vec3::Y));
        assert_eq!(registry.fields()[0].id, FieldId(3));
        assert_eq!(
            registry
                .sample(Vec3::ZERO, Vec3::ZERO, 1.0, FieldTargetMask::ALL)
                .force,
            Vec3::new(1.0, 1.0, 0.0)
        );
    }
}
