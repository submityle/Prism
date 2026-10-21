use glam::Vec3;

use super::{FieldBody, FieldEffect, FieldFalloff, FieldRegistry, FieldSample, FieldShape};

const EPS: f32 = 1.0e-6;

/// Independently evaluates the WGSL algorithm on CPU in registry/body order.
///
/// This intentionally does not delegate to core's sampling implementation:
/// tests compare this twin to that reference before checking real GPU parity.
#[must_use]
pub fn cpu_sample_fields(registry: &FieldRegistry, bodies: &[FieldBody]) -> Vec<FieldSample> {
    bodies
        .iter()
        .map(|body| {
            let mut result = FieldSample::default();
            if !body.position.is_finite() || !body.velocity.is_finite() || !body.mass.is_finite() {
                return result;
            }
            for field in registry.fields() {
                if field.target_mask.0 & body.target_mask.0 == 0 {
                    continue;
                }
                let Some(weight) = weight(field.shape, field.falloff, body.position) else {
                    continue;
                };
                let force = match field.effect {
                    FieldEffect::ConstantForce(value) => value,
                    FieldEffect::Radial { center, strength } => {
                        normalize(body.position - center) * strength
                    }
                    FieldEffect::Vortex {
                        center,
                        axis,
                        strength,
                    } => normalize(normalize(axis).cross(body.position - center)) * strength,
                    FieldEffect::Drag { coefficient } => {
                        -body.velocity * coefficient.max(0.0) * body.mass.max(0.0)
                    }
                    FieldEffect::Buoyancy {
                        fluid_density,
                        gravity,
                        displaced_volume,
                    } => -gravity * fluid_density.max(0.0) * displaced_volume.max(0.0),
                };
                let force = force * weight;
                if force.is_finite() {
                    result.force += force;
                }
            }
            result
        })
        .collect()
}

fn normalize(value: Vec3) -> Vec3 {
    let squared = value.length_squared();
    if value.is_finite() && squared.is_finite() && squared > EPS * EPS {
        value / squared.sqrt()
    } else {
        Vec3::ZERO
    }
}

fn weight(shape: FieldShape, falloff: FieldFalloff, position: Vec3) -> Option<f32> {
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
            let unit_axis = normalize(axis);
            if !apex.is_finite()
                || !axis.is_finite()
                || unit_axis == Vec3::ZERO
                || !length.is_finite()
                || length <= EPS
                || !(0.0..std::f32::consts::FRAC_PI_2).contains(&angle)
            {
                return None;
            }
            let delta = position - apex;
            let axial = delta.dot(unit_axis);
            if axial < 0.0 || axial > length {
                return None;
            }
            if (delta - unit_axis * axial).length() > axial * angle.tan() + EPS {
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
        FieldFalloff::InverseSquare { softening } => 1.0 / (1.0 + x * x / softening.max(EPS)),
    }
}
