use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use super::{Field, FieldBody, FieldEffect, FieldFalloff, FieldShape};

/// All records use vec4 lanes to make the Rust/WGSL storage stride explicit.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct PackedField {
    pub tags: [u32; 4], // shape, effect, falloff, target mask
    pub shape_a: [f32; 4],
    pub shape_b: [f32; 4],
    pub effect_a: [f32; 4],
    pub effect_b: [f32; 4],
    pub attenuation: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct PackedBody {
    pub position_mass: [f32; 4],
    pub velocity: [f32; 4],
    pub mask: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct PackedSample {
    pub force: [f32; 4],
    pub torque: [f32; 4],
}

fn lane(vector: Vec3, w: f32) -> [f32; 4] {
    [vector.x, vector.y, vector.z, w]
}

impl From<&FieldBody> for PackedBody {
    fn from(body: &FieldBody) -> Self {
        Self {
            position_mass: lane(body.position, body.mass),
            velocity: lane(body.velocity, 0.0),
            mask: [body.target_mask.0, 0, 0, 0],
        }
    }
}

impl From<&Field> for PackedField {
    fn from(field: &Field) -> Self {
        let mut out = Self::zeroed();
        out.tags[3] = field.target_mask.0;
        match field.shape {
            FieldShape::Global => {}
            FieldShape::Sphere { center, radius } => {
                out.tags[0] = 1;
                out.shape_a = lane(center, radius);
            }
            FieldShape::Aabb { min, max } => {
                out.tags[0] = 2;
                out.shape_a = lane(min, 0.0);
                out.shape_b = lane(max, 0.0);
            }
            FieldShape::Cone {
                apex,
                axis,
                angle,
                length,
            } => {
                out.tags[0] = 3;
                out.shape_a = lane(apex, length);
                out.shape_b = lane(axis, angle);
            }
        }
        match field.effect {
            FieldEffect::ConstantForce(value) => out.effect_a = lane(value, 0.0),
            FieldEffect::Radial { center, strength } => {
                out.tags[1] = 1;
                out.effect_a = lane(center, strength);
            }
            FieldEffect::Vortex {
                center,
                axis,
                strength,
            } => {
                out.tags[1] = 2;
                out.effect_a = lane(center, strength);
                out.effect_b = lane(axis, 0.0);
            }
            FieldEffect::Drag { coefficient } => {
                out.tags[1] = 3;
                out.effect_a[3] = coefficient;
            }
            FieldEffect::Buoyancy {
                fluid_density,
                gravity,
                displaced_volume,
            } => {
                out.tags[1] = 4;
                out.effect_a = lane(gravity, fluid_density);
                out.effect_b[3] = displaced_volume;
            }
        }
        match field.falloff {
            FieldFalloff::Constant => {}
            FieldFalloff::Linear => out.tags[2] = 1,
            FieldFalloff::InverseSquare { softening } => {
                out.tags[2] = 2;
                out.attenuation[0] = softening;
            }
        }
        out
    }
}
