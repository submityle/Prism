//! Canonical main-world light scene and native direct-light adapter.

use crate::lighting::{LightingDirectionalInput, LightingPunctualInput};
use prism_ecs::prelude::Resource;
use prism_render_architecture::lighting::{
    LightClass, LightEmission, LightGeometry, LightScene, LightSceneHandle, LightUnit,
};

/// Sole authoritative CPU light table in the application world.
#[derive(Clone, Debug, Default, Resource)]
pub struct LightingLightScene {
    /// Generation-safe retained light records.
    pub scene: LightScene,
}

/// Failure to represent a canonical light in the current direct-light ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LightingSceneExtractError {
    /// The current direct-light path does not represent this emitter class.
    UnsupportedClass {
        handle: LightSceneHandle,
        class: LightClass,
    },
    /// No explicit photometric conversion to the current ABI exists.
    UnsupportedUnit {
        handle: LightSceneHandle,
        unit: LightUnit,
    },
    /// A non-default layer mask cannot be represented by the current ABI.
    UnsupportedLayerMask { handle: LightSceneHandle, mask: u32 },
    /// The current cone ABI cannot preserve a distinct inner cone.
    UnsupportedSpotProfile { handle: LightSceneHandle },
    /// Renderer flags are not represented by the current direct-light ABI.
    UnsupportedFlags { handle: LightSceneHandle, bits: u32 },
    /// More records than the current GPU light buffers can represent.
    CapacityExceeded {
        directionals: usize,
        punctuals: usize,
    },
}

/// Derived direct-light ABI values.
#[derive(Clone, Debug, Default)]
pub struct NativeLights {
    /// Directionals in stable slot order.
    pub directionals: Vec<LightingDirectionalInput>,
    /// Point and spot lights in stable slot order.
    pub punctuals: Vec<LightingPunctualInput>,
}

/// Converts all enabled records or rejects the complete snapshot.
///
/// The canonical directional vector points along light propagation, matching
/// the documented rect/disk emission direction. The direct-light ABI instead
/// stores a vector from the shaded point toward the source, so directionals
/// are negated exactly once here. Punctual `direction` remains the emitted
/// direction consumed by the shader's `dot(light_direction, -to_light)` test.
pub fn derive_native_lights(scene: &LightScene) -> Result<NativeLights, LightingSceneExtractError> {
    let mut native = NativeLights::default();
    for (handle, record) in scene.iter() {
        if !record.visibility.enabled {
            continue;
        }
        if record.visibility.layer_mask != u32::MAX {
            return Err(LightingSceneExtractError::UnsupportedLayerMask {
                handle,
                mask: record.visibility.layer_mask,
            });
        }
        if record.flags.bits() != 0 {
            return Err(LightingSceneExtractError::UnsupportedFlags {
                handle,
                bits: record.flags.bits(),
            });
        }
        let emission = record.emission;
        match record.geometry {
            LightGeometry::Directional { direction } => {
                if emission.unit != LightUnit::Lux {
                    return Err(LightingSceneExtractError::UnsupportedUnit {
                        handle,
                        unit: emission.unit,
                    });
                }
                native.directionals.push(LightingDirectionalInput {
                    direction_to_light: direction.map(|v| -v),
                    visibility: 1.0,
                    illuminance: rgb(emission, emission.value),
                    _padding: 0.0,
                });
            }
            LightGeometry::Point {
                position, range, ..
            } => {
                let intensity = match emission.unit {
                    LightUnit::Candela => emission.value,
                    LightUnit::Lumen => emission.value / (4.0 * core::f32::consts::PI),
                    unit => {
                        return Err(LightingSceneExtractError::UnsupportedUnit { handle, unit })
                    }
                };
                native.punctuals.push(LightingPunctualInput {
                    position,
                    range,
                    intensity: rgb(emission, intensity),
                    spot_scale: 0.0,
                    direction: [0.0, 0.0, -1.0],
                    spot_offset: 1.0,
                    visibility: 1.0,
                    _padding: [0.0; 7],
                });
            }
            LightGeometry::Spot {
                position,
                direction,
                inner_angle,
                outer_angle,
                range,
                ..
            } => {
                let cos_inner = inner_angle.cos();
                let cos_outer = outer_angle.cos();
                let cosine_width = cos_inner - cos_outer;
                // Equal angles describe a hard edge, which the ABI's finite
                // linear cone ramp cannot represent.
                if cosine_width <= f32::EPSILON {
                    return Err(LightingSceneExtractError::UnsupportedSpotProfile { handle });
                }
                // The GPU multiplies attenuation by cone^2. Integrating this
                // profile over solid angle gives 2*pi*((1-cos(inner)) +
                // (cos(inner)-cos(outer))/3), converting total lumens to
                // peak candela for the actual inner/outer cone distribution.
                let effective_solid_angle =
                    2.0 * core::f32::consts::PI * ((1.0 - cos_inner) + cosine_width / 3.0);
                let intensity = match emission.unit {
                    LightUnit::Candela => emission.value,
                    LightUnit::Lumen => emission.value / effective_solid_angle,
                    unit => {
                        return Err(LightingSceneExtractError::UnsupportedUnit { handle, unit })
                    }
                };
                native.punctuals.push(LightingPunctualInput {
                    position,
                    range,
                    intensity: rgb(emission, intensity),
                    spot_scale: 1.0 / cosine_width,
                    direction,
                    spot_offset: -cos_outer / cosine_width,
                    visibility: 1.0,
                    _padding: [0.0; 7],
                });
            }
            geometry => {
                return Err(LightingSceneExtractError::UnsupportedClass {
                    handle,
                    class: geometry.class(),
                })
            }
        }
    }
    if native.directionals.len() > 1024 || native.punctuals.len() > 1024 {
        return Err(LightingSceneExtractError::CapacityExceeded {
            directionals: native.directionals.len(),
            punctuals: native.punctuals.len(),
        });
    }
    Ok(native)
}

fn rgb(emission: LightEmission, value: f32) -> [f32; 3] {
    emission.color.map(|component| component * value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::lighting::{
        LightClassification, LightEmission, LightFlags, LightMobility, LightPatch, LightRecord,
        LightVisibility,
    };

    fn point_record(value: f32) -> prism_render_architecture::lighting::LightRecord {
        LightRecord::new(
            LightClassification {
                class: LightClass::Point,
                mobility: LightMobility::Dynamic,
            },
            LightEmission::new([0.5, 1.0, 2.0], value, LightUnit::Candela),
            LightGeometry::Point {
                position: [1.0, 2.0, 3.0],
                radius: 0.1,
                range: 10.0,
            },
        )
    }

    #[test]
    fn canonical_insert_update_remove_and_generation_freshness() {
        let mut scene = LightScene::new();
        let (old, _) = scene.insert(point_record(2.0), 1).expect("insert");
        let extracted = derive_native_lights(&scene).expect("derive insert");
        assert_eq!(extracted.punctuals.len(), 1);
        assert_eq!(extracted.punctuals[0].intensity, [1.0, 2.0, 4.0]);
        let insert_sequence = scene.sequence();

        scene
            .update(
                old,
                LightPatch::new().with_emission(LightEmission::new(
                    [1.0; 3],
                    3.0,
                    LightUnit::Candela,
                )),
                2,
            )
            .expect("update");
        assert_eq!(
            derive_native_lights(&scene)
                .expect("derive update")
                .punctuals[0]
                .intensity,
            [3.0; 3]
        );
        assert!(scene.sequence() > insert_sequence);

        scene.remove(old, 3).expect("remove");
        assert!(derive_native_lights(&scene)
            .expect("derive remove")
            .punctuals
            .is_empty());
        assert!(scene.get(old).is_none());
        let (new, _) = scene.insert(point_record(4.0), 4).expect("reinsert");
        assert_eq!(old.slot(), new.slot());
        assert_ne!(old.generation(), new.generation());
        assert!(scene.get(old).is_none());
    }

    #[test]
    fn explicit_lumen_conversion_and_unsupported_emitter_rejection() {
        let mut point_scene = LightScene::new();
        let record = LightRecord::new(
            LightClassification {
                class: LightClass::Point,
                mobility: LightMobility::Dynamic,
            },
            LightEmission::new([1.0; 3], 4.0 * core::f32::consts::PI, LightUnit::Lumen),
            LightGeometry::Point {
                position: [0.0; 3],
                radius: 0.1,
                range: 0.0,
            },
        );
        point_scene.insert(record, 1).expect("point");
        assert_eq!(
            derive_native_lights(&point_scene)
                .expect("convert")
                .punctuals[0]
                .intensity,
            [1.0; 3]
        );

        let mut area_scene = LightScene::new();
        let area = LightRecord::new(
            LightClassification {
                class: LightClass::Rect,
                mobility: LightMobility::Static,
            },
            LightEmission::new([1.0; 3], 1.0, LightUnit::Nits),
            LightGeometry::Rect {
                position: [0.0; 3],
                direction: [0.0, -1.0, 0.0],
                right: [1.0, 0.0, 0.0],
                up: [0.0, 0.0, 1.0],
                width: 1.0,
                height: 1.0,
            },
        );
        area_scene.insert(area, 1).expect("valid area record");
        assert!(matches!(
            derive_native_lights(&area_scene),
            Err(LightingSceneExtractError::UnsupportedClass {
                class: LightClass::Rect,
                ..
            })
        ));
    }

    #[test]
    fn directional_sign_and_spot_inner_outer_lumen_normalization_match_shader_profile() {
        let mut directional_scene = LightScene::new();
        directional_scene
            .insert(
                LightRecord::new(
                    LightClassification {
                        class: LightClass::Directional,
                        mobility: LightMobility::Static,
                    },
                    LightEmission::new([1.0; 3], 5.0, LightUnit::Lux),
                    LightGeometry::Directional {
                        direction: [0.0, -1.0, 0.0],
                    },
                ),
                1,
            )
            .expect("directional");
        assert_eq!(
            derive_native_lights(&directional_scene)
                .expect("directional conversion")
                .directionals[0]
                .direction_to_light,
            [0.0, 1.0, 0.0]
        );

        let inner_angle = 0.2_f32;
        let outer_angle = 0.6_f32;
        let lumens = 120.0_f32;
        let cos_inner = inner_angle.cos();
        let cos_outer = outer_angle.cos();
        let width = cos_inner - cos_outer;
        let effective_solid_angle = 2.0 * core::f32::consts::PI * ((1.0 - cos_inner) + width / 3.0);
        let mut spot_scene = LightScene::new();
        spot_scene
            .insert(
                LightRecord::new(
                    LightClassification {
                        class: LightClass::Spot,
                        mobility: LightMobility::Dynamic,
                    },
                    LightEmission::new([1.0; 3], lumens, LightUnit::Lumen),
                    LightGeometry::Spot {
                        position: [0.0; 3],
                        direction: [0.0, 0.0, -1.0],
                        inner_angle,
                        outer_angle,
                        radius: 0.0,
                        range: 10.0,
                    },
                ),
                1,
            )
            .expect("spot");
        let spot = derive_native_lights(&spot_scene)
            .expect("spot conversion")
            .punctuals[0];
        assert!((spot.spot_scale - 1.0 / width).abs() < 1.0e-5);
        assert!((spot.spot_offset + cos_outer / width).abs() < 1.0e-5);
        assert!((spot.intensity[0] * effective_solid_angle - lumens).abs() < 1.0e-4);
    }

    #[test]
    fn unsupported_layer_masks_and_flags_fail_instead_of_being_ignored() {
        let mut scene = LightScene::new();
        let (handle, _) = scene.insert(point_record(1.0), 1).expect("insert");
        scene
            .update(
                handle,
                LightPatch::new().with_visibility(LightVisibility {
                    enabled: true,
                    layer_mask: 1,
                }),
                2,
            )
            .expect("set layer mask");
        assert!(matches!(
            derive_native_lights(&scene),
            Err(LightingSceneExtractError::UnsupportedLayerMask { mask: 1, .. })
        ));

        let mut flagged_scene = LightScene::new();
        let (handle, _) = flagged_scene.insert(point_record(1.0), 1).expect("insert");
        flagged_scene
            .update(
                handle,
                LightPatch::new().with_flags(LightFlags::CAST_SHADOW),
                2,
            )
            .expect("set flags");
        assert!(matches!(
            derive_native_lights(&flagged_scene),
            Err(LightingSceneExtractError::UnsupportedFlags { bits, .. })
                if bits == LightFlags::CAST_SHADOW.bits()
        ));
    }

    #[test]
    fn disabled_records_are_not_native_inputs() {
        let mut scene = LightScene::new();
        let (handle, _) = scene.insert(point_record(1.0), 1).expect("insert");
        scene
            .update(
                handle,
                LightPatch::new().with_visibility(LightVisibility {
                    enabled: false,
                    layer_mask: 0,
                }),
                2,
            )
            .expect("disable");
        assert!(derive_native_lights(&scene)
            .expect("derive")
            .punctuals
            .is_empty());
        assert_eq!((scene.sequence(), scene.scene_epoch()), (2, 2));
    }
}
