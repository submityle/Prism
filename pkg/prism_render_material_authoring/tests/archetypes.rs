#![allow(missing_docs, reason = "integration tests are self-describing")]

use prism_render_material_authoring::MaterialArchetype;

#[test]
fn archetypes_have_stable_keys_and_are_deterministic() {
    let keys = MaterialArchetype::ALL.map(MaterialArchetype::key);
    assert_eq!(
        keys,
        [
            "car_paint",
            "cloth",
            "foliage",
            "glass",
            "metal",
            "skin",
            "toon",
            "water"
        ]
    );
    for archetype in MaterialArchetype::ALL {
        assert_eq!(archetype.parameters(), archetype.parameters());
        assert!(!archetype.display_name().is_empty());
    }
}

#[test]
fn archetypes_populate_real_runtime_axes_without_non_finite_values() {
    for archetype in MaterialArchetype::ALL {
        let p = archetype.parameters();
        let scalars = [
            p.emission_luminance,
            p.base_metalness,
            p.specular_roughness,
            p.specular_ior,
            p.transmission_weight,
            p.thickness,
            p.coat_weight,
            p.fuzz_weight,
            p.subsurface_weight,
            p.index_of_refraction,
            p.npr_face_softness,
        ];
        assert!(scalars.into_iter().all(f32::is_finite), "{archetype:?}");
        assert!(p.base_color.into_iter().all(f32::is_finite));
        assert!(p.transmission_color.into_iter().all(f32::is_finite));
    }

    let metal = MaterialArchetype::Metal.parameters();
    assert_eq!(metal.base_metalness, 1.0);
    assert_eq!(metal.transmission_weight, 0.0);

    let glass = MaterialArchetype::Glass.parameters();
    assert_eq!(glass.transmission_weight, 1.0);
    assert!(glass.specular_roughness < 0.1);

    let cloth = MaterialArchetype::Cloth.parameters();
    assert!(cloth.fuzz_weight > 0.0);

    let water = MaterialArchetype::Water.parameters();
    assert!((water.index_of_refraction - 1.333).abs() < 1e-6);
}
