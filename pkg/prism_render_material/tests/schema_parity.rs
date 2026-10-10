//! S1 parity gate for the self-describing surface schema (design doc §17.5.8).
//!
//! The schema in `prism_material_schema` reverse-declares the hand-written
//! layout in `surface.rs`. This test asserts that the schema-derived layout is
//! byte-for-byte identical to the authored layout for every lobe subset, so
//! that stage S2 can generate the pack/unpack code from the schema without any
//! layout regression. Any drift between `surface.toml` and `surface.rs` (either
//! side moving) fails here.

use prism_material_schema::{surface_schema, FieldType, SurfaceSchema};
use prism_render_material::{
    GpuSurfaceParameters, LobeMask, SurfaceParameterBlock, SURFACE_CORE_WORDS, SURFACE_LOBE_WORDS,
};

/// The same authored fat parameters used by the `surface.rs` unit tests, so the
/// schema is checked against real packed bytes rather than synthetic data.
fn authored() -> GpuSurfaceParameters {
    GpuSurfaceParameters {
        base_color: [0.1, 0.2, 0.3, 1.0],
        emissive: [2.0, 0.0, 0.0, 1.0],
        metallic: 0.25,
        perceptual_roughness: 0.4,
        reflectance: 0.55,
        ambient_occlusion: 0.9,
        normal_scale: 0.8,
        alpha_cutoff: 0.37,
        transmission: 0.6,
        thickness: 1.5,
        clearcoat: 0.7,
        clearcoat_roughness: 0.15,
        anisotropy: 0.3,
        anisotropy_rotation: 0.2,
        sheen: 0.45,
        subsurface: 0.65,
        index_of_refraction: 1.33,
        dispersion: 0.05,
        face_softness: 0.25,
        _pad_face0: 0.0,
        _pad_face1: 0.0,
        _pad_face2: 0.0,
    }
}

/// Map a schema `(lobe_id, field_name)` to the authored component values. Core
/// fields pass `lobe = None`. This is the human-audited bridge the S1 stage
/// requires; S3 deletes it when the golden mirror is schema-generated.
fn authored_components(lobe: Option<&str>, field: &str, p: &GpuSurfaceParameters) -> Vec<f32> {
    match (lobe, field) {
        (None, "base_color") => p.base_color.to_vec(),
        (None, "metallic") => vec![p.metallic],
        (None, "perceptual_roughness") => vec![p.perceptual_roughness],
        (None, "reflectance") => vec![p.reflectance],
        (None, "ambient_occlusion") => vec![p.ambient_occlusion],
        (None, "normal_scale") => vec![p.normal_scale],
        (None, "alpha_cutoff") => vec![p.alpha_cutoff],
        (Some("emission"), "emissive") => p.emissive.to_vec(),
        (Some("clearcoat"), "clearcoat") => vec![p.clearcoat],
        (Some("clearcoat"), "clearcoat_roughness") => vec![p.clearcoat_roughness],
        (Some("anisotropy"), "anisotropy") => vec![p.anisotropy],
        (Some("anisotropy"), "anisotropy_rotation") => vec![p.anisotropy_rotation],
        (Some("sheen"), "sheen") => vec![p.sheen],
        (Some("subsurface"), "subsurface") => vec![p.subsurface],
        (Some("transmission"), "transmission") => vec![p.transmission],
        (Some("transmission"), "thickness") => vec![p.thickness],
        (Some("transmission"), "index_of_refraction") => vec![p.index_of_refraction],
        (Some("transmission"), "dispersion") => vec![p.dispersion],
        (Some("face"), "softness") => vec![p.face_softness],
        other => panic!("schema field {other:?} has no authored binding"),
    }
}

/// The `LobeMask` bits named by a schema `mask_const` string.
fn mask_const_bits(name: &str) -> u32 {
    match name {
        "EMISSION" => LobeMask::EMISSION.bits(),
        "CLEARCOAT" => LobeMask::CLEARCOAT.bits(),
        "ANISOTROPY" => LobeMask::ANISOTROPY.bits(),
        "SHEEN" => LobeMask::SHEEN.bits(),
        "SUBSURFACE" => LobeMask::SUBSURFACE.bits(),
        "TRANSMISSION" => LobeMask::TRANSMISSION.bits(),
        "FACE" => LobeMask::FACE.bits(),
        other => panic!("unknown mask_const {other}"),
    }
}

fn schema() -> SurfaceSchema {
    surface_schema()
}

#[test]
fn core_and_lobe_word_counts_match_surface_rs() {
    let schema = schema();
    assert_eq!(schema.core.words as usize, SURFACE_CORE_WORDS);
    for lobe in &schema.lobes {
        assert_eq!(
            lobe.words as usize, SURFACE_LOBE_WORDS,
            "lobe {} is not a 4-word quantum",
            lobe.id
        );
    }
    assert_eq!(schema.lobe_count(), LobeMask::COUNT);
}

#[test]
fn registry_slots_match_lobe_mask_constants() {
    let schema = schema();
    for lobe in &schema.lobes {
        assert_eq!(
            1u32 << lobe.registry_slot,
            mask_const_bits(&lobe.mask_const),
            "lobe {} registry_slot/bit mismatch",
            lobe.id
        );
    }
}

#[test]
fn packed_len_matches_for_every_lobe_subset() {
    let schema = schema();
    let params = authored();
    for raw in 0..(1u32 << LobeMask::COUNT) {
        let mask = LobeMask(raw);
        let block = SurfaceParameterBlock::from_full(&params, mask);
        assert_eq!(
            schema.packed_len_words(raw) as usize,
            block.packed_len_words(),
            "packed length mismatch for mask {raw:#09b}",
        );
    }
}

#[test]
fn schema_offsets_reproduce_packed_bytes_for_full_mask() {
    let schema = schema();
    let params = authored();
    let all = LobeMask::all();
    let block = SurfaceParameterBlock::from_full(&params, all);
    let words = block.pack();

    for placement in schema.placements(all.bits()) {
        if placement.is_pad {
            // Padding words must be zero in the packed image.
            for i in 0..placement.words {
                let w = words[(placement.word_offset + i) as usize];
                assert_eq!(w, 0, "pad word {} not zero", placement.word_offset + i);
            }
            continue;
        }
        let expected = authored_components(placement.lobe.as_deref(), &placement.name, &params);
        assert_eq!(
            expected.len(),
            placement.words as usize,
            "field {} component arity mismatch",
            placement.name
        );
        for (i, exp) in expected.iter().enumerate() {
            let w = words[placement.word_offset as usize + i];
            let got = f32::from_bits(w);
            assert_eq!(
                got, *exp,
                "field {:?}.{} word {} schema-derived value mismatch",
                placement.lobe, placement.name, i
            );
        }
    }
}

#[test]
fn schema_defaults_match_neutral_surface() {
    let schema = schema();
    let neutral = GpuSurfaceParameters::default();

    for field in &schema.core.fields {
        if field.ty == FieldType::Pad {
            continue;
        }
        let expected = authored_components(None, &field.name, &neutral);
        assert_eq!(
            field.default.components(),
            expected,
            "core field {} default != neutral surface",
            field.name
        );
    }
    for lobe in &schema.lobes {
        for field in &lobe.fields {
            if field.ty == FieldType::Pad {
                continue;
            }
            let expected = authored_components(Some(&lobe.id), &field.name, &neutral);
            assert_eq!(
                field.default.components(),
                expected,
                "lobe {}.{} default != neutral surface",
                lobe.id,
                field.name
            );
        }
    }
}
