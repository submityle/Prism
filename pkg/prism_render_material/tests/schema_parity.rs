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
    openpbr_conformance, project_to_openpbr, GpuSurfaceParameters, LobeMask, LobeRegion,
    OpenPbrConformance, SurfaceParameterBlock, SURFACE_CORE_WORDS, SURFACE_LOBE_WORDS,
    SURFACE_LOBE_WORDS_BY_SLOT,
};

/// The same authored fat parameters used by the `surface.rs` unit tests, so the
/// schema is checked against real packed bytes rather than synthetic data.
fn authored() -> GpuSurfaceParameters {
    GpuSurfaceParameters {
        base_color: [0.1, 0.2, 0.3, 1.0],
        emissive: [2.0, 0.0, 0.0, 1.0],
        metallic: 0.25,
        perceptual_roughness: 0.4,
        specular_weight: 0.65,
        ambient_occlusion: 0.9,
        normal_scale: 0.8,
        alpha_cutoff: 0.37,
        base_weight: 0.85,
        diffuse_roughness: 0.4,
        specular_color: [0.95, 0.97, 1.0, 1.0],
        specular_ior: 1.7,
        thin_walled: 0.0,
        transmission: 0.6,
        thickness: 1.5,
        clearcoat: 0.7,
        clearcoat_roughness: 0.15,
        coat_ior: 1.45,
        coat_color: [0.9, 0.95, 1.0, 1.0],
        coat_roughness_anisotropy: 0.35,
        anisotropy: 0.3,
        anisotropy_rotation: 0.2,
        sheen: 0.45,
        fuzz_roughness: 0.3,
        fuzz_color: [0.8, 0.85, 0.9, 1.0],
        subsurface: 0.65,
        subsurface_color: [0.9, 0.5, 0.4, 1.0],
        subsurface_radius: [1.0, 0.6, 0.4, 1.0],
        index_of_refraction: 1.33,
        dispersion: 0.05,
        transmission_color: [0.85, 0.95, 1.0, 1.0],
        face_softness: 0.25,
        _pad_face0: 0.0,
        _pad_face1: 0.0,
        _pad_face2: 0.0,
        _pad_tail: 0.0,
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
        (None, "specular_weight") => vec![p.specular_weight],
        (None, "specular_color") => p.specular_color.to_vec(),
        (None, "specular_ior") => vec![p.specular_ior],
        (None, "thin_walled") => vec![p.thin_walled],
        (None, "ambient_occlusion") => vec![p.ambient_occlusion],
        (None, "normal_scale") => vec![p.normal_scale],
        (None, "alpha_cutoff") => vec![p.alpha_cutoff],
        (None, "base_weight") => vec![p.base_weight],
        (None, "diffuse_roughness") => vec![p.diffuse_roughness],
        (Some("emission"), "emissive") => p.emissive.to_vec(),
        (Some("clearcoat"), "clearcoat") => vec![p.clearcoat],
        (Some("clearcoat"), "clearcoat_roughness") => vec![p.clearcoat_roughness],
        (Some("clearcoat"), "coat_ior") => vec![p.coat_ior],
        (Some("clearcoat"), "coat_color") => p.coat_color.to_vec(),
        (Some("clearcoat"), "coat_roughness_anisotropy") => vec![p.coat_roughness_anisotropy],
        (Some("anisotropy"), "anisotropy") => vec![p.anisotropy],
        (Some("anisotropy"), "anisotropy_rotation") => vec![p.anisotropy_rotation],
        (Some("sheen"), "sheen") => vec![p.sheen],
        (Some("sheen"), "fuzz_roughness") => vec![p.fuzz_roughness],
        (Some("sheen"), "fuzz_color") => p.fuzz_color.to_vec(),
        (Some("subsurface"), "subsurface") => vec![p.subsurface],
        (Some("subsurface"), "subsurface_color") => p.subsurface_color.to_vec(),
        (Some("subsurface"), "subsurface_radius") => p.subsurface_radius.to_vec(),
        (Some("transmission"), "transmission") => vec![p.transmission],
        (Some("transmission"), "thickness") => vec![p.thickness],
        (Some("transmission"), "index_of_refraction") => vec![p.index_of_refraction],
        (Some("transmission"), "dispersion") => vec![p.dispersion],
        (Some("transmission"), "transmission_color") => p.transmission_color.to_vec(),
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
        // Per-slot footprint is the packing stride (variable length, §4.4); the
        // schema's per-lobe `words` must match it exactly.
        assert_eq!(
            lobe.words as usize,
            SURFACE_LOBE_WORDS_BY_SLOT[lobe.registry_slot as usize],
            "lobe {} footprint disagrees with the generated per-slot table",
            lobe.id
        );
        assert!(
            lobe.words as usize <= SURFACE_LOBE_WORDS,
            "lobe {} exceeds the widest quantum",
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

/// §4.5 constraint 2 (bit-region isolation) parity gate, end to end from the
/// schema: every lobe's declared `namespace` must place its bit in the matching
/// `LobeMask` region, the two regions must exactly tile the defined lobes, and
/// the §4.5 constraint 1 conformance gate must hold for every schema-derived
/// pure-OpenPBR mask. This is the machine-checkable form of the "Prism über is
/// a real OpenPBR superset" contract — drift between the schema, the generated
/// lobe table, and the hand-written regions fails here.
#[test]
fn schema_namespaces_match_lobe_mask_regions() {
    let schema = surface_schema();

    let mut khr = LobeMask::default();
    let mut extension = LobeMask::default();
    for lobe in &schema.lobes {
        let bit = LobeMask(1 << lobe.registry_slot);
        let core = lobe.namespace.is_openpbr_core();
        assert_eq!(
            LobeMask::region_of(bit),
            Some(if core {
                LobeRegion::Canonical
            } else {
                LobeRegion::Extension
            }),
            "schema lobe {:?} namespace {:?} disagrees with its region",
            lobe.id,
            lobe.namespace
        );
        if core {
            khr = khr.union(bit);
        } else {
            extension = extension.union(bit);
        }
    }

    assert_eq!(LobeMask::KHR_REGION, khr);
    assert_eq!(LobeMask::EXTENSION_REGION, extension);
    assert_eq!(
        LobeMask::KHR_REGION.union(LobeMask::EXTENSION_REGION),
        LobeMask::all(),
        "regions must tile every defined lobe exactly"
    );
    assert_eq!(
        LobeMask::KHR_REGION.bits() & LobeMask::EXTENSION_REGION.bits(),
        0,
        "regions must be disjoint"
    );

    // Constraint 1, end to end: every pure-OpenPBR lobe subset is a lossless
    // fixed point of the downgrade projection.
    let authored = authored();
    for raw in 0..=khr.bits() {
        let mask = LobeMask(raw & khr.bits());
        let block = SurfaceParameterBlock::from_full(&authored, mask);
        assert_eq!(openpbr_conformance(&block), OpenPbrConformance::Conformant);
        let projection = project_to_openpbr(&block);
        assert!(projection.is_lossless());
        assert_eq!(projection.block.pack(), block.pack());
    }

    // Every extension lobe must be reported (never silently dropped).
    for lobe in &schema.lobes {
        if lobe.namespace.is_openpbr_core() {
            continue;
        }
        let bit = LobeMask(1 << lobe.registry_slot);
        let block = SurfaceParameterBlock::from_full(&authored, bit);
        let projection = project_to_openpbr(&block);
        assert_eq!(projection.dropped.len(), 1, "lobe {:?} not reported", lobe.id);
        assert_eq!(projection.dropped[0].uid, lobe.uid);
        assert!(!projection.is_lossless());
    }
}
