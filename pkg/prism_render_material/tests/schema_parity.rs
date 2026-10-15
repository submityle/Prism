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
    openpbr_conformance, project_to_openpbr, GpuAnisotropyLobe, GpuClearCoatLobe, GpuEmissionLobe,
    GpuGlintLobe, GpuMeasuredBrdfLobe, GpuNprFaceLobe, GpuSheenLobe, GpuSubsurfaceLobe,
    GpuSurfaceCore, GpuSurfaceParameters, GpuThinFilmLobe, GpuTransmissionLobe, LobeMask,
    LobeRegion, OpenPbrConformance, SurfaceParameterBlock, MATERIAL_ABI_VERSION,
    SURFACE_CORE_WORDS, SURFACE_LOBES, SURFACE_LOBE_WORDS, SURFACE_LOBE_WORDS_BY_SLOT,
};

#[test]
fn surface_schema_tracks_the_runtime_material_abi_version() {
    assert_eq!(surface_schema().abi_version, MATERIAL_ABI_VERSION);
}

/// The same authored fat parameters used by the `surface.rs` unit tests, so the
/// schema is checked against real packed bytes rather than synthetic data.
fn authored() -> GpuSurfaceParameters {
    GpuSurfaceParameters {
        base_color: [0.1, 0.2, 0.3, 1.0],
        emission_color: [2.0, 0.0, 0.0, 1.0],
        emission_luminance: 3.5,
        base_metalness: 0.25,
        specular_roughness: 0.4,
        specular_weight: 0.65,
        ambient_occlusion: 0.9,
        normal_scale: 0.8,
        alpha_cutoff: 0.37,
        base_weight: 0.85,
        base_diffuse_roughness: 0.4,
        specular_color: [0.95, 0.97, 1.0, 1.0],
        specular_ior: 1.7,
        thin_walled: 0.0,
        geometry_opacity: 0.9,
        transmission_weight: 0.6,
        thickness: 1.5,
        coat_weight: 0.7,
        coat_roughness: 0.15,
        coat_ior: 1.45,
        coat_color: [0.9, 0.95, 1.0, 1.0],
        coat_roughness_anisotropy: 0.35,
        coat_darkening: 0.6,
        anisotropy: 0.3,
        anisotropy_rotation: 0.2,
        fuzz_weight: 0.45,
        fuzz_roughness: 0.3,
        fuzz_color: [0.8, 0.85, 0.9, 1.0],
        subsurface_weight: 0.65,
        subsurface_color: [0.9, 0.5, 0.4, 1.0],
        subsurface_radius: 0.8,
        subsurface_radius_scale: [1.0, 0.6, 0.4, 1.0],
        subsurface_scatter_anisotropy: 0.7,
        index_of_refraction: 1.33,
        transmission_dispersion_scale: 0.05,
        transmission_color: [0.85, 0.95, 1.0, 1.0],
        transmission_scatter: [0.25, 0.15, 0.05, 1.0],
        transmission_scatter_anisotropy: -0.4,
        transmission_dispersion_abbe_number: 35.0,
        thin_film_weight: 0.7,
        thin_film_thickness: 0.35,
        thin_film_ior: 1.55,
        npr_face_softness: 0.25,
        npr_stylized_ramp_softness: 1.25,
        npr_stylized_specular_scale: 0.75,
        npr_stylized_rim_scale: 1.5,
        glint_weight: 0.6,
        glint_roughness: 0.22,
        glint_density: 12.0,
        glint_bin_half_width: 0.031,
        measured_weight: 0.45,
        measured_phi_d: 128.0,
        measured_scale: 2.5,
    }
}

/// Map a schema `(lobe_id, field_name)` to the authored component values. Core
/// fields pass `lobe = None`. This is the human-audited bridge the S1 stage
/// requires; S3 deletes it when the golden mirror is schema-generated.
fn authored_components(lobe: Option<&str>, field: &str, p: &GpuSurfaceParameters) -> Vec<f32> {
    match (lobe, field) {
        (None, "base_color") => p.base_color.to_vec(),
        (None, "base_metalness") => vec![p.base_metalness],
        (None, "specular_roughness") => vec![p.specular_roughness],
        (None, "specular_weight") => vec![p.specular_weight],
        (None, "specular_color") => p.specular_color.to_vec(),
        (None, "specular_ior") => vec![p.specular_ior],
        (None, "geometry_thin_walled") => vec![p.thin_walled],
        (None, "geometry_opacity") => vec![p.geometry_opacity],
        (None, "ambient_occlusion") => vec![p.ambient_occlusion],
        (None, "normal_scale") => vec![p.normal_scale],
        (None, "alpha_cutoff") => vec![p.alpha_cutoff],
        (None, "base_weight") => vec![p.base_weight],
        (None, "base_diffuse_roughness") => vec![p.base_diffuse_roughness],
        (Some("emission"), "emission_color") => p.emission_color.to_vec(),
        (Some("emission"), "emission_luminance") => vec![p.emission_luminance],
        (Some("clearcoat"), "coat_weight") => vec![p.coat_weight],
        (Some("clearcoat"), "coat_roughness") => vec![p.coat_roughness],
        (Some("clearcoat"), "coat_ior") => vec![p.coat_ior],
        (Some("clearcoat"), "coat_color") => p.coat_color.to_vec(),
        (Some("clearcoat"), "coat_roughness_anisotropy") => vec![p.coat_roughness_anisotropy],
        (Some("clearcoat"), "coat_darkening") => vec![p.coat_darkening],
        (Some("anisotropy"), "specular_roughness_anisotropy") => vec![p.anisotropy],
        (Some("anisotropy"), "anisotropy_rotation") => vec![p.anisotropy_rotation],
        (Some("sheen"), "fuzz_weight") => vec![p.fuzz_weight],
        (Some("sheen"), "fuzz_roughness") => vec![p.fuzz_roughness],
        (Some("sheen"), "fuzz_color") => p.fuzz_color.to_vec(),
        (Some("subsurface"), "subsurface_weight") => vec![p.subsurface_weight],
        (Some("subsurface"), "subsurface_color") => p.subsurface_color.to_vec(),
        (Some("subsurface"), "subsurface_radius") => vec![p.subsurface_radius],
        (Some("subsurface"), "subsurface_radius_scale") => p.subsurface_radius_scale.to_vec(),
        (Some("subsurface"), "subsurface_scatter_anisotropy") => {
            vec![p.subsurface_scatter_anisotropy]
        }
        (Some("transmission"), "transmission_weight") => vec![p.transmission_weight],
        (Some("transmission"), "transmission_depth") => vec![p.thickness],
        (Some("transmission"), "index_of_refraction") => vec![p.index_of_refraction],
        (Some("transmission"), "transmission_dispersion_scale") => {
            vec![p.transmission_dispersion_scale]
        }
        (Some("transmission"), "transmission_color") => p.transmission_color.to_vec(),
        (Some("transmission"), "transmission_scatter") => p.transmission_scatter.to_vec(),
        (Some("transmission"), "transmission_scatter_anisotropy") => {
            vec![p.transmission_scatter_anisotropy]
        }
        (Some("transmission"), "transmission_dispersion_abbe_number") => {
            vec![p.transmission_dispersion_abbe_number]
        }
        (Some("thin_film"), "thin_film_weight") => vec![p.thin_film_weight],
        (Some("thin_film"), "thin_film_thickness") => vec![p.thin_film_thickness],
        (Some("thin_film"), "thin_film_ior") => vec![p.thin_film_ior],
        (Some("npr_face"), "softness") => vec![p.npr_face_softness],
        (Some("npr_face"), "ramp_softness") => vec![p.npr_stylized_ramp_softness],
        (Some("npr_face"), "specular_scale") => vec![p.npr_stylized_specular_scale],
        (Some("npr_face"), "rim_scale") => vec![p.npr_stylized_rim_scale],
        (Some("glint"), "glint_weight") => vec![p.glint_weight],
        (Some("glint"), "glint_roughness") => vec![p.glint_roughness],
        (Some("glint"), "glint_density") => vec![p.glint_density],
        (Some("glint"), "glint_bin_half_width") => vec![p.glint_bin_half_width],
        (Some("measured_brdf"), "measured_weight") => vec![p.measured_weight],
        (Some("measured_brdf"), "measured_phi_d") => vec![p.measured_phi_d],
        (Some("measured_brdf"), "measured_scale") => vec![p.measured_scale],
        other => panic!("schema field {other:?} has no authored binding"),
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
            lobe.words as usize, SURFACE_LOBE_WORDS_BY_SLOT[lobe.registry_slot as usize],
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
    for (schema_lobe, generated_lobe) in schema.canonical_lobes().into_iter().zip(SURFACE_LOBES) {
        assert_eq!(
            1u32 << schema_lobe.registry_slot,
            generated_lobe.mask_bits,
            "lobe {} registry_slot/generated bit mismatch",
            schema_lobe.id
        );
        assert_eq!(
            generated_lobe.registry_slot, schema_lobe.registry_slot,
            "lobe {} registry slot mismatch",
            schema_lobe.id
        );
        assert_eq!(
            generated_lobe.words as u32, schema_lobe.words,
            "lobe {} word footprint mismatch",
            schema_lobe.id
        );
    }
}

#[test]
fn generated_surface_fields_match_schema_layout_metadata() {
    let schema = schema();
    let expected = schema
        .core
        .fields
        .iter()
        .filter(|field| field.ty != FieldType::Pad)
        .map(|field| {
            (
                None,
                field.name.as_str(),
                field.pack_slot,
                field.ty.words(),
                field.bit_region.as_str(),
            )
        })
        .chain(schema.canonical_lobes().into_iter().flat_map(|lobe| {
            lobe.fields
                .iter()
                .filter(|field| field.ty != FieldType::Pad)
                .map(move |field| {
                    (
                        Some(lobe.id.as_str()),
                        field.name.as_str(),
                        field.pack_slot,
                        field.ty.words(),
                        field.bit_region.as_str(),
                    )
                })
        }))
        .collect::<Vec<_>>();
    let actual = prism_render_material::SURFACE_FIELDS
        .iter()
        .map(|field| {
            (
                field.lobe,
                field.name,
                field.pack_slot,
                field.words,
                field.bit_region,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

fn typed_field_offset(lobe: Option<&str>, name: &str) -> (usize, usize) {
    macro_rules! field {
        ($ty:ty, $member:ident) => {
            (
                std::mem::offset_of!($ty, $member),
                std::mem::size_of::<$ty>(),
            )
        };
    }

    match (lobe, name) {
        (None, "base_color") => field!(GpuSurfaceCore, base_color),
        (None, "base_metalness") => field!(GpuSurfaceCore, base_metalness),
        (None, "specular_roughness") => field!(GpuSurfaceCore, specular_roughness),
        (None, "specular_weight") => field!(GpuSurfaceCore, specular_weight),
        (None, "ambient_occlusion") => field!(GpuSurfaceCore, ambient_occlusion),
        (None, "normal_scale") => field!(GpuSurfaceCore, normal_scale),
        (None, "alpha_cutoff") => field!(GpuSurfaceCore, alpha_cutoff),
        (None, "base_weight") => field!(GpuSurfaceCore, base_weight),
        (None, "base_diffuse_roughness") => field!(GpuSurfaceCore, base_diffuse_roughness),
        (None, "specular_color") => field!(GpuSurfaceCore, specular_color),
        (None, "specular_ior") => field!(GpuSurfaceCore, specular_ior),
        (None, "geometry_thin_walled") => field!(GpuSurfaceCore, thin_walled),
        (None, "geometry_opacity") => field!(GpuSurfaceCore, geometry_opacity),
        (Some("emission"), "emission_color") => field!(GpuEmissionLobe, emission_color),
        (Some("emission"), "emission_luminance") => field!(GpuEmissionLobe, emission_luminance),
        (Some("clearcoat"), "coat_weight") => field!(GpuClearCoatLobe, coat_weight),
        (Some("clearcoat"), "coat_roughness") => field!(GpuClearCoatLobe, coat_roughness),
        (Some("clearcoat"), "coat_ior") => field!(GpuClearCoatLobe, coat_ior),
        (Some("clearcoat"), "coat_color") => field!(GpuClearCoatLobe, coat_color),
        (Some("clearcoat"), "coat_roughness_anisotropy") => {
            field!(GpuClearCoatLobe, coat_roughness_anisotropy)
        }
        (Some("clearcoat"), "coat_darkening") => field!(GpuClearCoatLobe, coat_darkening),
        (Some("anisotropy"), "specular_roughness_anisotropy") => {
            field!(GpuAnisotropyLobe, anisotropy)
        }
        (Some("anisotropy"), "anisotropy_rotation") => {
            field!(GpuAnisotropyLobe, anisotropy_rotation)
        }
        (Some("sheen"), "fuzz_weight") => field!(GpuSheenLobe, fuzz_weight),
        (Some("sheen"), "fuzz_roughness") => field!(GpuSheenLobe, fuzz_roughness),
        (Some("sheen"), "fuzz_color") => field!(GpuSheenLobe, fuzz_color),
        (Some("subsurface"), "subsurface_weight") => field!(GpuSubsurfaceLobe, subsurface_weight),
        (Some("subsurface"), "subsurface_radius") => field!(GpuSubsurfaceLobe, subsurface_radius),
        (Some("subsurface"), "subsurface_color") => field!(GpuSubsurfaceLobe, subsurface_color),
        (Some("subsurface"), "subsurface_radius_scale") => {
            field!(GpuSubsurfaceLobe, subsurface_radius_scale)
        }
        (Some("subsurface"), "subsurface_scatter_anisotropy") => {
            field!(GpuSubsurfaceLobe, subsurface_scatter_anisotropy)
        }
        (Some("transmission"), "transmission_weight") => {
            field!(GpuTransmissionLobe, transmission_weight)
        }
        (Some("transmission"), "transmission_depth") => field!(GpuTransmissionLobe, thickness),
        (Some("transmission"), "index_of_refraction") => {
            field!(GpuTransmissionLobe, index_of_refraction)
        }
        (Some("transmission"), "transmission_dispersion_scale") => {
            field!(GpuTransmissionLobe, transmission_dispersion_scale)
        }
        (Some("transmission"), "transmission_color") => {
            field!(GpuTransmissionLobe, transmission_color)
        }
        (Some("transmission"), "transmission_scatter") => {
            field!(GpuTransmissionLobe, transmission_scatter)
        }
        (Some("transmission"), "transmission_scatter_anisotropy") => {
            field!(GpuTransmissionLobe, transmission_scatter_anisotropy)
        }
        (Some("transmission"), "transmission_dispersion_abbe_number") => {
            field!(GpuTransmissionLobe, transmission_dispersion_abbe_number)
        }
        (Some("thin_film"), "thin_film_weight") => field!(GpuThinFilmLobe, thin_film_weight),
        (Some("thin_film"), "thin_film_thickness") => {
            field!(GpuThinFilmLobe, thin_film_thickness)
        }
        (Some("thin_film"), "thin_film_ior") => field!(GpuThinFilmLobe, thin_film_ior),
        (Some("npr_face"), "softness") => field!(GpuNprFaceLobe, softness),
        (Some("npr_face"), "ramp_softness") => field!(GpuNprFaceLobe, ramp_softness),
        (Some("npr_face"), "specular_scale") => field!(GpuNprFaceLobe, specular_scale),
        (Some("npr_face"), "rim_scale") => field!(GpuNprFaceLobe, rim_scale),
        (Some("glint"), "glint_weight") => field!(GpuGlintLobe, glint_weight),
        (Some("glint"), "glint_roughness") => field!(GpuGlintLobe, glint_roughness),
        (Some("glint"), "glint_density") => field!(GpuGlintLobe, glint_density),
        (Some("glint"), "glint_bin_half_width") => field!(GpuGlintLobe, glint_bin_half_width),
        (Some("measured_brdf"), "measured_weight") => {
            field!(GpuMeasuredBrdfLobe, measured_weight)
        }
        (Some("measured_brdf"), "measured_phi_d") => field!(GpuMeasuredBrdfLobe, measured_phi_d),
        (Some("measured_brdf"), "measured_scale") => field!(GpuMeasuredBrdfLobe, measured_scale),
        other => panic!("schema field {other:?} has no typed ABI member"),
    }
}

#[test]
fn typed_surface_struct_offsets_match_schema_pack_slots() {
    let schema = schema();
    for field in schema
        .core
        .fields
        .iter()
        .filter(|field| field.ty != FieldType::Pad)
    {
        let (offset, size) = typed_field_offset(None, &field.name);
        assert_eq!(
            offset / std::mem::size_of::<u32>(),
            field.pack_slot as usize
        );
        assert!(offset + field.ty.words() as usize * std::mem::size_of::<u32>() <= size);
    }
    for lobe in schema.canonical_lobes() {
        for field in lobe
            .fields
            .iter()
            .filter(|field| field.ty != FieldType::Pad)
        {
            let (offset, size) = typed_field_offset(Some(&lobe.id), &field.name);
            assert_eq!(
                offset / std::mem::size_of::<u32>(),
                field.pack_slot as usize,
                "{}.{}/typed offset",
                lobe.id,
                field.name
            );
            assert!(
                offset + field.ty.words() as usize * std::mem::size_of::<u32>() <= size,
                "{}.{}/typed field exceeds struct",
                lobe.id,
                field.name
            );
        }
    }
}

#[test]
fn packed_len_matches_for_every_lobe_subset() {
    let schema = schema();
    let params = authored();
    for subset in 0..(1u32 << LobeMask::COUNT) {
        let raw = SURFACE_LOBES
            .iter()
            .enumerate()
            .fold(0u32, |mask, (index, lobe)| {
                if subset & (1 << index) != 0 {
                    mask | (1 << lobe.registry_slot)
                } else {
                    mask
                }
            });
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
        assert_eq!(
            projection.dropped.len(),
            1,
            "lobe {:?} not reported",
            lobe.id
        );
        assert_eq!(projection.dropped[0].uid, lobe.uid);
        assert!(!projection.is_lossless());
    }
}
