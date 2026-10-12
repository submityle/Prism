//! CPU-only verification: the `#[repr(C)]` layout contract, the byte packing,
//! the single-sourced WGSL compiling under `naga`, and the mirror-scope
//! documentation. The device-dependent parity test lives in
//! `tests/gpu_parity.rs`.

use core::mem::{offset_of, size_of};

use crate::params::{GpuSurfaceParams, BATCH_BYTES, MAX_BATCH, SURFACE_PARAMS_SIZE};
use crate::shader::{SURFACE_SHADER_WGSL, WORKGROUP_SIZE};
use crate::{MirrorScope, MIRROR_SCOPE};

/// The byte offset of every field the WGSL `SurfaceParams` struct declares,
/// asserted against the documented table in `params.rs`.
#[test]
fn repr_c_layout_matches_the_wgsl_offsets() {
    let expected: [(&str, usize); 26] = [
        ("base_color", 0),
        ("base_weight", 12),
        ("specular_color", 16),
        ("diffuse_roughness", 28),
        ("emissive", 32),
        ("metallic", 44),
        ("perceptual_roughness", 48),
        ("specular_weight", 52),
        ("specular_ior", 56),
        ("normal_variance", 60),
        ("anisotropy", 64),
        ("anisotropy_rotation", 68),
        ("reserved0", 72),
        ("reserved1", 76),
        ("normal", 80),
        ("normal_pad", 92),
        ("view", 96),
        ("view_pad", 108),
        ("tangent", 112),
        ("tangent_pad", 124),
        ("bitangent", 128),
        ("bitangent_pad", 140),
        ("direction", 144),
        ("visibility", 156),
        ("illuminance", 160),
        ("illuminance_pad", 172),
    ];

    let actual: [(&str, usize); 26] = [
        ("base_color", offset_of!(GpuSurfaceParams, base_color)),
        ("base_weight", offset_of!(GpuSurfaceParams, base_weight)),
        (
            "specular_color",
            offset_of!(GpuSurfaceParams, specular_color),
        ),
        (
            "diffuse_roughness",
            offset_of!(GpuSurfaceParams, diffuse_roughness),
        ),
        ("emissive", offset_of!(GpuSurfaceParams, emissive)),
        ("metallic", offset_of!(GpuSurfaceParams, metallic)),
        (
            "perceptual_roughness",
            offset_of!(GpuSurfaceParams, perceptual_roughness),
        ),
        (
            "specular_weight",
            offset_of!(GpuSurfaceParams, specular_weight),
        ),
        ("specular_ior", offset_of!(GpuSurfaceParams, specular_ior)),
        (
            "normal_variance",
            offset_of!(GpuSurfaceParams, normal_variance),
        ),
        ("anisotropy", offset_of!(GpuSurfaceParams, anisotropy)),
        (
            "anisotropy_rotation",
            offset_of!(GpuSurfaceParams, anisotropy_rotation),
        ),
        ("reserved0", offset_of!(GpuSurfaceParams, reserved0)),
        ("reserved1", offset_of!(GpuSurfaceParams, reserved1)),
        ("normal", offset_of!(GpuSurfaceParams, normal)),
        ("normal_pad", offset_of!(GpuSurfaceParams, normal_pad)),
        ("view", offset_of!(GpuSurfaceParams, view)),
        ("view_pad", offset_of!(GpuSurfaceParams, view_pad)),
        ("tangent", offset_of!(GpuSurfaceParams, tangent)),
        ("tangent_pad", offset_of!(GpuSurfaceParams, tangent_pad)),
        ("bitangent", offset_of!(GpuSurfaceParams, bitangent)),
        ("bitangent_pad", offset_of!(GpuSurfaceParams, bitangent_pad)),
        ("direction", offset_of!(GpuSurfaceParams, direction)),
        ("visibility", offset_of!(GpuSurfaceParams, visibility)),
        ("illuminance", offset_of!(GpuSurfaceParams, illuminance)),
        (
            "illuminance_pad",
            offset_of!(GpuSurfaceParams, illuminance_pad),
        ),
    ];

    assert_eq!(expected, actual, "the Rust layout drifted from params.rs");
    assert_eq!(size_of::<GpuSurfaceParams>(), SURFACE_PARAMS_SIZE);

    // Every vector member starts on a 16-byte offset, which is what makes the
    // WGSL `vec3<f32>` alignment and the `[f32; 3]` layout agree.
    for (name, offset) in actual {
        if matches!(
            name,
            "base_color"
                | "specular_color"
                | "emissive"
                | "normal"
                | "view"
                | "tangent"
                | "bitangent"
                | "direction"
                | "illuminance"
        ) {
            assert_eq!(offset % 16, 0, "vector field {name} is not 16-byte aligned");
        }
    }
}

/// The struct is a whole number of 16-byte blocks, so a uniform array of it has
/// the 16-byte stride WGSL requires.
#[test]
fn surface_params_fit_a_uniform_array_stride() {
    assert_eq!(SURFACE_PARAMS_SIZE % 16, 0);
    assert_eq!(GpuSurfaceParams::SIZE, SURFACE_PARAMS_SIZE);
    assert_eq!(
        GpuSurfaceParams::FLOATS * size_of::<f32>(),
        SURFACE_PARAMS_SIZE
    );
    assert_eq!(
        BATCH_BYTES,
        u64::from(MAX_BATCH) * SURFACE_PARAMS_SIZE as u64
    );
    // The batch fits inside the default 64 KiB uniform binding limit; that is
    // asserted at compile time in `params.rs`.
}

/// `to_bytes` writes each field little-endian at its documented offset.
#[test]
fn byte_packing_places_fields_at_their_offsets() {
    let params = GpuSurfaceParams {
        base_color: [0.25, 0.5, 0.75],
        base_weight: 0.5,
        specular_ior: 1.5,
        visibility: 0.75,
        ..Default::default()
    };
    let bytes = params.to_bytes();
    assert_eq!(bytes.len(), SURFACE_PARAMS_SIZE);

    let read = |offset: usize| {
        f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    };
    assert_eq!(read(0), 0.25);
    assert_eq!(read(4), 0.5);
    assert_eq!(read(8), 0.75);
    assert_eq!(read(12), 0.5, "base_weight");
    assert_eq!(read(56), 1.5, "specular_ior");
    assert_eq!(read(156), 0.75, "visibility");

    // The packing is the flat float list, in order.
    let floats = params.to_floats();
    assert_eq!(floats.len(), GpuSurfaceParams::FLOATS);
    for (index, value) in floats.iter().enumerate() {
        assert_eq!(read(index * 4), *value, "float {index} mismatch");
    }
}

/// The WGSL parses and validates under `naga`, so a syntax or type error cannot
/// hide behind a sandbox that has no adapter to compile it.
#[test]
fn shader_validates_under_naga() {
    let module = naga::front::wgsl::parse_str(SURFACE_SHADER_WGSL)
        .unwrap_or_else(|err| panic!("surface.wgsl failed to parse: {err:?}"));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .unwrap_or_else(|err| panic!("surface.wgsl failed to validate: {err:?}"));
}

/// The shader's batch capacity and workgroup size match the Rust constants, so
/// the dispatch math and the uniform buffer size cannot drift from the WGSL.
#[test]
fn shader_constants_match_the_rust_side() {
    assert!(
        SURFACE_SHADER_WGSL.contains("const MAX_BATCH: u32 = 128u;"),
        "surface.wgsl must declare MAX_BATCH = 128u to match params::MAX_BATCH",
    );
    assert_eq!(MAX_BATCH, 128);
    assert!(
        SURFACE_SHADER_WGSL.contains(&format!("@workgroup_size({WORKGROUP_SIZE})")),
        "surface.wgsl must declare @workgroup_size({WORKGROUP_SIZE})",
    );
}

/// The mirror-scope string lists every matched term and every excluded lobe, so
/// the documented scope cannot silently drift from the shader.
#[test]
fn mirror_scope_documents_matched_and_unmirrored_terms() {
    assert!(MIRROR_SCOPE.contains(MirrorScope::MATCHED_PREFIX));
    assert!(MIRROR_SCOPE.contains(MirrorScope::UNMIRRORED_PREFIX));
    assert!(MIRROR_SCOPE.contains(MirrorScope::GUARD_PREFIX));
    for term in MirrorScope::MATCHED_TERMS {
        assert!(MIRROR_SCOPE.contains(term), "mirror scope omits {term}");
    }
    for lobe in MirrorScope::UNMIRRORED_LOBES {
        assert!(
            MIRROR_SCOPE.contains(lobe),
            "mirror scope omits the excluded lobe {lobe}",
        );
    }
    for guard in MirrorScope::OMITTED_GUARDS {
        assert!(
            MIRROR_SCOPE.contains(guard),
            "mirror scope omits the guard note {guard}",
        );
    }
    // The excluded-lobe list is the OpenPBR set the brief names.
    for required in ["coat", "fuzz", "transmission", "subsurface"] {
        assert!(MirrorScope::UNMIRRORED_LOBES.contains(&required));
    }
}

/// The default surface is the neutral `OpenPBR` surface the CPU golden renders
/// for a missing material.
#[test]
fn default_matches_the_cpu_surface_defaults() {
    let params = GpuSurfaceParams::default();
    assert_eq!(params.base_color, [0.8; 3]);
    assert_eq!(params.base_weight, 1.0);
    assert_eq!(params.diffuse_roughness, 0.0);
    assert_eq!(params.metallic, 0.0);
    assert_eq!(params.perceptual_roughness, 0.3);
    assert_eq!(params.specular_weight, 1.0);
    assert_eq!(params.specular_color, [1.0; 3]);
    assert_eq!(params.specular_ior, 1.5);
    assert_eq!(params.normal_variance, 0.0);
    assert_eq!(params.anisotropy, 0.0);
    assert_eq!(params.anisotropy_rotation, 0.0);
    assert_eq!(params.emissive, [0.0; 3]);
    assert_eq!(params.normal, [0.0, 1.0, 0.0]);
    assert_eq!(params.view, [0.0, 1.0, 0.0]);
    assert_eq!(params.tangent, [1.0, 0.0, 0.0]);
    assert_eq!(params.bitangent, [0.0, 0.0, -1.0]);
    assert_eq!(params.direction, [0.0, 1.0, 0.0]);
    assert_eq!(params.illuminance, [1.0; 3]);
    assert_eq!(params.visibility, 1.0);
}
