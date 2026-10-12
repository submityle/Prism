//! OpenPBR Surface importer for external materials (design doc §4.4).
//!
//! External materials — Bevy `StandardMaterial` today, glTF/`KHR_*` material
//! extensions, MaterialX and USD importers later — are mapped through ONE
//! format adapter onto the canonical OpenPBR Surface parameter block
//! ([`import_openpbr_surface`]). The packaging step ([`import_standard_material`])
//! is format-agnostic, so there is no per-format 1:1 translation layer.

use bevy_asset::{AssetId, Handle};
use bevy_color::ColorToComponents;
use bevy_image::Image;
use bevy_material::AlphaMode;
use bevy_pbr::StandardMaterial;
use prism_render_architecture::abi::GenerationalHandle;

use crate::{
    GpuMaterialTexture, GpuSurfaceParameters, Illumination, MaterialDomain, MaterialFeatureFlags,
    MaterialRecord, MaterialRenderClass,
};

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextureSemantic {
    BaseColor,
    Emissive,
    MetallicRoughness,
    Normal,
    Occlusion,
    ClearCoat,
    ClearCoatRoughness,
    ClearCoatNormal,
    /// Baked signed-distance-field face-shadow map (stylized NPR). Sampled by
    /// `shading_resolve.wesl::shade_toon` with a light-azimuth-driven U mirror
    /// (`SEMANTIC_FACE_SDF` in `material_sample.wesl`). Its `as u32`
    /// discriminant is the shared [`crate::FACE_SHADOW_SDF_SEMANTIC`].
    FaceShadowSdf,
}

// The enum discriminants are the ABI semantic bytes; keep `FaceShadowSdf`
// pinned to the crate-wide constant its non-bevy producer and every shader
// consumer share.
const _: () = assert!(TextureSemantic::FaceShadowSdf as u32 == crate::FACE_SHADOW_SDF_SEMANTIC);

pub trait StandardMaterialTextureResolver {
    fn resolve(&mut self, image: AssetId<Image>, semantic: TextureSemantic) -> GpuMaterialTexture;
}

/// Import an external `StandardMaterial` (Bevy / glTF / KHR material extensions)
/// into the canonical **OpenPBR Surface** parameter set (design doc §4.4).
///
/// This is the single format-adapter step: every external material — Bevy
/// `StandardMaterial` today, glTF/MaterialX/USD importers later — is mapped onto
/// the same über parameter block, and the record plumbing below is
/// format-agnostic. There is deliberately no per-format 1:1 translation layer.
///
/// `alpha_cutoff` is applied to the geometry group's cutout threshold.
#[must_use]
pub fn import_openpbr_surface(
    material: &StandardMaterial,
    alpha_cutoff: f32,
) -> GpuSurfaceParameters {
    GpuSurfaceParameters {
        base_color: material.base_color.to_linear().to_f32_array(),
        emissive: material.emissive.to_f32_array(),
        metallic: material.metallic,
        perceptual_roughness: material.perceptual_roughness,
        // `OpenPBR` specular group. Bevy's `reflectance` is the
        // `KHR_materials_specular`-style `F0 = 0.16*r^2` remap, so invert it
        // back to a physical IOR (`ior = (1+sqrt(F0))/(1-sqrt(F0))`) and the
        // authored dielectric F0 is preserved exactly: `reflectance = 0.5`
        // lands on the `ior = 1.5` default (both `F0 = 0.04`). Bevy's own `ior`
        // drives refraction through the transmission lobe.
        specular_ior: ior_from_f0(0.16 * material.reflectance * material.reflectance),
        ambient_occlusion: 1.0,
        normal_scale: 1.0,
        alpha_cutoff,
        transmission: material
            .specular_transmission
            .max(material.diffuse_transmission),
        thickness: material.thickness,
        clearcoat: material.clearcoat,
        clearcoat_roughness: material.clearcoat_perceptual_roughness,
        anisotropy: material.anisotropy_strength,
        anisotropy_rotation: material.anisotropy_rotation,
        index_of_refraction: material.ior,
        ..Default::default()
    }
}

/// Import an external `StandardMaterial` into a runtime [`MaterialRecord`].
///
/// The OpenPBR Surface parameters come from [`import_openpbr_surface`]; this
/// function only adds the format-independent packaging (render class, features,
/// closure mask, texture rows, illumination axis). It replaces the former
/// per-format `lower_standard_material` translator (design doc §4.4).
pub fn import_standard_material(
    handle: GenerationalHandle,
    revision: u32,
    material: &StandardMaterial,
    textures: &mut impl StandardMaterialTextureResolver,
) -> MaterialRecord {
    let mut features = MaterialFeatureFlags::default();
    if material.double_sided {
        features |= MaterialFeatureFlags::DOUBLE_SIDED;
    }
    if material.normal_map_texture.is_some() {
        features |= MaterialFeatureFlags::NORMAL_MAP;
    }
    if material.emissive != bevy_color::LinearRgba::BLACK {
        features |= MaterialFeatureFlags::EMISSIVE;
    }
    if material.specular_transmission > 0.0 || material.diffuse_transmission > 0.0 {
        features |= MaterialFeatureFlags::TRANSMISSION;
    }
    let (render_class, alpha_cutoff) = match material.alpha_mode {
        AlphaMode::Opaque => (opaque_class(material.double_sided), 0.5),
        AlphaMode::Mask(cutoff) => {
            features |= MaterialFeatureFlags::ALPHA_MASK;
            (
                if material.double_sided {
                    MaterialRenderClass::MaskedTwoSided
                } else {
                    MaterialRenderClass::Masked
                },
                cutoff,
            )
        }
        AlphaMode::AlphaToCoverage => {
            features |= MaterialFeatureFlags::ALPHA_MASK;
            (
                if material.double_sided {
                    MaterialRenderClass::MaskedTwoSided
                } else {
                    MaterialRenderClass::Masked
                },
                0.5,
            )
        }
        AlphaMode::Add => (MaterialRenderClass::Additive, 0.0),
        AlphaMode::Blend | AlphaMode::Premultiplied | AlphaMode::Multiply => {
            (MaterialRenderClass::Transparent, 0.0)
        }
    };
    let render_class = if features.contains(MaterialFeatureFlags::TRANSMISSION) {
        MaterialRenderClass::Transmissive
    } else {
        render_class
    };
    let mut texture_rows = Vec::new();
    push_texture(
        &mut texture_rows,
        &material.base_color_texture,
        TextureSemantic::BaseColor,
        textures,
    );
    push_texture(
        &mut texture_rows,
        &material.emissive_texture,
        TextureSemantic::Emissive,
        textures,
    );
    push_texture(
        &mut texture_rows,
        &material.metallic_roughness_texture,
        TextureSemantic::MetallicRoughness,
        textures,
    );
    push_texture(
        &mut texture_rows,
        &material.normal_map_texture,
        TextureSemantic::Normal,
        textures,
    );
    push_texture(
        &mut texture_rows,
        &material.occlusion_texture,
        TextureSemantic::Occlusion,
        textures,
    );
    MaterialRecord {
        handle,
        revision,
        domain: MaterialDomain::Surface,
        render_class,
        // Style axis is orthogonal to the closure graph: clearcoat is a lobe
        // (carried by `closure_mask`), not an illumination.
        illumination: if material.unlit {
            Illumination::Unlit
        } else {
            Illumination::Lit
        },
        features,
        closure_mask: closure_mask(material),
        surface: import_openpbr_surface(material, alpha_cutoff),
        textures: texture_rows,
        custom_program: None,
    }
}

fn opaque_class(two_sided: bool) -> MaterialRenderClass {
    if two_sided {
        MaterialRenderClass::OpaqueTwoSided
    } else {
        MaterialRenderClass::Opaque
    }
}

/// Inverse of the dielectric `F0 = ((n-1)/(n+1))^2` remap:
/// `n = (1 + sqrt(F0)) / (1 - sqrt(F0))`, clamped to the `OpenPBR`
/// `specular_ior` range `[1, 3]`.
///
/// Used by [`import_openpbr_surface`] to fold Bevy's `reflectance` (which is
/// itself the `0.16 * r^2` KHR-style F0 remap) into a physical IOR without
/// changing the authored Fresnel.
fn ior_from_f0(f0: f32) -> f32 {
    let root = f0.clamp(0.0, 0.96).sqrt();
    ((1.0 + root) / (1.0 - root)).clamp(1.0, 3.0)
}

fn closure_mask(material: &StandardMaterial) -> u32 {
    1 | (u32::from(material.clearcoat > 0.0) << 3)
        | (u32::from(material.specular_transmission > 0.0) << 6)
        | (u32::from(material.emissive != bevy_color::LinearRgba::BLACK) << 7)
}
fn push_texture(
    rows: &mut Vec<GpuMaterialTexture>,
    image: &Option<Handle<Image>>,
    semantic: TextureSemantic,
    resolver: &mut impl StandardMaterialTextureResolver,
) {
    if let Some(image) = image {
        rows.push(resolver.resolve(image.id(), semantic));
    }
}
