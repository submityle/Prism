use bevy_asset::{AssetId, Handle};
use bevy_color::ColorToComponents;
use bevy_image::Image;
use bevy_material::AlphaMode;
use bevy_pbr::StandardMaterial;
use prism_render_architecture::abi::GenerationalHandle;

use crate::{
    GpuMaterialTexture, GpuSurfaceParameters, MaterialDomain, MaterialFeatureFlags, MaterialRecord,
    MaterialRenderClass, MaterialShadingModel,
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
}

pub trait StandardMaterialTextureResolver {
    fn resolve(&mut self, image: AssetId<Image>, semantic: TextureSemantic) -> GpuMaterialTexture;
}

pub fn lower_standard_material(
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
        shading_model: if material.unlit {
            MaterialShadingModel::Unlit
        } else if material.clearcoat > 0.0 {
            MaterialShadingModel::ClearCoat
        } else {
            MaterialShadingModel::Principled
        },
        features,
        closure_mask: closure_mask(material),
        surface: GpuSurfaceParameters {
            base_color: material.base_color.to_linear().to_f32_array(),
            emissive: material.emissive.to_f32_array(),
            metallic: material.metallic,
            perceptual_roughness: material.perceptual_roughness,
            reflectance: material.reflectance,
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
        },
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
