//! Scene-local adapter from Bevy StandardMaterial to Prism's canonical material ABI.\n//!\n//! This module is intentionally owned by render-scene: the canonical material\n//! crate remains independent of Bevy and exposes only the stable ABI.\n\n//! `OpenPBR` Surface importer for external materials (design doc §4.4).
//!
//! External materials — Bevy `StandardMaterial` today, glTF/`KHR_*` material
//! extensions, `MaterialX` and USD importers later — are mapped through ONE
//! format adapter onto the canonical `OpenPBR` Surface parameter block
//! ([`import_openpbr_surface`]). The packaging step ([`import_standard_material`])
//! is format-agnostic, so there is no per-format 1:1 translation layer.

use bevy_asset::{AssetId, Handle};
use bevy_color::ColorToComponents;
use bevy_image::Image;
use bevy_material::AlphaMode;
use bevy_mesh::UvChannel;
use bevy_pbr::StandardMaterial;
use prism_render_architecture::abi::GenerationalHandle;
use prism_render_architecture::render_fabric::RenderDomain;

use prism_render_material::{
    fallback_material_record, GpuMaterialTexture, GpuSurfaceParameters, MaterialDomain,
    MaterialFeatureFlags, MaterialRecord, MaterialRenderClass, ShadingPlan,
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
    /// discriminant is the shared [`prism_render_material::FACE_SHADOW_SDF_SEMANTIC`].
    FaceShadowSdf,
    /// Measured (tabulated) BRDF/BTF table (§17.6). Its discriminant is the
    /// shared [`prism_render_material::MEASURED_BRDF_SEMANTIC`]; the table's atlas layout is
    /// documented on that constant.
    MeasuredBrdf,
    /// Region-illumination mask. Its discriminant is the shared
    /// [`prism_render_material::ILLUMINATION_MASK_SEMANTIC`]; the visibility raster consumes it
    /// to select Lit/Stylized per region without changing the material ABI.
    IlluminationMask,
}

impl TextureSemantic {
    /// Whether the semantic is folded into the sampled OpenPBR surface by the
    /// shared CPU/WESL texture consumer.
    pub const fn is_surface_folded(self) -> bool {
        matches!(
            self,
            Self::BaseColor
                | Self::Emissive
                | Self::MetallicRoughness
                | Self::Normal
                | Self::Occlusion
                | Self::ClearCoat
                | Self::ClearCoatRoughness
                | Self::ClearCoatNormal
        )
    }

    /// Whether the semantic is intentionally consumed by a dedicated pass
    /// rather than by the generic surface-factor fold.
    pub const fn has_dedicated_consumer(self) -> bool {
        matches!(
            self,
            Self::FaceShadowSdf | Self::MeasuredBrdf | Self::IlluminationMask
        )
    }

    /// Whether the current Bevy `StandardMaterial` exposes a source image for
    /// this semantic in this adapter. Feature-gated Bevy extensions are not
    /// listed here because this crate does not enable those dependency
    /// features; they must not be represented by guessed rows.
    pub const fn has_standard_material_source(self) -> bool {
        matches!(
            self,
            Self::BaseColor
                | Self::Emissive
                | Self::MetallicRoughness
                | Self::Normal
                | Self::Occlusion
        )
    }
}

const STANDARD_MATERIAL_TEXTURE_SEMANTICS: [TextureSemantic; 5] = [
    TextureSemantic::BaseColor,
    TextureSemantic::Emissive,
    TextureSemantic::MetallicRoughness,
    TextureSemantic::Normal,
    TextureSemantic::Occlusion,
];

// The enum discriminants are the ABI semantic bytes; keep `FaceShadowSdf`
// pinned to the crate-wide constant its non-bevy producer and every shader
// consumer share.
const _: () = assert!(
    TextureSemantic::FaceShadowSdf as u32 == prism_render_material::FACE_SHADOW_SDF_SEMANTIC
);
const _: () =
    assert!(TextureSemantic::MeasuredBrdf as u32 == prism_render_material::MEASURED_BRDF_SEMANTIC);
const _: () = assert!(
    TextureSemantic::IlluminationMask as u32 == prism_render_material::ILLUMINATION_MASK_SEMANTIC
);
const _: () = assert!(TextureSemantic::BaseColor.is_surface_folded());
const _: () = assert!(TextureSemantic::FaceShadowSdf.has_dedicated_consumer());

/// Resolves authored image handles into material texture rows.
///
/// A resolver that has no page-table-backed VT integration must return
/// [`GpuMaterialTexture::non_virtual`] rows. The production render-scene
/// resolver follows that rule explicitly; this trait does not imply that an
/// image has been virtualized merely because the ABI can carry VT metadata.
pub trait StandardMaterialTextureResolver {
    fn resolve(&mut self, image: AssetId<Image>, semantic: TextureSemantic) -> GpuMaterialTexture;
}

/// Immutable material-side UV state shared by resolve and VT feedback.
///
/// UV0 transforms are fully supported here. A texture authored against UV1 is
/// marked in the mask; the render-scene admission path either supplies the
/// real UV1 stream or explicitly aliases UV0 when that stream is absent, so
/// resolve and VT feedback use the same channel decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MaterialSurfaceContext {
    pub(crate) uv_channel_mask: u32,
    pub(crate) uv_matrix_x: [f32; 2],
    pub(crate) uv_matrix_y: [f32; 2],
    pub(crate) uv_translation: [f32; 2],
}

impl Default for MaterialSurfaceContext {
    fn default() -> Self {
        Self {
            uv_channel_mask: 0,
            uv_matrix_x: [1.0, 0.0],
            uv_matrix_y: [0.0, 1.0],
            uv_translation: [0.0, 0.0],
        }
    }
}

impl MaterialSurfaceContext {
    pub(crate) fn transform_uv(self, uv: [f32; 2]) -> [f32; 2] {
        [
            self.uv_matrix_x[0] * uv[0] + self.uv_matrix_y[0] * uv[1] + self.uv_translation[0],
            self.uv_matrix_x[1] * uv[0] + self.uv_matrix_y[1] * uv[1] + self.uv_translation[1],
        ]
    }

    pub(crate) fn transform_gradient(self, gradient: [f32; 2]) -> [f32; 2] {
        [
            self.uv_matrix_x[0] * gradient[0] + self.uv_matrix_y[0] * gradient[1],
            self.uv_matrix_x[1] * gradient[0] + self.uv_matrix_y[1] * gradient[1],
        ]
    }

    pub(crate) fn uses_uv1(self, semantic: TextureSemantic) -> bool {
        self.uv_channel_mask & (1_u32 << semantic as u32) != 0
    }
}

/// Lower the StandardMaterial UV transform and channel choices for the
/// texture rows emitted by this adapter.
pub(crate) fn material_surface_context(material: &StandardMaterial) -> MaterialSurfaceContext {
    let transform = material.uv_transform;
    let mut uv_channel_mask = 0;
    let mut mark_uv1 = |semantic: TextureSemantic, channel: &UvChannel| {
        if *channel == UvChannel::Uv1 {
            uv_channel_mask |= 1_u32 << semantic as u32;
        }
    };
    mark_uv1(TextureSemantic::BaseColor, &material.base_color_channel);
    mark_uv1(TextureSemantic::Emissive, &material.emissive_channel);
    mark_uv1(
        TextureSemantic::MetallicRoughness,
        &material.metallic_roughness_channel,
    );
    mark_uv1(TextureSemantic::Normal, &material.normal_map_channel);
    mark_uv1(TextureSemantic::Occlusion, &material.occlusion_channel);
    MaterialSurfaceContext {
        uv_channel_mask,
        uv_matrix_x: [transform.matrix2.x_axis.x, transform.matrix2.x_axis.y],
        uv_matrix_y: [transform.matrix2.y_axis.x, transform.matrix2.y_axis.y],
        uv_translation: [transform.translation.x, transform.translation.y],
    }
}
/// Import an external `StandardMaterial` (Bevy / glTF / KHR material extensions)
/// into the canonical **`OpenPBR` Surface** parameter set (design doc §4.4).
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
    // `OpenPBR` splits emission into `emission_color * emission_luminance`;
    // Bevy authors a single radiance colour, so factor it into a normalised
    // colour and its luminance (the max channel). The product — and therefore
    // the rendered emission — is preserved exactly.
    let emissive = material.emissive.to_f32_array();
    let emission_luminance = emissive[0].max(emissive[1]).max(emissive[2]);
    let emission_color = if emission_luminance > 0.0 {
        [
            emissive[0] / emission_luminance,
            emissive[1] / emission_luminance,
            emissive[2] / emission_luminance,
            emissive[3],
        ]
    } else {
        emissive
    };
    GpuSurfaceParameters {
        base_color: material.base_color.to_linear().to_f32_array(),
        emission_color,
        emission_luminance,
        base_metalness: material.metallic,
        specular_roughness: material.perceptual_roughness,
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
        transmission_weight: material
            .specular_transmission
            .max(material.diffuse_transmission),
        thickness: material.thickness,
        coat_weight: material.clearcoat,
        coat_roughness: material.clearcoat_perceptual_roughness,
        anisotropy: material.anisotropy_strength,
        anisotropy_rotation: material.anisotropy_rotation,
        index_of_refraction: material.ior,
        ..Default::default()
    }
}

/// Import an external `StandardMaterial` into a runtime [`MaterialRecord`].
///
/// The `OpenPBR` Surface parameters come from [`import_openpbr_surface`]; this
/// function only adds the format-independent packaging (render class, features,
/// closure mask, texture rows, illumination axis). It replaces the former
/// per-format standard-material translator (design doc §4.4).
pub fn import_standard_material(
    handle: GenerationalHandle,
    revision: u32,
    material: &StandardMaterial,
    textures: &mut impl StandardMaterialTextureResolver,
) -> MaterialRecord {
    // The canonical texture row ABI has no independent normal-map Y sign. A
    // flipped DirectX normal cannot be represented by `normal_scale` (which
    // scales both tangent axes), so publishing the row would be a wrong-image
    // silent downgrade. Use the explicit inactive/fallback record instead.
    if material.flip_normal_map_y && material.normal_map_texture.is_some() {
        return fallback_material_record(handle, revision as u64);
    }

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
    for semantic in STANDARD_MATERIAL_TEXTURE_SEMANTICS {
        if let Some(image) = standard_material_texture(material, semantic) {
            push_texture(&mut texture_rows, image, semantic, textures);
        }
    }
    MaterialRecord {
        handle,
        revision,
        domain: MaterialDomain::Surface,
        render_domain: RenderDomain::Surface,
        render_class,
        // Response selection is a plan, not a raster/blend class. The standard
        // material importer only needs the built-in Lit/Unlit programs.
        response_plan: if material.unlit {
            ShadingPlan::single_builtin(prism_render_material::BuiltinProgram::Unlit)
        } else {
            ShadingPlan::default()
        },
        features,
        closure_mask: closure_mask(material),
        surface: import_openpbr_surface(material, alpha_cutoff),
        textures: texture_rows,
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
    image: &Handle<Image>,
    semantic: TextureSemantic,
    resolver: &mut impl StandardMaterialTextureResolver,
) {
    assert!(
        semantic.is_surface_folded() && semantic.has_standard_material_source(),
        "StandardMaterial emitted a semantic without a CPU/WESL surface fold"
    );
    rows.push(resolver.resolve(image.id(), semantic));
}

fn standard_material_texture(
    material: &StandardMaterial,
    semantic: TextureSemantic,
) -> Option<&Handle<Image>> {
    match semantic {
        TextureSemantic::BaseColor => material.base_color_texture.as_ref(),
        TextureSemantic::Emissive => material.emissive_texture.as_ref(),
        TextureSemantic::MetallicRoughness => material.metallic_roughness_texture.as_ref(),
        TextureSemantic::Normal => material.normal_map_texture.as_ref(),
        TextureSemantic::Occlusion => material.occlusion_texture.as_ref(),
        TextureSemantic::ClearCoat
        | TextureSemantic::ClearCoatRoughness
        | TextureSemantic::ClearCoatNormal
        | TextureSemantic::FaceShadowSdf
        | TextureSemantic::MeasuredBrdf
        | TextureSemantic::IlluminationMask => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::uuid::Uuid;
    use bevy_math::{Affine2, Mat2, Vec2};

    struct RecordingResolver {
        calls: Vec<(AssetId<Image>, TextureSemantic)>,
    }

    impl StandardMaterialTextureResolver for RecordingResolver {
        fn resolve(
            &mut self,
            image: AssetId<Image>,
            semantic: TextureSemantic,
        ) -> GpuMaterialTexture {
            let index = self.calls.len() as u32 + 1;
            self.calls.push((image, semantic));
            GpuMaterialTexture::non_virtual(index, 1, semantic as u32, index)
        }
    }

    #[test]
    fn surface_context_preserves_uv_transform_and_marks_unavailable_uv1() {
        let mut material = StandardMaterial::default();
        material.uv_transform = Affine2::from_mat2_translation(
            Mat2::from_cols(Vec2::new(2.0, 3.0), Vec2::new(5.0, 7.0)),
            Vec2::new(11.0, 13.0),
        );
        material.normal_map_channel = UvChannel::Uv1;

        let context = material_surface_context(&material);
        assert_eq!(context.transform_uv([0.25, 0.5]), [14.0, 17.25]);
        assert_eq!(context.transform_gradient([0.5, 0.25]), [2.25, 3.25]);
        assert!(context.uses_uv1(TextureSemantic::Normal));
        assert!(!context.uses_uv1(TextureSemantic::BaseColor));
    }

    #[test]
    fn standard_material_emits_exactly_the_supported_folded_texture_semantics() {
        let handles = [
            Handle::<Image>::from(Uuid::from_u128(1)),
            Handle::<Image>::from(Uuid::from_u128(2)),
            Handle::<Image>::from(Uuid::from_u128(3)),
            Handle::<Image>::from(Uuid::from_u128(4)),
            Handle::<Image>::from(Uuid::from_u128(5)),
        ];
        let material = StandardMaterial {
            base_color_texture: Some(handles[0].clone()),
            emissive_texture: Some(handles[1].clone()),
            metallic_roughness_texture: Some(handles[2].clone()),
            normal_map_texture: Some(handles[3].clone()),
            occlusion_texture: Some(handles[4].clone()),
            ..Default::default()
        };
        let mut resolver = RecordingResolver { calls: Vec::new() };
        let record =
            import_standard_material(GenerationalHandle::new(7, 1), 3, &material, &mut resolver);

        let semantics: Vec<_> = record.textures.iter().map(|row| row.semantic).collect();
        assert_eq!(
            semantics,
            STANDARD_MATERIAL_TEXTURE_SEMANTICS
                .iter()
                .map(|semantic| *semantic as u32)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            resolver.calls.len(),
            STANDARD_MATERIAL_TEXTURE_SEMANTICS.len()
        );
        assert!(STANDARD_MATERIAL_TEXTURE_SEMANTICS
            .iter()
            .all(
                |semantic| semantic.is_surface_folded() && semantic.has_standard_material_source()
            ));
        assert!(!TextureSemantic::ClearCoat.has_standard_material_source());
        assert!(!TextureSemantic::FaceShadowSdf.is_surface_folded());
    }

    #[test]
    fn flipped_normal_map_fails_closed_instead_of_publishing_wrong_tangent_axes() {
        let material = StandardMaterial {
            normal_map_texture: Some(Handle::from(Uuid::from_u128(6))),
            flip_normal_map_y: true,
            ..Default::default()
        };
        let mut resolver = RecordingResolver { calls: Vec::new() };
        let record =
            import_standard_material(GenerationalHandle::new(8, 2), 4, &material, &mut resolver);

        assert!(record.textures.is_empty());
        assert_eq!(record.surface, GpuSurfaceParameters::default());
        assert!(resolver.calls.is_empty());
    }
}
