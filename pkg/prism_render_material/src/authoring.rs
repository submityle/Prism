//! Producer-side authoring helpers that emit [`MaterialRecord`]s for the
//! optional über-BSDF lobes whose consumers live in the render shaders.
//!
//! These keep the byte-exact ABI contract honest end to end: every helper here
//! is paired with a live shader consumer (for face shadow, the
//! `shading_resolve.wesl::shade_toon` terminator), so no committed material bit
//! is written without something reading it.

use crate::{GpuMaterialTexture, MaterialFeatureFlags, MaterialRecord};

/// ABI semantic tag for the baked face-shadow signed-distance-field map.
///
/// This is the single source of truth mirrored by every consumer:
/// `TextureSemantic::FaceShadowSdf` (the Bevy bridge enum), `SEMANTIC_FACE_SDF`
/// in both `material_sample.wesl` and `shading_resolve.wesl`. A face material's
/// SDF row carries this in [`GpuMaterialTexture::semantic`] so the toon
/// integrator can find it by scanning the material's texture rows.
pub const FACE_SHADOW_SDF_SEMANTIC: u32 = 8;

/// ABI semantic tag for a measured (tabulated) BRDF/BTF table (§17.6).
///
/// A measured material's table ships as an ordinary texture row carrying this
/// tag, exactly like the face-shadow SDF row above: the lobe finds it by
/// scanning the material's own texture list, so the ABI needs no slot index
/// (which would go stale whenever the bindless heap is repacked). Mirrored by
/// `TextureSemantic::MeasuredBrdf` (the Bevy bridge enum), `SEMANTIC_MEASURED_BRDF`
/// in `measured_lobe.wesl` and the shading-side texture semantics.
///
/// The table's atlas layout is part of the contract: width = `theta_h`
/// resolution, height = `theta_d` resolution × `phi_d` resolution, and texel
/// `(i, j * phi_d + k)` holds grid cell `(i, j, k)`.
pub const MEASURED_BRDF_SEMANTIC: u32 = 9;

/// ABI semantic tag for a region-illumination mask.
///
/// The mask is a single-channel texture: `0` selects the physically based
/// `Lit` response and `1` selects `Stylized` for that region. The visibility
/// raster samples it at the interpolated UV and writes the selector into the
/// visibility metadata so the classification pass can bin Lit and Stylized
/// regions into their respective waves even when they share one material.
pub const ILLUMINATION_MASK_SEMANTIC: u32 = 10;

impl MaterialRecord {
    /// Turns this record into a stylized (NPR) face-shadow producer.
    ///
    /// Sets the [`MaterialFeatureFlags::FACE_SHADOW`] bit (which lights up the
    /// `LobeMask::NPR_FACE` lobe and packs `npr_face_softness` into the parameter
    /// block), records the terminator half-window `softness`, and appends the
    /// baked SDF map as a [`FACE_SHADOW_SDF_SEMANTIC`] texture row.
    ///
    /// `sdf_slot`/`sdf_generation` address the SDF map in the bindless texture
    /// heap (the same `index`/`generation` pair a resolver produces for any
    /// other map); `sampler_index` selects its sampler. `softness` is stored
    /// verbatim and clamped to `[0, 0.5]` on the GPU by `evaluate_face_shadow`.
    ///
    /// Idempotent on the feature bit, but always appends a row, so call it once
    /// per material.
    pub fn enable_face_shadow(
        &mut self,
        sdf_slot: u32,
        sdf_generation: u32,
        sampler_index: u32,
        softness: f32,
    ) -> &mut Self {
        self.features |= MaterialFeatureFlags::FACE_SHADOW;
        self.surface.npr_face_softness = softness;
        self.textures.push(GpuMaterialTexture::non_virtual(
            sdf_slot,
            sdf_generation,
            FACE_SHADOW_SDF_SEMANTIC,
            sampler_index,
        ));
        self
    }

    /// Marks this record as a region-illumination producer.
    ///
    /// Sets [`MaterialFeatureFlags::REGION_ILLUMINATION`] and appends the
    /// mask texture row carrying [`ILLUMINATION_MASK_SEMANTIC`]. The mask is
    /// sampled in the visibility raster; the material header's `illumination`
    /// still selects the base response for regions where the mask is absent.
    pub fn enable_region_illumination(
        &mut self,
        mask_slot: u32,
        mask_generation: u32,
        sampler_index: u32,
    ) -> &mut Self {
        self.features |= MaterialFeatureFlags::REGION_ILLUMINATION;
        self.textures.push(GpuMaterialTexture::non_virtual(
            mask_slot,
            mask_generation,
            ILLUMINATION_MASK_SEMANTIC,
            sampler_index,
        ));
        self
    }

    /// Marks this record as a continuous Lit/NPR response-composition
    /// producer. The texture row is shared with the binary region mask, but
    /// the feature bit is intentionally distinct so the raster/classification
    /// path cannot silently treat a continuous weight as a binary selector.
    pub fn enable_illumination_blend(
        &mut self,
        mask_slot: u32,
        mask_generation: u32,
        sampler_index: u32,
    ) -> &mut Self {
        self.features |= MaterialFeatureFlags::ILLUMINATION_BLEND;
        self.textures.push(GpuMaterialTexture::non_virtual(
            mask_slot,
            mask_generation,
            ILLUMINATION_MASK_SEMANTIC,
            sampler_index,
        ));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        decode_illumination_blend_weight, fallback_material_record, LobeMask,
        ILLUMINATION_BLEND_SENTINEL, ILLUMINATION_BLEND_UNORM_MAX,
    };
    use prism_render_architecture::abi::GenerationalHandle;

    fn base_record() -> MaterialRecord {
        fallback_material_record(GenerationalHandle::new(7, 3), 1)
    }

    #[test]
    fn enable_face_shadow_sets_feature_lobe_softness_and_row() {
        let mut record = base_record();
        let words_before = record.packed_parameters().packed_len_words();

        record.enable_face_shadow(42, 5, 2, 0.3125);

        // Feature bit is live in the record and its serialized header.
        assert!(record.features.contains(MaterialFeatureFlags::FACE_SHADOW));
        let header = record.header(0, 0, 0);
        assert_eq!(
            header.feature_flags & MaterialFeatureFlags::FACE_SHADOW.0,
            MaterialFeatureFlags::FACE_SHADOW.0,
        );

        // The FACE lobe is present and mirrored into the header lobe mask.
        assert!(record.lobe_mask().contains(LobeMask::NPR_FACE));
        assert_eq!(header.lobe_mask, record.lobe_mask().bits());

        // Softness is stored and packed (the FACE lobe adds four words).
        assert_eq!(record.surface.npr_face_softness, 0.3125);
        let words_after = record.packed_parameters().packed_len_words();
        assert_eq!(words_after, words_before + 4);

        // The SDF row carries the shared semantic tag and bindless address.
        let row = record
            .textures
            .iter()
            .find(|t| t.semantic == FACE_SHADOW_SDF_SEMANTIC)
            .expect("face shadow SDF row present");
        assert_eq!(row.index, 42);
        assert_eq!(row.generation, 5);
        assert_eq!(row.sampler_index, 2);
    }

    #[test]
    fn softness_round_trips_through_the_packed_face_lobe() {
        let mut record = base_record();
        record.enable_face_shadow(1, 1, 0, 0.4);
        let block = record.packed_parameters();
        let restored = block.to_full();
        assert_eq!(restored.npr_face_softness, 0.4);
    }

    #[test]
    fn enable_region_illumination_sets_feature_and_mask_row() {
        let mut record = base_record();
        record.enable_region_illumination(17, 4, 3);

        assert!(record
            .features
            .contains(MaterialFeatureFlags::REGION_ILLUMINATION));
        let row = record
            .textures
            .iter()
            .find(|texture| texture.semantic == ILLUMINATION_MASK_SEMANTIC)
            .expect("illumination mask row present");
        assert_eq!(row.index, 17);
        assert_eq!(row.generation, 4);
        assert_eq!(row.sampler_index, 3);
    }

    #[test]
    fn continuous_illumination_blend_has_a_distinct_feature_bit() {
        let mut record = base_record();
        record.enable_illumination_blend(17, 4, 3);
        assert!(record
            .features
            .contains(MaterialFeatureFlags::ILLUMINATION_BLEND));
        assert!(!record
            .features
            .contains(MaterialFeatureFlags::REGION_ILLUMINATION));
        assert_ne!(
            MaterialFeatureFlags::ILLUMINATION_BLEND.0,
            MaterialFeatureFlags::REGION_ILLUMINATION.0
        );
    }

    #[test]
    fn continuous_illumination_metadata_has_a_bounded_unorm_and_sentinel() {
        assert_eq!(decode_illumination_blend_weight(0), Some(0.0));
        assert_eq!(
            decode_illumination_blend_weight(ILLUMINATION_BLEND_UNORM_MAX),
            Some(1.0)
        );
        assert_eq!(
            decode_illumination_blend_weight(ILLUMINATION_BLEND_SENTINEL),
            None
        );
        assert_eq!(decode_illumination_blend_weight(1 << 16), None);
    }

    #[test]
    fn stylized_modifiers_activate_the_npr_lobe_without_face_shadow() {
        let mut record = base_record();
        record.surface.npr_stylized_specular_scale = 0.5;
        assert!(record.lobe_mask().contains(LobeMask::NPR_FACE));
        assert_eq!(record.packed_parameters().face.specular_scale, 0.5);
        assert!(!record.features.contains(MaterialFeatureFlags::FACE_SHADOW));
    }
}
