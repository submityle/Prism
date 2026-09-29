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

impl MaterialRecord {
    /// Turns this record into a stylized (NPR) face-shadow producer.
    ///
    /// Sets the [`MaterialFeatureFlags::FACE_SHADOW`] bit (which lights up the
    /// `LobeMask::FACE` lobe and packs `face_softness` into the parameter
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
        self.surface.face_softness = softness;
        self.textures.push(GpuMaterialTexture {
            index: sdf_slot,
            generation: sdf_generation,
            semantic: FACE_SHADOW_SDF_SEMANTIC,
            sampler_index,
        });
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{fallback_material_record, LobeMask};
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
        let header = record.header(0, 0, 0, 0);
        assert_eq!(
            header.feature_flags & MaterialFeatureFlags::FACE_SHADOW.0,
            MaterialFeatureFlags::FACE_SHADOW.0,
        );

        // The FACE lobe is present and mirrored into the header lobe mask.
        assert!(record.lobe_mask().contains(LobeMask::FACE));
        assert_eq!(header.lobe_mask, record.lobe_mask().bits());

        // Softness is stored and packed (the FACE lobe adds four words).
        assert_eq!(record.surface.face_softness, 0.3125);
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
        assert_eq!(restored.face_softness, 0.4);
    }
}
