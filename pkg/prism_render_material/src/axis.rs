//! Orthogonal material style axes.
//!
//! These replace the former mutually-exclusive `MaterialShadingModel` enum
//! (see `docs/prism_material_pipeline_design_zh.md` §3). A material is
//! now described by independent axes:
//!
//! * `domain`        — Surface / Decal / Volume / `PostProcess` (see `record.rs`)
//! * `closure graph` — *what* the BSDF looks like (`ir::ClosureGraph`)
//! * `illumination`  — *how* shared lighting data is interpreted (this module)
//! * `blend`         — the `MaterialRenderClass` family
//!
//! The legal permutation of these axes is condensed into a
//! [`SpecializationId`], the stable key that classification buckets on today
//! and that the WESL specialization pipeline keys on later.

/// The *style* axis: how a surface interprets the shared lighting data.
///
/// This is orthogonal to the closure graph and to the blend family. NPR is a
/// value on this axis (`Stylized`), **not** a shading model competing with
/// `Principled`, and **not** the same tier as `Water` (which is a closure /
/// subsystem). This orthogonality is the whole point of the refactor.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Illumination {
    /// Physically-based light response (BRDF integration).
    #[default]
    Lit,
    /// Stylized / NPR light response (ramp quantization, stepped shadows,
    /// stylized specular, ...). Consumes the same lighting data as `Lit`.
    Stylized,
    /// Emissive-only; ignores scene lighting entirely.
    Unlit,
    /// Project-injected custom light response (custom WESL closure + pass).
    Custom,
}

impl Illumination {
    /// Reconstruct from the raw `u32` stored in `GpuMaterialHeader`.
    pub const fn from_u32(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Lit),
            1 => Some(Self::Stylized),
            2 => Some(Self::Unlit),
            3 => Some(Self::Custom),
            _ => None,
        }
    }
}

/// Deterministic identity of one compiled shader permutation.
///
/// Packs the legal combination of orthogonal axes so classification, the GPU
/// work-plan binning, and the WESL specialization pipeline all share
/// one stable key.
///
/// Bit layout (LSB → MSB):
/// * bits `0..8`   — `illumination`
/// * bits `8..40`  — `closure_mask` (32 closure-kind bits)
/// * bits `40..48` — `render_class` (blend / domain family)
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SpecializationId(pub u64);

impl SpecializationId {
    /// Build a specialization key from the orthogonal axes.
    ///
    /// `closure_mask` is the **live lobe set** (the derived `LobeMask` bits), not
    /// the raw authored closure bits: feature flags fold extra lobes in, and the
    /// permutation must follow the live set.
    pub const fn new(illumination: Illumination, closure_mask: u32, render_class: u32) -> Self {
        let illum = (illumination as u64) & 0xFF;
        let closure = (closure_mask as u64) << 8;
        let class = ((render_class as u64) & 0xFF) << 40;
        Self(illum | closure | class)
    }

    /// Raw illumination bits.
    pub const fn illumination_bits(self) -> u32 {
        (self.0 & 0xFF) as u32
    }

    /// Recover the illumination axis value.
    pub const fn illumination(self) -> Option<Illumination> {
        Illumination::from_u32(self.illumination_bits())
    }

    /// The live lobe set packed into the key (the derived `LobeMask` bits).
    pub const fn closure_mask(self) -> u32 {
        ((self.0 >> 8) & 0xFFFF_FFFF) as u32
    }

    /// The blend / render-class family packed into the key.
    pub const fn render_class_bits(self) -> u32 {
        ((self.0 >> 40) & 0xFF) as u32
    }

    /// Low 32 bits, for splitting into two `u32` GPU header slots.
    pub const fn low(self) -> u32 {
        self.0 as u32
    }

    /// High 32 bits, for splitting into two `u32` GPU header slots.
    pub const fn high(self) -> u32 {
        (self.0 >> 32) as u32
    }

    /// The compact `u32` form of this key used as a draw-bin class.
    ///
    /// Layout (LSB → MSB): `illumination` `0..2`, `render_class` `2..10`, live
    /// lobe set `10..27`. A legal lobe mask uses only bits `0..17` — the `LobeMask`
    /// vocabulary has eight kinds and the highest, `NPR_FACE`, sits at bit `16` —
    /// so the projection is collision-free for any legal material: a bin keyed on
    /// this class carries the **full** permutation, not just the
    /// illumination/render-class pair.
    ///
    /// Binning is therefore correct for materials that differ only in their live
    /// lobes (e.g. one with a face-shadow or clearcoat lobe) — collapsing them
    /// onto one class batched different pipelines together
    /// (`prism_material_pipeline_refactor_zh.md` §8.5).
    #[must_use]
    pub const fn bin_class(self) -> u32 {
        (self.illumination_bits() & 0x3)
            | ((self.render_class_bits() & 0xFF) << 2)
            | ((self.closure_mask() & 0x1_FFFF) << 10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn illumination_from_u32_covers_all_valid_values() {
        assert_eq!(Illumination::from_u32(0), Some(Illumination::Lit));
        assert_eq!(Illumination::from_u32(1), Some(Illumination::Stylized));
        assert_eq!(Illumination::from_u32(2), Some(Illumination::Unlit));
        assert_eq!(Illumination::from_u32(3), Some(Illumination::Custom));
    }

    #[test]
    fn illumination_from_u32_rejects_out_of_range() {
        assert_eq!(Illumination::from_u32(4), None);
        assert_eq!(Illumination::from_u32(u32::MAX), None);
    }

    #[test]
    fn illumination_default_is_lit() {
        assert_eq!(Illumination::default(), Illumination::Lit);
    }

    #[test]
    fn specialization_id_default_is_zero() {
        assert_eq!(SpecializationId::default(), SpecializationId(0));
    }

    #[test]
    fn specialization_id_round_trips_each_axis() {
        // Distinct non-overlapping bit patterns in each field prove the packing
        // masks and shifts do not bleed into neighbouring axes.
        let closure_mask = 0xABCD_1234_u32;
        let render_class = 0x5A_u32;
        let id = SpecializationId::new(Illumination::Custom, closure_mask, render_class);
        assert_eq!(id.illumination(), Some(Illumination::Custom));
        assert_eq!(id.illumination_bits(), Illumination::Custom as u32);
        assert_eq!(id.closure_mask(), closure_mask);
        assert_eq!(id.render_class_bits(), render_class);
    }

    #[test]
    fn specialization_id_masks_render_class_to_eight_bits() {
        // Only the low eight bits of the render class survive the packing.
        let id = SpecializationId::new(Illumination::Lit, 0, 0x1FF);
        assert_eq!(id.render_class_bits(), 0xFF);
    }

    #[test]
    fn specialization_id_low_high_split_reconstructs_key() {
        let id = SpecializationId::new(Illumination::Stylized, 0xFFFF_FFFF, 0x7F);
        let rebuilt = (u64::from(id.high()) << 32) | u64::from(id.low());
        assert_eq!(SpecializationId(rebuilt), id);
    }

    #[test]
    fn specialization_id_is_deterministic() {
        let a = SpecializationId::new(Illumination::Unlit, 42, 3);
        let b = SpecializationId::new(Illumination::Unlit, 42, 3);
        assert_eq!(a, b);
    }

    #[test]
    fn bin_class_carries_the_live_lobe_set() {
        let plain = SpecializationId::new(Illumination::Lit, 0, 0);
        // `LobeMask::CLEARCOAT` (bit 1) and `LobeMask::NPR_FACE` (bit 16).
        let clearcoat = SpecializationId::new(Illumination::Lit, 1 << 1, 0);
        let face_shadow = SpecializationId::new(Illumination::Lit, 1 << 16, 0);
        assert_ne!(
            plain.bin_class(),
            clearcoat.bin_class(),
            "a differing lobe yields a different bin class"
        );
        assert_ne!(plain.bin_class(), face_shadow.bin_class());
        assert_eq!(
            plain.bin_class(),
            SpecializationId::new(Illumination::Lit, 0, 0).bin_class(),
            "the bin class is deterministic"
        );

        // Illumination and render_class still separate bins on their own.
        assert_ne!(
            plain.bin_class(),
            SpecializationId::new(Illumination::Stylized, 0, 0).bin_class()
        );
        assert_ne!(
            plain.bin_class(),
            SpecializationId::new(Illumination::Lit, 0, 1).bin_class()
        );

        // The projection fits the documented field widths (27 bits total).
        let widest = SpecializationId::new(Illumination::Custom, 0x1_FFFF, 0xFF);
        assert_eq!(widest.bin_class() >> 27, 0, "bin class stays within 27 bits");
    }
}
