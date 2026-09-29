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

    /// The closure-kind bitmask packed into the key.
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
}
