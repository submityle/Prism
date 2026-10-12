//! Über-BSDF surface parameters split into a compact core plus optional
//! per-lobe blobs (design doc §3.3).
//!
//! The former [`GpuSurfaceParameters`](crate::GpuSurfaceParameters) is an
//! 18-field fat struct carried per pixel regardless of which lobes a material
//! actually uses. That is both a performance sink (every pixel pays for
//! clearcoat/sheen/subsurface/transmission it may never touch) and a modelling
//! lie (it implies every surface has every lobe).
//!
//! The über-BSDF is anchored on an OpenPBR-style core that *every* lit surface
//! carries. Everything else — emission, clearcoat, anisotropy, sheen,
//! subsurface, transmission — is an optional lobe. A [`LobeMask`] records which
//! lobes are live; [`SurfaceParameterBlock::pack`] serializes the core followed
//! by *only the present lobes* in canonical bit order, and
//! [`SurfaceParameterBlock::unpack`] reverses it. The compiled specialization
//! (and the `lobe_mask` stored in
//! [`GpuMaterialHeader`](crate::GpuMaterialHeader)) decides which lobes a shader
//! reads, so a plain dielectric costs 12 words instead of 24.

use crate::GpuSurfaceParameters;
// The word-count facts are generated from `schema/surface.toml` (design doc
// §17.5, S2) and live in `surface_layout.rs`. Re-exporting them here keeps the
// long-standing public path `prism_render_material::SURFACE_CORE_WORDS` stable
// while making the schema the single source of truth; the typed structs below
// are checked against this layout by `layout_matches_generated_table`.
pub use crate::surface_layout::{
    SURFACE_CORE_WORDS, SURFACE_LOBE_WORDS, SURFACE_LOBE_WORDS_BY_SLOT,
};

/// The compact über-BSDF core carried by every surface.
///
/// This is the OpenPBR-anchored base every lit/stylized/unlit surface needs. It
/// is deliberately free of clearcoat/sheen/subsurface/transmission/anisotropy —
/// those live in optional lobes so a material only pays for what it uses.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSurfaceCore {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub perceptual_roughness: f32,
    pub reflectance: f32,
    pub ambient_occlusion: f32,
    pub normal_scale: f32,
    pub alpha_cutoff: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

impl Default for GpuSurfaceCore {
    fn default() -> Self {
        Self {
            base_color: [1.0; 4],
            metallic: 0.0,
            perceptual_roughness: 0.5,
            reflectance: 0.5,
            ambient_occlusion: 1.0,
            normal_scale: 1.0,
            alpha_cutoff: 0.5,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }
}

/// Emissive radiance lobe (`illumination`-independent self-illumination).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuEmissionLobe {
    pub emissive: [f32; 4],
}

/// Clearcoat lobe: a second, smoother specular layer over the base.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuClearCoatLobe {
    pub clearcoat: f32,
    pub clearcoat_roughness: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

/// Anisotropy lobe: directional roughness for brushed metal / hair-like highlights.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuAnisotropyLobe {
    pub anisotropy: f32,
    pub anisotropy_rotation: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

/// Sheen lobe: retro-reflective fabric rim response.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSheenLobe {
    pub sheen: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

/// Subsurface lobe: wrapped diffuse approximation of shallow scattering.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSubsurfaceLobe {
    pub subsurface: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

/// Transmission lobe: refractive dielectric transport (glass/liquid).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTransmissionLobe {
    pub transmission: f32,
    pub thickness: f32,
    pub index_of_refraction: f32,
    pub dispersion: f32,
}

/// Face-shadow lobe: stylized SDF-driven directional face shading (anime/toon).
///
/// The lobe carries only the terminator softness; the SDF map itself is a
/// bindless texture bound through the `SEMANTIC_FACE_SDF` slot, and the
/// per-instance orientation used to flip/threshold the SDF comes from the
/// instance world matrix at shade time (no extra per-instance ABI).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuFaceLobe {
    pub softness: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

/// Bitset recording which optional lobes a surface carries.
///
/// The bit order is the canonical serialization order used by
/// [`SurfaceParameterBlock::pack`]: present lobes are written low bit first.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LobeMask(pub u32);

impl LobeMask {
    pub const EMISSION: Self = Self(1 << 0);
    pub const CLEARCOAT: Self = Self(1 << 1);
    pub const ANISOTROPY: Self = Self(1 << 2);
    pub const SHEEN: Self = Self(1 << 3);
    pub const SUBSURFACE: Self = Self(1 << 4);
    pub const TRANSMISSION: Self = Self(1 << 5);
    pub const FACE: Self = Self(1 << 6);

    /// The `OpenPBR` specification-core bit region (§4.5 constraint 2).
    ///
    /// Canonical (`namespace == "KHR"`) lobes occupy this low, fixed segment.
    /// A mask that fits entirely inside [`Self::KHR_REGION`] describes a
    /// *pure `OpenPBR`* surface, so pruning it to this region is the identity
    /// (see [`crate::project_to_openpbr`]) and its BSDF response is the
    /// conformance reference the `OpenPBR` gate compares against.
    ///
    /// This mask is the runtime mirror of the schema's `namespace` column; the
    /// schema-parity gate asserts the two never drift.
    pub const KHR_REGION: Self =
        Self(Self::EMISSION.0 | Self::CLEARCOAT.0 | Self::ANISOTROPY.0 | Self::SHEEN.0
            | Self::SUBSURFACE.0 | Self::TRANSMISSION.0);

    /// The Prism extension bit region (§4.5 constraint 2).
    ///
    /// Every non-`KHR` lobe (`PRISM`/`EXT`/`X_STUDIO`) occupies this high
    /// segment. It is disjoint from [`Self::KHR_REGION`], so a private lobe can
    /// never alias a specification-half field, and a downstream pure-OpenPBR
    /// consumer simply ignores anything outside [`Self::KHR_REGION`].
    pub const EXTENSION_REGION: Self = Self(Self::FACE.0);

    /// Number of distinct lobe bits defined; also the number of iterations of
    /// [`Self::iter_present`] over a fully populated mask.
    pub const COUNT: u32 = 7;

    /// Raw bits.
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Whether every bit in `other` is set here.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Union.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Count of present lobes (drives the packed length).
    pub const fn present_count(self) -> u32 {
        (self.0 & Self::all().0).count_ones()
    }

    /// The mask with every defined lobe bit set.
    pub const fn all() -> Self {
        Self((1 << Self::COUNT) - 1)
    }

    /// Keep only the bits inside the `OpenPBR` specification-core region.
    ///
    /// This is the loss-free half of [`crate::project_to_openpbr`]'s projection
    /// and the identity on any [`Self::is_pure_openpbr`] mask.
    pub const fn canonical_region(self) -> Self {
        Self(self.0 & Self::KHR_REGION.0)
    }

    /// Keep only the bits outside the `OpenPBR` specification-core region,
    /// i.e. the Prism/extension lobes a pure-OpenPBR downgrade must drop.
    pub const fn extension_region(self) -> Self {
        Self(self.0 & !Self::KHR_REGION.0)
    }

    /// Whether every set bit lives in the `OpenPBR` specification-core region.
    ///
    /// This is the precondition for the `OpenPBR` conformance gate (§4.5
    /// constraint 1): a pure-OpenPBR surface must reproduce the reference
    /// response bit-for-bit, and [`crate::project_to_openpbr`] must leave it
    /// untouched. Undefined bits are treated as non-conformant, so a corrupted
    /// mask can never masquerade as pure `OpenPBR`.
    pub const fn is_pure_openpbr(self) -> bool {
        self.0 & !Self::KHR_REGION.0 == 0
    }

    /// Which defined region `bit` belongs to, or `None` when `bit` is not a
    /// single defined lobe bit.
    pub const fn region_of(bit: Self) -> Option<LobeRegion> {
        if bit.0.count_ones() != 1 || bit.0 & Self::all().0 == 0 {
            return None;
        }
        if bit.0 & Self::KHR_REGION.0 != 0 {
            Some(LobeRegion::Canonical)
        } else {
            Some(LobeRegion::Extension)
        }
    }

    /// Canonical low-bit-first ordering of the defined lobes.
    const fn canonical() -> [Self; Self::COUNT as usize] {
        [
            Self::EMISSION,
            Self::CLEARCOAT,
            Self::ANISOTROPY,
            Self::SHEEN,
            Self::SUBSURFACE,
            Self::TRANSMISSION,
            Self::FACE,
        ]
    }
}

/// Which of the two disjoint bit regions a lobe belongs to (§4.5 constraint 2).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LobeRegion {
    /// The `OpenPBR` specification core (`namespace == "KHR"`).
    Canonical,
    /// The Prism extension region (`PRISM`/`EXT`/`X_STUDIO`).
    Extension,
}

/// The decoded über-BSDF surface: the always-present core plus every lobe value.
///
/// Lobe *values* are always materialized (so authoring/CPU code can read them
/// uniformly), but `lobe_mask` decides which lobes are considered live and are
/// therefore serialized by [`Self::pack`]. Lobes whose bit is clear serialize to
/// nothing and decode back to their neutral [`Default`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SurfaceParameterBlock {
    pub core: GpuSurfaceCore,
    pub lobe_mask: LobeMask,
    pub emission: GpuEmissionLobe,
    pub clearcoat: GpuClearCoatLobe,
    pub anisotropy: GpuAnisotropyLobe,
    pub sheen: GpuSheenLobe,
    pub subsurface: GpuSubsurfaceLobe,
    pub transmission: GpuTransmissionLobe,
    pub face: GpuFaceLobe,
}

/// Reasons an [`SurfaceParameterBlock::unpack`] can fail.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SurfaceUnpackError {
    /// The word slice length did not match `core + present lobes` for the mask.
    WrongLength { expected: usize, got: usize },
}

impl core::fmt::Display for SurfaceUnpackError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WrongLength { expected, got } => write!(
                f,
                "packed surface parameters length mismatch: expected {expected} words, got {got}"
            ),
        }
    }
}

impl std::error::Error for SurfaceUnpackError {}

impl SurfaceParameterBlock {
    /// Build a block from the decoded fat parameters, keeping only `lobe_mask`
    /// lobes live. Lobe values are copied verbatim regardless of the mask so a
    /// caller can inspect them, but only masked-in lobes will be serialized.
    pub fn from_full(params: &GpuSurfaceParameters, lobe_mask: LobeMask) -> Self {
        Self {
            core: GpuSurfaceCore {
                base_color: params.base_color,
                metallic: params.metallic,
                perceptual_roughness: params.perceptual_roughness,
                reflectance: params.reflectance,
                ambient_occlusion: params.ambient_occlusion,
                normal_scale: params.normal_scale,
                alpha_cutoff: params.alpha_cutoff,
                _pad0: 0.0,
                _pad1: 0.0,
            },
            lobe_mask,
            emission: GpuEmissionLobe {
                emissive: params.emissive,
            },
            clearcoat: GpuClearCoatLobe {
                clearcoat: params.clearcoat,
                clearcoat_roughness: params.clearcoat_roughness,
                _pad0: 0.0,
                _pad1: 0.0,
            },
            anisotropy: GpuAnisotropyLobe {
                anisotropy: params.anisotropy,
                anisotropy_rotation: params.anisotropy_rotation,
                _pad0: 0.0,
                _pad1: 0.0,
            },
            sheen: GpuSheenLobe {
                sheen: params.sheen,
                _pad0: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            },
            subsurface: GpuSubsurfaceLobe {
                subsurface: params.subsurface,
                _pad0: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            },
            transmission: GpuTransmissionLobe {
                transmission: params.transmission,
                thickness: params.thickness,
                index_of_refraction: params.index_of_refraction,
                dispersion: params.dispersion,
            },
            face: GpuFaceLobe {
                softness: params.face_softness,
                _pad0: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            },
        }
    }

    /// Reconstruct the fat parameters. The result starts from the canonical
    /// neutral [`GpuSurfaceParameters::default`] and overwrites the core plus
    /// only the present lobes, so an absent lobe reads back as its true neutral
    /// value (e.g. IOR `1.5`) and never leaks stale authored fields.
    pub fn to_full(&self) -> GpuSurfaceParameters {
        let mut full = GpuSurfaceParameters {
            base_color: self.core.base_color,
            metallic: self.core.metallic,
            perceptual_roughness: self.core.perceptual_roughness,
            reflectance: self.core.reflectance,
            ambient_occlusion: self.core.ambient_occlusion,
            normal_scale: self.core.normal_scale,
            alpha_cutoff: self.core.alpha_cutoff,
            ..GpuSurfaceParameters::default()
        };
        if self.lobe_mask.contains(LobeMask::EMISSION) {
            full.emissive = self.emission.emissive;
        }
        if self.lobe_mask.contains(LobeMask::CLEARCOAT) {
            full.clearcoat = self.clearcoat.clearcoat;
            full.clearcoat_roughness = self.clearcoat.clearcoat_roughness;
        }
        if self.lobe_mask.contains(LobeMask::ANISOTROPY) {
            full.anisotropy = self.anisotropy.anisotropy;
            full.anisotropy_rotation = self.anisotropy.anisotropy_rotation;
        }
        if self.lobe_mask.contains(LobeMask::SHEEN) {
            full.sheen = self.sheen.sheen;
        }
        if self.lobe_mask.contains(LobeMask::SUBSURFACE) {
            full.subsurface = self.subsurface.subsurface;
        }
        if self.lobe_mask.contains(LobeMask::TRANSMISSION) {
            full.transmission = self.transmission.transmission;
            full.thickness = self.transmission.thickness;
            full.index_of_refraction = self.transmission.index_of_refraction;
            full.dispersion = self.transmission.dispersion;
        }
        if self.lobe_mask.contains(LobeMask::FACE) {
            full.face_softness = self.face.softness;
        }
        full
    }

    /// Number of `u32` words [`Self::pack`] will emit for this block's mask.
    pub fn packed_len_words(&self) -> usize {
        SURFACE_CORE_WORDS
            + LobeMask::canonical()
                .into_iter()
                .filter(|bit| self.lobe_mask.contains(*bit))
                .map(lobe_word_count)
                .sum::<usize>()
    }

    /// Serialized byte size (always a multiple of 16).
    pub fn packed_size_bytes(&self) -> usize {
        self.packed_len_words() * size_of::<u32>()
    }

    /// The words of a lobe, selected by its mask bit.
    ///
    /// The slice length is the lobe's own `u32` footprint (its typed struct size
    /// equals the schema's `words`), so the pack cursor walks variable-length
    /// lobes correctly (§4.4 variable-length packing).
    fn lobe_words(&self, bit: LobeMask) -> &[u32] {
        match bit {
            LobeMask::EMISSION => bytemuck::cast_slice(core::slice::from_ref(&self.emission)),
            LobeMask::CLEARCOAT => bytemuck::cast_slice(core::slice::from_ref(&self.clearcoat)),
            LobeMask::ANISOTROPY => bytemuck::cast_slice(core::slice::from_ref(&self.anisotropy)),
            LobeMask::SHEEN => bytemuck::cast_slice(core::slice::from_ref(&self.sheen)),
            LobeMask::SUBSURFACE => bytemuck::cast_slice(core::slice::from_ref(&self.subsurface)),
            LobeMask::TRANSMISSION => {
                bytemuck::cast_slice(core::slice::from_ref(&self.transmission))
            }
            LobeMask::FACE => bytemuck::cast_slice(core::slice::from_ref(&self.face)),
            _ => &[],
        }
    }

    /// Serialize the core followed by only the present lobes, low bit first.
    pub fn pack(&self) -> Vec<u32> {
        let mut words = Vec::with_capacity(self.packed_len_words());
        words.extend_from_slice(bytemuck::cast_slice(core::slice::from_ref(&self.core)));
        for bit in LobeMask::canonical() {
            if self.lobe_mask.contains(bit) {
                words.extend_from_slice(self.lobe_words(bit));
            }
        }
        words
    }

    /// Reverse [`Self::pack`]: read the core, then each present lobe in
    /// canonical order. Absent lobes decode to their neutral defaults.
    pub fn unpack(lobe_mask: LobeMask, words: &[u32]) -> Result<Self, SurfaceUnpackError> {
        let expected = SURFACE_CORE_WORDS
            + LobeMask::canonical()
                .into_iter()
                .filter(|bit| lobe_mask.contains(*bit))
                .map(lobe_word_count)
                .sum::<usize>();
        if words.len() != expected {
            return Err(SurfaceUnpackError::WrongLength {
                expected,
                got: words.len(),
            });
        }
        let core: GpuSurfaceCore =
            *bytemuck::from_bytes(bytemuck::cast_slice(&words[..SURFACE_CORE_WORDS]));
        let mut block = Self {
            core,
            lobe_mask,
            ..Self::default()
        };
        let mut cursor = SURFACE_CORE_WORDS;
        for bit in LobeMask::canonical() {
            if !lobe_mask.contains(bit) {
                continue;
            }
            let lobe_len = lobe_word_count(bit);
            let slice = bytemuck::cast_slice(&words[cursor..cursor + lobe_len]);
            match bit {
                LobeMask::EMISSION => block.emission = *bytemuck::from_bytes(slice),
                LobeMask::CLEARCOAT => block.clearcoat = *bytemuck::from_bytes(slice),
                LobeMask::ANISOTROPY => block.anisotropy = *bytemuck::from_bytes(slice),
                LobeMask::SHEEN => block.sheen = *bytemuck::from_bytes(slice),
                LobeMask::SUBSURFACE => block.subsurface = *bytemuck::from_bytes(slice),
                LobeMask::TRANSMISSION => block.transmission = *bytemuck::from_bytes(slice),
                LobeMask::FACE => block.face = *bytemuck::from_bytes(slice),
                _ => {}
            }
            cursor += lobe_len;
        }
        Ok(block)
    }
}

/// The `u32` word footprint of the lobe whose mask bit is `bit`.
///
/// Slots are dense and low-bit-first (schema invariant, asserted by
/// `layout_matches_generated_table`), so a single defined bit indexes the
/// generated per-slot table directly. Callers always pass a single defined bit;
/// an undefined bit maps to slot `32`, which the table saturates to `0`.
fn lobe_word_count(bit: LobeMask) -> usize {
    debug_assert!(
        bit.bits().count_ones() == 1,
        "lobe_word_count expects a single lobe bit, got {bit:?}"
    );
    SURFACE_LOBE_WORDS_BY_SLOT
        .get(bit.bits().trailing_zeros() as usize)
        .copied()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn authored() -> GpuSurfaceParameters {
        GpuSurfaceParameters {
            base_color: [0.1, 0.2, 0.3, 1.0],
            emissive: [2.0, 0.0, 0.0, 1.0],
            metallic: 0.25,
            perceptual_roughness: 0.4,
            reflectance: 0.55,
            ambient_occlusion: 0.9,
            normal_scale: 0.8,
            alpha_cutoff: 0.37,
            transmission: 0.6,
            thickness: 1.5,
            clearcoat: 0.7,
            clearcoat_roughness: 0.15,
            anisotropy: 0.3,
            anisotropy_rotation: 0.2,
            sheen: 0.45,
            subsurface: 0.65,
            index_of_refraction: 1.33,
            dispersion: 0.05,
            face_softness: 0.25,
            _pad_face0: 0.0,
            _pad_face1: 0.0,
            _pad_face2: 0.0,
        }
    }

    #[test]
    fn layout_matches_generated_table() {
        use crate::surface_layout::{SURFACE_LOBES, SURFACE_LOBE_COUNT, SURFACE_LOBE_WORDS_BY_SLOT};

        // The hand-written typed enum must agree with the schema-generated
        // table in count, bit order, and per-lobe stride — this is the S2
        // guard that keeps `surface.rs` from drifting away from the schema.
        assert_eq!(LobeMask::COUNT, SURFACE_LOBE_COUNT);
        assert_eq!(LobeMask::canonical().len(), SURFACE_LOBES.len());
        for (mask, lobe) in LobeMask::canonical().iter().zip(SURFACE_LOBES.iter()) {
            // Canonical ordering is low-bit-first, so bit index == pack slot.
            assert_eq!(mask.bits(), 1 << lobe.registry_slot);
            // The per-slot stride is the source of truth for variable-length
            // packing; `SURFACE_LOBE_WORDS` is only the widest quantum.
            assert_eq!(
                lobe.words,
                SURFACE_LOBE_WORDS_BY_SLOT[lobe.registry_slot as usize]
            );
            assert!(lobe.words <= SURFACE_LOBE_WORDS);
        }
        // The all-lobes mask is exactly the OR of every generated slot.
        let generated_all = SURFACE_LOBES
            .iter()
            .fold(0u32, |acc, l| acc | (1 << l.registry_slot));
        assert_eq!(LobeMask::all().bits(), generated_all);
    }

    #[test]
    fn region_masks_match_the_schema_namespaces() {
        use crate::surface_layout::SURFACE_LOBES;

        // §4.5 constraint 2: the hand-written region masks must be exactly the
        // `KHR` (canonical) and non-`KHR` (extension) partitions of the
        // schema-generated lobe table. Drift here would let a private lobe
        // alias the specification region, or let the OpenPBR conformance gate
        // accept a non-OpenPBR field.
        let mut khr = LobeMask::default();
        let mut extension = LobeMask::default();
        for lobe in SURFACE_LOBES {
            let bit = LobeMask(1 << lobe.registry_slot);
            assert_eq!(LobeMask::region_of(bit), Some(expected_region(lobe.namespace)));
            if lobe.namespace == "KHR" {
                khr = khr.union(bit);
            } else {
                extension = extension.union(bit);
            }
        }
        assert_eq!(LobeMask::KHR_REGION, khr);
        assert_eq!(LobeMask::EXTENSION_REGION, extension);
        // The two regions are disjoint and together tile the defined bits.
        assert_eq!(LobeMask::KHR_REGION.bits() & LobeMask::EXTENSION_REGION.bits(), 0);
        assert_eq!(
            LobeMask::KHR_REGION.union(LobeMask::EXTENSION_REGION),
            LobeMask::all()
        );
    }

    fn expected_region(namespace: &str) -> LobeRegion {
        if namespace == "KHR" {
            LobeRegion::Canonical
        } else {
            LobeRegion::Extension
        }
    }

    #[test]
    fn pure_openpbr_masks_are_exactly_the_canonical_region() {
        assert!(LobeMask::default().is_pure_openpbr());
        assert!(LobeMask::all().contains(LobeMask::FACE));
        assert!(!LobeMask::all().is_pure_openpbr());
        for bit in LobeMask::canonical() {
            assert_eq!(bit.is_pure_openpbr(), bit != LobeMask::FACE);
            assert_eq!(bit.canonical_region() == bit, bit != LobeMask::FACE);
        }
        // Undefined bits must never pass the conformance precondition.
        assert!(!LobeMask(1 << 31).is_pure_openpbr());
        assert!(!LobeMask(1 << LobeMask::COUNT).is_pure_openpbr());
    }

    #[test]
    fn core_and_lobes_are_sixteen_byte_quanta() {
        assert_eq!(size_of::<GpuSurfaceCore>(), SURFACE_CORE_WORDS * 4);
        assert_eq!(size_of::<GpuSurfaceCore>() % 16, 0);
        // Every typed lobe struct's byte size must equal its schema-declared
        // per-slot footprint: this is the invariant that lets the pack cursor
        // advance by `SURFACE_LOBE_WORDS_BY_SLOT[slot]` (variable length) and
        // stay byte-exact with the WESL decoder's `PRISM_LOBE_WORDS_*`.
        let expected = [
            (LobeMask::EMISSION, size_of::<GpuEmissionLobe>()),
            (LobeMask::CLEARCOAT, size_of::<GpuClearCoatLobe>()),
            (LobeMask::ANISOTROPY, size_of::<GpuAnisotropyLobe>()),
            (LobeMask::SHEEN, size_of::<GpuSheenLobe>()),
            (LobeMask::SUBSURFACE, size_of::<GpuSubsurfaceLobe>()),
            (LobeMask::TRANSMISSION, size_of::<GpuTransmissionLobe>()),
            (LobeMask::FACE, size_of::<GpuFaceLobe>()),
        ];
        for (bit, bytes) in expected {
            assert_eq!(
                bytes,
                lobe_word_count(bit) * 4,
                "lobe {bit:?} struct size disagrees with its schema footprint"
            );
            assert_eq!(bytes % 16, 0);
        }
    }

    #[test]
    fn bare_dielectric_packs_core_only() {
        let block = SurfaceParameterBlock::from_full(&authored(), LobeMask::default());
        assert_eq!(block.packed_len_words(), SURFACE_CORE_WORDS);
        // 48 bytes vs the 96-byte fat struct: half the per-pixel traffic.
        assert_eq!(block.packed_size_bytes(), 48);
        assert!(block.packed_size_bytes() < size_of::<GpuSurfaceParameters>());
    }

    #[test]
    fn full_mask_packs_core_plus_every_lobe() {
        let block = SurfaceParameterBlock::from_full(&authored(), LobeMask::all());
        assert_eq!(
            block.packed_len_words(),
            SURFACE_CORE_WORDS
                + crate::surface_layout::SURFACE_LOBE_WORDS_BY_SLOT
                    .iter()
                    .sum::<usize>()
        );
        assert_eq!(block.pack().len(), block.packed_len_words());
    }

    #[test]
    fn pack_unpack_round_trips_for_every_lobe_subset() {
        let params = authored();
        for raw in 0..(1u32 << LobeMask::COUNT) {
            let mask = LobeMask(raw);
            let block = SurfaceParameterBlock::from_full(&params, mask);
            let words = block.pack();
            assert_eq!(words.len(), block.packed_len_words());
            let restored = SurfaceParameterBlock::unpack(mask, &words).unwrap();
            assert_eq!(restored.lobe_mask, mask);
            // Re-packing is idempotent: unpack(pack(x)) preserves the byte image.
            assert_eq!(restored.pack(), words);
            // The decoded fat view survives a round trip for present lobes and
            // falls back to neutral defaults for absent ones.
            assert_eq!(restored.to_full(), block.to_full());
        }
    }

    #[test]
    fn absent_lobes_decode_to_neutral_defaults() {
        let params = authored();
        // Emission live, everything else masked out.
        let block = SurfaceParameterBlock::from_full(&params, LobeMask::EMISSION);
        let full = block.to_full();
        assert_eq!(full.emissive, params.emissive);
        // Transmission/clearcoat/etc. were authored non-zero but are absent, so
        // they must read back as their neutral defaults, not the stale values.
        assert_eq!(full.transmission, 0.0);
        assert_eq!(full.clearcoat, 0.0);
        assert_eq!(full.index_of_refraction, 1.5);
    }

    #[test]
    fn unpack_rejects_wrong_length() {
        let err = SurfaceParameterBlock::unpack(LobeMask::all(), &[0; 3]).unwrap_err();
        assert!(matches!(err, SurfaceUnpackError::WrongLength { .. }));
    }
}
