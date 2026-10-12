use crate::axis::{Illumination, SpecializationId};
use crate::ir::ClosureKind;
use crate::surface::{LobeMask, SurfaceParameterBlock};
use core::ops::{BitOr, BitOrAssign};
use prism_render_architecture::abi::GenerationalHandle;

/// Material ABI version. Bumped to 5 by the first v5-only layout change: the
/// `clearcoat` lobe grew to carry the `OpenPBR` `coat_ior` / `coat_color` fields
/// (design doc §4.4 / P1-f), so a v4 reader can no longer decode the packed
/// block. The remaining v5 header/classification work (§8, P1-b/P1-h) lands on
/// top of this marker rather than re-issuing it.
pub const MATERIAL_ABI_VERSION: u32 = 5;
pub const FALLBACK_MATERIAL_HANDLE: GenerationalHandle = GenerationalHandle {
    index: 0,
    generation: 0,
};

/// The material domain axis. Orthogonal to `Illumination` and to the closure
/// graph; decides which pipeline family consumes the material.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MaterialDomain {
    #[default]
    Surface,
    Decal,
    Volume,
    PostProcess,
}

/// The blend / raster family axis.
///
/// This intentionally no longer carries `Npr*` / `Custom*` variants: those were
/// the Cartesian product of `blend × illumination` flattened into one enum.
/// Style now lives on the orthogonal [`Illumination`] axis, so a stylized
/// opaque material is simply `render_class = Opaque, illumination = Stylized`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MaterialRenderClass {
    #[default]
    Opaque,
    OpaqueTwoSided,
    Masked,
    MaskedTwoSided,
    Transmissive,
    Transparent,
    Additive,
    Volume,
    Hair,
    Water,
    Decal,
}

impl MaterialRenderClass {
    /// Recovers a render class from its `repr(u32)` discriminant.
    ///
    /// The inverse of the `as u32` cast the GPU header stores, so a value read
    /// back from a header (or decoded from a packed bin class) maps to the enum.
    #[must_use]
    pub const fn from_u32(value: u32) -> Option<Self> {
        Some(match value {
            0 => Self::Opaque,
            1 => Self::OpaqueTwoSided,
            2 => Self::Masked,
            3 => Self::MaskedTwoSided,
            4 => Self::Transmissive,
            5 => Self::Transparent,
            6 => Self::Additive,
            7 => Self::Volume,
            8 => Self::Hair,
            9 => Self::Water,
            10 => Self::Decal,
            _ => return None,
        })
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MaterialFeatureFlags(pub u32);

impl MaterialFeatureFlags {
    pub const DOUBLE_SIDED: Self = Self(1 << 0);
    pub const ALPHA_MASK: Self = Self(1 << 1);
    pub const TRANSMISSION: Self = Self(1 << 2);
    pub const NORMAL_MAP: Self = Self(1 << 3);
    pub const EMISSIVE: Self = Self(1 << 4);
    pub const COMPLEX_CLOSURE: Self = Self(1 << 5);
    pub const FACE_SHADOW: Self = Self(1 << 6);
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl BitOr for MaterialFeatureFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl BitOrAssign for MaterialFeatureFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// GPU-visible material header. The former `shading_model` field is gone;
/// style is carried by `illumination`, and the compiled permutation identity is
/// carried by `specialization_low`/`specialization_high` (a split
/// [`SpecializationId`]).
///
/// The runtime is **graph-free** (design §6.2): there is no serialized closure
/// IR. Deferred/RT consumers evaluate the same packed über
/// [`SurfaceParameterBlock`] the raster path does — one execution path, only the
/// *light source* switches (§8.5) — so the header carries no graph reference.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMaterialHeader {
    pub generation: u32,
    pub revision: u32,
    pub illumination: u32,
    pub render_class: u32,
    pub feature_flags: u32,
    pub closure_mask: u32,
    /// Word offset of this material's packed surface block inside the shared
    /// variable-length parameter word heap (ABI v4). No longer an element
    /// index into a fixed-stride `GpuSurfaceParameters` array: the scene packs
    /// only the über-BSDF core plus the live lobes (see [`crate::
    /// SurfaceParameterBlock`]), so blocks are variable width and this is a
    /// `u32`-word address decoded with `lobe_mask`.
    pub parameter_offset: u32,
    /// Byte length of the packed surface block at `parameter_offset`
    /// (`packed_size_bytes()` = `(SURFACE_CORE_WORDS + Σ present-lobe
    /// footprints) * 4`; lobes are variable length, so this is core + the
    /// present lobes' own word counts, not a fixed stride per lobe).
    pub parameter_size: u32,
    pub texture_offset: u32,
    pub texture_count: u32,
    pub sampler_offset: u32,
    pub sampler_count: u32,
    pub custom_program: u32,
    /// `1` for a live material, `0` for a dormant placeholder.
    ///
    /// Spelled `is_active`, not `active`: WGSL reserves `active`, so the
    /// WESL mirrors (`PrismMaterialHeader` / the standalone `MaterialHeader`)
    /// are forced to `is_active`, and the ABI's canonical Rust name follows
    /// suit. `tests/material_header_abi.rs` pins both sides to this exact
    /// spelling.
    pub is_active: u32,
    pub material_epoch_low: u32,
    pub material_epoch_high: u32,
    pub specialization_low: u32,
    pub specialization_high: u32,
    /// Which optional über-BSDF lobes this material carries
    /// ([`LobeMask`](crate::LobeMask) bits). Drives packed-parameter decode:
    /// shaders read `parameter_size / 4` words at `parameter_offset` and
    /// expand the core + present lobes back to a full surface using this mask
    /// (`material_unpack.wesl::prism_unpack_surface`, the byte-exact twin of
    /// [`crate::SurfaceParameterBlock::unpack`]).
    pub lobe_mask: u32,
    /// PCM extension digest, `used` half, slots `0..32` (§5.3, P1-h).
    ///
    /// The non-`KHR` capabilities this material **uses**. Optional capabilities
    /// a target lacks are gracefully ignored (the material projects to a
    /// variant without them), exactly like a glTF extension. Derived from
    /// `lobe_mask` by the namespace of each registered lobe, so it is `0` for a
    /// pure-`OpenPBR` material.
    ///
    /// The digest is four words — `used` and `required`, each split low/high
    /// ([`prism_material_pcm::HEADER_DIGEST_WORDS`]) — so all 64 registry slots
    /// are addressable.
    pub extensions_used_low: u32,
    /// PCM extension digest, `used` half, slots `32..64` (§5.3).
    pub extensions_used_high: u32,
    /// PCM extension digest, `required` half, slots `0..32` (§5.3). A target
    /// whose capability set does not satisfy every entry here is a *load-time
    /// error* (like a Vulkan required extension), never a silent runtime
    /// surprise.
    ///
    /// Policy: a **stable** extension is part of the engine's contract
    /// (required when present); an **experimental** one is optional and degrades
    /// gracefully ([`crate::capability::negotiate_digest`]).
    pub extensions_required_low: u32,
    /// PCM extension digest, `required` half, slots `32..64` (§5.3).
    pub extensions_required_high: u32,
}

impl GpuMaterialHeader {
    /// Recover the packed specialization key.
    pub const fn specialization(&self) -> SpecializationId {
        SpecializationId(((self.specialization_high as u64) << 32) | self.specialization_low as u64)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSurfaceParameters {
    pub base_color: [f32; 4],
    pub emission_color: [f32; 4],
    /// `OpenPBR` `emission_luminance` (nits): radiance = emissive * luminance.
    pub emission_luminance: f32,
    pub base_metalness: f32,
    pub specular_roughness: f32,
    /// `OpenPBR` `specular_weight`: scales the dielectric specular lobe.
    pub specular_weight: f32,
    pub ambient_occlusion: f32,
    pub normal_scale: f32,
    pub alpha_cutoff: f32,
    /// `OpenPBR` `base_weight`: scales the base diffuse + specular layer.
    pub base_weight: f32,
    /// `OpenPBR` `diffuse_roughness`: Oren-Nayar micro-slope roughness of the
    /// base diffuse lobe (`0` = Lambert).
    pub base_diffuse_roughness: f32,
    /// `OpenPBR` `specular_color`: per-channel tint of the dielectric specular.
    pub specular_color: [f32; 4],
    /// `OpenPBR` `specular_ior`: dielectric specular interface index.
    pub specular_ior: f32,
    /// `OpenPBR` `geometry_thin_walled`: single-interface open surface.
    pub thin_walled: f32,
    /// `OpenPBR` `geometry_opacity` in [0, 1].
    pub geometry_opacity: f32,
    pub transmission_weight: f32,
    pub thickness: f32,
    pub coat_weight: f32,
    pub coat_roughness: f32,
    /// `OpenPBR` `coat_ior`: relative index of refraction of the coat interface.
    pub coat_ior: f32,
    /// `OpenPBR` `coat_color`: per-channel tint of the coat highlight.
    pub coat_color: [f32; 4],
    /// `OpenPBR` `coat_roughness_anisotropy` in `[0, 1]`.
    pub coat_roughness_anisotropy: f32,
    /// `OpenPBR` `coat_darkening` in `[0, 1]`.
    pub coat_darkening: f32,
    pub anisotropy: f32,
    pub anisotropy_rotation: f32,
    pub fuzz_weight: f32,
    /// `OpenPBR` `fuzz_roughness`: independent width of the fuzz highlight.
    pub fuzz_roughness: f32,
    /// `OpenPBR` `fuzz_color`: per-channel tint of the fuzz lobe.
    pub fuzz_color: [f32; 4],
    pub subsurface_weight: f32,
    /// `OpenPBR` `subsurface_color`: tint of the scattered light.
    pub subsurface_color: [f32; 4],
    /// `OpenPBR` `subsurface_radius`: scalar scattering radius.
    pub subsurface_radius: f32,
    /// `OpenPBR` `subsurface_radius_scale`: per-channel split (color3).
    pub subsurface_radius_scale: [f32; 4],
    /// `OpenPBR` `scatter_anisotropy` in `[-1, 1]`.
    pub subsurface_scatter_anisotropy: f32,
    pub index_of_refraction: f32,
    pub transmission_dispersion_scale: f32,
    /// `OpenPBR` `transmission_color`: tint of the refracted light.
    pub transmission_color: [f32; 4],
    /// `OpenPBR` `transmission_scatter` (color3, w unused).
    pub transmission_scatter: [f32; 4],
    /// `OpenPBR` `thin_film_weight` / `_thickness` (µm) / `_ior`.
    pub thin_film_weight: f32,
    pub thin_film_thickness: f32,
    pub thin_film_ior: f32,
    /// `OpenPBR` `transmission_scatter_anisotropy` in `[-1, 1]`.
    pub transmission_scatter_anisotropy: f32,
    /// `OpenPBR` `transmission_dispersion_abbe_number` (`V_d`).
    pub transmission_dispersion_abbe_number: f32,
    /// Stylized face-shadow terminator softness (FACE lobe). Padding keeps the
    /// fat authoring view 16-byte aligned; only `npr_face_softness` is packed.
    pub npr_face_softness: f32,
    pub _pad_face0: f32,
    pub _pad_face1: f32,
    pub _pad_face2: f32,
    /// Tail padding keeping the fat authoring view a 16-byte multiple as the
    /// coat group grows. Carries no data.
    pub _pad_tail: [f32; 7],
}

impl Default for GpuSurfaceParameters {
    fn default() -> Self {
        Self {
            // The neutral surface is the `OpenPBR` default surface: every field
            // below takes the specification default so a material that authors
            // nothing is spec-conformant (design doc §4.4 / roadmap P1-f2).
            base_color: [0.8, 0.8, 0.8, 1.0],
            emission_color: [1.0; 4],
            emission_luminance: 0.0,
            base_metalness: 0.0,
            specular_roughness: 0.3,
            specular_weight: 1.0,
            ambient_occlusion: 1.0,
            normal_scale: 1.0,
            alpha_cutoff: 0.5,
            base_weight: 1.0,
            base_diffuse_roughness: 0.0,
            specular_color: [1.0; 4],
            specular_ior: 1.5,
            thin_walled: 0.0,
            geometry_opacity: 1.0,
            transmission_weight: 0.0,
            thickness: 0.0,
            coat_weight: 0.0,
            coat_roughness: 0.0,
            coat_ior: 1.6,
            coat_color: [1.0; 4],
            coat_roughness_anisotropy: 0.0,
            coat_darkening: 1.0,
            anisotropy: 0.0,
            anisotropy_rotation: 0.0,
            fuzz_weight: 0.0,
            fuzz_roughness: 0.5,
            fuzz_color: [1.0; 4],
            subsurface_weight: 0.0,
            subsurface_color: [0.8, 0.8, 0.8, 1.0],
            subsurface_radius: 1.0,
            subsurface_radius_scale: [1.0, 0.5, 0.25, 1.0],
            subsurface_scatter_anisotropy: 0.0,
            index_of_refraction: 1.5,
            transmission_dispersion_scale: 0.0,
            transmission_color: [1.0; 4],
            transmission_scatter: [0.0, 0.0, 0.0, 1.0],
            thin_film_weight: 0.0,
            thin_film_thickness: 0.5,
            thin_film_ior: 1.4,
            transmission_scatter_anisotropy: 0.0,
            transmission_dispersion_abbe_number: 20.0,
            npr_face_softness: 0.1,
            _pad_face0: 0.0,
            _pad_face1: 0.0,
            _pad_face2: 0.0,
            _pad_tail: [0.0; 7],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMaterialTexture {
    pub index: u32,
    pub generation: u32,
    pub semantic: u32,
    pub sampler_index: u32,
}

pub fn inactive_material_header(generation: u32) -> GpuMaterialHeader {
    GpuMaterialHeader {
        generation,
        custom_program: u32::MAX,
        ..Default::default()
    }
}

/// Slot zero is a permanent, generation-zero principled fallback. Consumers
/// can safely use it while an asynchronously loaded material is unavailable.
pub fn fallback_material_header(epoch: u64) -> GpuMaterialHeader {
    let spec = SpecializationId::new(Illumination::Lit, 1, MaterialRenderClass::Opaque as u32);
    GpuMaterialHeader {
        generation: 0,
        illumination: Illumination::Lit as u32,
        render_class: MaterialRenderClass::Opaque as u32,
        closure_mask: 1,
        parameter_size: SurfaceParameterBlock::from_full(
            &GpuSurfaceParameters::default(),
            LobeMask::default(),
        )
        .packed_size_bytes() as u32,
        custom_program: u32::MAX,
        is_active: 1,
        material_epoch_low: epoch as u32,
        material_epoch_high: (epoch >> 32) as u32,
        specialization_low: spec.low(),
        specialization_high: spec.high(),
        ..Default::default()
    }
}

pub fn fallback_material_record(handle: GenerationalHandle, revision: u64) -> MaterialRecord {
    MaterialRecord {
        handle,
        revision: revision as u32,
        domain: MaterialDomain::Surface,
        render_class: MaterialRenderClass::Opaque,
        illumination: Illumination::Lit,
        features: MaterialFeatureFlags::default(),
        closure_mask: 1,
        surface: GpuSurfaceParameters::default(),
        textures: Vec::new(),
        custom_program: None,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaterialRecord {
    pub handle: GenerationalHandle,
    pub revision: u32,
    pub domain: MaterialDomain,
    pub render_class: MaterialRenderClass,
    pub illumination: Illumination,
    pub features: MaterialFeatureFlags,
    pub closure_mask: u32,
    pub surface: GpuSurfaceParameters,
    pub textures: Vec<GpuMaterialTexture>,
    pub custom_program: Option<u32>,
}

impl MaterialRecord {
    /// The deterministic specialization identity for this record's axes.
    ///
    /// Keyed on the **derived** [`lobe_mask`](Self::lobe_mask), not the raw
    /// authored `closure_mask`: feature flags fold extra lobes into the live set
    /// (`EMISSIVE` / `TRANSMISSION` / `FACE_SHADOW`), and the shader permutation
    /// must follow the live lobes. Keying on the raw closure bits let two
    /// materials that differ only in such a flag collide on one permutation
    /// (e.g. one with a face-shadow lobe and one without), which is exactly what
    /// the "key = lobe set + feature flags" invariant forbids
    /// (`prism_material_pipeline_refactor_zh.md` §8.5).
    pub fn specialization(&self) -> SpecializationId {
        SpecializationId::new(
            self.illumination,
            self.lobe_mask().bits(),
            self.render_class as u32,
        )
    }

    /// Which optional über-BSDF lobes this record carries.
    ///
    /// Lobes with a dedicated closure kind (emission, clearcoat, sheen,
    /// subsurface, transmission) are driven by `closure_mask`; anisotropy has
    /// no closure kind of its own, so it is derived from a non-zero authored
    /// anisotropy amount. This is the mask serialized into the packed
    /// parameter block and mirrored into [`GpuMaterialHeader::lobe_mask`].
    pub fn lobe_mask(&self) -> LobeMask {
        let has = |kind: ClosureKind| self.closure_mask & (1 << kind as u32) != 0;
        let mut mask = LobeMask::default();
        if has(ClosureKind::Emission) || self.features.contains(MaterialFeatureFlags::EMISSIVE) {
            mask = mask.union(LobeMask::EMISSION);
        }
        if has(ClosureKind::ClearCoat) {
            mask = mask.union(LobeMask::CLEARCOAT);
        }
        if has(ClosureKind::Sheen) {
            mask = mask.union(LobeMask::SHEEN);
        }
        if has(ClosureKind::Subsurface) {
            mask = mask.union(LobeMask::SUBSURFACE);
        }
        if has(ClosureKind::Transmission)
            || self.features.contains(MaterialFeatureFlags::TRANSMISSION)
        {
            mask = mask.union(LobeMask::TRANSMISSION);
        }
        if self.surface.anisotropy != 0.0 || self.surface.anisotropy_rotation != 0.0 {
            mask = mask.union(LobeMask::ANISOTROPY);
        }
        if self.features.contains(MaterialFeatureFlags::FACE_SHADOW) {
            mask = mask.union(LobeMask::NPR_FACE);
        }
        mask
    }

    /// The compact core-plus-present-lobes view of this record's surface
    /// (design doc §3.3). [`SurfaceParameterBlock::pack`] serializes only the
    /// live lobes, so a plain dielectric costs 12 words instead of 24.
    pub fn packed_parameters(&self) -> SurfaceParameterBlock {
        SurfaceParameterBlock::from_full(&self.surface, self.lobe_mask())
    }

    pub fn header(
        &self,
        parameter_offset: u32,
        texture_offset: u32,
        epoch: u64,
    ) -> GpuMaterialHeader {
        let spec = self.specialization();
        let digest = crate::capability::digest_for(self.lobe_mask());
        GpuMaterialHeader {
            generation: self.handle.generation,
            revision: self.revision,
            illumination: self.illumination as u32,
            render_class: self.render_class as u32,
            feature_flags: self.features.0,
            closure_mask: self.closure_mask,
            parameter_offset,
            parameter_size: self.packed_parameters().packed_size_bytes() as u32,
            texture_offset,
            texture_count: self.textures.len() as u32,
            sampler_offset: texture_offset,
            sampler_count: self.textures.len() as u32,
            custom_program: self.custom_program.unwrap_or(u32::MAX),
            is_active: 1,
            material_epoch_low: epoch as u32,
            material_epoch_high: (epoch >> 32) as u32,
            specialization_low: spec.low(),
            specialization_high: spec.high(),
            lobe_mask: self.lobe_mask().bits(),
            // PCM (§5.3): `extensions_used` is the non-`KHR` subset of
            // `lobe_mask`, `extensions_required` the subset of that the schema
            // marks `stable`. Both come from the *derivation*, not from a field
            // an author could set: the catalogue in `capability.rs` is the only
            // place the policy lives, so a header can never claim an extension
            // the material does not actually carry.
            extensions_used_low: digest.used_low(),
            extensions_used_high: digest.used_high(),
            extensions_required_low: digest.required_low(),
            extensions_required_high: digest.required_high(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn closure_bit(kind: ClosureKind) -> u32 {
        1 << kind as u32
    }

    fn record(handle: GenerationalHandle, closure_mask: u32, textures: usize) -> MaterialRecord {
        MaterialRecord {
            handle,
            revision: 3,
            domain: MaterialDomain::Surface,
            render_class: MaterialRenderClass::Opaque,
            illumination: Illumination::Lit,
            features: MaterialFeatureFlags::default(),
            closure_mask,
            surface: GpuSurfaceParameters::default(),
            textures: vec![GpuMaterialTexture::default(); textures],
            custom_program: None,
        }
    }

    #[test]
    fn feature_flags_union_and_contains() {
        let mut flags = MaterialFeatureFlags::default();
        assert!(!flags.contains(MaterialFeatureFlags::EMISSIVE));
        flags |= MaterialFeatureFlags::EMISSIVE;
        let combined = flags | MaterialFeatureFlags::DOUBLE_SIDED;
        assert!(combined.contains(MaterialFeatureFlags::EMISSIVE));
        assert!(combined.contains(MaterialFeatureFlags::DOUBLE_SIDED));
        assert!(!combined.contains(MaterialFeatureFlags::TRANSMISSION));
    }

    #[test]
    fn lobe_mask_is_derived_from_closure_bits_and_anisotropy() {
        let handle = GenerationalHandle::new(1, 1);
        let mut value = record(handle, closure_bit(ClosureKind::Emission), 0);
        value.surface.anisotropy = 0.5;
        let mask = value.lobe_mask();
        assert!(mask.contains(LobeMask::EMISSION));
        assert!(mask.contains(LobeMask::ANISOTROPY));
        assert!(!mask.contains(LobeMask::SHEEN));
    }

    #[test]
    fn specialization_follows_the_derived_lobe_set_not_the_raw_closure_bits() {
        // Two records with identical axes/closure bits but a differing feature
        // flag differ in their live lobe set, so they must not share a
        // permutation: keying on the raw `closure_mask` collided them.
        let handle = GenerationalHandle::new(1, 1);
        let plain = record(handle, 0, 0);
        let mut face_shadow = record(handle, 0, 0);
        face_shadow.features |= MaterialFeatureFlags::FACE_SHADOW;

        assert_ne!(
            plain.lobe_mask(),
            face_shadow.lobe_mask(),
            "the feature flag adds a live lobe"
        );
        assert_ne!(
            plain.specialization(),
            face_shadow.specialization(),
            "a differing live lobe set yields a different permutation key"
        );

        // A feature-driven lobe that also has a closure bit keys the same way.
        let mut emissive = record(handle, 0, 0);
        emissive.features |= MaterialFeatureFlags::EMISSIVE;
        let by_closure = record(handle, closure_bit(ClosureKind::Emission), 0);
        assert_eq!(
            emissive.specialization(),
            by_closure.specialization(),
            "the same live lobe keys identically however it was authored"
        );
    }

    #[test]
    fn emissive_feature_flag_alone_marks_the_emission_lobe() {
        let handle = GenerationalHandle::new(1, 1);
        let mut value = record(handle, 0, 0);
        value.features |= MaterialFeatureFlags::EMISSIVE;
        assert!(value.lobe_mask().contains(LobeMask::EMISSION));
    }

    #[test]
    fn header_packs_specialization_and_counts() {
        let handle = GenerationalHandle::new(2, 7);
        let value = record(handle, closure_bit(ClosureKind::ClearCoat), 2);
        let epoch = 0x1_0000_0002_u64;
        let header = value.header(64, 128, epoch);
        assert_eq!(header.is_active, 1);
        assert_eq!(header.generation, 7);
        assert_eq!(header.texture_count, 2);
        assert_eq!(header.parameter_offset, 64);
        assert_eq!(header.texture_offset, 128);
        assert_eq!(header.custom_program, u32::MAX);
        // The split epoch and specialization key survive the round trip.
        let rebuilt_epoch =
            (u64::from(header.material_epoch_high) << 32) | u64::from(header.material_epoch_low);
        assert_eq!(rebuilt_epoch, epoch);
        assert_eq!(header.specialization(), value.specialization());
    }

    #[test]
    fn fallback_record_is_a_lit_principled_surface() {
        let handle = GenerationalHandle::new(0, 0);
        let value = fallback_material_record(handle, 42);
        assert_eq!(value.revision, 42);
        assert_eq!(value.domain, MaterialDomain::Surface);
        assert_eq!(value.illumination, Illumination::Lit);
        assert_eq!(value.render_class, MaterialRenderClass::Opaque);
        assert_eq!(value.closure_mask, 1);
        assert!(value.textures.is_empty());
    }

    #[test]
    fn inactive_header_is_generation_tagged_and_dormant() {
        let header = inactive_material_header(5);
        assert_eq!(header.generation, 5);
        assert_eq!(header.is_active, 0);
        assert_eq!(header.custom_program, u32::MAX);
    }

    #[test]
    fn render_class_round_trips_through_its_discriminant() {
        let classes = [
            MaterialRenderClass::Opaque,
            MaterialRenderClass::OpaqueTwoSided,
            MaterialRenderClass::Masked,
            MaterialRenderClass::MaskedTwoSided,
            MaterialRenderClass::Transmissive,
            MaterialRenderClass::Transparent,
            MaterialRenderClass::Additive,
            MaterialRenderClass::Volume,
            MaterialRenderClass::Hair,
            MaterialRenderClass::Water,
            MaterialRenderClass::Decal,
        ];
        for (value, class) in classes.into_iter().enumerate() {
            assert_eq!(
                class as u32, value as u32,
                "{class:?} is the {value}th dense discriminant"
            );
            assert_eq!(
                MaterialRenderClass::from_u32(value as u32),
                Some(class),
                "{class:?} round trips"
            );
        }
        assert_eq!(
            MaterialRenderClass::from_u32(classes.len() as u32),
            None,
            "unassigned rejected"
        );
        assert_eq!(
            MaterialRenderClass::from_u32(u32::MAX),
            None,
            "garbage rejected"
        );
    }

    #[test]
    fn fallback_header_specialization_matches_axes() {
        let header = fallback_material_header(9);
        let expected =
            SpecializationId::new(Illumination::Lit, 1, MaterialRenderClass::Opaque as u32);
        assert_eq!(header.specialization(), expected);
        assert_eq!(header.is_active, 1);
    }
}
