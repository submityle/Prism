use crate::axis::{Illumination, SpecializationId};
use crate::ir::ClosureKind;
use crate::surface::{LobeMask, SurfaceParameterBlock};
use core::ops::{BitOr, BitOrAssign};
use prism_render_architecture::abi::GenerationalHandle;

pub const MATERIAL_ABI_VERSION: u32 = 4;
pub const MAX_MATERIAL_TEXTURES: usize = 8;
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
/// [`SpecializationId`]). `closure_graph_offset` points at the serialized
/// closure IR for RT/deferred consumption.
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
    /// (`packed_size_bytes()` = `(12 + present_lobes*4) * 4`).
    pub parameter_size: u32,
    pub texture_offset: u32,
    pub texture_count: u32,
    pub sampler_offset: u32,
    pub sampler_count: u32,
    pub custom_program: u32,
    pub active: u32,
    pub material_epoch_low: u32,
    pub material_epoch_high: u32,
    pub closure_graph_offset: u32,
    pub specialization_low: u32,
    pub specialization_high: u32,
    /// Which optional über-BSDF lobes this material carries
    /// ([`LobeMask`](crate::LobeMask) bits). Drives packed-parameter decode:
    /// shaders read `parameter_size / 4` words at `parameter_offset` and
    /// expand the core + present lobes back to a full surface using this mask
    /// (`material_unpack.wesl::prism_unpack_surface`, the byte-exact twin of
    /// [`crate::SurfaceParameterBlock::unpack`]).
    pub lobe_mask: u32,
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
    pub emissive: [f32; 4],
    pub metallic: f32,
    pub perceptual_roughness: f32,
    pub reflectance: f32,
    pub ambient_occlusion: f32,
    pub normal_scale: f32,
    pub alpha_cutoff: f32,
    pub transmission: f32,
    pub thickness: f32,
    pub clearcoat: f32,
    pub clearcoat_roughness: f32,
    pub anisotropy: f32,
    pub anisotropy_rotation: f32,
    pub sheen: f32,
    pub subsurface: f32,
    pub index_of_refraction: f32,
    pub dispersion: f32,
    /// Stylized face-shadow terminator softness (FACE lobe). Padding keeps the
    /// fat authoring view 16-byte aligned; only `face_softness` is packed.
    pub face_softness: f32,
    pub _pad_face0: f32,
    pub _pad_face1: f32,
    pub _pad_face2: f32,
}

impl Default for GpuSurfaceParameters {
    fn default() -> Self {
        Self {
            base_color: [1.0; 4],
            emissive: [0.0; 4],
            metallic: 0.0,
            perceptual_roughness: 0.5,
            reflectance: 0.5,
            ambient_occlusion: 1.0,
            normal_scale: 1.0,
            alpha_cutoff: 0.5,
            transmission: 0.0,
            thickness: 0.0,
            clearcoat: 0.0,
            clearcoat_roughness: 0.5,
            anisotropy: 0.0,
            anisotropy_rotation: 0.0,
            sheen: 0.0,
            subsurface: 0.0,
            index_of_refraction: 1.5,
            dispersion: 0.0,
            face_softness: 0.1,
            _pad_face0: 0.0,
            _pad_face1: 0.0,
            _pad_face2: 0.0,
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
        active: 1,
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
    pub fn specialization(&self) -> SpecializationId {
        SpecializationId::new(
            self.illumination,
            self.closure_mask,
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
            mask = mask.union(LobeMask::FACE);
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
        closure_graph_offset: u32,
        epoch: u64,
    ) -> GpuMaterialHeader {
        let spec = self.specialization();
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
            active: 1,
            material_epoch_low: epoch as u32,
            material_epoch_high: (epoch >> 32) as u32,
            closure_graph_offset,
            specialization_low: spec.low(),
            specialization_high: spec.high(),
            lobe_mask: self.lobe_mask().bits(),
        }
    }

    pub fn fixed_texture_rows(&self) -> [GpuMaterialTexture; MAX_MATERIAL_TEXTURES] {
        let mut rows = [GpuMaterialTexture::default(); MAX_MATERIAL_TEXTURES];
        let count = self.textures.len().min(MAX_MATERIAL_TEXTURES);
        rows[..count].copy_from_slice(&self.textures[..count]);
        rows
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
        let header = value.header(64, 128, 256, epoch);
        assert_eq!(header.active, 1);
        assert_eq!(header.generation, 7);
        assert_eq!(header.texture_count, 2);
        assert_eq!(header.parameter_offset, 64);
        assert_eq!(header.texture_offset, 128);
        assert_eq!(header.closure_graph_offset, 256);
        assert_eq!(header.custom_program, u32::MAX);
        // The split epoch and specialization key survive the round trip.
        let rebuilt_epoch =
            (u64::from(header.material_epoch_high) << 32) | u64::from(header.material_epoch_low);
        assert_eq!(rebuilt_epoch, epoch);
        assert_eq!(header.specialization(), value.specialization());
    }

    #[test]
    fn fixed_texture_rows_truncate_without_panicking() {
        let handle = GenerationalHandle::new(3, 1);
        let value = record(handle, 0, MAX_MATERIAL_TEXTURES + 3);
        let rows = value.fixed_texture_rows();
        assert_eq!(rows.len(), MAX_MATERIAL_TEXTURES);
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
        assert_eq!(header.active, 0);
        assert_eq!(header.custom_program, u32::MAX);
    }

    #[test]
    fn fallback_header_specialization_matches_axes() {
        let header = fallback_material_header(9);
        let expected =
            SpecializationId::new(Illumination::Lit, 1, MaterialRenderClass::Opaque as u32);
        assert_eq!(header.specialization(), expected);
        assert_eq!(header.active, 1);
    }
}
