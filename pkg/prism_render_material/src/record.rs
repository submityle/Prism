use core::ops::{BitOr, BitOrAssign};
use prism_render_architecture::abi::GenerationalHandle;

pub const MATERIAL_ABI_VERSION: u32 = 1;
pub const MAX_MATERIAL_TEXTURES: usize = 8;
pub const FALLBACK_MATERIAL_HANDLE: GenerationalHandle = GenerationalHandle {
    index: 0,
    generation: 0,
};

#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MaterialDomain {
    #[default]
    Surface,
    Decal,
    Volume,
    PostProcess,
}

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
    NprOpaque,
    NprTransparent,
    CustomOpaque,
    CustomTransparent,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MaterialShadingModel {
    #[default]
    Principled,
    Unlit,
    Subsurface,
    ClearCoat,
    Cloth,
    Hair,
    Water,
    Npr,
    Custom,
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

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMaterialHeader {
    pub generation: u32,
    pub revision: u32,
    pub shading_model: u32,
    pub render_class: u32,
    pub feature_flags: u32,
    pub closure_mask: u32,
    pub parameter_offset: u32,
    pub parameter_size: u32,
    pub texture_offset: u32,
    pub texture_count: u32,
    pub sampler_offset: u32,
    pub sampler_count: u32,
    pub custom_program: u32,
    pub active: u32,
    pub material_epoch_low: u32,
    pub material_epoch_high: u32,
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
    GpuMaterialHeader {
        generation: 0,
        shading_model: MaterialShadingModel::Principled as u32,
        render_class: MaterialRenderClass::Opaque as u32,
        parameter_size: size_of::<GpuSurfaceParameters>() as u32,
        custom_program: u32::MAX,
        active: 1,
        material_epoch_low: epoch as u32,
        material_epoch_high: (epoch >> 32) as u32,
        ..Default::default()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaterialRecord {
    pub handle: GenerationalHandle,
    pub revision: u32,
    pub domain: MaterialDomain,
    pub render_class: MaterialRenderClass,
    pub shading_model: MaterialShadingModel,
    pub features: MaterialFeatureFlags,
    pub closure_mask: u32,
    pub surface: GpuSurfaceParameters,
    pub textures: Vec<GpuMaterialTexture>,
    pub custom_program: Option<u32>,
}

impl MaterialRecord {
    pub fn header(
        &self,
        parameter_offset: u32,
        texture_offset: u32,
        epoch: u64,
    ) -> GpuMaterialHeader {
        GpuMaterialHeader {
            generation: self.handle.generation,
            revision: self.revision,
            shading_model: self.shading_model as u32,
            render_class: self.render_class as u32,
            feature_flags: self.features.0,
            closure_mask: self.closure_mask,
            parameter_offset,
            parameter_size: size_of::<GpuSurfaceParameters>() as u32,
            texture_offset,
            texture_count: self.textures.len() as u32,
            sampler_offset: texture_offset,
            sampler_count: self.textures.len() as u32,
            custom_program: self.custom_program.unwrap_or(u32::MAX),
            active: 1,
            material_epoch_low: epoch as u32,
            material_epoch_high: (epoch >> 32) as u32,
        }
    }

    pub fn fixed_texture_rows(&self) -> [GpuMaterialTexture; MAX_MATERIAL_TEXTURES] {
        let mut rows = [GpuMaterialTexture::default(); MAX_MATERIAL_TEXTURES];
        let count = self.textures.len().min(MAX_MATERIAL_TEXTURES);
        rows[..count].copy_from_slice(&self.textures[..count]);
        rows
    }
}
