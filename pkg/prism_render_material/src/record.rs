use crate::execution::MaterialExecutionBinding;
use crate::ir::ClosureKind;
use crate::shading_plan::ShadingPlan;
use crate::surface::{LobeMask, SurfaceParameterBlock};
use core::ops::{BitOr, BitOrAssign};
use prism_render_architecture::abi::GenerationalHandle;
pub use prism_render_architecture::render_fabric::RenderDomain;

/// Material ABI version. Bumped to 8 when the header gained an explicit
/// render-domain word alongside the five-word
/// dense material execution binding (plan, response program, package and the
/// full 64-bit permutation index). The binding is a table identity, never a
/// hash or a truncated specialization surrogate.
///
/// The earlier v5-only layout change was the first version in which the
/// `clearcoat` lobe grew to carry the `OpenPBR` `coat_ior` / `coat_color` fields
/// (design doc §4.4 / P1-f), so a v4 reader can no longer decode the packed
/// block. The remaining v5 header/classification work (§8, P1-b/P1-h) lands on
/// top of this marker rather than re-issuing it.
pub const MATERIAL_ABI_VERSION: u32 = 8;

/// Sentinel for every unbound material-execution word.
pub const MATERIAL_EXECUTION_BINDING_SENTINEL: u32 = u32::MAX;

/// Sentinel in [`GpuMaterialTexture::virtual_texture_id`] for a row that is
/// not backed by a virtual texture. This is deliberately not zero: zero is a
/// valid texture key for implementations that allocate keys densely.
pub const NO_VIRTUAL_TEXTURE: u32 = u32::MAX;
/// Reserved row marker for the normalized non-bindless material atlas.
pub const PORTABLE_TEXTURE_ATLAS: u32 = u32::MAX - 1;
pub const FALLBACK_MATERIAL_HANDLE: GenerationalHandle = GenerationalHandle {
    index: 0,
    generation: 0,
};

/// The material domain axis. Orthogonal to the response plan and closure graph;
/// it decides which pipeline family consumes the material.
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
/// This intentionally does not carry response-program variants: response
/// execution is selected by [`ShadingPlan`], not by raster/blend classification.
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
    /// The material carries a region-illumination mask texture. The visibility
    /// raster samples it per pixel and writes the Lit/Stylized selector into
    /// the visibility metadata's spare word; classification then bins each
    /// region into the requested response without changing the material ABI.
    pub const REGION_ILLUMINATION: Self = Self(1 << 7);
    /// The surface belongs to the dedicated water feature family. Water is a
    /// surface domain; this bit selects its response bucket.
    pub const WATER: Self = Self(1 << 8);
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

/// GPU-visible material header. Response and pipeline identity is carried only
/// by the verified dense execution binding at the tail of this record; the
/// legacy illumination/custom/specialization header words are intentionally not
/// part of ABI8.
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
    pub render_class: u32,
    /// Runtime object domain, orthogonal to the seven blend/raster classes.
    pub render_domain: u32,
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
    /// Runtime dynamic channel `wetness` (§17.1), an `f32` bitcast into the word.
    ///
    /// The four dynamic words are the **only** part of the header a runtime state
    /// change touches: the material system rewrites them in place (marking the
    /// row dirty so the next upload carries them) without republishing the
    /// record, changing the `generation` the visibility buffer validated, or
    /// perturbing the compiled permutation. `0.0` is the exact identity for every
    /// channel, so a material that never drives one is bit-for-bit unchanged.
    pub dynamic_wetness: u32,
    /// Runtime dynamic channel `snow` (§17.1), an `f32` bitcast into the word.
    pub dynamic_snow: u32,
    /// Runtime dynamic channel `damage` (§17.1), an `f32` bitcast into the word.
    pub dynamic_damage: u32,
    /// Runtime dynamic channel `dissolve` (§17.1), an `f32` bitcast into the word.
    pub dynamic_dissolve: u32,
    /// Dense index into the material execution plan table, or the all-ones
    /// sentinel when no verified execution binding is attached.
    pub execution_plan_index: u32,
    /// Dense index into the response-program table, or the all-ones sentinel.
    pub execution_program_index: u32,
    /// Dense index into the verified shader-package table, or the all-ones
    /// sentinel.
    pub execution_package_index: u32,
    /// Low word of the complete checked permutation index.
    pub execution_permutation_low: u32,
    /// High word of the complete checked permutation index.
    pub execution_permutation_high: u32,
}

/// The four runtime dynamic channels of §17.1 — the axes the PCM catalogue
/// registers as `kind = channel`.
///
/// These are *inputs*, not authored über parameters. The bake freezes the static
/// surface; a runtime state change then rewrites four header words and the
/// shading path modulates the frozen fields analytically. That is what keeps
/// "wet car / drying puddle / melting snow / burning away" out of the
/// permutation space (§8.4) while staying deterministic and branch-free: the
/// modulation is a closed-form remap of existing fields, not a new lobe.
///
/// Channel semantics (all in `[0, 1]`, `0` = exact identity):
/// * `wetness` — a water film fills the pores and sits on top: darkens the base,
///   smooths both the diffuse and the specular lobe, raises the specular
///   interface toward water's index, and mats the fuzz down.
/// * `snow` — a bright, rough top layer: drives the base toward white, kills
///   metalness, and coarsens the highlight into a diffuse-dominant response.
/// * `damage` — chipped/oxidised wear: darkens and desaturates the base, breaks
///   the metal interface up, and roughens the surface.
/// * `dissolve` — a coverage cut against a per-pixel mask: shrinks the geometry
///   opacity and lights an emissive rim along the cut line.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DynamicChannels {
    /// Water film coverage.
    pub wetness: f32,
    /// Snow cover.
    pub snow: f32,
    /// Wear/chip coverage.
    pub damage: f32,
    /// Dissolve progress against the per-pixel mask.
    pub dissolve: f32,
}

impl DynamicChannels {
    /// Every channel inert. The modulation functions are exact identities here,
    /// so a material that never drives a channel is bit-for-bit unchanged.
    pub const NONE: Self = Self {
        wetness: 0.0,
        snow: 0.0,
        damage: 0.0,
        dissolve: 0.0,
    };

    /// Sanitise and pack into the header's four dynamic words.
    ///
    /// Runtime state is *outside* the schema validators, so the contract is
    /// enforced here instead: a non-finite channel becomes inert (`0`) and the
    /// rest are clamped into `[0, 1]`. The shading path can then consume the
    /// words unguarded — no hot-path `is_nan` per channel.
    #[must_use]
    pub fn to_words(self) -> [u32; 4] {
        [
            sanitize_channel(self.wetness).to_bits(),
            sanitize_channel(self.snow).to_bits(),
            sanitize_channel(self.damage).to_bits(),
            sanitize_channel(self.dissolve).to_bits(),
        ]
    }

    /// Unpack the header's four dynamic words.
    #[must_use]
    pub const fn from_words(words: [u32; 4]) -> Self {
        Self {
            wetness: f32::from_bits(words[0]),
            snow: f32::from_bits(words[1]),
            damage: f32::from_bits(words[2]),
            dissolve: f32::from_bits(words[3]),
        }
    }

    /// Whether every channel is inert.
    #[must_use]
    pub fn is_inert(self) -> bool {
        self == Self::NONE
    }
}

/// Clamp one runtime channel into `[0, 1]`, mapping a non-finite value to inert.
#[must_use]
fn sanitize_channel(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

impl GpuMaterialHeader {
    /// Sentinel used by the ABI for an unbound execution identity.
    pub const EXECUTION_BINDING_SENTINEL: u32 = MATERIAL_EXECUTION_BINDING_SENTINEL;

    /// Returns the five ABI words in the same order as
    /// [`MaterialExecutionBinding::to_words`].
    #[must_use]
    pub const fn execution_binding_words(&self) -> [u32; 5] {
        [
            self.execution_plan_index,
            self.execution_program_index,
            self.execution_package_index,
            self.execution_permutation_low,
            self.execution_permutation_high,
        ]
    }

    /// Whether this header carries a complete execution binding rather than
    /// the all-ones unbound sentinel.
    #[must_use]
    pub const fn has_execution_binding(&self) -> bool {
        self.execution_plan_index != MATERIAL_EXECUTION_BINDING_SENTINEL
            && self.execution_program_index != MATERIAL_EXECUTION_BINDING_SENTINEL
            && self.execution_package_index != MATERIAL_EXECUTION_BINDING_SENTINEL
            && self.execution_permutation_low != MATERIAL_EXECUTION_BINDING_SENTINEL
            && self.execution_permutation_high != MATERIAL_EXECUTION_BINDING_SENTINEL
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
    /// Prism glint lobe (§17.6): blends a discrete (counting) microfacet NDF
    /// into the specular lobe. `glint_weight == 0` is an exact identity.
    pub glint_weight: f32,
    /// Orientation roughness of the countable flakes (independent of the base
    /// `specular_roughness`).
    pub glint_roughness: f32,
    /// Facets per unit² of the shade-time surface-cell space (the UV cell the
    /// glint seed quantises); multiplied by the per-pixel footprint area.
    pub glint_density: f32,
    /// Angular half-width (radians) of the half-vector acceptance bin.
    pub glint_bin_half_width: f32,
    /// Measured (tabulated) BRDF/BTF lobe (§17.6): blend weight, the bound
    /// table's difference-azimuth resolution, and the table-to-radiance scale.
    pub measured_weight: f32,
    pub measured_phi_d: f32,
    pub measured_scale: f32,
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
            // Glint defaults: weight 0 keeps the lobe inert, density 4/unit² and
            // a ~2.9° acceptance bin are the values the shading-module tests are
            // baselined against.
            glint_weight: 0.0,
            glint_roughness: 0.1,
            glint_density: 4.0,
            glint_bin_half_width: 0.05,
            measured_weight: 0.0,
            measured_phi_d: 180.0,
            measured_scale: 1.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMaterialTexture {
    /// Bindless texture heap slot used by the current non-VT sampling path.
    pub index: u32,
    /// Generation of [`index`] when this row was published.
    pub generation: u32,
    /// Stable material texture semantic (base color, normal, etc.).
    pub semantic: u32,
    /// Bindless sampler heap slot paired with [`index`].
    pub sampler_index: u32,
    /// Stable virtual-texture resource key. [`NO_VIRTUAL_TEXTURE`] means that
    /// VT is disabled for this row; no VT page lookup is implied by the other
    /// fields.
    pub virtual_texture_id: u32,
    /// Mip-zero width in texels, meaningful only when VT is enabled.
    pub virtual_texture_width: u32,
    /// Mip-zero height in texels, meaningful only when VT is enabled.
    pub virtual_texture_height: u32,
    /// Square VT page edge in texels, meaningful only when VT is enabled.
    pub virtual_texture_page_size: u32,
}

impl Default for GpuMaterialTexture {
    fn default() -> Self {
        Self {
            index: 0,
            generation: 0,
            semantic: 0,
            sampler_index: 0,
            virtual_texture_id: NO_VIRTUAL_TEXTURE,
            virtual_texture_width: 0,
            virtual_texture_height: 0,
            virtual_texture_page_size: 0,
        }
    }
}

impl GpuMaterialTexture {
    /// Build an ordinary bindless row with virtual texturing explicitly off.
    #[must_use]
    pub const fn non_virtual(
        index: u32,
        generation: u32,
        semantic: u32,
        sampler_index: u32,
    ) -> Self {
        Self {
            index,
            generation,
            semantic,
            sampler_index,
            virtual_texture_id: NO_VIRTUAL_TEXTURE,
            virtual_texture_width: 0,
            virtual_texture_height: 0,
            virtual_texture_page_size: 0,
        }
    }

    /// Build a row for the normalized non-bindless material atlas. The same
    /// row remains a normal direct bindless row in the bindless specialization;
    /// the shader only interprets the marker on the portable variant.
    #[must_use]
    pub const fn with_portable_atlas(
        index: u32,
        generation: u32,
        semantic: u32,
        sampler_index: u32,
        width: u32,
        height: u32,
        page_size: u32,
    ) -> Self {
        Self {
            index,
            generation,
            semantic,
            sampler_index,
            virtual_texture_id: PORTABLE_TEXTURE_ATLAS,
            virtual_texture_width: width,
            virtual_texture_height: height,
            virtual_texture_page_size: page_size,
        }
    }

    /// Build a row carrying VT metadata. This records metadata only; a VT
    /// page table, residency service, and shader consumer are still required
    /// before a renderer may sample it as a virtual texture.
    #[must_use]
    pub const fn with_virtual_texture(
        index: u32,
        generation: u32,
        semantic: u32,
        sampler_index: u32,
        virtual_texture_id: u32,
        virtual_texture_width: u32,
        virtual_texture_height: u32,
        virtual_texture_page_size: u32,
    ) -> Self {
        Self {
            index,
            generation,
            semantic,
            sampler_index,
            virtual_texture_id,
            virtual_texture_width,
            virtual_texture_height,
            virtual_texture_page_size,
        }
    }
}

pub fn inactive_material_header(generation: u32) -> GpuMaterialHeader {
    GpuMaterialHeader {
        generation,
        render_domain: RenderDomain::Surface as u32,
        execution_plan_index: MATERIAL_EXECUTION_BINDING_SENTINEL,
        execution_program_index: MATERIAL_EXECUTION_BINDING_SENTINEL,
        execution_package_index: MATERIAL_EXECUTION_BINDING_SENTINEL,
        execution_permutation_low: MATERIAL_EXECUTION_BINDING_SENTINEL,
        execution_permutation_high: MATERIAL_EXECUTION_BINDING_SENTINEL,
        ..Default::default()
    }
}

/// Slot zero is a permanent, generation-zero principled fallback. Consumers
/// can safely use it while an asynchronously loaded material is unavailable.
pub fn fallback_material_header(epoch: u64) -> GpuMaterialHeader {
    GpuMaterialHeader {
        generation: 0,
        render_domain: RenderDomain::Surface as u32,
        render_class: MaterialRenderClass::Opaque as u32,
        closure_mask: 1,
        parameter_size: SurfaceParameterBlock::from_full(
            &GpuSurfaceParameters::default(),
            LobeMask::default(),
        )
        .packed_size_bytes() as u32,
        execution_plan_index: MATERIAL_EXECUTION_BINDING_SENTINEL,
        execution_program_index: MATERIAL_EXECUTION_BINDING_SENTINEL,
        execution_package_index: MATERIAL_EXECUTION_BINDING_SENTINEL,
        execution_permutation_low: MATERIAL_EXECUTION_BINDING_SENTINEL,
        execution_permutation_high: MATERIAL_EXECUTION_BINDING_SENTINEL,
        is_active: 1,
        material_epoch_low: epoch as u32,
        material_epoch_high: (epoch >> 32) as u32,
        ..Default::default()
    }
}

pub fn fallback_material_record(handle: GenerationalHandle, revision: u64) -> MaterialRecord {
    MaterialRecord {
        handle,
        revision: revision as u32,
        domain: MaterialDomain::Surface,
        render_domain: RenderDomain::Surface,
        render_class: MaterialRenderClass::Opaque,
        response_plan: ShadingPlan::default(),
        features: MaterialFeatureFlags::default(),
        closure_mask: 1,
        surface: GpuSurfaceParameters::default(),
        textures: Vec::new(),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaterialRecord {
    pub handle: GenerationalHandle,
    pub revision: u32,
    pub domain: MaterialDomain,
    /// Runtime object domain used by pipeline-state selection.
    pub render_domain: RenderDomain,
    pub render_class: MaterialRenderClass,
    /// The response/composition plan selected by material lowering. It is
    /// authoring data; verified dense execution identities live in the registry
    /// binding and in the ABI8 execution words.
    pub response_plan: ShadingPlan,
    pub features: MaterialFeatureFlags,
    pub closure_mask: u32,
    pub surface: GpuSurfaceParameters,
    pub textures: Vec<GpuMaterialTexture>,
}

impl MaterialRecord {
    /// Stable coarse ordering for visibility work. This is only a sort hint;
    /// pipeline identity comes from the resolved execution/pass tables.
    #[must_use]
    pub const fn response_sort_key(&self) -> u8 {
        match self.response_plan {
            ShadingPlan::Single { program } => match program {
                crate::ProgramRef::Builtin(crate::BuiltinProgram::Lit) => 0,
                crate::ProgramRef::Builtin(crate::BuiltinProgram::Npr) => 1,
                crate::ProgramRef::Builtin(crate::BuiltinProgram::Unlit) => 2,
                crate::ProgramRef::Registered(_) => 3,
            },
            ShadingPlan::MaskBlend { .. } => 4,
        }
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
        self.header_with_execution(parameter_offset, texture_offset, epoch, None)
    }

    /// Builds a header and copies a verified execution binding without
    /// re-encoding its identities. `None` preserves the ABI's all-ones
    /// unbound sentinel in all five execution words.
    pub fn header_with_execution(
        &self,
        parameter_offset: u32,
        texture_offset: u32,
        epoch: u64,
        execution: Option<MaterialExecutionBinding>,
    ) -> GpuMaterialHeader {
        let digest = crate::capability::digest_for(self.lobe_mask());
        let [execution_plan_index, execution_program_index, execution_package_index, execution_permutation_low, execution_permutation_high] =
            execution
                .map(MaterialExecutionBinding::to_words)
                .unwrap_or([MATERIAL_EXECUTION_BINDING_SENTINEL; 5]);
        GpuMaterialHeader {
            generation: self.handle.generation,
            revision: self.revision,
            render_class: self.render_class as u32,
            render_domain: self.render_domain as u32,
            feature_flags: self.features.0,
            closure_mask: self.closure_mask,
            parameter_offset,
            parameter_size: self.packed_parameters().packed_size_bytes() as u32,
            texture_offset,
            texture_count: self.textures.len() as u32,
            sampler_offset: texture_offset,
            sampler_count: self.textures.len() as u32,
            is_active: 1,
            material_epoch_low: epoch as u32,
            material_epoch_high: (epoch >> 32) as u32,
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
            // §17.1 dynamic channels are *runtime* state, not part of the bake,
            // so a freshly built header starts inert; the registry injects the
            // live values when it uploads the row (see
            // `MaterialRegistry::gpu_tables`).
            dynamic_wetness: 0,
            dynamic_snow: 0,
            dynamic_damage: 0,
            dynamic_dissolve: 0,
            execution_plan_index,
            execution_program_index,
            execution_package_index,
            execution_permutation_low,
            execution_permutation_high,
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
            render_domain: RenderDomain::Surface,
            render_class: MaterialRenderClass::Opaque,
            response_plan: ShadingPlan::default(),
            features: MaterialFeatureFlags::default(),
            closure_mask,
            surface: GpuSurfaceParameters::default(),
            textures: vec![GpuMaterialTexture::default(); textures],
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
        assert_ne!(plain.lobe_mask(), face_shadow.lobe_mask());

        // A feature-driven lobe that also has a closure bit keys the same way.
        let mut emissive = record(handle, 0, 0);
        emissive.features |= MaterialFeatureFlags::EMISSIVE;
        let by_closure = record(handle, closure_bit(ClosureKind::Emission), 0);
        assert_eq!(
            emissive.lobe_mask(),
            by_closure.lobe_mask(),
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
        assert_eq!(
            header.execution_binding_words(),
            [MATERIAL_EXECUTION_BINDING_SENTINEL; 5]
        );
        assert!(!header.has_execution_binding());
        // The split epoch and specialization key survive the round trip.
        let rebuilt_epoch =
            (u64::from(header.material_epoch_high) << 32) | u64::from(header.material_epoch_low);
        assert_eq!(rebuilt_epoch, epoch);
    }

    #[test]
    fn fallback_record_is_a_lit_principled_surface() {
        let handle = GenerationalHandle::new(0, 0);
        let value = fallback_material_record(handle, 42);
        assert_eq!(value.revision, 42);
        assert_eq!(value.domain, MaterialDomain::Surface);
        assert_eq!(value.response_plan, ShadingPlan::default());
        assert_eq!(value.render_class, MaterialRenderClass::Opaque);
        assert_eq!(value.closure_mask, 1);
        assert!(value.textures.is_empty());
    }

    #[test]
    fn inactive_header_is_generation_tagged_and_dormant() {
        let header = inactive_material_header(5);
        assert_eq!(header.generation, 5);
        assert_eq!(header.is_active, 0);
        assert_eq!(
            header.execution_binding_words(),
            [MATERIAL_EXECUTION_BINDING_SENTINEL; 5]
        );
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
    fn fallback_header_keeps_execution_identity_unbound() {
        let header = fallback_material_header(9);
        assert_eq!(header.is_active, 1);
        assert!(!header.has_execution_binding());
        assert_eq!(
            header.execution_binding_words(),
            [MATERIAL_EXECUTION_BINDING_SENTINEL; 5]
        );
    }
}
