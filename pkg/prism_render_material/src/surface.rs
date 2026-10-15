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
    pub base_metalness: f32,
    pub specular_roughness: f32,
    /// `OpenPBR` `specular_weight`: scales the dielectric specular lobe. It
    /// replaces the former `reflectance` IOR proxy.
    pub specular_weight: f32,
    pub ambient_occlusion: f32,
    pub normal_scale: f32,
    pub alpha_cutoff: f32,
    /// `OpenPBR` `base_weight`: scales the base diffuse + specular layer.
    pub base_weight: f32,
    /// `OpenPBR` `diffuse_roughness`: micro-slope roughness of the base diffuse
    /// lobe (Oren-Nayar). `0` is exactly Lambert.
    pub base_diffuse_roughness: f32,
    /// `OpenPBR` `specular_color`: per-channel tint of the dielectric specular
    /// (white = the identity; metals use `base_color`).
    pub specular_color: [f32; 4],
    /// `OpenPBR` `specular_ior`: relative index of the dielectric specular
    /// interface. `1.5` gives `F0 = 0.04`, the value the removed
    /// `reflectance = 0.5` produced.
    pub specular_ior: f32,
    /// `OpenPBR` `geometry_thin_walled`: a single-interface, open surface whose
    /// transmission lobe skips the volume absorption. `0.0` keeps the solid slab.
    pub thin_walled: f32,
    /// `OpenPBR` `geometry_opacity` in `[0, 1]`: multiplied into the alpha the
    /// blend/mask path reads. `1.0` is the exact identity.
    pub geometry_opacity: f32,
    pub _pad2: f32,
}

impl Default for GpuSurfaceCore {
    fn default() -> Self {
        // `OpenPBR` default surface (design doc §4.4 / roadmap P1-f2): the core
        // carries the specification defaults so an authoring-free material is
        // spec-conformant.
        Self {
            base_color: [0.8, 0.8, 0.8, 1.0],
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
            _pad2: 0.0,
        }
    }
}

/// Emissive radiance lobe (`illumination`-independent self-illumination).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuEmissionLobe {
    /// `OpenPBR` `emission_color` (Prism's long-standing `emissive` name).
    pub emission_color: [f32; 4],
    /// `OpenPBR` `emission_luminance` in nits: emission radiance is
    /// `emissive * emission_luminance`.
    pub emission_luminance: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

impl Default for GpuEmissionLobe {
    fn default() -> Self {
        Self {
            emission_color: [1.0; 4],
            emission_luminance: 0.0,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }
}

/// Clearcoat lobe: a second, smoother specular layer over the base.
///
/// The `OpenPBR` `coat` group. `coat_ior` drives the coat interface Fresnel
/// (default `1.5`, reproducing the historical fixed `F0 = 0.04`) and
/// `coat_color` tints the coat highlight (default white = identity). This is
/// the first lobe completed against the `OpenPBR` field set (§4.4); the remaining
/// coat parameters (`coat_darkening`, `coat_roughness_anisotropy`,
/// `coat_normal`) are still to be added.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuClearCoatLobe {
    pub coat_weight: f32,
    pub coat_roughness: f32,
    /// `OpenPBR` relative index of refraction of the coat interface.
    pub coat_ior: f32,
    pub _pad0: f32,
    /// `OpenPBR` per-channel tint of the coat highlight (white = clear lacquer).
    pub coat_color: [f32; 4],
    /// `OpenPBR` `coat_roughness_anisotropy` in `[0, 1]`: elongates the coat
    /// highlight along the shared surface tangent frame. `0` is isotropic.
    pub coat_roughness_anisotropy: f32,
    pub _pad1: f32,
    pub _pad2: f32,
    pub _pad3: f32,
    /// `OpenPBR` `coat_darkening` in `[0, 1]`: how much of the coat's internal
    /// reflection is lost, darkening the base. `0.0` keeps the historical
    /// two-pass coat response exactly.
    pub coat_darkening: f32,
    pub _pad4: f32,
    pub _pad5: f32,
    pub _pad6: f32,
}

impl Default for GpuClearCoatLobe {
    fn default() -> Self {
        Self {
            coat_weight: 0.0,
            coat_roughness: 0.0,
            coat_ior: 1.6,
            _pad0: 0.0,
            coat_color: [1.0; 4],
            coat_roughness_anisotropy: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
            coat_darkening: 1.0,
            _pad4: 0.0,
            _pad5: 0.0,
            _pad6: 0.0,
        }
    }
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

/// Sheen (fuzz) lobe: retro-reflective fabric rim response.
///
/// The `OpenPBR` `fuzz` group. `sheen` is the fuzz weight, `fuzz_roughness` the
/// independent highlight width (default `0.5`) and `fuzz_color` the per-channel
/// tint (default white = achromatic, the historical behaviour).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSheenLobe {
    pub fuzz_weight: f32,
    /// `OpenPBR` `fuzz_roughness` in `[0, 1]`.
    pub fuzz_roughness: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    /// `OpenPBR` `fuzz_color` (white = achromatic fuzz).
    pub fuzz_color: [f32; 4],
}

impl Default for GpuSheenLobe {
    fn default() -> Self {
        Self {
            fuzz_weight: 0.0,
            fuzz_roughness: 0.5,
            _pad0: 0.0,
            _pad1: 0.0,
            fuzz_color: [1.0; 4],
        }
    }
}

/// Subsurface lobe: wrapped diffuse approximation of shallow scattering.
///
/// The `OpenPBR` subsurface group. `subsurface_color` tints the scattered light
/// (white = identity) and `subsurface_radius` is the per-channel scattering
/// extent (white = identity, documented as *relative* in this real-time lobe).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSubsurfaceLobe {
    pub subsurface_weight: f32,
    pub _pad0: f32,
    /// `OpenPBR` `subsurface_radius`: scalar scattering radius (relative in this
    /// real-time lobe; the specification default is 1).
    pub subsurface_radius: f32,
    pub _pad1: f32,
    /// `OpenPBR` `subsurface_color` (white = identity).
    pub subsurface_color: [f32; 4],
    /// `OpenPBR` `subsurface_radius_scale` (color3, w unused): the per-channel
    /// split of the radius, default (1.0, 0.5, 0.25) per the specification.
    pub subsurface_radius_scale: [f32; 4],
    /// `OpenPBR` `scatter_anisotropy` in `[-1, 1]`: forward (+) / backward (-)
    /// bias of the transmitted lobe. `0.0` is the historical constant.
    pub subsurface_scatter_anisotropy: f32,
    pub _pad2: f32,
    pub _pad3: f32,
    pub _pad4: f32,
}

impl Default for GpuSubsurfaceLobe {
    fn default() -> Self {
        Self {
            subsurface_weight: 0.0,
            _pad0: 0.0,
            subsurface_radius: 1.0,
            _pad1: 0.0,
            subsurface_color: [0.8, 0.8, 0.8, 1.0],
            subsurface_radius_scale: [1.0, 0.5, 0.25, 1.0],
            subsurface_scatter_anisotropy: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
            _pad4: 0.0,
        }
    }
}

/// Transmission lobe: refractive dielectric transport (glass/liquid).
///
/// The `OpenPBR` transmission group. `transmission_color` tints the refracted
/// light (white = identity).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTransmissionLobe {
    pub transmission_weight: f32,
    pub thickness: f32,
    pub index_of_refraction: f32,
    pub transmission_dispersion_scale: f32,
    /// `OpenPBR` `transmission_color` (white = identity).
    pub transmission_color: [f32; 4],
    /// `OpenPBR` `transmission_scatter` (color3, w unused): per-channel fraction
    /// of the absorbed energy re-scattered instead of lost. Black is pure
    /// Beer-Lambert absorption.
    pub transmission_scatter: [f32; 4],
    /// `OpenPBR` `transmission_scatter_anisotropy` in `[-1, 1]` (`0` =
    /// isotropic in-scatter).
    pub transmission_scatter_anisotropy: f32,
    /// `OpenPBR` `transmission_dispersion_abbe_number` (`V_d`).
    pub transmission_dispersion_abbe_number: f32,
    pub _pad3: f32,
    pub _pad4: f32,
}

impl Default for GpuTransmissionLobe {
    fn default() -> Self {
        Self {
            transmission_weight: 0.0,
            thickness: 0.0,
            index_of_refraction: 1.5,
            transmission_dispersion_scale: 0.0,
            transmission_color: [1.0; 4],
            transmission_scatter: [0.0, 0.0, 0.0, 1.0],
            transmission_scatter_anisotropy: 0.0,
            transmission_dispersion_abbe_number: 20.0,
            _pad3: 0.0,
            _pad4: 0.0,
        }
    }
}

/// Thin-film lobe: the `OpenPBR` iridescent film over the base.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuThinFilmLobe {
    pub thin_film_weight: f32,
    /// Film thickness in micrometres (specification default 0.5).
    pub thin_film_thickness: f32,
    /// Film refractive index (specification default 1.4).
    pub thin_film_ior: f32,
    pub _pad0: f32,
}

impl Default for GpuThinFilmLobe {
    fn default() -> Self {
        // `OpenPBR` thin-film defaults: weight 0 (inert), thickness 0.5 µm,
        // IOR 1.4. `weight == 0` keeps the whole lobe inert regardless.
        Self {
            thin_film_weight: 0.0,
            thin_film_thickness: 0.5,
            thin_film_ior: 1.4,
            _pad0: 0.0,
        }
    }
}

/// Face-shadow lobe: stylized SDF-driven directional face shading (anime/toon).
///
/// The lobe carries only the terminator softness; the SDF map itself is a
/// bindless texture bound through the `SEMANTIC_FACE_SDF` slot, and the
/// per-instance orientation used to flip/threshold the SDF comes from the
/// instance world matrix at shade time (no extra per-instance ABI).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuNprFaceLobe {
    pub softness: f32,
    pub ramp_softness: f32,
    pub specular_scale: f32,
    pub rim_scale: f32,
}

/// Glint lobe: the discrete (counting) microfacet estimator of §17.6.
///
/// Real metallic flake, snow, sand and glitter expose a *finite* number of
/// mirror-like facets per pixel footprint, so the highlight twinkles instead of
/// being a smooth statistical lobe. The estimator counts how many facets land in
/// the half-vector query bin, which is **unbiased** — its ensemble mean is the
/// smooth GGX NDF — so `glint_weight` re-shapes the specular lobe without adding
/// or removing energy. It is view- and space-dependent by construction, so it is
/// never an object-space-cacheable field.
///
/// The two quantities the estimator needs that are *not* authored parameters —
/// the surface cell the seed quantises and the per-pixel footprint area — are
/// shade-time inputs supplied alongside the frame (see
/// `prism_render_shading::lighting::SurfaceSample`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuGlintLobe {
    pub glint_weight: f32,
    pub glint_roughness: f32,
    pub glint_density: f32,
    pub glint_bin_half_width: f32,
}

impl Default for GpuGlintLobe {
    fn default() -> Self {
        // Exactly the schema defaults (`surface.toml`), so an absent lobe reads
        // back as its true neutral value rather than a zeroed struct — the same
        // three-way alignment (`schema` ↔ `GpuSurfaceParameters` ↔ lobe struct)
        // the P1-f2 default-alignment gate checks for every other lobe.
        Self {
            glint_weight: 0.0,
            glint_roughness: 0.1,
            glint_density: 4.0,
            glint_bin_half_width: 0.05,
        }
    }
}

/// Measured (tabulated) BRDF/BTF lobe: the data-driven, non-AI member of the
/// open vocabulary (§17.6).
///
/// The table itself is *not* in the ABI: it ships as an ordinary material
/// texture (a `provider`, bound with the measured-BRDF semantic), and the shader
/// fetches its eight trilinear corners with integer texel loads. What the lobe
/// carries is only the authored response: the blend weight, the table's
/// difference-azimuth resolution, and a scale that puts the measured data's own
/// normalisation into the renderer's units.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMeasuredBrdfLobe {
    pub measured_weight: f32,
    pub measured_phi_d: f32,
    pub measured_scale: f32,
    pub _pad0: f32,
}

impl Default for GpuMeasuredBrdfLobe {
    fn default() -> Self {
        // Exactly the schema defaults, so an absent lobe reads back neutral.
        Self {
            measured_weight: 0.0,
            measured_phi_d: 180.0,
            measured_scale: 1.0,
            _pad0: 0.0,
        }
    }
}

/// Bitset recording which optional lobes a surface carries.
///
/// The bit order is the canonical serialization order used by
/// [`SurfaceParameterBlock::pack`]: present lobes are written low bit first.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LobeMask(pub u32);

impl LobeMask {
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
    pub thin_film: GpuThinFilmLobe,
    pub face: GpuNprFaceLobe,
    pub glint: GpuGlintLobe,
    pub measured: GpuMeasuredBrdfLobe,
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
                base_metalness: params.base_metalness,
                specular_roughness: params.specular_roughness,
                specular_weight: params.specular_weight,
                ambient_occlusion: params.ambient_occlusion,
                normal_scale: params.normal_scale,
                alpha_cutoff: params.alpha_cutoff,
                base_weight: params.base_weight,
                base_diffuse_roughness: params.base_diffuse_roughness,
                specular_color: params.specular_color,
                specular_ior: params.specular_ior,
                thin_walled: params.thin_walled,
                geometry_opacity: params.geometry_opacity,
                _pad2: 0.0,
            },
            lobe_mask,
            emission: GpuEmissionLobe {
                emission_color: params.emission_color,
                emission_luminance: params.emission_luminance,
                _pad0: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            },
            clearcoat: GpuClearCoatLobe {
                coat_weight: params.coat_weight,
                coat_roughness: params.coat_roughness,
                coat_ior: params.coat_ior,
                _pad0: 0.0,
                coat_color: params.coat_color,
                coat_roughness_anisotropy: params.coat_roughness_anisotropy,
                _pad1: 0.0,
                _pad2: 0.0,
                _pad3: 0.0,
                coat_darkening: params.coat_darkening,
                _pad4: 0.0,
                _pad5: 0.0,
                _pad6: 0.0,
            },
            anisotropy: GpuAnisotropyLobe {
                anisotropy: params.anisotropy,
                anisotropy_rotation: params.anisotropy_rotation,
                _pad0: 0.0,
                _pad1: 0.0,
            },
            sheen: GpuSheenLobe {
                fuzz_weight: params.fuzz_weight,
                fuzz_roughness: params.fuzz_roughness,
                _pad0: 0.0,
                _pad1: 0.0,
                fuzz_color: params.fuzz_color,
            },
            subsurface: GpuSubsurfaceLobe {
                subsurface_weight: params.subsurface_weight,
                _pad0: 0.0,
                subsurface_radius: params.subsurface_radius,
                _pad1: 0.0,
                subsurface_color: params.subsurface_color,
                subsurface_radius_scale: params.subsurface_radius_scale,
                subsurface_scatter_anisotropy: params.subsurface_scatter_anisotropy,
                _pad2: 0.0,
                _pad3: 0.0,
                _pad4: 0.0,
            },
            transmission: GpuTransmissionLobe {
                transmission_weight: params.transmission_weight,
                thickness: params.thickness,
                index_of_refraction: params.index_of_refraction,
                transmission_dispersion_scale: params.transmission_dispersion_scale,
                transmission_color: params.transmission_color,
                transmission_scatter: params.transmission_scatter,
                transmission_scatter_anisotropy: params.transmission_scatter_anisotropy,
                transmission_dispersion_abbe_number: params.transmission_dispersion_abbe_number,
                _pad3: 0.0,
                _pad4: 0.0,
            },
            thin_film: GpuThinFilmLobe {
                thin_film_weight: params.thin_film_weight,
                thin_film_thickness: params.thin_film_thickness,
                thin_film_ior: params.thin_film_ior,
                _pad0: 0.0,
            },
            face: GpuNprFaceLobe {
                softness: params.npr_face_softness,
                ramp_softness: params.npr_stylized_ramp_softness,
                specular_scale: params.npr_stylized_specular_scale,
                rim_scale: params.npr_stylized_rim_scale,
            },
            glint: GpuGlintLobe {
                glint_weight: params.glint_weight,
                glint_roughness: params.glint_roughness,
                glint_density: params.glint_density,
                glint_bin_half_width: params.glint_bin_half_width,
            },
            measured: GpuMeasuredBrdfLobe {
                measured_weight: params.measured_weight,
                measured_phi_d: params.measured_phi_d,
                measured_scale: params.measured_scale,
                _pad0: 0.0,
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
            base_metalness: self.core.base_metalness,
            specular_roughness: self.core.specular_roughness,
            specular_weight: self.core.specular_weight,
            ambient_occlusion: self.core.ambient_occlusion,
            normal_scale: self.core.normal_scale,
            alpha_cutoff: self.core.alpha_cutoff,
            base_weight: self.core.base_weight,
            base_diffuse_roughness: self.core.base_diffuse_roughness,
            specular_color: self.core.specular_color,
            specular_ior: self.core.specular_ior,
            thin_walled: self.core.thin_walled,
            geometry_opacity: self.core.geometry_opacity,
            ..GpuSurfaceParameters::default()
        };
        if self.lobe_mask.contains(LobeMask::EMISSION) {
            full.emission_color = self.emission.emission_color;
            full.emission_luminance = self.emission.emission_luminance;
        }
        if self.lobe_mask.contains(LobeMask::CLEARCOAT) {
            full.coat_weight = self.clearcoat.coat_weight;
            full.coat_roughness = self.clearcoat.coat_roughness;
            full.coat_ior = self.clearcoat.coat_ior;
            full.coat_color = self.clearcoat.coat_color;
            full.coat_roughness_anisotropy = self.clearcoat.coat_roughness_anisotropy;
            full.coat_darkening = self.clearcoat.coat_darkening;
        }
        if self.lobe_mask.contains(LobeMask::ANISOTROPY) {
            full.anisotropy = self.anisotropy.anisotropy;
            full.anisotropy_rotation = self.anisotropy.anisotropy_rotation;
        }
        if self.lobe_mask.contains(LobeMask::SHEEN) {
            full.fuzz_weight = self.sheen.fuzz_weight;
            full.fuzz_roughness = self.sheen.fuzz_roughness;
            full.fuzz_color = self.sheen.fuzz_color;
        }
        if self.lobe_mask.contains(LobeMask::SUBSURFACE) {
            full.subsurface_weight = self.subsurface.subsurface_weight;
            full.subsurface_color = self.subsurface.subsurface_color;
            full.subsurface_radius = self.subsurface.subsurface_radius;
            full.subsurface_radius_scale = self.subsurface.subsurface_radius_scale;
            full.subsurface_scatter_anisotropy = self.subsurface.subsurface_scatter_anisotropy;
        }
        if self.lobe_mask.contains(LobeMask::TRANSMISSION) {
            full.transmission_weight = self.transmission.transmission_weight;
            full.thickness = self.transmission.thickness;
            full.index_of_refraction = self.transmission.index_of_refraction;
            full.transmission_dispersion_scale = self.transmission.transmission_dispersion_scale;
            full.transmission_color = self.transmission.transmission_color;
            full.transmission_scatter = self.transmission.transmission_scatter;
            full.transmission_scatter_anisotropy =
                self.transmission.transmission_scatter_anisotropy;
            full.transmission_dispersion_abbe_number =
                self.transmission.transmission_dispersion_abbe_number;
        }
        if self.lobe_mask.contains(LobeMask::THIN_FILM) {
            full.thin_film_weight = self.thin_film.thin_film_weight;
            full.thin_film_thickness = self.thin_film.thin_film_thickness;
            full.thin_film_ior = self.thin_film.thin_film_ior;
        }
        if self.lobe_mask.contains(LobeMask::NPR_FACE) {
            full.npr_face_softness = self.face.softness;
            full.npr_stylized_ramp_softness = self.face.ramp_softness;
            full.npr_stylized_specular_scale = self.face.specular_scale;
            full.npr_stylized_rim_scale = self.face.rim_scale;
        }
        if self.lobe_mask.contains(LobeMask::GLINT) {
            full.glint_weight = self.glint.glint_weight;
            full.glint_roughness = self.glint.glint_roughness;
            full.glint_density = self.glint.glint_density;
            full.glint_bin_half_width = self.glint.glint_bin_half_width;
        }
        if self.lobe_mask.contains(LobeMask::MEASURED_BRDF) {
            full.measured_weight = self.measured.measured_weight;
            full.measured_phi_d = self.measured.measured_phi_d;
            full.measured_scale = self.measured.measured_scale;
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
            LobeMask::THIN_FILM => bytemuck::cast_slice(core::slice::from_ref(&self.thin_film)),
            LobeMask::NPR_FACE => bytemuck::cast_slice(core::slice::from_ref(&self.face)),
            LobeMask::GLINT => bytemuck::cast_slice(core::slice::from_ref(&self.glint)),
            LobeMask::MEASURED_BRDF => bytemuck::cast_slice(core::slice::from_ref(&self.measured)),
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
                LobeMask::THIN_FILM => block.thin_film = *bytemuck::from_bytes(slice),
                LobeMask::NPR_FACE => block.face = *bytemuck::from_bytes(slice),
                LobeMask::GLINT => block.glint = *bytemuck::from_bytes(slice),
                LobeMask::MEASURED_BRDF => block.measured = *bytemuck::from_bytes(slice),
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
            emission_color: [2.0, 0.0, 0.0, 1.0],
            emission_luminance: 3.5,
            base_metalness: 0.25,
            specular_roughness: 0.4,
            specular_weight: 0.65,
            ambient_occlusion: 0.9,
            normal_scale: 0.8,
            alpha_cutoff: 0.37,
            base_weight: 0.85,
            base_diffuse_roughness: 0.4,
            specular_color: [0.95, 0.97, 1.0, 1.0],
            specular_ior: 1.7,
            thin_walled: 0.0,
            geometry_opacity: 0.9,
            transmission_weight: 0.6,
            thickness: 1.5,
            coat_weight: 0.7,
            coat_roughness: 0.15,
            coat_ior: 1.45,
            coat_color: [0.9, 0.95, 1.0, 1.0],
            coat_roughness_anisotropy: 0.35,
            coat_darkening: 0.6,
            anisotropy: 0.3,
            anisotropy_rotation: 0.2,
            fuzz_weight: 0.45,
            fuzz_roughness: 0.3,
            fuzz_color: [0.8, 0.85, 0.9, 1.0],
            subsurface_weight: 0.65,
            subsurface_color: [0.9, 0.5, 0.4, 1.0],
            subsurface_radius: 0.8,
            subsurface_radius_scale: [1.0, 0.6, 0.4, 1.0],
            subsurface_scatter_anisotropy: 0.7,
            index_of_refraction: 1.33,
            transmission_dispersion_scale: 0.05,
            transmission_color: [0.85, 0.95, 1.0, 1.0],
            transmission_scatter: [0.25, 0.15, 0.05, 1.0],
            transmission_scatter_anisotropy: -0.4,
            transmission_dispersion_abbe_number: 35.0,
            thin_film_weight: 0.7,
            thin_film_thickness: 0.35,
            thin_film_ior: 1.55,
            npr_face_softness: 0.25,
            npr_stylized_ramp_softness: 1.25,
            npr_stylized_specular_scale: 0.75,
            npr_stylized_rim_scale: 1.5,
            glint_weight: 0.6,
            glint_roughness: 0.22,
            glint_density: 12.0,
            glint_bin_half_width: 0.031,
            measured_weight: 0.45,
            measured_phi_d: 128.0,
            measured_scale: 2.5,
        }
    }

    #[test]
    fn layout_matches_generated_table() {
        use crate::surface_layout::{
            SURFACE_LOBES, SURFACE_LOBE_COUNT, SURFACE_LOBE_WORDS_BY_SLOT,
        };

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
            assert_eq!(
                LobeMask::region_of(bit),
                Some(expected_region(lobe.namespace))
            );
            if lobe.namespace == "KHR" {
                khr = khr.union(bit);
            } else {
                extension = extension.union(bit);
            }
        }
        assert_eq!(LobeMask::KHR_REGION, khr);
        assert_eq!(LobeMask::EXTENSION_REGION, extension);
        // The two regions are disjoint and together tile the defined bits.
        assert_eq!(
            LobeMask::KHR_REGION.bits() & LobeMask::EXTENSION_REGION.bits(),
            0
        );
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
        assert!(LobeMask::all().contains(LobeMask::EXTENSION_REGION));
        assert!(!LobeMask::all().is_pure_openpbr());
        // Derived from the region partition rather than a lobe name list, so a
        // newly-registered extension lobe cannot silently fail to be covered.
        for bit in LobeMask::canonical() {
            let canonical = LobeMask::region_of(bit) == Some(LobeRegion::Canonical);
            assert_eq!(bit.is_pure_openpbr(), canonical, "{bit:?}");
            assert_eq!(bit.canonical_region() == bit, canonical, "{bit:?}");
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
            (LobeMask::THIN_FILM, size_of::<GpuThinFilmLobe>()),
            (LobeMask::NPR_FACE, size_of::<GpuNprFaceLobe>()),
            (LobeMask::GLINT, size_of::<GpuGlintLobe>()),
            (LobeMask::MEASURED_BRDF, size_of::<GpuMeasuredBrdfLobe>()),
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
        // 80 bytes of packed block vs the much larger fat authoring struct.
        assert_eq!(block.packed_size_bytes(), SURFACE_CORE_WORDS * 4);
        assert!(block.packed_size_bytes() < size_of::<GpuSurfaceParameters>());
    }

    #[test]
    fn full_mask_packs_core_plus_every_lobe() {
        let block = SurfaceParameterBlock::from_full(&authored(), LobeMask::all());
        assert_eq!(
            block.packed_len_words(),
            SURFACE_CORE_WORDS + SURFACE_LOBE_WORDS_BY_SLOT.iter().sum::<usize>()
        );
        assert_eq!(block.pack().len(), block.packed_len_words());
    }

    #[test]
    fn pack_unpack_round_trips_for_every_lobe_subset() {
        let params = authored();
        for subset in 0..(1u32 << LobeMask::COUNT) {
            let mask = LobeMask::canonical().into_iter().enumerate().fold(
                LobeMask::default(),
                |mask, (index, bit)| {
                    if subset & (1 << index) != 0 {
                        mask.union(bit)
                    } else {
                        mask
                    }
                },
            );
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
        assert_eq!(full.emission_color, params.emission_color);
        // Transmission/clearcoat/etc. were authored non-zero but are absent, so
        // they must read back as their neutral defaults, not the stale values.
        assert_eq!(full.transmission_weight, 0.0);
        assert_eq!(full.coat_weight, 0.0);
        assert_eq!(full.index_of_refraction, 1.5);
    }

    #[test]
    fn unpack_rejects_wrong_length() {
        let err = SurfaceParameterBlock::unpack(LobeMask::all(), &[0; 3]).unwrap_err();
        assert!(matches!(err, SurfaceUnpackError::WrongLength { .. }));
    }
}
