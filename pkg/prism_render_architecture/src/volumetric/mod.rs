//! Volumetric cloud / atmospheric-volume subsystem contracts (AAA next-gen
//! volumetric cloud + atmospheric engine).
//!
//! Volumetric clouds are a first-class rendering subsystem in the same sense as
//! cloth, hair, particles, and water: they own their own *geometry* (a
//! procedural 3D density field, a weather map, and a cloud-domain bounding
//! volume rather than a triangle mesh), their own *simulation* (weather-system
//! evolution, wind-driven advection, growth/decay, precipitation and storm
//! coupling), and their own *special rendering* (ray-marched single scattering,
//! multiple-scattering approximations, adaptive volumetric self-shadowing, god
//! rays, and aerial-perspective hookup). The pipeline mirrors production
//! volumetric engines at the *algorithm* level only, without reusing any of
//! their code: `UE5` Volumetric Clouds + Sky Atmosphere, Guerrilla `Nubis`
//! (Perlin-Worley modelling, coverage/type/height gradients, `detail erosion`,
//! powder, energy-conserving ray-march), Frostbite physical atmosphere
//! (pre-integrated transmittance / multi-scatter `LUT`, aerial-perspective
//! froxels), Decima / `RDR2` weather systems, Wrenninge/Fong production volume
//! rendering (`octave scattering`, anisotropic phase, offline energy reference),
//! Intel `AVSM` / `NVIDIA` deep shadows, and `PBRT`/`Mitsuba` volume path
//! tracing.
//!
//! Module layout follows the design doc section 14, one concern per file:
//!
//! - [`math`] — hand-rolled `exp`/`ln`/`pow`/trig approximations, [`Vec2`],
//!   [`Vec3`], and the shared `EPS` constants (the determinism policy allows
//!   only `sqrt` among the float intrinsics).
//! - [`budget`] — per-frame ray-march / modelling / upsample / multi-scatter
//!   budget arbitration, the volumetric analogue of the shared deformation
//!   scheduler.
//! - [`noise`] — Perlin / Worley / `curl` noise primitives (deterministic,
//!   seedable, reproducible).
//! - [`modeling`] — coverage / cloud-type / height-gradient modulation and
//!   `detail erosion` `remap` (energy-preserving, pure).
//! - [`weather`] — semi-Lagrangian weather-map advection, the sky-state machine,
//!   and precipitation classification.
//! - [`raymarch`] — adaptive step / empty-space-skipping / early-out decisions.
//! - [`scatter`] — Henyey-Greenstein dual-lobe + Draine phase, powder, and
//!   `octave scattering` weights.
//! - [`multiscatter`] — pre-integrated environment multi-scatter `LUT` axes and
//!   spatial irradiance-probe interpolation.
//! - [`avsm`] — adaptive volumetric shadow-map control-point compression and
//!   curve queries.
//! - [`cloud_lod`] — step / resolution / imposter `LOD` bucket decisions.
//! - [`temporal`] — temporal reprojection / upsampling plan and history clamp.
//! - [`shadow`] — cloud-shadow casting and god-ray injection plan.
//! - [`atmosphere`] — aerial-perspective blend weights (samples the shared
//!   atmosphere `LUT`, never reimplements it).
//! - [`spectral`] — spectral atmosphere / night-sky consumption parameters.
//! - [`storm`] — cumulonimbus vertical development (`anvil` / gravity wave /
//!   `virga` / pyrocumulus) state.
//! - [`coupling`] — density carving / terrain occlusion / cloud-shadow-to-
//!   atmosphere two-way coupling (deterministic).
//! - [`fog`] — unified volumetric fog / contrail sources / froxel injection.
//! - [`reference`] — delta/ratio-tracking single-scatter / transmittance
//!   reference truth for offline calibration (deterministic, verifiable).
//!
//! **Frontend orthogonality.** The `PBR`, `NPR`, custom, and hybrid frontends
//! diverge *only* in their lighting response (see [`ShadingFrontend`]). Every
//! frontend — `NPR` included — consumes the exact same shared advanced base:
//! `Lumen`-style hybrid `GI`, `ReSTIR` `DI`/`GI`, virtual shadow-map cloud
//! shadows, froxel volumetrics, the atmosphere-scattering `LUT` service, the
//! path-tracing reference, and temporal upsampling. This is encoded in
//! [`SharedBaseServices`] and asserted by [`ShadingFrontend::shared_base`], so
//! "does `NPR` miss any next-gen capability?" has a compile-checked answer: no.
//! Virtual geometry is intentionally *not* in the base — clouds have no triangle
//! mesh and use the density field + ray-march instead — which is why this
//! subsystem does not touch `DeformationKind` either.
//!
//! Only classical numerical methods are used; there is no `AI`/`ML` anywhere.
//! The `GPU` `WESL` kernels and frame-graph wiring are documented as scaffolding
//! and marked "not machine-verified" where the contract signatures anticipate
//! them (the sandbox has no `GPU`); the `CPU`-verifiable pure functions and
//! scheduling are the verification target.

pub mod budget;
pub mod math;

// The remaining design-doc section 14 modules (noise, modeling, weather,
// raymarch, scatter, multiscatter, avsm, cloud_lod, temporal, shadow,
// atmosphere, spectral, storm, coupling, fog, reference) plus the cross-module
// integration_tests are declared here as each lands, so every commit compiles.

pub use math::{Vec2, Vec3, EPS, EPS_LEN_SQ};

/// Version of the volumetric subsystem contracts in this module.
pub const VOLUMETRIC_ARCHITECTURE_VERSION: u32 = 1;

/// Stable identity of one cloud layer (a `Cumulus` deck, `Cirrus` sheet, ...).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CloudLayerHandle(pub u32);

/// Stable identity of one weather-map tile driving global cloud distribution.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WeatherMapHandle(pub u32);

/// Stable identity of one two-way interaction source (aircraft carving a hole,
/// contrail emitter, fire/explosion thermal column feeding pyrocumulus).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct InteractionSourceHandle(pub u32);

/// Stable identity of one multi-scatter irradiance probe within a cloud domain.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ProbeHandle(pub u32);

/// The four canonical cloud kinds, each with a distinct height band, modelling
/// signature, and dynamic behaviour (design section 2).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CloudKind {
    /// Low-layer puffy fair-weather cloud: high coverage contrast, strong
    /// `detail erosion`.
    Cumulus,
    /// Low-to-mid uniform overcast sheet: low contrast, flat spread.
    Stratus,
    /// High-layer thin wispy cloud advected into streaks by `curl` noise.
    Cirrus,
    /// Deep-convective storm cloud spanning all layers, the precipitation
    /// source with full vertical development (design section 9b).
    Cumulonimbus,
}

impl CloudKind {
    /// A canonical `(min, max)` altitude band in metres for this kind, the
    /// default the authoring layer starts from before per-asset overrides.
    #[must_use]
    pub fn default_height_band(self) -> (f32, f32) {
        match self {
            CloudKind::Cumulus => (1500.0, 4000.0),
            CloudKind::Stratus => (600.0, 2000.0),
            CloudKind::Cirrus => (6000.0, 12000.0),
            CloudKind::Cumulonimbus => (1000.0, 12000.0),
        }
    }

    /// `true` for kinds that can produce precipitation (only `Cumulonimbus` in
    /// the base model; other kinds signal drizzle only through the weather map).
    #[must_use]
    pub fn precipitation_capable(self) -> bool {
        matches!(self, CloudKind::Cumulonimbus)
    }

    /// `true` for kinds driven by the storm vertical-development state machine
    /// (`updraft` / `anvil` / overshooting top); only `Cumulonimbus`.
    #[must_use]
    pub fn vertically_developed(self) -> bool {
        matches!(self, CloudKind::Cumulonimbus)
    }

    /// `true` for high-layer kinds whose shape is dominated by `curl`-advected
    /// streaks rather than puffy `Worley` erosion.
    #[must_use]
    pub fn is_high_layer(self) -> bool {
        matches!(self, CloudKind::Cirrus)
    }
}

/// The geometry of one cloud layer: its altitude band, domain extent, and the
/// resolution of the density field baked for far/perf tiers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudGeometry {
    /// Lower altitude of the layer (world Y, metres).
    pub height_min: f32,
    /// Upper altitude of the layer (world Y, metres).
    pub height_max: f32,
    /// Half-extent of the cloud domain bounding volume (metres).
    pub domain_half_extent: Vec3,
    /// Per-side resolution of the optional baked density cache (`0` disables
    /// baking and samples the noise on demand).
    pub density_resolution: u32,
}

impl CloudGeometry {
    /// Thickness of the layer (`height_max - height_min`), clamped non-negative.
    #[must_use]
    pub fn thickness(self) -> f32 {
        (self.height_max - self.height_min).max(0.0)
    }

    /// Normalised height `0..=1` of a world-Y sample within the layer band,
    /// saturated outside the band. Drives the height gradient in [`modeling`].
    #[must_use]
    pub fn height_fraction(self, world_y: f32) -> f32 {
        let t = self.thickness();
        if t < EPS {
            return 0.0;
        }
        math::saturate((world_y - self.height_min) / t)
    }
}

/// The procedural modelling parameters of one cloud layer (design section 4).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudModeling {
    /// Base coverage `0..=1`; combined with the weather-map coverage channel.
    pub coverage: f32,
    /// Cloud type `0..=1` interpolating stratus (0) to cumulus (1) shape.
    pub cloud_type: f32,
    /// Low-frequency Perlin-Worley base frequency (cycles per domain).
    pub base_frequency: f32,
    /// High-frequency `detail erosion` frequency (cycles per domain).
    pub detail_frequency: f32,
    /// `detail erosion` strength `0..=1`; how hard the high-frequency `Worley`
    /// noise eats the cloud edges via `remap`.
    pub detail_erosion: f32,
    /// `curl` advection strength for wispy streaking (metres of displacement).
    pub curl_strength: f32,
}

/// The optical / physical medium properties of one cloud layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudMedium {
    /// Extinction coefficient (per metre) at unit density — Beer-Lambert sigma.
    pub extinction: f32,
    /// Single-scattering albedo `0..=1` (clouds are near-white, ~0.99).
    pub scattering_albedo: f32,
    /// Henyey-Greenstein anisotropy `g` in `(-1, 1)`; forward-biased for clouds.
    pub anisotropy_g: f32,
}

/// The weather / dynamics parameters of one cloud layer (design section 9).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudWeatherParams {
    /// Horizontal wind vector (metres/second) advecting the weather map.
    pub wind: Vec2,
    /// Sky-state evolution rate `0..=1` per second (coverage lerp speed).
    pub evolution_rate: f32,
    /// Precipitation trigger threshold on the weather-map precip channel.
    pub precip_threshold: f32,
}

/// The shading parameters of one cloud layer, routed to the chosen frontend.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudShading {
    /// Lighting-response frontend.
    pub frontend: ShadingFrontend,
    /// Silver-lining strength `0..=1` for grazing-sun edge brightening.
    pub silver_lining: f32,
    /// Powder-effect strength `0..=1` for dense-cloud dark-edge darkening.
    pub powder: f32,
    /// Ambient (sky/ground) light ratio `0..=1` mixed into scattering.
    pub ambient_ratio: f32,
}

/// The unified authoring/runtime description of one cloud layer.
///
/// Carries the cloud kind, its geometry, procedural modelling, optical medium,
/// weather dynamics, and shading frontend. Fine-grained per-module parameters
/// (noise octave tables, ray-march step counts, storm-stage curves) live in the
/// sibling modules that own them; this struct is the routing contract every
/// module reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudLayer {
    /// Stable identity of this layer.
    pub handle: CloudLayerHandle,
    /// Cloud kind.
    pub kind: CloudKind,
    /// Altitude band, domain extent, density-cache resolution.
    pub geometry: CloudGeometry,
    /// Procedural modelling parameters.
    pub modeling: CloudModeling,
    /// Optical medium properties.
    pub medium: CloudMedium,
    /// Weather / dynamics parameters.
    pub weather: CloudWeatherParams,
    /// Shading frontend and its parameters.
    pub shading: CloudShading,
}

/// One sample of the weather map (design section 2): the `RGBA` channels that
/// drive global cloud distribution. `R` = coverage, `G` = cloud type, `B` =
/// precipitation intensity, `A` = wind disturbance.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WeatherSample {
    /// Coverage `0..=1` (`R` channel): how much cloud fills this cell.
    pub coverage: f32,
    /// Cloud type `0..=1` (`G` channel): stratus (0) to cumulus (1) shape bias.
    pub cloud_type: f32,
    /// Precipitation intensity `0..=1` (`B` channel).
    pub precipitation: f32,
    /// Wind disturbance `0..=1` (`A` channel): local turbulence/gust strength.
    pub wind_disturbance: f32,
}

impl WeatherSample {
    /// Builds a sample from raw `RGBA` channel values, each saturated to
    /// `0..=1` so downstream modulation never sees out-of-range weather.
    #[must_use]
    pub fn from_rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self {
            coverage: math::saturate(r),
            cloud_type: math::saturate(g),
            precipitation: math::saturate(b),
            wind_disturbance: math::saturate(a),
        }
    }

    /// Packs the sample back into an `RGBA` array (round-trips [`from_rgba`] for
    /// in-range inputs).
    #[must_use]
    pub fn to_rgba(self) -> [f32; 4] {
        [
            self.coverage,
            self.cloud_type,
            self.precipitation,
            self.wind_disturbance,
        ]
    }

    /// Component-wise linear blend toward `other` by `t`, the primitive the
    /// semi-Lagrangian advection and sky-state machine build on.
    #[must_use]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            coverage: math::lerp(self.coverage, other.coverage, t),
            cloud_type: math::lerp(self.cloud_type, other.cloud_type, t),
            precipitation: math::lerp(self.precipitation, other.precipitation, t),
            wind_disturbance: math::lerp(self.wind_disturbance, other.wind_disturbance, t),
        }
    }
}

/// The set of shared advanced base services a volumetric frontend consumes.
///
/// These services live in sibling crates/modules (`lighting`, `virtual_shadow`,
/// the atmosphere service, `ray_scene`, `temporal_upscale`); the volumetric
/// subsystem only *consumes* them and never reimplements them. Virtual geometry
/// is deliberately absent — clouds have no triangle mesh — which is the one
/// expected difference from the mesh subsystems. Bit operations are integer
/// comparisons, never `f32` equality.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SharedBaseServices(u16);

impl SharedBaseServices {
    /// `Lumen`-style hybrid global illumination (ambient sky/ground light).
    pub const HYBRID_GI: Self = Self(1 << 0);
    /// `ReSTIR` direct/indirect resampling for many lights (sun/moon/lightning).
    pub const RESTIR: Self = Self(1 << 1);
    /// Virtual shadow-map cloud shadows onto the world.
    pub const VSM_SHADOW: Self = Self(1 << 2);
    /// Froxel volumetrics for god rays / crepuscular light-shaft injection.
    pub const FROXEL_VOLUME: Self = Self(1 << 3);
    /// Atmosphere-scattering `LUT` service (transmittance / multi-scatter /
    /// aerial perspective) — consumed, never rewritten.
    pub const ATMOSPHERE_LUT: Self = Self(1 << 4);
    /// Offline ray-traced / path-tracing reference for calibration only.
    pub const RT_REFERENCE: Self = Self(1 << 5);
    /// Temporal upsampling / anti-aliasing history.
    pub const TEMPORAL_UPSAMPLE: Self = Self(1 << 6);

    /// The empty set.
    pub const NONE: Self = Self(0);

    /// The full shared advanced base — the seven services above.
    pub const ALL: Self =
        Self((1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6));

    /// Number of distinct services in the full base.
    pub const COUNT: u32 = 7;

    /// Raw bit pattern.
    #[must_use]
    pub const fn bits(self) -> u16 {
        self.0
    }

    /// Union of two service sets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// `true` when `self` contains every service in `other`.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Count of services present in this set.
    #[must_use]
    pub const fn len(self) -> u32 {
        self.0.count_ones()
    }

    /// `true` when no service is present.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// The four shading frontends. They diverge only in the lighting response;
/// geometry, simulation, `AVSM` self-shadowing, multiple scattering, and the
/// entire shared advanced base ([`SharedBaseServices`]) are identical across
/// all four — so top-tier `NPR` clouds are at capability parity with `PBR`
/// (design section 5 / 9e).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ShadingFrontend {
    /// Physically based: Henyey-Greenstein dual-lobe + Draine phase, single +
    /// multiple scattering, powder, silver lining, spectral sunset, aerial
    /// perspective; calibrated against the path-tracing reference.
    Pbr,
    /// Stylized (illumination axis): ramp-quantized lighting, toon layer blocks,
    /// hand-drawn cloud outlines, painterly strokes, ink diffusion, 2.5D
    /// parallax puffiness, stylized crepuscular shafts (design section 9e).
    Npr,
    /// Author-injected phase/lighting closure compiled into a `WESL`
    /// specialization, without changing the kernel.
    Custom,
    /// Per-layer / per-region blend of the above across a stacked cloud domain.
    Hybrid,
}

impl ShadingFrontend {
    /// The shared advanced base this frontend consumes.
    ///
    /// Every frontend consumes the full base: `NPR` is *not* a reduced path.
    /// The frontends differ only in how they interpret the resulting lighting,
    /// so this returns [`SharedBaseServices::ALL`] unconditionally.
    #[must_use]
    pub fn shared_base(self) -> SharedBaseServices {
        SharedBaseServices::ALL
    }

    /// `true` when this frontend consumes the entire shared advanced base.
    /// Holds for all four frontends by construction.
    #[must_use]
    pub fn shares_full_base(self) -> bool {
        self.shared_base().contains(SharedBaseServices::ALL)
    }
}

/// Per-frame caps arbitrated by [`budget::plan_volumetric`], the volumetric
/// analogue of [`crate::deformation::DeformationBudget`]. Each quota is
/// independent so one saturated stage never starves another.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VolumetricBudget {
    /// Ray-march sample dispatches admitted per frame (view integration).
    pub raymarch_samples_per_frame: u32,
    /// Density-field modelling voxels evaluated/baked per frame.
    pub modeling_voxels_per_frame: u32,
    /// Temporal-upsample / reprojection pixels resolved per frame.
    pub upsample_pixels_per_frame: u32,
    /// Multi-scatter probe / `LUT` cells updated per frame.
    pub multiscatter_cells_per_frame: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloud_kind_classification() {
        assert!(CloudKind::Cumulonimbus.precipitation_capable());
        assert!(!CloudKind::Cumulus.precipitation_capable());
        assert!(CloudKind::Cumulonimbus.vertically_developed());
        assert!(!CloudKind::Stratus.vertically_developed());
        assert!(CloudKind::Cirrus.is_high_layer());
        for kind in [
            CloudKind::Cumulus,
            CloudKind::Stratus,
            CloudKind::Cirrus,
            CloudKind::Cumulonimbus,
        ] {
            let (lo, hi) = kind.default_height_band();
            assert!(hi > lo, "height band must be ordered for {kind:?}");
        }
    }

    #[test]
    fn geometry_height_fraction_is_saturated_and_ordered() {
        let g = CloudGeometry {
            height_min: 1000.0,
            height_max: 3000.0,
            domain_half_extent: Vec3::splat(5000.0),
            density_resolution: 128,
        };
        assert_eq!(g.thickness(), 2000.0);
        assert_eq!(g.height_fraction(0.0), 0.0);
        assert_eq!(g.height_fraction(2000.0), 0.5);
        assert_eq!(g.height_fraction(9000.0), 1.0);
        // Degenerate band never divides by zero.
        let flat = CloudGeometry {
            height_min: 1000.0,
            height_max: 1000.0,
            domain_half_extent: Vec3::ZERO,
            density_resolution: 0,
        };
        assert_eq!(flat.height_fraction(1000.0), 0.0);
    }

    #[test]
    fn weather_sample_round_trips_and_saturates() {
        let s = WeatherSample::from_rgba(0.3, 1.5, -0.2, 0.7);
        assert_eq!(s.coverage, 0.3);
        assert_eq!(s.cloud_type, 1.0);
        assert_eq!(s.precipitation, 0.0);
        assert_eq!(s.wind_disturbance, 0.7);
        assert_eq!(s.to_rgba(), [0.3, 1.0, 0.0, 0.7]);
        let mid = WeatherSample::from_rgba(0.0, 0.0, 0.0, 0.0)
            .lerp(WeatherSample::from_rgba(1.0, 1.0, 1.0, 1.0), 0.5);
        assert_eq!(mid.to_rgba(), [0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn shared_base_set_algebra() {
        let a = SharedBaseServices::HYBRID_GI.union(SharedBaseServices::RESTIR);
        assert!(a.contains(SharedBaseServices::HYBRID_GI));
        assert!(!a.contains(SharedBaseServices::VSM_SHADOW));
        assert_eq!(a.len(), 2);
        assert!(SharedBaseServices::NONE.is_empty());
        assert_eq!(SharedBaseServices::ALL.len(), SharedBaseServices::COUNT);
    }

    #[test]
    fn every_frontend_shares_the_full_advanced_base() {
        // The core contract: NPR is not a reduced path. All four frontends
        // consume the entire shared advanced base.
        for frontend in [
            ShadingFrontend::Pbr,
            ShadingFrontend::Npr,
            ShadingFrontend::Custom,
            ShadingFrontend::Hybrid,
        ] {
            assert!(frontend.shares_full_base());
            assert_eq!(frontend.shared_base(), SharedBaseServices::ALL);
            for service in [
                SharedBaseServices::HYBRID_GI,
                SharedBaseServices::RESTIR,
                SharedBaseServices::VSM_SHADOW,
                SharedBaseServices::FROXEL_VOLUME,
                SharedBaseServices::ATMOSPHERE_LUT,
                SharedBaseServices::RT_REFERENCE,
                SharedBaseServices::TEMPORAL_UPSAMPLE,
            ] {
                assert!(frontend.shared_base().contains(service));
            }
        }
    }
}
