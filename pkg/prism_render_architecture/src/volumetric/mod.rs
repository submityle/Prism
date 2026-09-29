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

pub mod atmosphere;
pub mod avsm;
pub mod budget;
pub mod cloud_lod;
pub mod coupling;
pub mod fog;
pub mod math;
pub mod modeling;
pub mod multiscatter;
pub mod noise;
pub mod raymarch;
pub mod reference;
pub mod scatter;
pub mod shadow;
pub mod spectral;
pub mod storm;
pub mod temporal;
pub mod weather;

#[cfg(test)]
mod integration_tests;

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
    /// Top-tier next-gen `NPR` stylization vector (design section 9e).
    /// Consulted only when [`Self::frontend`] resolves to the `NPR` or
    /// hybrid path; `PBR` and custom frontends ignore it. It is always
    /// present so a layer can switch frontends at runtime without
    /// reshaping its shading contract.
    pub npr: NprStyle,
    /// Per-height `PBR`↔`NPR` blend mask for the hybrid frontend
    /// (design section 5.3/5.4). Ignored by the non-hybrid frontends.
    pub hybrid_mix: HybridLayerMix,
}

/// Top-tier next-gen `NPR` cloud stylization vector (design section 9e).
///
/// These are the *illumination-response and post-process* knobs that pull
/// stylized clouds to capability parity with the `PBR` frontend. The geometry,
/// simulation, `AVSM` self-shadowing, multiple scattering, `GI`, atmosphere,
/// and temporal upsampling feeding an `NPR` layer are byte-for-byte identical
/// to the `PBR` path (see [`ShadingFrontend`]); only the final lighting
/// interpretation and screen-space stylization differ. Every axis is a pure,
/// deterministic, artist-facing scalar (or a small integer cadence), so the
/// whole struct is `CPU`-testable with no `GPU` state.
///
/// The eight design-section-9e axes map to the fields as follows: painterly
/// oil strokes ([`Self::painterly`]); eastern ink-wash diffusion with dry-brush
/// break-up ([`Self::ink_diffusion`], [`Self::flying_white`]); 2.5D parallax
/// puffiness ([`Self::parallax_layers`], [`Self::parallax_scale`]); stylized
/// crepuscular shafts ([`Self::crepuscular_threshold`],
/// [`Self::crepuscular_gain`]); hand-drawn silver rim and toon specular blocks
/// ([`Self::silver_rim`], [`Self::toon_specular`]); warm/cool tone-grading ramp
/// ([`Self::ramp_warm`], [`Self::ramp_cool`]); silhouette outline plus interior
/// ink lines ([`Self::outline_width`], [`Self::interior_ink`]); and `on-twos`
/// stepped animation cadence ([`Self::on_twos`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NprStyle {
    /// Painterly / oil-brush strength `0..=1` (flow-aligned `Kuwahara`-style
    /// smoothing of the lit result into hand-painted patches).
    pub painterly: f32,
    /// Ink-wash diffusion edge width `0..=1`: how far the cloud silhouette
    /// bleeds outward in an eastern-ink bloom.
    pub ink_diffusion: f32,
    /// Flying-white (dry-brush) break-up amount `0..=1` fracturing the ink edge
    /// so it is not a solid stroke.
    pub flying_white: f32,
    /// 2.5D parallax-puffiness slab count: pseudo-volume cel layers offset by
    /// view parallax to fake thickness. Zero is treated as a single slab.
    pub parallax_layers: u32,
    /// Per-slab parallax offset scale `0..=1` driving the puffiness magnitude.
    pub parallax_scale: f32,
    /// Stylized crepuscular (god-ray) cull threshold `0..=1`: raw shaft
    /// intensity below this is discretized to zero for hand-drawn banding.
    pub crepuscular_threshold: f32,
    /// Stylized crepuscular shaft gain `0..=1` applied above the threshold.
    pub crepuscular_gain: f32,
    /// Hand-drawn silver-rim strength `0..=1` (thresholded grazing rim light).
    pub silver_rim: f32,
    /// Toon specular-block strength `0..=1` (quantized highlight patches).
    pub toon_specular: f32,
    /// Warm (lit-side) end of the tone-grading ramp, linear `RGB`.
    pub ramp_warm: Vec3,
    /// Cool (shadow-side) end of the tone-grading ramp, linear `RGB`.
    pub ramp_cool: Vec3,
    /// Silhouette outline width `0..=1` driven by depth/density gradient.
    pub outline_width: f32,
    /// Interior ink-line density `0..=1` (layered inner strokes).
    pub interior_ink: f32,
    /// `on-twos` animation cadence: hold each simulated frame for this many
    /// display frames (1 = full rate, 2 = classic `on-twos`). Zero is 1.
    pub on_twos: u32,
}

impl NprStyle {
    /// A fully disabled style: no stylization, an identity warm→cool ramp
    /// (white lit, black shadow), one parallax slab, full-rate animation. A
    /// layer that carries this on an `NPR` frontend renders like a plain
    /// grayscale lighting readout.
    pub const DISABLED: Self = Self {
        painterly: 0.0,
        ink_diffusion: 0.0,
        flying_white: 0.0,
        parallax_layers: 1,
        parallax_scale: 0.0,
        crepuscular_threshold: 0.0,
        crepuscular_gain: 0.0,
        silver_rim: 0.0,
        toon_specular: 0.0,
        ramp_warm: Vec3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        ramp_cool: Vec3::ZERO,
        outline_width: 0.0,
        interior_ink: 0.0,
        on_twos: 1,
    };

    /// A Ghibli-flavoured preset: soft painterly patches, warm sunlit tops
    /// fading to cool blue-grey shadow, gentle silver rim, a light outline, and
    /// classic `on-twos` cadence.
    #[must_use]
    pub fn ghibli() -> Self {
        Self {
            painterly: 0.65,
            ink_diffusion: 0.15,
            flying_white: 0.1,
            parallax_layers: 4,
            parallax_scale: 0.35,
            crepuscular_threshold: 0.35,
            crepuscular_gain: 0.8,
            silver_rim: 0.7,
            toon_specular: 0.5,
            ramp_warm: Vec3::new(1.0, 0.93, 0.78),
            ramp_cool: Vec3::new(0.32, 0.4, 0.55),
            outline_width: 0.25,
            interior_ink: 0.2,
            on_twos: 2,
        }
    }

    /// An eastern ink-wash (`shui-mo`) preset: strong diffusion edges with
    /// dry-brush break-up, monochrome cool ramp, heavy outline, minimal
    /// specular, and stepped animation.
    #[must_use]
    pub fn ink_wash() -> Self {
        Self {
            painterly: 0.2,
            ink_diffusion: 0.85,
            flying_white: 0.6,
            parallax_layers: 2,
            parallax_scale: 0.15,
            crepuscular_threshold: 0.5,
            crepuscular_gain: 0.4,
            silver_rim: 0.15,
            toon_specular: 0.1,
            ramp_warm: Vec3::new(0.92, 0.92, 0.9),
            ramp_cool: Vec3::new(0.12, 0.13, 0.16),
            outline_width: 0.6,
            interior_ink: 0.55,
            on_twos: 3,
        }
    }

    /// Returns a copy with every axis clamped into its valid range: strengths
    /// into `[0, 1]`, cadence counts to at least 1. Idempotent and never
    /// produces `NaN`, so downstream stylization can assume well-formed input.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            painterly: math::saturate(self.painterly),
            ink_diffusion: math::saturate(self.ink_diffusion),
            flying_white: math::saturate(self.flying_white),
            parallax_layers: self.parallax_layers.max(1),
            parallax_scale: math::saturate(self.parallax_scale),
            crepuscular_threshold: math::saturate(self.crepuscular_threshold),
            crepuscular_gain: math::saturate(self.crepuscular_gain),
            silver_rim: math::saturate(self.silver_rim),
            toon_specular: math::saturate(self.toon_specular),
            ramp_warm: self.ramp_warm,
            ramp_cool: self.ramp_cool,
            outline_width: math::saturate(self.outline_width),
            interior_ink: math::saturate(self.interior_ink),
            on_twos: self.on_twos.max(1),
        }
    }

    /// Grades a scalar lit term `0..=1` through the warm/cool tone ramp,
    /// returning the stylized colour. `lit == 0` yields [`Self::ramp_cool`],
    /// `lit == 1` yields [`Self::ramp_warm`], with a linear blend between; the
    /// input is saturated so out-of-range lighting cannot extrapolate past the
    /// ramp ends.
    #[must_use]
    pub fn ramp_color(self, lit: f32) -> Vec3 {
        self.ramp_cool.lerp(self.ramp_warm, math::saturate(lit))
    }

    /// Discretizes a raw physical god-ray intensity `0..=1` into a stylized
    /// shaft weight. Intensity at or below [`Self::crepuscular_threshold`] is
    /// culled to zero (hand-drawn banding); above it, the remainder is
    /// remapped to `[0, 1]` and scaled by [`Self::crepuscular_gain`]. The
    /// result is a non-decreasing function of `raw` in `[0, 1]`.
    #[must_use]
    pub fn crepuscular_shaft(self, raw: f32) -> f32 {
        let raw = math::saturate(raw);
        if raw <= self.crepuscular_threshold {
            return 0.0;
        }
        let above = math::remap(raw, self.crepuscular_threshold, 1.0, 0.0, 1.0);
        math::saturate(above * math::saturate(self.crepuscular_gain))
    }

    /// Combines depth and density silhouette gradients into a `[0, 1]` outline
    /// weight scaled by [`Self::outline_width`]. The stronger of the two
    /// gradients drives the edge so either a depth discontinuity or a density
    /// cliff raises an outline.
    #[must_use]
    pub fn outline_weight(self, depth_gradient: f32, density_gradient: f32) -> f32 {
        let edge = depth_gradient.max(density_gradient).max(0.0);
        math::saturate(math::saturate(self.outline_width) * edge)
    }

    /// Thresholded hand-drawn silver-rim weight from a grazing term
    /// `grazing` (`0..=1`, e.g. `1 - |view·light|`). Below the rim onset the
    /// weight is zero; above it the rim ramps smoothly to
    /// [`Self::silver_rim`]. Deterministic and in `[0, 1]`.
    #[must_use]
    pub fn silver_rim_weight(self, grazing: f32) -> f32 {
        let onset = 0.6;
        let band = math::smoothstep(onset, 1.0, math::saturate(grazing));
        math::saturate(self.silver_rim) * band
    }

    /// Holds the simulation clock to the `on-twos` cadence: the returned
    /// display frame is the most recent multiple of the cadence at or before
    /// `frame`, so animation updates in discrete steps like hand-drawn
    /// two-frame holds. A zero cadence is treated as full rate.
    #[must_use]
    pub fn on_twos_frame(self, frame: u64) -> u64 {
        let cadence = self.on_twos.max(1) as u64;
        frame - (frame % cadence)
    }

    /// Parallax offset for one 2.5D puffiness slab. Slab `layer_index` (0 at
    /// the base) is displaced along the view direction by a fraction of
    /// [`Self::parallax_scale`] proportional to its depth in the stack, faking
    /// cel-animation thickness. Out-of-range indices are clamped to the top
    /// slab so the call never panics.
    #[must_use]
    pub fn parallax_offset(self, layer_index: u32, view_xy: Vec2) -> Vec2 {
        let slabs = self.parallax_layers.max(1);
        let clamped = layer_index.min(slabs - 1);
        let depth = clamped as f32 / slabs as f32;
        view_xy.scale(depth * math::saturate(self.parallax_scale))
    }
}

/// Per-height `PBR`↔`NPR` blend mask for the hybrid frontend (design section
/// 5.3/5.4). It answers "how stylized is this layer at a given height?" so a
/// stacked cloud domain can run physical low cumulus and stylized high cirrus
/// within one layer, cross-fading smoothly rather than hard-switching.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HybridLayerMix {
    /// Height fraction `0..=1` at or below which the layer is fully `PBR`.
    pub pbr_below: f32,
    /// Height fraction `0..=1` at or above which the layer is fully `NPR`.
    pub npr_above: f32,
}

impl HybridLayerMix {
    /// A degenerate all-`PBR` mask (no stylization anywhere), the neutral
    /// default a non-hybrid layer carries.
    pub const ALL_PBR: Self = Self {
        pbr_below: 1.0,
        npr_above: 1.0,
    };

    /// The `NPR` weight `0..=1` at the given height fraction, ramping smoothly
    /// from 0 at [`Self::pbr_below`] to 1 at [`Self::npr_above`]. Monotonically
    /// non-decreasing in `height_fraction`; an inverted or collapsed band
    /// degrades to a hard step via [`math::smoothstep`] instead of dividing by
    /// zero.
    #[must_use]
    pub fn npr_weight(self, height_fraction: f32) -> f32 {
        math::smoothstep(
            self.pbr_below,
            self.npr_above,
            math::saturate(height_fraction),
        )
    }

    /// The complementary `PBR` weight `1 - npr_weight`, in `[0, 1]`.
    #[must_use]
    pub fn pbr_weight(self, height_fraction: f32) -> f32 {
        1.0 - self.npr_weight(height_fraction)
    }
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

    #[test]
    fn npr_style_presets_sanitize_into_range() {
        for style in [NprStyle::DISABLED, NprStyle::ghibli(), NprStyle::ink_wash()] {
            let s = style.sanitized();
            assert_eq!(s, s.sanitized(), "sanitize is idempotent");
            for v in [
                s.painterly,
                s.ink_diffusion,
                s.flying_white,
                s.parallax_scale,
                s.crepuscular_threshold,
                s.crepuscular_gain,
                s.silver_rim,
                s.toon_specular,
                s.outline_width,
                s.interior_ink,
            ] {
                assert!((0.0..=1.0).contains(&v), "strength axis in range: {v}");
            }
            assert!(s.parallax_layers >= 1);
            assert!(s.on_twos >= 1);
        }
    }

    #[test]
    fn npr_style_clamps_out_of_range_input() {
        let wild = NprStyle {
            painterly: 5.0,
            ink_diffusion: -2.0,
            flying_white: f32::NAN.max(2.0),
            parallax_layers: 0,
            parallax_scale: 9.0,
            crepuscular_threshold: -0.5,
            crepuscular_gain: 3.0,
            silver_rim: -1.0,
            toon_specular: 4.0,
            ramp_warm: Vec3::splat(2.0),
            ramp_cool: Vec3::splat(-1.0),
            outline_width: 7.0,
            interior_ink: -3.0,
            on_twos: 0,
        };
        let s = wild.sanitized();
        assert_eq!(s.painterly, 1.0);
        assert_eq!(s.ink_diffusion, 0.0);
        assert_eq!(s.crepuscular_threshold, 0.0);
        assert_eq!(s.parallax_layers, 1);
        assert_eq!(s.on_twos, 1);
    }

    #[test]
    fn npr_ramp_color_interpolates_cool_to_warm() {
        let s = NprStyle::ghibli();
        let close = |a: Vec3, b: Vec3| {
            (a.x - b.x).abs() < 1e-5 && (a.y - b.y).abs() < 1e-5 && (a.z - b.z).abs() < 1e-5
        };
        assert_eq!(s.ramp_color(0.0), s.ramp_cool);
        assert!(close(s.ramp_color(1.0), s.ramp_warm));
        // Out-of-range lit terms saturate rather than extrapolate.
        assert_eq!(s.ramp_color(-1.0), s.ramp_cool);
        assert!(close(s.ramp_color(2.0), s.ramp_warm));
        let mid = s.ramp_color(0.5);
        assert!(mid.x > s.ramp_cool.x && mid.x < s.ramp_warm.x);
    }

    #[test]
    fn npr_crepuscular_shaft_thresholds_and_is_monotone() {
        let s = NprStyle {
            crepuscular_threshold: 0.3,
            crepuscular_gain: 1.0,
            ..NprStyle::DISABLED
        };
        assert_eq!(s.crepuscular_shaft(0.0), 0.0);
        assert_eq!(s.crepuscular_shaft(0.3), 0.0);
        assert!(s.crepuscular_shaft(0.6) > 0.0);
        // Non-decreasing above the threshold and bounded to [0, 1].
        let mut prev = 0.0;
        let mut x = 0.0;
        while x <= 1.0 {
            let w = s.crepuscular_shaft(x);
            assert!((0.0..=1.0).contains(&w));
            assert!(w + 1e-6 >= prev, "monotone non-decreasing at {x}");
            prev = w;
            x += 0.05;
        }
        assert!((s.crepuscular_shaft(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn npr_outline_and_silver_rim_are_bounded() {
        let s = NprStyle {
            outline_width: 0.5,
            silver_rim: 0.8,
            ..NprStyle::DISABLED
        };
        assert_eq!(s.outline_weight(0.0, 0.0), 0.0);
        // Negative gradients cannot produce a negative outline.
        assert_eq!(s.outline_weight(-3.0, -1.0), 0.0);
        assert!(s.outline_weight(2.0, 0.0) <= 1.0);
        assert!(s.outline_weight(0.0, 2.0) <= 1.0);
        // Silver rim is zero below onset and rises to at most `silver_rim`.
        assert_eq!(s.silver_rim_weight(0.0), 0.0);
        assert!(s.silver_rim_weight(1.0) <= 0.8 + 1e-6);
        assert!(s.silver_rim_weight(1.0) > s.silver_rim_weight(0.65));
    }

    #[test]
    fn npr_on_twos_holds_frames() {
        let s = NprStyle {
            on_twos: 3,
            ..NprStyle::DISABLED
        };
        assert_eq!(s.on_twos_frame(0), 0);
        assert_eq!(s.on_twos_frame(1), 0);
        assert_eq!(s.on_twos_frame(2), 0);
        assert_eq!(s.on_twos_frame(3), 3);
        assert_eq!(s.on_twos_frame(7), 6);
        // A zero cadence is treated as full rate.
        let full = NprStyle {
            on_twos: 0,
            ..NprStyle::DISABLED
        };
        assert_eq!(full.on_twos_frame(42), 42);
    }

    #[test]
    fn npr_parallax_offset_grows_with_slab_depth_and_clamps() {
        let s = NprStyle {
            parallax_layers: 4,
            parallax_scale: 0.5,
            ..NprStyle::DISABLED
        };
        let dir = Vec2::new(1.0, 0.0);
        let base = s.parallax_offset(0, dir);
        let top = s.parallax_offset(3, dir);
        assert_eq!(base.x, 0.0);
        assert!(top.x > base.x);
        // Out-of-range slab index clamps to the top slab (no panic).
        assert_eq!(s.parallax_offset(99, dir), top);
    }

    #[test]
    fn hybrid_layer_mix_is_monotone_and_complementary() {
        let mix = HybridLayerMix {
            pbr_below: 0.3,
            npr_above: 0.7,
        };
        assert_eq!(mix.npr_weight(0.0), 0.0);
        assert_eq!(mix.npr_weight(0.3), 0.0);
        assert_eq!(mix.npr_weight(0.7), 1.0);
        assert_eq!(mix.npr_weight(1.0), 1.0);
        let mut prev = 0.0;
        let mut h = 0.0;
        while h <= 1.0 {
            let w = mix.npr_weight(h);
            assert!((0.0..=1.0).contains(&w));
            assert!(w + 1e-6 >= prev, "npr weight monotone at {h}");
            assert!(
                (mix.pbr_weight(h) + w - 1.0).abs() < 1e-6,
                "weights sum to one"
            );
            prev = w;
            h += 0.05;
        }
        // Inverted band degrades to a hard step without dividing by zero.
        let inverted = HybridLayerMix {
            pbr_below: 0.8,
            npr_above: 0.2,
        };
        let w = inverted.npr_weight(0.5);
        assert!((0.0..=1.0).contains(&w));
        assert_eq!(HybridLayerMix::ALL_PBR.npr_weight(0.5), 0.0);
    }
}
