//! Data-driven water authoring: the `WaterBodyAsset` and its compiler.
//!
//! A water body is authored as plain data (see the engine design doc section 2)
//! and *compiled* into the numeric solver parameters the sibling modules read
//! plus a shader specialization key. The authored asset is a serialization-
//! shaped POD — a host editor or a `RON`/`serde` layer in a sibling crate owns
//! the on-disk format; this zero-dependency crate owns the schema and the
//! deterministic compile step, never a file parser.
//!
//! Compilation does three things, all as pure deterministic functions:
//!
//! 1. **Validate and normalize.** Out-of-range values are clamped (steepness,
//!    spreads, gradient weights into `0..=1`; the index of refraction to at
//!    least `1.0`), and combinations that cannot run — a `Volume` body on a
//!    height-field solver, a zero-resolution grid — are rejected rather than
//!    silently producing garbage.
//! 2. **Resolve solver parameters.** The authored physics/wave/shading fields
//!    become a flat [`CompiledWaterParams`] the solver and shading planners can
//!    consume without re-deriving anything.
//! 3. **Build the specialization key.** The enabled features (cascades, foam,
//!    caustics, spectral dispersion, underwater scattering, breaking, wetness,
//!    incompressibility) plus the solver and frontend ordinals pack into a
//!    [`WaterSpecializationKey`], the `WESL` specialization key named in the
//!    design doc. Equal assets always produce an equal key, so the shader
//!    permutation cache is stable.
//!
//! Only `sqrt` among the float intrinsics is used elsewhere in the subsystem;
//! this module uses none. There are no `f32` equality tests (magnitudes are
//! compared against [`EPS`]) and no AI/ML.

use super::{ShadingFrontend, SolverKind, Vec3, WaterKind, EPS};

/// Ocean spectral model selector (authoring value).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SpectrumModel {
    /// `Phillips` spectrum: the classic fully-developed-sea model.
    Phillips,
    /// `JONSWAP`: fetch-limited, sharper peak than `Pierson-Moskowitz`.
    Jonswap,
    /// `Pierson-Moskowitz`: fully-developed open-ocean spectrum.
    PiersonMoskowitz,
}

/// Domain / interaction boundary condition (authoring value).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BoundaryCondition {
    /// Waves reflect off the domain edge.
    Reflective,
    /// Waves leave the domain (absorbing / radiating edge).
    Open,
    /// The domain wraps, for tiling spectral oceans.
    Periodic,
    /// A no-slip solid wall (containers, pools, pipes).
    SolidWall,
}

/// Geometry authoring block (design doc section 2: geometry).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeometrySpec {
    /// Half-extent of the simulation domain, in meters.
    pub domain_half_extent: Vec3,
    /// Grid resolution per side (cascade/height-field, or reconstruction grid).
    pub grid_resolution: u32,
    /// Maximum particle count for `Volume` bodies (`0` for non-particle).
    pub particle_cap: u32,
}

/// Physical-property authoring block (design doc section 2: physics).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhysicalSpec {
    /// Fluid density in `kg/m^3` (water is ~`1000`).
    pub density: f32,
    /// Kinematic viscosity (higher is more syrup-like).
    pub viscosity: f32,
    /// Surface-tension coefficient.
    pub surface_tension: f32,
    /// Still-water reference level (world Y the surface relaxes to).
    pub still_water_level: f32,
    /// Per-channel `Beer-Lambert` extinction (RGB), 1/meter.
    pub extinction: Vec3,
    /// Volumetric scattering coefficient for underwater light transport.
    pub scattering: f32,
    /// Index of refraction (`IOR`); water is ~`1.33`.
    pub ior: f32,
    /// Spectral dispersion coefficient (per-wavelength `IOR` offset scale).
    pub dispersion_coeff: f32,
}

/// Wave-shape authoring block (design doc section 2: wave form).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaveSpec {
    /// Ocean spectrum model.
    pub spectrum: SpectrumModel,
    /// Wind speed driving the spectrum, m/s.
    pub wind_speed: f32,
    /// Wind direction in radians.
    pub wind_direction: f32,
    /// Requested spectral cascade levels (clamped and solver-gated on compile).
    pub cascade_count: u32,
    /// Largest cascade tile size in meters.
    pub cascade_scale: f32,
    /// `Gerstner`/choppiness steepness in `0..=1`.
    pub steepness: f32,
    /// Directional spreading in `0..=1` (`0` unidirectional, `1` isotropic).
    pub directional_spread: f32,
    /// Jacobian/steepness fold threshold above which a crest breaks, in
    /// `0..=1`.
    pub breaking_threshold: f32,
}

/// Interaction authoring block (design doc section 2: interaction).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InteractionSpec {
    /// Number of injectable interaction sources (boats, impacts, rain).
    pub injectable_sources: u32,
    /// Domain-edge boundary condition.
    pub boundary: BoundaryCondition,
}

/// Shading authoring block (design doc section 2: shading).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadingSpec {
    /// Lighting-response frontend (`PBR`/`NPR`/custom/hybrid).
    pub frontend: ShadingFrontend,
    /// Foam coverage threshold in `0..=1`.
    pub foam_threshold: f32,
    /// Foam persistence (decay time scale); `0` disables foam.
    pub foam_persistence: f32,
    /// Caustic intensity; near `0` disables caustics.
    pub caustic_intensity: f32,
    /// Underwater visibility distance, meters.
    pub visibility: f32,
    /// Shallow-water color-gradient curve weight in `0..=1`.
    pub shallow_gradient: f32,
    /// Wetness darkening-gradient curve weight in `0..=1`; `0` disables wetness.
    pub wetness_gradient: f32,
}

/// A fully authored water body (design doc section 2: `WaterBodyAsset`).
///
/// This is the complete data-driven description an artist edits. It is pure
/// data; [`WaterBodyAsset::compile`] turns it into runtime parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterBodyAsset {
    /// Geometry class (`Ocean`/`Surface`/`Volume`).
    pub kind: WaterKind,
    /// Requested solver bucket.
    pub solver: SolverKind,
    /// Geometry block.
    pub geometry: GeometrySpec,
    /// Physical-property block.
    pub physical: PhysicalSpec,
    /// Wave-shape block.
    pub wave: WaveSpec,
    /// Interaction block.
    pub interaction: InteractionSpec,
    /// Shading block.
    pub shading: ShadingSpec,
}

/// A compiled `WESL` specialization key.
///
/// A packed feature/solver/frontend bitset; equal assets compile to an equal
/// key so the shader-permutation cache stays stable. Comparison is integer
/// equality, never `f32`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WaterSpecializationKey(u32);

impl WaterSpecializationKey {
    const SOLVER_SHIFT: u32 = 0; // 3 bits
    const FRONTEND_SHIFT: u32 = 3; // 2 bits
    const CASCADES: u32 = 1 << 5;
    const FOAM: u32 = 1 << 6;
    const CAUSTICS: u32 = 1 << 7;
    const DISPERSION: u32 = 1 << 8;
    const UNDERWATER: u32 = 1 << 9;
    const BREAKING: u32 = 1 << 10;
    const WETNESS: u32 = 1 << 11;
    const INCOMPRESSIBLE: u32 = 1 << 12;

    /// The raw packed bits (for a shader-cache lookup or a stable hash).
    #[must_use]
    pub fn bits(self) -> u32 {
        self.0
    }

    /// `true` when `flag` (one of the feature constants) is set.
    #[must_use]
    fn has(self, flag: u32) -> bool {
        self.0 & flag != 0
    }

    /// `true` when the cascaded-displacement path is specialized in.
    #[must_use]
    pub fn has_cascades(self) -> bool {
        self.has(Self::CASCADES)
    }

    /// `true` when foam advection/shading is specialized in.
    #[must_use]
    pub fn has_foam(self) -> bool {
        self.has(Self::FOAM)
    }

    /// `true` when caustics are specialized in.
    #[must_use]
    pub fn has_caustics(self) -> bool {
        self.has(Self::CAUSTICS)
    }

    /// `true` when spectral dispersion is specialized in.
    #[must_use]
    pub fn has_dispersion(self) -> bool {
        self.has(Self::DISPERSION)
    }

    /// `true` when underwater scattering is specialized in.
    #[must_use]
    pub fn has_underwater(self) -> bool {
        self.has(Self::UNDERWATER)
    }

    /// `true` when breaking-wave handling is specialized in.
    #[must_use]
    pub fn has_breaking(self) -> bool {
        self.has(Self::BREAKING)
    }

    /// `true` when wetness/shoreline darkening is specialized in.
    #[must_use]
    pub fn has_wetness(self) -> bool {
        self.has(Self::WETNESS)
    }

    /// `true` when the solver enforces incompressibility (`PBF`/`FLIP`).
    #[must_use]
    pub fn is_incompressible(self) -> bool {
        self.has(Self::INCOMPRESSIBLE)
    }
}

fn solver_ordinal(solver: SolverKind) -> u32 {
    match solver {
        SolverKind::SpectralIfft => 0,
        SolverKind::Gerstner => 1,
        SolverKind::ShallowWater => 2,
        SolverKind::Pbf => 3,
        SolverKind::FlipApic => 4,
    }
}

fn frontend_ordinal(frontend: ShadingFrontend) -> u32 {
    match frontend {
        ShadingFrontend::Pbr => 0,
        ShadingFrontend::Npr => 1,
        ShadingFrontend::Custom => 2,
        ShadingFrontend::Hybrid => 3,
    }
}

/// The compiled runtime parameters of a water body.
///
/// A flat, validated snapshot: routing (`kind`/`solver`/`frontend`), the
/// resolved grid and cascade counts, the clamped physics and shading scalars,
/// and the [`WaterSpecializationKey`]. The sibling solver/shading planners read
/// these directly; nothing here needs re-derivation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompiledWaterParams {
    /// Geometry class.
    pub kind: WaterKind,
    /// Resolved solver bucket.
    pub solver: SolverKind,
    /// Lighting-response frontend.
    pub frontend: ShadingFrontend,
    /// Grid resolution per side.
    pub grid_resolution: u32,
    /// Cascade levels after solver gating (`0` for non-spectral bodies).
    pub cascade_count: u32,
    /// Particle cap after solver gating (`0` for non-particle bodies).
    pub particle_cap: u32,
    /// Domain half-extent in meters.
    pub domain_half_extent: Vec3,
    /// Still-water reference level.
    pub still_water_level: f32,
    /// Fluid density, `kg/m^3`.
    pub density: f32,
    /// Kinematic viscosity.
    pub viscosity: f32,
    /// Index of refraction (at least `1.0`).
    pub ior: f32,
    /// Spectral dispersion coefficient (clamped non-negative).
    pub dispersion_coeff: f32,
    /// Per-channel extinction (clamped non-negative).
    pub extinction: Vec3,
    /// Scattering coefficient (clamped non-negative).
    pub scattering: f32,
    /// Clamped steepness in `0..=1`.
    pub steepness: f32,
    /// Clamped directional spread in `0..=1`.
    pub directional_spread: f32,
    /// Clamped foam threshold in `0..=1`.
    pub foam_threshold: f32,
    /// Foam persistence (clamped non-negative).
    pub foam_persistence: f32,
    /// Caustic intensity (clamped non-negative).
    pub caustic_intensity: f32,
    /// Underwater visibility distance (clamped positive).
    pub visibility: f32,
    /// The `WESL` specialization key.
    pub spec_key: WaterSpecializationKey,
}

/// Returns `true` when `solver` can drive a body of geometry class `kind`.
///
/// `Ocean` runs spectral `IFFT` or `Gerstner`; `Surface` runs shallow water or
/// `Gerstner`; `Volume` runs the particle solvers (`PBF`, `FLIP`/`APIC`).
#[must_use]
pub fn solver_matches_kind(kind: WaterKind, solver: SolverKind) -> bool {
    match kind {
        WaterKind::Ocean => matches!(solver, SolverKind::SpectralIfft | SolverKind::Gerstner),
        WaterKind::Surface => {
            matches!(solver, SolverKind::ShallowWater | SolverKind::Gerstner)
        }
        WaterKind::Volume => solver.is_particle_based(),
    }
}

fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

fn non_negative(v: f32) -> f32 {
    v.max(0.0)
}

impl WaterBodyAsset {
    /// Compiles the authored asset into [`CompiledWaterParams`].
    ///
    /// Returns `None` when the asset cannot run: a solver that does not match
    /// the geometry class, a grid resolution below `2`, a non-positive
    /// density, or a `Volume` body with a zero particle cap. Everything else is
    /// clamped into a usable range, so a compiled asset is always well-formed.
    #[must_use]
    pub fn compile(&self) -> Option<CompiledWaterParams> {
        if !solver_matches_kind(self.kind, self.solver) {
            return None;
        }
        if self.geometry.grid_resolution < 2 {
            return None;
        }
        if self.physical.density <= EPS {
            return None;
        }
        if self.kind == WaterKind::Volume && self.geometry.particle_cap == 0 {
            return None;
        }

        // Cascades only exist for the spectral ocean path; everything else
        // resolves to zero so downstream code never allocates stray cascades.
        let cascade_count = if self.solver == SolverKind::SpectralIfft {
            self.wave.cascade_count.max(1)
        } else {
            0
        };
        let particle_cap = if self.solver.is_particle_based() {
            self.geometry.particle_cap
        } else {
            0
        };

        let steepness = clamp01(self.wave.steepness);
        let directional_spread = clamp01(self.wave.directional_spread);
        let breaking_threshold = clamp01(self.wave.breaking_threshold);
        let foam_threshold = clamp01(self.shading.foam_threshold);
        let foam_persistence = non_negative(self.shading.foam_persistence);
        let caustic_intensity = non_negative(self.shading.caustic_intensity);
        let dispersion_coeff = non_negative(self.physical.dispersion_coeff);
        let scattering = non_negative(self.physical.scattering);
        let wetness_gradient = clamp01(self.shading.wetness_gradient);
        let visibility = self.shading.visibility.max(EPS);
        let ior = self.physical.ior.max(1.0);
        let extinction = Vec3::new(
            non_negative(self.physical.extinction.x),
            non_negative(self.physical.extinction.y),
            non_negative(self.physical.extinction.z),
        );

        let spec_key = Self::build_key(
            self.kind,
            self.solver,
            self.shading.frontend,
            cascade_count,
            foam_persistence,
            foam_threshold,
            caustic_intensity,
            dispersion_coeff,
            scattering,
            breaking_threshold,
            wetness_gradient,
        );

        Some(CompiledWaterParams {
            kind: self.kind,
            solver: self.solver,
            frontend: self.shading.frontend,
            grid_resolution: self.geometry.grid_resolution,
            cascade_count,
            particle_cap,
            domain_half_extent: self.geometry.domain_half_extent,
            still_water_level: self.physical.still_water_level,
            density: self.physical.density,
            viscosity: non_negative(self.physical.viscosity),
            ior,
            dispersion_coeff,
            extinction,
            scattering,
            steepness,
            directional_spread,
            foam_threshold,
            foam_persistence,
            caustic_intensity,
            visibility,
            spec_key,
        })
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "The specialization key is a flat packing of already-validated scalars; grouping them into an intermediate struct would only duplicate CompiledWaterParams."
    )]
    fn build_key(
        kind: WaterKind,
        solver: SolverKind,
        frontend: ShadingFrontend,
        cascade_count: u32,
        foam_persistence: f32,
        foam_threshold: f32,
        caustic_intensity: f32,
        dispersion_coeff: f32,
        scattering: f32,
        breaking_threshold: f32,
        wetness_gradient: f32,
    ) -> WaterSpecializationKey {
        let mut bits = solver_ordinal(solver) << WaterSpecializationKey::SOLVER_SHIFT;
        bits |= frontend_ordinal(frontend) << WaterSpecializationKey::FRONTEND_SHIFT;
        if cascade_count > 0 {
            bits |= WaterSpecializationKey::CASCADES;
        }
        // Foam needs both a finite coverage threshold and a non-zero decay.
        if foam_persistence > EPS && foam_threshold < 1.0 - EPS {
            bits |= WaterSpecializationKey::FOAM;
        }
        if caustic_intensity > EPS {
            bits |= WaterSpecializationKey::CAUSTICS;
        }
        if dispersion_coeff > EPS {
            bits |= WaterSpecializationKey::DISPERSION;
        }
        if scattering > EPS {
            bits |= WaterSpecializationKey::UNDERWATER;
        }
        // Breaking only applies to the height-field/displacement surfaces.
        if solver.is_height_field() && breaking_threshold > EPS && breaking_threshold < 1.0 - EPS {
            bits |= WaterSpecializationKey::BREAKING;
        }
        if wetness_gradient > EPS {
            bits |= WaterSpecializationKey::WETNESS;
        }
        if solver.is_incompressible() {
            bits |= WaterSpecializationKey::INCOMPRESSIBLE;
        }
        // `kind` is implied by the solver bucket but kept in the signature so a
        // future kind-specific permutation has a seam; it does not add bits.
        let _ = kind;
        WaterSpecializationKey(bits)
    }

    /// A calm `PBR` deep-ocean preset (spectral `IFFT`, four cascades).
    #[must_use]
    pub fn ocean_default(domain_half_extent: Vec3) -> Self {
        Self {
            kind: WaterKind::Ocean,
            solver: SolverKind::SpectralIfft,
            geometry: GeometrySpec {
                domain_half_extent,
                grid_resolution: 256,
                particle_cap: 0,
            },
            physical: PhysicalSpec {
                density: 1025.0,
                viscosity: 0.0,
                surface_tension: 0.072,
                still_water_level: 0.0,
                extinction: Vec3::new(0.45, 0.07, 0.03),
                scattering: 0.2,
                ior: 1.33,
                dispersion_coeff: 0.02,
            },
            wave: WaveSpec {
                spectrum: SpectrumModel::Jonswap,
                wind_speed: 9.0,
                wind_direction: 0.0,
                cascade_count: 4,
                cascade_scale: 512.0,
                steepness: 0.6,
                directional_spread: 0.3,
                breaking_threshold: 0.65,
            },
            interaction: InteractionSpec {
                injectable_sources: 8,
                boundary: BoundaryCondition::Periodic,
            },
            shading: ShadingSpec {
                frontend: ShadingFrontend::Pbr,
                foam_threshold: 0.5,
                foam_persistence: 2.0,
                caustic_intensity: 0.4,
                visibility: 12.0,
                shallow_gradient: 0.5,
                wetness_gradient: 0.0,
            },
        }
    }

    /// An interactive `PBR` river/lake preset (shallow water).
    #[must_use]
    pub fn river_default(domain_half_extent: Vec3) -> Self {
        Self {
            kind: WaterKind::Surface,
            solver: SolverKind::ShallowWater,
            geometry: GeometrySpec {
                domain_half_extent,
                grid_resolution: 128,
                particle_cap: 0,
            },
            physical: PhysicalSpec {
                density: 1000.0,
                viscosity: 0.001,
                surface_tension: 0.072,
                still_water_level: 0.0,
                extinction: Vec3::new(0.3, 0.1, 0.08),
                scattering: 0.35,
                ior: 1.33,
                dispersion_coeff: 0.0,
            },
            wave: WaveSpec {
                spectrum: SpectrumModel::Phillips,
                wind_speed: 2.0,
                wind_direction: 0.0,
                cascade_count: 0,
                cascade_scale: 0.0,
                steepness: 0.2,
                directional_spread: 0.5,
                breaking_threshold: 0.5,
            },
            interaction: InteractionSpec {
                injectable_sources: 16,
                boundary: BoundaryCondition::Open,
            },
            shading: ShadingSpec {
                frontend: ShadingFrontend::Pbr,
                foam_threshold: 0.4,
                foam_persistence: 1.0,
                caustic_intensity: 0.6,
                visibility: 5.0,
                shallow_gradient: 0.8,
                wetness_gradient: 0.7,
            },
        }
    }

    /// A film-grade `PBR` splash/pool preset (`FLIP`/`APIC` volume).
    #[must_use]
    pub fn volume_default(domain_half_extent: Vec3, particle_cap: u32) -> Self {
        Self {
            kind: WaterKind::Volume,
            solver: SolverKind::FlipApic,
            geometry: GeometrySpec {
                domain_half_extent,
                grid_resolution: 128,
                particle_cap: particle_cap.max(1),
            },
            physical: PhysicalSpec {
                density: 1000.0,
                viscosity: 0.001,
                surface_tension: 0.072,
                still_water_level: 0.0,
                extinction: Vec3::new(0.35, 0.09, 0.05),
                scattering: 0.3,
                ior: 1.33,
                dispersion_coeff: 0.0,
            },
            wave: WaveSpec {
                spectrum: SpectrumModel::Phillips,
                wind_speed: 0.0,
                wind_direction: 0.0,
                cascade_count: 0,
                cascade_scale: 0.0,
                steepness: 0.0,
                directional_spread: 0.0,
                breaking_threshold: 0.0,
            },
            interaction: InteractionSpec {
                injectable_sources: 4,
                boundary: BoundaryCondition::SolidWall,
            },
            shading: ShadingSpec {
                frontend: ShadingFrontend::Pbr,
                foam_threshold: 0.3,
                foam_persistence: 0.8,
                caustic_intensity: 0.5,
                visibility: 4.0,
                shallow_gradient: 0.3,
                wetness_gradient: 0.6,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extent() -> Vec3 {
        Vec3::new(100.0, 10.0, 100.0)
    }

    #[test]
    fn ocean_preset_compiles_with_cascades_and_pbr() {
        let asset = WaterBodyAsset::ocean_default(extent());
        let c = asset.compile().expect("ocean preset must compile");
        assert_eq!(c.kind, WaterKind::Ocean);
        assert_eq!(c.solver, SolverKind::SpectralIfft);
        assert_eq!(c.cascade_count, 4);
        assert!(c.spec_key.has_cascades());
        assert!(c.spec_key.has_foam());
        assert!(c.spec_key.has_caustics());
        assert!(c.spec_key.has_dispersion());
        assert!(c.spec_key.has_underwater());
        assert!(!c.spec_key.is_incompressible());
        assert_eq!(c.particle_cap, 0);
    }

    #[test]
    fn volume_preset_is_incompressible_without_cascades() {
        let asset = WaterBodyAsset::volume_default(extent(), 100_000);
        let c = asset.compile().expect("volume preset must compile");
        assert!(c.spec_key.is_incompressible());
        assert!(!c.spec_key.has_cascades());
        assert_eq!(c.cascade_count, 0);
        assert_eq!(c.particle_cap, 100_000);
        // Breaking is a height-field feature, never set for a particle volume.
        assert!(!c.spec_key.has_breaking());
    }

    #[test]
    fn river_preset_enables_breaking_and_wetness() {
        let asset = WaterBodyAsset::river_default(extent());
        let c = asset.compile().expect("river preset must compile");
        assert_eq!(c.solver, SolverKind::ShallowWater);
        assert!(c.spec_key.has_breaking());
        assert!(c.spec_key.has_wetness());
        assert!(!c.spec_key.has_cascades());
        assert!(!c.spec_key.has_dispersion());
    }

    #[test]
    fn mismatched_solver_and_kind_is_rejected() {
        let mut asset = WaterBodyAsset::ocean_default(extent());
        asset.solver = SolverKind::Pbf; // a particle solver on an Ocean body
        assert!(asset.compile().is_none());
        assert!(!solver_matches_kind(WaterKind::Ocean, SolverKind::Pbf));
        assert!(solver_matches_kind(WaterKind::Volume, SolverKind::Pbf));
    }

    #[test]
    fn zero_resolution_and_zero_density_are_rejected() {
        let mut asset = WaterBodyAsset::river_default(extent());
        asset.geometry.grid_resolution = 1;
        assert!(asset.compile().is_none());
        let mut asset2 = WaterBodyAsset::river_default(extent());
        asset2.physical.density = 0.0;
        assert!(asset2.compile().is_none());
    }

    #[test]
    fn volume_with_zero_particle_cap_is_rejected() {
        let mut asset = WaterBodyAsset::volume_default(extent(), 1);
        asset.geometry.particle_cap = 0;
        assert!(asset.compile().is_none());
    }

    #[test]
    fn out_of_range_scalars_are_clamped() {
        let mut asset = WaterBodyAsset::ocean_default(extent());
        asset.wave.steepness = 3.0;
        asset.wave.directional_spread = -1.0;
        asset.shading.foam_threshold = 9.0;
        asset.physical.ior = 0.5;
        asset.physical.dispersion_coeff = -2.0;
        asset.physical.extinction = Vec3::new(-1.0, -2.0, -3.0);
        let c = asset.compile().expect("clamping must still compile");
        assert!((c.steepness - 1.0).abs() < EPS);
        assert!(c.directional_spread.abs() < EPS);
        assert!((c.foam_threshold - 1.0).abs() < EPS);
        assert!((c.ior - 1.0).abs() < EPS);
        assert!(c.dispersion_coeff.abs() < EPS);
        assert!(c.extinction.x.abs() < EPS && c.extinction.y.abs() < EPS);
        // A clamped-to-one foam threshold means no foam permutation.
        assert!(!c.spec_key.has_foam());
    }

    #[test]
    fn foam_disabled_when_persistence_zero() {
        let mut asset = WaterBodyAsset::ocean_default(extent());
        asset.shading.foam_persistence = 0.0;
        let c = asset.compile().expect("compiles");
        assert!(!c.spec_key.has_foam());
    }

    #[test]
    fn npr_frontend_shares_full_base_and_changes_key() {
        let mut asset = WaterBodyAsset::ocean_default(extent());
        let pbr = asset.compile().expect("pbr compiles");
        asset.shading.frontend = ShadingFrontend::Npr;
        let npr = asset.compile().expect("npr compiles");
        // NPR is a different shader permutation but the same feature set.
        assert_ne!(pbr.spec_key.bits(), npr.spec_key.bits());
        assert!(asset.shading.frontend.shares_full_base());
    }

    #[test]
    fn compile_is_deterministic() {
        let asset = WaterBodyAsset::ocean_default(extent());
        assert_eq!(asset.compile(), asset.compile());
    }
}
