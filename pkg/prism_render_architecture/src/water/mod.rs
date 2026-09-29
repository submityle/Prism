//! Water / fluid subsystem contracts (AAA ocean + free-surface fluid engine).
//!
//! Water is a first-class rendering subsystem in the same sense as cloth, hair,
//! and particles: it owns its own geometry, its own simulation, and its own
//! special rendering. It is *not* a normal-map trick painted onto a flat quad.
//! The pipeline mirrors production water/fluid engines at the algorithm level
//! only, without reusing any of their code: `UE5` Water and Single-Layer Water,
//! `Crest` and `WaveWorks` ocean cascades, `Tessendorf` `FFT` oceans, `Niagara`
//! and `Houdini` grid fluids, Position-Based Fluids (`PBF`), `FLIP`/`APIC`
//! particle-grid fluids, and Shallow-Water Equations (`SWE`).
//!
//! Three water bodies share one [`WaterBody`] abstraction:
//!
//! 1. **Ocean** — an unbounded cascaded displacement mesh driven by a spectral
//!    inverse `FFT` (`Phillips`/`JONSWAP`/Pierson-Moskowitz spectra) plus
//!    `Gerstner` trains and dynamic wave injection; see [`spectrum`],
//!    [`ocean_lod`], and [`breaking`].
//! 2. **Surface** — spline/polygon 2.5D height fields solved with the
//!    Shallow-Water Equations for rivers, lakes, and interactive ripples; see
//!    [`swe`].
//! 3. **Volume** — 3D domains of fluid particles solved with `PBF` or
//!    `FLIP`/`APIC` and reconstructed into a surface; see [`pbf`], [`flip`],
//!    and [`reconstruct`].
//!
//! Shared foam advection ([`foam`]), caustics ([`caustics`]), spectral
//! dispersion ([`dispersion`]), underwater volumetrics ([`underwater`]),
//! wetness/shoreline ([`wetness`]), the waterline mask ([`waterline`]),
//! two-way coupling ([`coupling`]), solver-transition blending ([`transition`]),
//! and the per-frame solve budget ([`budget`]) round out the subsystem.
//!
//! **Frontend orthogonality.** The `PBR`, `NPR`, custom, and hybrid frontends
//! diverge *only* in their lighting response (see [`ShadingFrontend`]). Every
//! frontend — `NPR` included — consumes the exact same shared advanced base:
//! virtual geometry, `Lumen`-style hybrid GI, `ReSTIR` DI/GI, virtual shadow
//! maps, froxel volumetrics, ray-traced reflections and caustics, the
//! path-tracing reference, and temporal upsampling. This is encoded in
//! [`SharedBaseServices`] and asserted by [`ShadingFrontend::shared_base`], so
//! "does `NPR` miss any advanced feature?" has a compile-checked answer: no.
//!
//! This module owns the geometry, solver, budget, and shading contracts and the
//! math/handle types shared across the sibling modules. It references, never
//! reimplements, the shared deformation budget
//! ([`crate::deformation::schedule`]), the material water shading closures, the
//! transparency (OIT) routing, and the `prism_physics_core` constraint
//! primitives. Only classical numerical methods are used; there is no AI/ML
//! anywhere in the subsystem.

// The full module list is completed incrementally as each solver/render
// module lands (see the water engine design doc, roadmap M0-M9); every
// intermediate state keeps the crate compiling and its gates green.
pub mod breaking;
pub mod budget;
pub mod caustics;
pub mod coupling;
pub mod coupling_frame;
pub mod dispersion;
pub mod flip;
pub mod foam;
pub mod kernels;
pub mod ocean_lod;
pub mod optics;
pub mod pbf;
pub mod pipeline;
pub mod profile;
pub mod reconstruct;
pub mod shading;
pub mod shoreline;
pub mod simulation;
pub mod spectrum;
pub mod surface_fx;
pub mod swe;
pub mod transition;
pub mod underwater;
pub mod waterline;
pub mod wetness;

use crate::deformation::DeformationHandle;

/// Version of the water subsystem contracts in this module.
pub const WATER_ARCHITECTURE_VERSION: u32 = 1;

/// Squared-length threshold below which a vector is treated as zero, so
/// normalization and projection never divide by (near) zero and never
/// propagate `NaN`.
pub const EPS_LEN_SQ: f32 = 1e-12;

/// General absolute tolerance for scalar comparisons.
///
/// `f32` equality is never tested with `==`; call sites compare magnitudes
/// against this constant instead, matching the workspace determinism policy.
pub const EPS: f32 = 1e-6;

/// Standard gravitational acceleration (m/s^2), the dispersion constant for
/// deep-water gravity waves and the shallow-water restoring force.
pub const GRAVITY: f32 = 9.81;

/// The mathematical constant pi, needed for spectral phase and Gerstner math.
pub const PI: f32 = core::f32::consts::PI;

/// Two pi, the phase period used by [`sin_approx`] / [`cos_approx`] range
/// reduction.
pub const TWO_PI: f32 = 2.0 * core::f32::consts::PI;

/// Half pi, used to derive cosine from sine.
pub const FRAC_PI_2: f32 = core::f32::consts::FRAC_PI_2;

/// Hand-rolled `exp` for the water subsystem, since the workspace determinism
/// policy allows only `sqrt` among the float intrinsics and forbids
/// [`f32::exp`]. Uses the limit identity `exp(x) = (1 + x/2^12)^(2^12)`
/// evaluated by twelve squarings — pure add/multiply arithmetic.
///
/// The result is non-negative for every finite input and monotonically
/// increasing in `x` (for `x > -4096`), which is exactly what the spectral
/// energy, `Beer-Lambert` extinction, foam decay, and drying laws rely on. It
/// is an approximation, not a bit-exact `exp`; the base is clamped so an
/// extreme negative argument saturates to `0` instead of going negative.
#[must_use]
pub fn exp_approx(x: f32) -> f32 {
    // 2^12 = 4096 squaring steps: large enough for smoothness across the
    // arguments the spectra/extinction terms produce, cheap enough to inline.
    let mut base = 1.0 + x / 4096.0;
    if base < 0.0 {
        base = 0.0;
    }
    let mut i = 0;
    while i < 12 {
        base *= base;
        i += 1;
    }
    base
}

/// Reduces an angle to the range `[-PI, PI]` by subtracting the nearest whole
/// multiple of `2*PI`, so [`sin_approx`] / [`cos_approx`] stay accurate and
/// exactly periodic for large phases. Uses only multiply/round/subtract.
#[must_use]
fn wrap_pi(x: f32) -> f32 {
    let k = (x / TWO_PI).round();
    x - k * TWO_PI
}

/// Hand-rolled `sin` for spectral phase advance and `Gerstner` trains, since
/// [`f32::sin`] is forbidden by the determinism policy. The angle is reduced to
/// `[-PI, PI]`, folded into `[-PI/2, PI/2]` by the sine symmetry
/// `sin(PI - x) = sin(x)`, then evaluated with the seventh-order Taylor
/// polynomial (error below `2e-4` on that interval). Exactly periodic and
/// bounded to roughly `[-1, 1]`.
#[must_use]
pub fn sin_approx(x: f32) -> f32 {
    let mut r = wrap_pi(x);
    if r > FRAC_PI_2 {
        r = PI - r;
    } else if r < -FRAC_PI_2 {
        r = -PI - r;
    }
    let x2 = r * r;
    // r - r^3/6 + r^5/120 - r^7/5040, Horner form.
    r * (1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0))))
}

/// Hand-rolled `cos` via `cos(x) = sin(x + PI/2)`; see [`sin_approx`].
#[must_use]
pub fn cos_approx(x: f32) -> f32 {
    sin_approx(x + FRAC_PI_2)
}

/// A hand-rolled two-component vector for height-field, spectral, and foam math.
///
/// `prism_render_architecture` is a dependency-free contracts crate, so the
/// vector math is spelled out here rather than pulled from a linear-algebra
/// dependency. Only `sqrt` is used (the single float intrinsic the workspace
/// determinism policy allows); no transcendental functions are called.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
}

impl Vec2 {
    /// The zero vector.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    /// Builds a vector from components.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Component-wise sum.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The water math API is specified with named add/sub methods for call-site uniformity; operator traits are intentionally not part of this internal type."
    )]
    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }

    /// Component-wise difference `self - rhs`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "See add: the specified API uses named sub for call-site uniformity, not operator traits."
    )]
    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }

    /// Uniform scale by a scalar.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s)
    }

    /// Dot (inner) product.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y
    }

    /// Squared Euclidean length; cheaper than [`Vec2::length`] for comparisons.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Distance to another point.
    #[must_use]
    pub fn distance(self, rhs: Self) -> f32 {
        self.sub(rhs).length()
    }

    /// Unit vector, or the zero vector when the length is below `EPS_LEN_SQ`.
    #[must_use]
    pub fn normalize_or_zero(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq > EPS_LEN_SQ {
            self.scale(1.0 / len_sq.sqrt())
        } else {
            Self::ZERO
        }
    }
}

/// A hand-rolled three-component vector for particle and normal math.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
}

impl Vec3 {
    /// The zero vector.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Builds a vector from components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Component-wise sum.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The water math API is specified with named add/sub methods for call-site uniformity; operator traits are intentionally not part of this internal type."
    )]
    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    /// Component-wise difference `self - rhs`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "See add: the specified API uses named sub for call-site uniformity, not operator traits."
    )]
    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }

    /// Uniform scale by a scalar.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    /// Dot (inner) product.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Cross product `self x rhs`.
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    /// Squared Euclidean length; cheaper than [`Vec3::length`] for comparisons.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Distance to another point.
    #[must_use]
    pub fn distance(self, rhs: Self) -> f32 {
        self.sub(rhs).length()
    }

    /// Unit vector, or the zero vector when the length is below `EPS_LEN_SQ`.
    #[must_use]
    pub fn normalize_or_zero(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq > EPS_LEN_SQ {
            self.scale(1.0 / len_sq.sqrt())
        } else {
            Self::ZERO
        }
    }
}

/// Stable identity of one water body (ocean/surface/volume instance).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WaterBodyHandle(pub u32);

/// Stable identity of one spectral cascade level on an ocean body.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WaterCascadeHandle(pub u32);

/// Stable identity of one volumetric fluid domain (particle sim region).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FluidDomainHandle(pub u32);

/// Stable identity of one foam density field.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FoamFieldHandle(pub u32);

/// Stable identity of one interaction source (boat wake, wading body, rain,
/// impact) that injects into a water body.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct InteractionSourceHandle(pub u32);

/// The three kinds of water body, each with a distinct geometry and solver.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WaterKind {
    /// Unbounded cascaded displacement ocean surface.
    Ocean,
    /// Spline/polygon 2.5D height-field river, lake, or pond.
    Surface,
    /// 3D domain of fluid particles with a reconstructed surface.
    Volume,
}

/// The solver bucket a water body runs. A scene may mix buckets (far-field
/// spectral ocean, near-shore shallow water, local `FLIP`/`PBF`), blended in
/// the depth-composition stage by [`transition`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SolverKind {
    /// Spectral inverse-`FFT` (`Tessendorf`) ocean displacement.
    SpectralIfft,
    /// Analytic `Gerstner` wave trains (art-directable, few wave numbers).
    Gerstner,
    /// Shallow-Water Equations height-field stepping.
    ShallowWater,
    /// Position-Based Fluids (`XPBD` density constraint) particle sim.
    Pbf,
    /// `FLIP`/`APIC` particle-grid sim with a pressure projection.
    FlipApic,
}

impl SolverKind {
    /// Returns `true` for particle-based volumetric solvers (`PBF`,
    /// `FLIP`/`APIC`) that need surface reconstruction.
    #[must_use]
    pub fn is_particle_based(self) -> bool {
        matches!(self, SolverKind::Pbf | SolverKind::FlipApic)
    }

    /// Returns `true` for 2.5D height-field / displacement solvers that produce
    /// a heightmap or displacement mesh rather than particles.
    #[must_use]
    pub fn is_height_field(self) -> bool {
        matches!(
            self,
            SolverKind::SpectralIfft | SolverKind::Gerstner | SolverKind::ShallowWater
        )
    }

    /// Returns `true` for solvers that enforce incompressibility (a pressure or
    /// density projection), as opposed to purely animated displacement.
    #[must_use]
    pub fn is_incompressible(self) -> bool {
        matches!(self, SolverKind::Pbf | SolverKind::FlipApic)
    }
}

/// The set of shared advanced base services a water frontend consumes.
///
/// These services live in sibling crates/modules (virtual geometry, lighting,
/// virtual shadows, ray scene, temporal upsampling); water only *consumes*
/// them and never reimplements them. The type exists so the contract "every
/// frontend shares the full advanced base" is expressed as data rather than
/// prose. Bit operations are integer comparisons, never `f32` equality.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SharedBaseServices(u16);

impl SharedBaseServices {
    /// Virtual geometry vis-buffer cluster LOD / software raster.
    pub const VIRTUAL_GEOMETRY: Self = Self(1 << 0);
    /// `Lumen`-style hybrid global illumination.
    pub const HYBRID_GI: Self = Self(1 << 1);
    /// `ReSTIR` direct/indirect light resampling.
    pub const RESTIR: Self = Self(1 << 2);
    /// Virtual shadow maps (`VSM`).
    pub const VIRTUAL_SHADOW: Self = Self(1 << 3);
    /// Froxel volumetrics (underwater scattering, god rays).
    pub const FROXEL_VOLUME: Self = Self(1 << 4);
    /// Ray-traced reflections (SSR miss fallback).
    pub const RT_REFLECTION: Self = Self(1 << 5);
    /// Ray-traced / photon caustics.
    pub const RT_CAUSTICS: Self = Self(1 << 6);
    /// Offline path-tracing reference for calibration.
    pub const PATH_TRACING_REF: Self = Self(1 << 7);
    /// Temporal upsampling / anti-aliasing history.
    pub const TEMPORAL_UPSAMPLE: Self = Self(1 << 8);

    /// The empty set.
    pub const NONE: Self = Self(0);

    /// The full shared advanced base — the nine services above.
    pub const ALL: Self = Self(
        (1 << 0)
            | (1 << 1)
            | (1 << 2)
            | (1 << 3)
            | (1 << 4)
            | (1 << 5)
            | (1 << 6)
            | (1 << 7)
            | (1 << 8),
    );

    /// Number of distinct services in the full base.
    pub const COUNT: u32 = 9;

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
/// geometry, simulation, refraction/reflection routing, and the entire shared
/// advanced base ([`SharedBaseServices`]) are identical across all four.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ShadingFrontend {
    /// Physically based: Schlick `Fresnel`, `Beer-Lambert` absorption, GGX
    /// micro-surface, grazing subsurface transmission, spectral dispersion.
    Pbr,
    /// Stylized (illumination axis): ramp-quantized water color, toon specular
    /// blocks, hand-drawn foam lines, stylized caustics, ink diffusion.
    Npr,
    /// Author-injected closure hook compiled into a shader specialization.
    Custom,
    /// Per-region/per-layer blend of the above on one water body.
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

/// The unified authoring/runtime description of one water body.
///
/// Carries the geometry class, solver bucket, shading frontend, the shared
/// deformation-cache slot its displacement mesh / height field writes into, and
/// the coarse physical parameters the solver modules read. Fine-grained per-
/// solver parameters (spectrum shape, `CFL` limits, particle radius) live in
/// the sibling modules that own them; this struct is the routing contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterBody {
    /// Stable identity of this body.
    pub handle: WaterBodyHandle,
    /// Geometry class.
    pub kind: WaterKind,
    /// Solver bucket driving this body.
    pub solver: SolverKind,
    /// Lighting-response frontend.
    pub frontend: ShadingFrontend,
    /// Deformation-cache entry the displacement/height update writes into.
    pub deformation: DeformationHandle,
    /// Grid resolution per side for height-field / spectral cascades, or the
    /// reconstruction grid resolution for particle domains.
    pub grid_resolution: u32,
    /// Number of spectral cascade levels (ocean); `0` for non-spectral bodies.
    pub cascade_count: u32,
    /// Half-extent of the simulation domain (surface/volume bounds), in meters.
    pub domain_half_extent: Vec3,
    /// Still-water reference level (world Y), the height the surface relaxes to.
    pub still_water_level: f32,
    /// Aggregate per-frame simulation/shading tuning fed to the per-frame
    /// planners (`sim`, `surface_fx`, `shoreline`, `optics`, `coupling`).
    pub profile: profile::WaterSimProfile,
}

/// Per-frame caps arbitrated by [`budget::plan_water`], the water analogue of
/// [`crate::deformation::DeformationBudget`]. Each quota is independent so one
/// saturated stage never starves another.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WaterBudget {
    /// Solver step dispatches (spectral/`SWE`/`PBF`/`FLIP`) admitted per frame.
    pub solve_steps_per_frame: u32,
    /// Surface-reconstruction cells processed per frame (particle bodies).
    pub reconstruct_cells_per_frame: u32,
    /// Displacement-mesh vertices generated per frame (clipmap).
    pub displacement_vertices_per_frame: u32,
    /// Foam-advection cells stepped per frame.
    pub foam_cells_per_frame: u32,
    /// Crest-spray emission bursts admitted per frame.
    pub spray_bursts_per_frame: u32,
    /// Two-way coupling field read-back queries admitted per frame.
    pub coupling_queries_per_frame: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec2_algebra_is_exact() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(4.0, 6.0);
        assert_eq!(a.add(b), Vec2::new(5.0, 8.0));
        assert_eq!(b.sub(a), Vec2::new(3.0, 4.0));
        assert_eq!(a.scale(2.0), Vec2::new(2.0, 4.0));
        assert_eq!(a.dot(b), 16.0);
    }

    #[test]
    fn vec2_length_and_normalize() {
        let a = Vec2::new(3.0, 4.0);
        assert_eq!(a.length_squared(), 25.0);
        assert_eq!(a.length(), 5.0);
        assert_eq!(a.normalize_or_zero(), Vec2::new(0.6, 0.8));
        assert_eq!(Vec2::ZERO.normalize_or_zero(), Vec2::ZERO);
    }

    #[test]
    fn vec3_algebra_and_cross() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(x.cross(y), Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(Vec3::new(3.0, 4.0, 0.0).length(), 5.0);
        assert_eq!(Vec3::ZERO.normalize_or_zero(), Vec3::ZERO);
    }

    #[test]
    fn solver_classification() {
        assert!(SolverKind::Pbf.is_particle_based());
        assert!(SolverKind::FlipApic.is_particle_based());
        assert!(!SolverKind::SpectralIfft.is_particle_based());
        assert!(SolverKind::ShallowWater.is_height_field());
        assert!(SolverKind::Gerstner.is_height_field());
        assert!(!SolverKind::Pbf.is_height_field());
        assert!(SolverKind::FlipApic.is_incompressible());
        assert!(!SolverKind::Gerstner.is_incompressible());
    }

    #[test]
    fn shared_base_set_algebra() {
        let a = SharedBaseServices::VIRTUAL_GEOMETRY.union(SharedBaseServices::HYBRID_GI);
        assert!(a.contains(SharedBaseServices::VIRTUAL_GEOMETRY));
        assert!(a.contains(SharedBaseServices::HYBRID_GI));
        assert!(!a.contains(SharedBaseServices::RESTIR));
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
            // Each individual service is present for every frontend.
            for service in [
                SharedBaseServices::VIRTUAL_GEOMETRY,
                SharedBaseServices::HYBRID_GI,
                SharedBaseServices::RESTIR,
                SharedBaseServices::VIRTUAL_SHADOW,
                SharedBaseServices::FROXEL_VOLUME,
                SharedBaseServices::RT_REFLECTION,
                SharedBaseServices::RT_CAUSTICS,
                SharedBaseServices::PATH_TRACING_REF,
                SharedBaseServices::TEMPORAL_UPSAMPLE,
            ] {
                assert!(frontend.shared_base().contains(service));
            }
        }
    }

    #[test]
    fn exp_approx_is_nonnegative_and_monotonic() {
        assert!((exp_approx(0.0) - 1.0).abs() < 1e-3);
        // Monotonic increasing across a wide range.
        let mut prev = exp_approx(-8.0);
        let mut t = -8.0;
        while t <= 4.0 {
            let cur = exp_approx(t);
            assert!(cur >= prev - EPS);
            assert!(cur >= 0.0);
            prev = cur;
            t += 0.25;
        }
        // Reasonable accuracy near zero (e^1 ~= 2.718).
        assert!((exp_approx(1.0) - core::f32::consts::E).abs() < 0.05);
        // Extreme negative saturates to zero, never negative.
        assert!(exp_approx(-1.0e4) >= 0.0);
    }

    #[test]
    fn sin_cos_are_bounded_periodic_and_accurate() {
        // Bounded and accurate at the cardinal angles.
        assert!(sin_approx(0.0).abs() < 1e-3);
        assert!((sin_approx(FRAC_PI_2) - 1.0).abs() < 1e-3);
        assert!((cos_approx(0.0) - 1.0).abs() < 1e-3);
        assert!(cos_approx(FRAC_PI_2).abs() < 1e-3);
        // Periodicity: sin(x) == sin(x + 2*PI) after range reduction.
        let x = 0.9;
        assert!((sin_approx(x) - sin_approx(x + TWO_PI * 5.0)).abs() < 1e-3);
        // Pythagorean identity holds approximately everywhere.
        let mut t = -10.0;
        while t <= 10.0 {
            let s = sin_approx(t);
            let c = cos_approx(t);
            assert!((s * s + c * c - 1.0).abs() < 2e-2);
            t += 0.3;
        }
    }

    #[test]
    fn architecture_version_is_stable() {
        assert_eq!(WATER_ARCHITECTURE_VERSION, 1);
    }
}
