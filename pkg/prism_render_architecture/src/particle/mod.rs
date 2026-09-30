//! Ember particle-engine subsystem contracts (AAA GPU-driven VFX engine).
//!
//! Particles are a first-class rendering subsystem, distinct from cloth and
//! hair: the primitive is a *pooled, GPU-resident particle* whose lifetime is
//! measured in frames, not a triangle-mesh vertex or a strand. The pipeline
//! mirrors production VFX engines (Unreal `Niagara`, Unity `VFX Graph`,
//! `PopcornFX`, `EmberGen`, `Houdini` Pyro, and `Frostbite`'s FX stack) at the
//! algorithm level, without reusing any of their code. It follows the layered
//! concept model of the engine design (design §4): a *system* owns *emitters*,
//! an emitter runs ordered *stages*, and each emitter drives one or more
//! *renderers*.
//!
//! 1. **Pooling** — every emitter owns a fixed-capacity Structure-of-Arrays
//!    particle pool with a free-list allocator, atomic counters, and an
//!    optional prefix-sum compaction (design §5.2, §11); see [`pool`].
//! 2. **Emission** — rate / burst / distribution modules turn an emitter's
//!    per-frame spawn budget into concrete spawn slots seeded from the pool's
//!    free list, inheriting emitter velocity (design §8.2); see [`emitter`].
//! 3. **Simulation** — a substepping integrator (semi-implicit Euler / Verlet /
//!    `RK2`) advances particles under a force library, ages them against
//!    over-life curves, and draws all randomness from a stateless hash RNG so
//!    the `CPU` and `GPU` paths agree bit for bit (design §8, §25, §29); see
//!    [`simulation`].
//! 4. **Simulation stages** — the generalized iteration-domain scheduler that
//!    unifies per-particle work, spatial-hash neighborhoods, grid-fluid voxel
//!    solves, and graph-colored `XPBD` constraint batches (design §7, §10);
//!    see [`stages`]. Constraint solving aligns with the shared physics kernel
//!    rather than reimplementing it.
//! 5. **Sorting & culling** — sort-key quantization, the additive/radix/bitonic
//!    strategy matrix, frustum / distance / `HZB` culling decisions, bounds
//!    reduction, and significance-based sleeping (design §12, §13); see
//!    [`sort_cull`].
//! 6. **LOD** — a screen-coverage quality ladder with a platform-profile matrix
//!    and a deterministic degradation staircase, emitting the deformation
//!    request that charges simulation against the shared budget (design §28);
//!    see [`lod`]. Like cloth and hair, this subsystem only emits requests
//!    through [`crate::deformation::schedule`]; it does not own the budget.
//! 7. **Shading** — the four-equal-citizens shading router (`Unlit` / `PBR` /
//!    `NPR` / custom / hybrid), motion-vector requirements, `OIT` routing, and
//!    volumetric six-way lighting parameters (design §16-§21); see [`shading`].
//!
//! This module owns the shared contract types the sibling modules build on: the
//! hand-rolled vector math, the stable handles, the shading-model enum, and the
//! small policy enums (simulation space, integrator, sort strategy, iteration
//! domain). The `GPU` draw/compute kernels and the `WESL` shader codegen are out
//! of scope for this `CPU`-verifiable contract layer and are documented as
//! "pending the GPU backend" where the contract signatures anticipate them.

pub mod alpha_erosion;
pub mod ao_sample;
pub mod atlas_packing;
pub mod attributes;
pub mod audio_spectrum;
pub mod authoring;
pub mod billboard_atlas;
pub mod bitonic_sort;
pub mod bloom_threshold;
pub mod boids;
pub mod bounds;
pub mod bvh;
pub mod camera;
pub mod chromatic_aberration;
pub mod collision;
pub mod color_gradient;
pub mod compression;
pub mod constraints;
pub mod contact_shadow;
pub mod curl_noise;
pub mod curves;
pub mod decal;
pub mod depth_of_field;
pub mod determinism;
pub mod draw_pass_buffers;
pub mod dual_backend;
pub mod emitter;
pub mod emitter_pass_buffers;
pub mod event_pass_buffers;
pub mod events;
pub mod feedback;
pub mod flipbook_blend;
pub mod fluid;
pub mod forces;
pub mod frame_pipeline;
pub mod fresnel_rim;
pub mod gi_probe;
pub mod gpu_dispatch;
pub mod gpu_layout;
pub mod gpu_prefix_scan;
pub mod gpu_radix_histogram;
pub mod gpu_reduce;
pub mod gpu_scan_segmented;
pub mod gpu_timer_query;
pub mod graph;
pub mod heat_distortion;
pub mod indirect_dispatch;
pub mod indirect_draw;
pub mod instancing;
pub mod light_clustered;
pub mod lod;
pub mod luminance_hist;
pub mod mesh_emission;
pub mod mesh_renderer;
pub mod modules;
pub mod motion_blur;
pub mod motion_vectors;
pub mod noise;
pub mod occlusion;
pub mod oit;
pub mod orientation_basis;
pub mod perf_budget;
pub mod pipeline_layout;
pub mod platform;
pub mod point_cache;
pub mod pool;
pub mod property_binder;
pub mod raytrace;
pub mod readback;
pub mod replay;
pub mod renderers;
pub mod ribbon_geometry;
pub mod ribbon_trail;
pub mod sdf;
pub mod serialization;
pub mod shading;
pub mod sim_pass_buffers;
pub mod sim_space;
pub mod simulation;
pub mod soft_particle;
pub mod sort_cull;
pub mod sort_pass_buffers;
pub mod spatial_hash;
pub mod spawn_pass_buffers;
pub mod spline;
pub mod sprite_stretch;
pub mod sss_wrap;
pub mod stability;
pub mod stages;
pub mod stats_overlay;
pub mod suballocator;
pub mod subframe_spawn;
pub mod temporal_dither;
pub mod time_control;
pub mod tonemap;
pub mod uv_animation;
pub mod validation;
pub mod vector_field;
pub mod vignette_mask;
pub mod volume_march;
pub mod volumetrics;
pub mod warmup;
pub mod wind_field;
pub mod worley;

#[cfg(test)]
mod integration_tests;

/// Squared-length threshold below which a vector is treated as zero, so
/// normalization and force accumulation never divide by (near) zero and never
/// propagate `NaN`. Matches the sibling cloth/hair subsystems.
pub const EPS_LEN_SQ: f32 = 1e-12;

/// A hand-rolled three-component vector.
///
/// `prism_render_architecture` is a dependency-free contracts crate, so the
/// particle math is spelled out here rather than pulled from a linear-algebra
/// dependency. Only `sqrt` is used (allowed by the workspace determinism lint);
/// no transcendental functions are called, keeping the `CPU` reference bit-
/// reproducible against a future `GPU` kernel.
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

    /// Uniform vector with every component set to `v`.
    #[must_use]
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
    }

    /// Component-wise sum.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API is specified with named add/sub methods for call-site uniformity, matching the sibling cloth module; operator traits are intentionally not part of this internal type."
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

    /// Component-wise (Hadamard) product.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "See add: the specified API uses a named component-wise mul (Hadamard product) for call-site uniformity, not the scalar-overloading Mul operator trait."
    )]
    pub fn mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }

    /// Dot (inner) product.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Cross product `self × rhs`.
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    /// Squared Euclidean length; cheaper than [`Vec3::length`] when only
    /// comparisons are needed.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Squared distance between two points.
    #[must_use]
    pub fn distance_squared(self, rhs: Self) -> f32 {
        self.sub(rhs).length_squared()
    }

    /// Euclidean distance between two points.
    #[must_use]
    pub fn distance(self, rhs: Self) -> f32 {
        self.sub(rhs).length()
    }

    /// Component-wise minimum, used to grow an axis-aligned bounds.
    #[must_use]
    pub fn min(self, rhs: Self) -> Self {
        Self::new(self.x.min(rhs.x), self.y.min(rhs.y), self.z.min(rhs.z))
    }

    /// Component-wise maximum, used to grow an axis-aligned bounds.
    #[must_use]
    pub fn max(self, rhs: Self) -> Self {
        Self::new(self.x.max(rhs.x), self.y.max(rhs.y), self.z.max(rhs.z))
    }

    /// Returns the unit vector along `self`, or [`Vec3::ZERO`] when `self` is
    /// (numerically) the zero vector, so normalization never yields `NaN`.
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

/// Identifies one particle *system*: a compiled effect asset instance (design
/// §4). A system owns one or more emitters.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ParticleSystemHandle(pub u32);

/// Identifies one *emitter*: a single class of particles within a system (for
/// example the flame core, the sparks, or the smoke of an explosion).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EmitterHandle(pub u32);

/// A non-recursive shading basis used both directly and as the two lobes of a
/// [`EmberShadingModel::Hybrid`] blend.
///
/// Splitting the basis out of [`EmberShadingModel`] keeps the hybrid variant
/// flat (no heap indirection in a `forbid(unsafe_code)`, `alloc`-light contract)
/// while still expressing "physical base plus stylized overlay" blends.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ShadingBasis {
    /// Unlit emissive/additive response (cheapest, common for energy effects).
    Unlit,
    /// Physically based response through the shared `PBR` closure (design §17).
    Pbr,
    /// Stylized response through the shared `NPR` closure (design §18).
    Npr,
    /// A user-authored `WESL` closure referenced by a stable handle (design §19).
    Custom(u32),
}

/// The shading model selected for an emitter (or an individual particle).
///
/// All four responses are equal citizens (design §16): they share the geometry,
/// culling, sorting, temporal, volumetric, and shadow services and differ only
/// in how they respond to light. `Hybrid` blends a `base` and an `overlay`
/// basis by a `weight` in `0..=1`, realizing "physical lighting plus stylized
/// rim/gradient" transitions (design §19).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EmberShadingModel {
    /// Traditional additive / self-emissive shading (design §16).
    Unlit,
    /// Physically based lighting through the shared `PBR` closure (design §17).
    Pbr,
    /// Stylized lighting through the shared `NPR` closure (design §18).
    Npr,
    /// A user `WESL` closure injected through the material extension point.
    Custom(u32),
    /// A per-particle blend of two bases weighted by `weight` in `0..=1`.
    Hybrid {
        /// The base lobe evaluated first (for example a `PBR` response).
        base: ShadingBasis,
        /// The overlay lobe blended on top (for example an `NPR` rim).
        overlay: ShadingBasis,
        /// Blend weight in `0..=1`; `0` is pure `base`, `1` is pure `overlay`.
        weight: f32,
    },
}

impl EmberShadingModel {
    /// Returns `true` when the model needs lighting data (clustered lights,
    /// shadows, `GI`) evaluated for it. `Unlit` and a pure-`Unlit` custom blend
    /// do not; every lit or hybrid model does.
    #[must_use]
    pub fn needs_lighting(self) -> bool {
        match self {
            EmberShadingModel::Unlit => false,
            EmberShadingModel::Pbr | EmberShadingModel::Npr | EmberShadingModel::Custom(_) => true,
            EmberShadingModel::Hybrid { base, overlay, .. } => {
                basis_needs_lighting(base) || basis_needs_lighting(overlay)
            }
        }
    }
}

/// Whether a single [`ShadingBasis`] lobe consumes lighting data.
#[must_use]
fn basis_needs_lighting(basis: ShadingBasis) -> bool {
    match basis {
        ShadingBasis::Unlit => false,
        ShadingBasis::Pbr | ShadingBasis::Npr | ShadingBasis::Custom(_) => true,
    }
}

/// The frame of reference an emitter simulates in (design §26).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SimSpace {
    /// Particles follow the owning transform (a hand-held torch).
    Local,
    /// Particles detach into world space once spawned (a trail left behind).
    World,
    /// Spawn in local space, then update in world space.
    Hybrid,
}

/// The time integrator advancing particle motion (design §25).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum IntegratorKind {
    /// Semi-implicit (symplectic) Euler: the stable, cheap default.
    SemiImplicitEuler,
    /// Position Verlet: velocity is implied by the position history, which
    /// pairs naturally with positional (`XPBD`) constraints.
    Verlet,
    /// Second-order Runge-Kutta (midpoint) for higher accuracy.
    Rk2,
}

/// How a renderer sorts its particles before compositing (design §12).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SortStrategy {
    /// Order-independent: additive / premultiplied / opaque needs no sort.
    None,
    /// Route through the scene's shared order-independent transparency path.
    SharedOit,
    /// Standalone one-sweep radix sort on a quantized view-depth key (many
    /// particles).
    ViewDepthRadix,
    /// Standalone bitonic sort on a view-depth key (few particles).
    ViewDepthBitonic,
}

impl SortStrategy {
    /// Returns `true` when the strategy performs an explicit depth sort (as
    /// opposed to order-independent blending or shared `OIT`).
    #[must_use]
    pub fn is_explicit_sort(self) -> bool {
        matches!(
            self,
            SortStrategy::ViewDepthRadix | SortStrategy::ViewDepthBitonic
        )
    }
}

/// The dispatch domain of a simulation stage (design §7).
///
/// Each stage declares the domain it iterates so the scheduler can pick the
/// dispatch count and insert the minimal barriers between stages.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum IterationDomain {
    /// One invocation per live particle (regular forces, integration, aging).
    PerParticle,
    /// One invocation per spatial-hash cell (neighbor build/query, flocking).
    PerNeighborCell,
    /// One invocation per 3D grid voxel (voxel fluid: advect/diffuse/project).
    PerGridVoxel,
    /// One invocation per constraint batch (`XPBD`/`VBD` iteration).
    PerConstraint,
    /// One invocation per buffered event (spawn-from-event stages).
    PerEvent,
    /// A user-specified fixed invocation count (generic compute).
    Custom(u32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec3_basic_algebra_is_exact() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.add(b), Vec3::new(5.0, 7.0, 9.0));
        assert_eq!(b.sub(a), Vec3::new(3.0, 3.0, 3.0));
        assert_eq!(a.scale(2.0), Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(a.mul(b), Vec3::new(4.0, 10.0, 18.0));
        assert_eq!(a.dot(b), 32.0);
        assert_eq!(Vec3::splat(2.0), Vec3::new(2.0, 2.0, 2.0));
    }

    #[test]
    fn vec3_cross_is_right_handed() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(x.cross(y), Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn vec3_length_and_distance() {
        let a = Vec3::new(3.0, 4.0, 0.0);
        assert_eq!(a.length_squared(), 25.0);
        assert_eq!(a.length(), 5.0);
        assert_eq!(Vec3::ZERO.distance(a), 5.0);
    }

    #[test]
    fn vec3_min_max_grow_bounds() {
        let a = Vec3::new(1.0, -2.0, 3.0);
        let b = Vec3::new(-1.0, 2.0, 0.0);
        assert_eq!(a.min(b), Vec3::new(-1.0, -2.0, 0.0));
        assert_eq!(a.max(b), Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn normalize_zero_vector_is_zero_not_nan() {
        let n = Vec3::ZERO.normalize_or_zero();
        assert_eq!(n, Vec3::ZERO);
        let unit = Vec3::new(0.0, 5.0, 0.0).normalize_or_zero();
        assert_eq!(unit, Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn shading_model_lighting_requirements() {
        assert!(!EmberShadingModel::Unlit.needs_lighting());
        assert!(EmberShadingModel::Pbr.needs_lighting());
        assert!(EmberShadingModel::Npr.needs_lighting());
        assert!(EmberShadingModel::Custom(3).needs_lighting());
        assert!(EmberShadingModel::Hybrid {
            base: ShadingBasis::Unlit,
            overlay: ShadingBasis::Npr,
            weight: 0.5,
        }
        .needs_lighting());
        assert!(!EmberShadingModel::Hybrid {
            base: ShadingBasis::Unlit,
            overlay: ShadingBasis::Unlit,
            weight: 0.25,
        }
        .needs_lighting());
    }

    #[test]
    fn sort_strategy_explicit_sort_flag() {
        assert!(!SortStrategy::None.is_explicit_sort());
        assert!(!SortStrategy::SharedOit.is_explicit_sort());
        assert!(SortStrategy::ViewDepthRadix.is_explicit_sort());
        assert!(SortStrategy::ViewDepthBitonic.is_explicit_sort());
    }
}
