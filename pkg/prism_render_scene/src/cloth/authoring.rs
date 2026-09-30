//! Ergonomic public authoring for spawnable cloth garments.
//!
//! The render-world [`ClothGarment`](super::garment::ClothGarment) component is
//! deliberately a resident, device-shaped record: its collider, backstop and
//! embed slices are stored pre-packed in their `#[repr(C)]` mirror form so the
//! per-frame prepare stage can upload them with a plain `bytemuck` cast and never
//! re-pack on the hot path. That layout is efficient for the renderer but hostile
//! to authoring by hand, so this module is the one public front door a game uses
//! to describe a garment with the readable, CPU-golden architecture types and get
//! back a component it can spawn on an entity.
//!
//! [`ClothGarmentBuilder`] accepts the same authored types the deterministic
//! solver in `prism_render_architecture::cloth` owns:
//!
//! * particles as [`ClothParticle`] (position, velocity and inverse mass, where a
//!   non-positive inverse mass pins the particle), or a whole welded
//!   [`PolygonGarmentMesh`] straight out of the polygon garment builder;
//! * mixed-kind [`Constraint`]s and dihedral [`BendingConstraint`] hinges, left
//!   unsorted because the prepare stage partitions and graph-colors them;
//! * analytic [`BodyCollider`] proxies, painted [`Backstop`] planes and
//!   render-vertex [`BarycentricBinding`] embeds, which this builder packs into
//!   their device mirrors through [`super::pack`];
//! * a [`FabricMaterial`] whose surface friction and aerodynamic drag flow into
//!   the garment's solver scalars.
//!
//! The builder never fabricates simulation state: an empty builder builds an
//! empty garment, and every optional pass (self-collision, strain limiting,
//! aerodynamics, embedding) stays off until its inputs are actually supplied, so
//! the honest no-op contract the prepare and dispatch stages rely on is preserved
//! end to end.

use prism_render_architecture::cloth::asset::FabricMaterial;
use prism_render_architecture::cloth::bending::BendingConstraint;
use prism_render_architecture::cloth::collision::{Backstop, BodyCollider};
use prism_render_architecture::cloth::embed::BarycentricBinding;
use prism_render_architecture::cloth::polygon_garment::PolygonGarmentMesh;
use prism_render_architecture::cloth::{ClothParticle, Constraint};

use super::abi::{GpuClothBackstop, GpuClothCollider, GpuClothEmbedBinding};
use super::garment::ClothGarment;
use super::pack::{pack_backstops, pack_colliders, pack_embed_bindings};

/// Default full-frame timestep: one 60 Hz frame.
const DEFAULT_DT: f32 = 1.0 / 60.0;
/// Default XPBD substeps per frame; more substeps stiffen the response.
const DEFAULT_SUBSTEPS: u32 = 8;
/// Default constraint-projection iterations per substep.
const DEFAULT_ITERATIONS: u32 = 4;
/// Default constant downward acceleration (metres per second squared).
const DEFAULT_GRAVITY: [f32; 3] = [0.0, -9.81, 0.0];

/// Fluent builder that turns authored, CPU-golden cloth data into a spawnable
/// [`ClothGarment`] component.
///
/// Start from raw particles ([`ClothGarmentBuilder::from_particles`]) or a welded
/// [`PolygonGarmentMesh`] ([`ClothGarmentBuilder::from_mesh`]), chain the optional
/// setters in any order, then call [`ClothGarmentBuilder::build`]. Collider,
/// backstop and embed slices are packed into their device mirrors as they are
/// supplied, so [`ClothGarmentBuilder::build`] is a cheap move with no further
/// conversion.
///
/// # Examples
///
/// ```ignore
/// use prism_render_scene::ClothGarmentBuilder;
/// use prism_render_architecture::cloth::{ClothParticle, Vec3};
///
/// let particles = [
///     ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
///     ClothParticle::new(Vec3::new(1.0, 0.0, 0.0), 1.0),
/// ];
/// let garment = ClothGarmentBuilder::from_particles(&particles).build();
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct ClothGarmentBuilder {
    positions: Vec<[f32; 4]>,
    velocities: Vec<[f32; 4]>,
    constraints: Vec<Constraint>,
    bending: Vec<BendingConstraint>,
    triangles: Vec<[u32; 3]>,
    wind_velocity: [f32; 3],
    wind_turbulence: f32,
    aero_drag: f32,
    aero_lift: f32,
    aero_air_density: f32,
    friction: f32,
    colliders: Vec<GpuClothCollider>,
    backstops: Vec<GpuClothBackstop>,
    embed_bindings: Vec<GpuClothEmbedBinding>,
    render_vertex_count: u32,
    hash_cell_count: u32,
    gravity: [f32; 3],
    dt: f32,
    substeps: u32,
    iterations: u32,
    damping: f32,
    strain_limit: f32,
    self_thickness: f32,
    self_cell_size: f32,
}

impl Default for ClothGarmentBuilder {
    /// An empty garment with sensible integrator defaults (60 Hz timestep, eight
    /// substeps, four iterations, earth gravity) and every optional pass off.
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            velocities: Vec::new(),
            constraints: Vec::new(),
            bending: Vec::new(),
            triangles: Vec::new(),
            wind_velocity: [0.0; 3],
            wind_turbulence: 0.0,
            aero_drag: 0.0,
            aero_lift: 0.0,
            aero_air_density: 0.0,
            friction: 0.0,
            colliders: Vec::new(),
            backstops: Vec::new(),
            embed_bindings: Vec::new(),
            render_vertex_count: 0,
            hash_cell_count: 0,
            gravity: DEFAULT_GRAVITY,
            dt: DEFAULT_DT,
            substeps: DEFAULT_SUBSTEPS,
            iterations: DEFAULT_ITERATIONS,
            damping: 0.0,
            strain_limit: 0.0,
            self_thickness: 0.0,
            self_cell_size: 0.0,
        }
    }
}

/// Splits a [`ClothParticle`] into the position (with inverse mass in `.w`) and
/// velocity rows the resident garment stores.
#[must_use]
fn split_particle(particle: &ClothParticle) -> ([f32; 4], [f32; 4]) {
    let position = [
        particle.position.x,
        particle.position.y,
        particle.position.z,
        particle.inverse_mass,
    ];
    let velocity = [
        particle.velocity.x,
        particle.velocity.y,
        particle.velocity.z,
        0.0,
    ];
    (position, velocity)
}

impl ClothGarmentBuilder {
    /// Starts a builder from a slice of authored particles.
    ///
    /// Each particle's inverse mass is packed into the position `.w` (a
    /// non-positive value pins it), matching the resident particle-pool layout.
    #[must_use]
    pub fn from_particles(particles: &[ClothParticle]) -> Self {
        let mut positions = Vec::with_capacity(particles.len());
        let mut velocities = Vec::with_capacity(particles.len());
        for particle in particles {
            let (position, velocity) = split_particle(particle);
            positions.push(position);
            velocities.push(velocity);
        }
        Self {
            positions,
            velocities,
            ..Self::default()
        }
    }

    /// Starts a builder from a welded [`PolygonGarmentMesh`], seeding the
    /// particles, the woven constraint graph and the render/collision triangle
    /// membrane in one step.
    ///
    /// Bending hinges, colliders, backstops and embeds are not implied by the
    /// mesh; add them with the matching setters when the garment needs them.
    #[must_use]
    pub fn from_mesh(mesh: &PolygonGarmentMesh) -> Self {
        Self {
            constraints: mesh.constraints.clone(),
            triangles: mesh.triangles.clone(),
            ..Self::from_particles(&mesh.particles)
        }
    }

    /// Replaces the authored constraint set (mixed distance and attachment kinds,
    /// in any order).
    #[must_use]
    pub fn constraints(mut self, constraints: Vec<Constraint>) -> Self {
        self.constraints = constraints;
        self
    }

    /// Sets the dihedral bending hinges.
    #[must_use]
    pub fn bending(mut self, bending: Vec<BendingConstraint>) -> Self {
        self.bending = bending;
        self
    }

    /// Sets the sim-mesh triangles the aerodynamic gather pass integrates wind
    /// over. An empty set leaves aerodynamics disabled.
    #[must_use]
    pub fn triangles(mut self, triangles: Vec<[u32; 3]>) -> Self {
        self.triangles = triangles;
        self
    }

    /// Packs and stores the analytic body-collision proxies.
    #[must_use]
    pub fn colliders(mut self, colliders: &[BodyCollider]) -> Self {
        self.colliders = pack_colliders(colliders);
        self
    }

    /// Packs and stores the painted backstop planes; `backstops[i]` constrains
    /// particle `i`, so the slice order is index-aligned with the particles.
    #[must_use]
    pub fn backstops(mut self, backstops: &[Backstop]) -> Self {
        self.backstops = pack_backstops(backstops);
        self
    }

    /// Packs and stores the render-vertex embed bindings and records the render
    /// vertex count that sizes the skinned output pool.
    ///
    /// `bindings[i]` drives render vertex `i`; the render vertex count defaults to
    /// the binding count but can be overridden with
    /// [`ClothGarmentBuilder::render_vertex_count`] when the render mesh carries
    /// trailing vertices with no binding.
    #[must_use]
    pub fn embed_bindings(mut self, bindings: &[BarycentricBinding]) -> Self {
        self.embed_bindings = pack_embed_bindings(bindings);
        self.render_vertex_count = bindings.len() as u32;
        self
    }

    /// Overrides the render vertex count that sizes the embed output pool.
    #[must_use]
    pub fn render_vertex_count(mut self, count: u32) -> Self {
        self.render_vertex_count = count;
        self
    }

    /// Applies a [`FabricMaterial`]: its (sanitized) surface friction grips the
    /// garment against colliders and its drag seeds the aerodynamic drag
    /// coefficient.
    #[must_use]
    pub fn material(mut self, material: FabricMaterial) -> Self {
        let material = material.sanitized();
        self.friction = material.friction;
        self.aero_drag = material.drag;
        self
    }

    /// Sets the steady world-space wind velocity and per-triangle turbulence
    /// strength; a zero turbulence disables the jitter.
    #[must_use]
    pub fn wind(mut self, velocity: [f32; 3], turbulence: f32) -> Self {
        self.wind_velocity = velocity;
        self.wind_turbulence = turbulence;
        self
    }

    /// Sets the aerodynamic drag (normal) and lift (in-plane) coefficients.
    #[must_use]
    pub fn aerodynamics(mut self, drag: f32, lift: f32) -> Self {
        self.aero_drag = drag;
        self.aero_lift = lift;
        self
    }

    /// Sets the fluid (air) density that scales the quadratic aerodynamic term.
    ///
    /// The default of `0` keeps the linear (historical) drag/lift model; a
    /// positive value opts into the UE5 `Chaos`-style quadratic (airspeed-
    /// squared) model, where the per-face force additionally scales by
    /// `0.5 * air_density * relative_wind_magnitude`.
    #[must_use]
    pub fn air_density(mut self, density: f32) -> Self {
        self.aero_air_density = density;
        self
    }

    /// Enables uniform-grid self-collision with the given separation thickness,
    /// grid cell edge and hash-table cell count. A zero cell count keeps
    /// self-collision disabled.
    #[must_use]
    pub fn self_collision(mut self, thickness: f32, cell_size: f32, hash_cell_count: u32) -> Self {
        self.self_thickness = thickness;
        self.self_cell_size = cell_size;
        self.hash_cell_count = hash_cell_count;
        self
    }

    /// Sets the constant external acceleration (gravity).
    #[must_use]
    pub fn gravity(mut self, gravity: [f32; 3]) -> Self {
        self.gravity = gravity;
        self
    }

    /// Sets the full-frame timestep in seconds.
    #[must_use]
    pub fn timestep(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }

    /// Sets the XPBD substep and per-substep iteration counts. The prepare stage
    /// clamps each to at least one.
    #[must_use]
    pub fn solver_iterations(mut self, substeps: u32, iterations: u32) -> Self {
        self.substeps = substeps;
        self.iterations = iterations;
        self
    }

    /// Sets the velocity damping in `[0, 1]`.
    #[must_use]
    pub fn damping(mut self, damping: f32) -> Self {
        self.damping = damping;
        self
    }

    /// Sets the strain limit: the maximum fractional stretch a structural edge
    /// may keep. A non-positive value disables the strain-limiter pass.
    #[must_use]
    pub fn strain_limit(mut self, strain_limit: f32) -> Self {
        self.strain_limit = strain_limit;
        self
    }

    /// Consumes the builder and produces the spawnable [`ClothGarment`] component.
    #[must_use]
    pub fn build(self) -> ClothGarment {
        ClothGarment {
            positions: self.positions,
            velocities: self.velocities,
            constraints: self.constraints,
            bending: self.bending,
            triangles: self.triangles,
            wind_velocity: self.wind_velocity,
            wind_turbulence: self.wind_turbulence,
            aero_drag: self.aero_drag,
            aero_lift: self.aero_lift,
            aero_air_density: self.aero_air_density,
            friction: self.friction,
            colliders: self.colliders,
            backstops: self.backstops,
            embed_bindings: self.embed_bindings,
            render_vertex_count: self.render_vertex_count,
            hash_cell_count: self.hash_cell_count,
            gravity: self.gravity,
            dt: self.dt,
            substeps: self.substeps,
            iterations: self.iterations,
            damping: self.damping,
            strain_limit: self.strain_limit,
            self_thickness: self.self_thickness,
            self_cell_size: self.self_cell_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::cloth::polygon_garment::PolygonGarmentMesh;
    use prism_render_architecture::cloth::{Compliance, ConstraintKind, Vec3};

    fn sample_particles() -> [ClothParticle; 3] {
        [
            ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
            ClothParticle::new(Vec3::new(1.0, 2.0, 3.0), 0.5),
            ClothParticle {
                position: Vec3::new(-1.0, -2.0, -3.0),
                velocity: Vec3::new(4.0, 5.0, 6.0),
                inverse_mass: 2.0,
            },
        ]
    }

    #[test]
    fn from_particles_packs_position_inverse_mass_and_velocity() {
        let garment = ClothGarmentBuilder::from_particles(&sample_particles()).build();

        assert_eq!(garment.positions.len(), 3);
        assert_eq!(garment.velocities.len(), 3);
        // Pinned particle: inverse mass 0 in position .w, zero velocity.
        assert_eq!(garment.positions[0], [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(garment.velocities[0], [0.0, 0.0, 0.0, 0.0]);
        // Movable particle: inverse mass in .w.
        assert_eq!(garment.positions[1], [1.0, 2.0, 3.0, 0.5]);
        // Explicit velocity is carried into the velocity row with a zero pad.
        assert_eq!(garment.positions[2], [-1.0, -2.0, -3.0, 2.0]);
        assert_eq!(garment.velocities[2], [4.0, 5.0, 6.0, 0.0]);
    }

    #[test]
    fn from_mesh_seeds_particles_constraints_and_triangles() {
        let mesh = PolygonGarmentMesh {
            particles: sample_particles().to_vec(),
            constraints: vec![Constraint::new(
                0,
                1,
                1.0,
                Compliance::RIGID,
                ConstraintKind::Stretch,
            )],
            triangles: vec![[0, 1, 2]],
            ..PolygonGarmentMesh::default()
        };

        let garment = ClothGarmentBuilder::from_mesh(&mesh).build();

        assert_eq!(garment.positions.len(), 3);
        assert_eq!(garment.constraints.len(), 1);
        assert_eq!(garment.triangles, vec![[0, 1, 2]]);
    }

    #[test]
    fn colliders_setter_matches_pack_colliders() {
        let colliders = [
            BodyCollider::Sphere {
                center: Vec3::new(1.0, 2.0, 3.0),
                radius: 0.5,
            },
            BodyCollider::HalfSpace {
                normal: Vec3::new(0.0, 1.0, 0.0),
                offset: -1.0,
            },
        ];
        let garment = ClothGarmentBuilder::from_particles(&sample_particles())
            .colliders(&colliders)
            .build();

        assert_eq!(garment.colliders, pack_colliders(&colliders));
    }

    #[test]
    fn backstops_setter_matches_pack_backstops() {
        let backstops = [Backstop {
            origin: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            distance: 0.25,
        }];
        let garment = ClothGarmentBuilder::from_particles(&sample_particles())
            .backstops(&backstops)
            .build();

        assert_eq!(garment.backstops, pack_backstops(&backstops));
    }

    #[test]
    fn embed_bindings_setter_packs_and_counts_render_vertices() {
        let bindings = [
            BarycentricBinding::new([0, 1, 2], (0.5, 0.25, 0.25), 0.01),
            BarycentricBinding::new([2, 1, 0], (0.2, 0.3, 0.5), -0.02),
        ];
        let garment = ClothGarmentBuilder::from_particles(&sample_particles())
            .embed_bindings(&bindings)
            .build();

        assert_eq!(garment.embed_bindings, pack_embed_bindings(&bindings));
        assert_eq!(garment.render_vertex_count, 2);
    }

    #[test]
    fn render_vertex_count_override_beats_the_binding_count() {
        let bindings = [BarycentricBinding::new([0, 1, 2], (1.0, 0.0, 0.0), 0.0)];
        let garment = ClothGarmentBuilder::from_particles(&sample_particles())
            .embed_bindings(&bindings)
            .render_vertex_count(7)
            .build();

        assert_eq!(garment.render_vertex_count, 7);
    }

    #[test]
    fn material_sets_sanitized_friction_and_drag() {
        // Friction above one and negative drag both clamp during sanitization.
        let material = FabricMaterial {
            friction: 2.0,
            drag: -1.0,
            ..FabricMaterial::default()
        };
        let garment = ClothGarmentBuilder::from_particles(&sample_particles())
            .material(material)
            .build();

        assert!((garment.friction - 1.0).abs() <= 1e-6);
        assert!((garment.aero_drag - 0.0).abs() <= 1e-6);
    }

    #[test]
    fn self_collision_setter_enables_the_pass() {
        let garment = ClothGarmentBuilder::from_particles(&sample_particles())
            .self_collision(0.01, 0.05, 64)
            .build();

        assert!((garment.self_thickness - 0.01).abs() <= 1e-6);
        assert!((garment.self_cell_size - 0.05).abs() <= 1e-6);
        assert_eq!(garment.hash_cell_count, 64);
    }

    #[test]
    fn wind_and_aerodynamics_setters_flow_into_the_garment() {
        let garment = ClothGarmentBuilder::from_particles(&sample_particles())
            .wind([1.0, 0.0, -2.0], 0.3)
            .aerodynamics(0.7, 0.4)
            .build();

        assert_eq!(garment.wind_velocity, [1.0, 0.0, -2.0]);
        assert!((garment.wind_turbulence - 0.3).abs() <= 1e-6);
        assert!((garment.aero_drag - 0.7).abs() <= 1e-6);
        assert!((garment.aero_lift - 0.4).abs() <= 1e-6);
        // The air density defaults to the linear model until opted in.
        assert!(garment.aero_air_density.abs() <= 1e-6);
        let quadratic = ClothGarmentBuilder::from_particles(&sample_particles())
            .aerodynamics(0.7, 0.4)
            .air_density(1.225)
            .build();
        assert!((quadratic.aero_air_density - 1.225).abs() <= 1e-6);
    }

    #[test]
    fn integrator_setters_override_the_defaults() {
        let garment = ClothGarmentBuilder::from_particles(&sample_particles())
            .gravity([0.0, -1.0, 0.0])
            .timestep(0.5)
            .solver_iterations(3, 6)
            .damping(0.2)
            .strain_limit(0.1)
            .build();

        assert_eq!(garment.gravity, [0.0, -1.0, 0.0]);
        assert!((garment.dt - 0.5).abs() <= 1e-6);
        assert_eq!(garment.substeps, 3);
        assert_eq!(garment.iterations, 6);
        assert!((garment.damping - 0.2).abs() <= 1e-6);
        assert!((garment.strain_limit - 0.1).abs() <= 1e-6);
    }

    #[test]
    fn default_builder_builds_an_empty_no_op_garment() {
        let garment = ClothGarmentBuilder::default().build();

        assert!(garment.positions.is_empty());
        assert!(garment.constraints.is_empty());
        assert!(garment.colliders.is_empty());
        assert!(garment.backstops.is_empty());
        assert!(garment.embed_bindings.is_empty());
        // Every optional pass stays off until its inputs are supplied.
        assert_eq!(garment.hash_cell_count, 0);
        assert_eq!(garment.render_vertex_count, 0);
        assert!((garment.strain_limit).abs() <= 1e-6);
        assert!(garment.triangles.is_empty());
        // Integrator defaults are the documented 60 Hz / 8 substep / 4 iteration set.
        assert!((garment.dt - DEFAULT_DT).abs() <= 1e-6);
        assert_eq!(garment.substeps, DEFAULT_SUBSTEPS);
        assert_eq!(garment.iterations, DEFAULT_ITERATIONS);
    }
}
