//! The stateless Vertex Block Descent solver.
//!
//! [`VbdSolver`] advances a set of particles coupled by a [`SpringSet`] one
//! frame at a time. Each frame is split into equal substeps; within a substep
//! the implicit-Euler inertial target is formed once and then a fixed number of
//! Gauss-Seidel *vertex-block* sweeps refine the positions. Every sweep visits
//! each dynamic vertex, assembles its local `3x3` [`VertexSystem`] from the
//! inertial term and its incident springs, solves `H dx = f`, and applies the
//! update immediately (Gauss-Seidel, so later vertices in the same sweep see
//! the new positions). Velocities are recovered from the net substep motion.
//!
//! The solver holds no mutable state, so a single instance can advance many
//! bodies. The vertex→incident-spring adjacency is rebuilt at the start of each
//! [`step`](VbdSolver::step) from the supplied [`SpringSet`], which keeps the
//! solver robust to callers that mutate topology between frames.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! substepped inertial target, per-vertex Gauss-Seidel block descent, and
//! velocity recovery are the formulation published by Chen et al., "Vertex
//! Block Descent" (2024).

use glam::Vec3;

use crate::math::scalar::Real;
use crate::soft::particle::ParticleStorage;

use super::config::VbdConfig;
use super::element::SpringSet;
use super::coloring::VbdColoring;
use super::system::VertexSystem;

/// A stateless Vertex Block Descent solver for springs, cloth, and rope.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct VbdSolver;

impl VbdSolver {
    /// Creates a solver. The solver holds no state; it is a zero-sized handle to
    /// the stepping logic.
    #[must_use]
    pub const fn new() -> VbdSolver {
        VbdSolver
    }

    /// Returns a short, stable name identifying the solver.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        "vertex-block-descent"
    }

    /// Advances `particles` coupled by `springs` by `dt` seconds using the
    /// Vertex Block Descent scheme configured by `config`.
    ///
    /// Does nothing when there are no particles or when `dt` is non-positive.
    /// The substep and iteration counts are clamped to at least `1`.
    pub fn step(
        &self,
        particles: &mut ParticleStorage,
        springs: &SpringSet,
        config: &VbdConfig,
        dt: Real,
    ) {
        if particles.is_empty() || dt <= 0.0 {
            return;
        }
        let substeps = config.substeps.max(1);
        let h = dt / substeps as Real;
        if h <= 0.0 {
            return;
        }

        let adjacency = Adjacency::build(particles.len(), springs);
        // Inertial targets, reused across sweeps within a substep.
        let mut targets = vec![Vec3::ZERO; particles.len()];

        for _ in 0..substeps {
            self.substep(particles, springs, &adjacency, config, h, &mut targets);
        }
    }

    /// Advances `particles` exactly like [`step`](Self::step) but sweeps
    /// vertices in the colour-major order described by `coloring` instead of
    /// natural index order.
    ///
    /// This is the parallel-friendly twin of [`step`](Self::step): the
    /// `coloring` (from [`color_springs`](super::coloring::color_springs))
    /// partitions the vertices so that within one colour no two vertices share a
    /// spring, so relaxing a colour's vertices in any order yields the same
    /// per-vertex result while colours are applied one after another
    /// (Gauss-Seidel across colours). That is exactly the schedule a GPU
    /// dispatch runs (one dispatch per colour), so this method is the
    /// bit-for-bit CPU golden for the GPU VBD kernels.
    ///
    /// Everything else — inertial prediction, the per-vertex `3x3` solve,
    /// velocity recovery, and the empty / non-positive-`dt` guards — is
    /// identical to [`step`](Self::step); only the sweep order differs. The
    /// `coloring` should have been built from the same `springs` and vertex
    /// count; a vertex missing from the colour order is simply not swept.
    pub fn step_colored(
        &self,
        particles: &mut ParticleStorage,
        springs: &SpringSet,
        config: &VbdConfig,
        coloring: &VbdColoring,
        dt: Real,
    ) {
        if particles.is_empty() || dt <= 0.0 {
            return;
        }
        let substeps = config.substeps.max(1);
        let h = dt / substeps as Real;
        if h <= 0.0 {
            return;
        }

        let adjacency = Adjacency::build(particles.len(), springs);
        let mut targets = vec![Vec3::ZERO; particles.len()];

        for _ in 0..substeps {
            self.substep_ordered(
                particles,
                springs,
                &adjacency,
                Some(coloring),
                config,
                h,
                &mut targets,
            );
        }
    }

    /// Runs one substep: predict the inertial target, sweep the vertices, then
    /// recover velocities from the net motion over the substep.
    fn substep(
        &self,
        particles: &mut ParticleStorage,
        springs: &SpringSet,
        adjacency: &Adjacency,
        config: &VbdConfig,
        h: Real,
        targets: &mut [Vec3],
    ) {
        self.substep_ordered(particles, springs, adjacency, None, config, h, targets);
    }

    /// Shared substep used by both the natural-order [`step`](Self::step) and
    /// the colour-ordered [`step_colored`](Self::step_colored). When `coloring`
    /// is `Some` the Gauss-Seidel sweeps visit vertices colour-major; otherwise
    /// they visit vertices in natural index order. Prediction and velocity
    /// recovery are identical in both cases.
    #[expect(
        clippy::too_many_arguments,
        reason = "a substep genuinely needs particles, springs, adjacency, optional coloring, config, step, and target scratch"
    )]
    fn substep_ordered(
        &self,
        particles: &mut ParticleStorage,
        springs: &SpringSet,
        adjacency: &Adjacency,
        coloring: Option<&VbdColoring>,
        config: &VbdConfig,
        h: Real,
        targets: &mut [Vec3],
    ) {
        {
            let columns = particles.columns_mut();
            // Snapshot the start-of-substep positions into prev, form the
            // inertial target, and warm-start the iterate at the target.
            #[expect(
                clippy::needless_range_loop,
                reason = "index addresses several parallel particle columns at once"
            )]
            for i in 0..columns.positions.len() {
                columns.prev_positions[i] = columns.positions[i];
                if columns.inverse_masses[i] == 0.0 {
                    targets[i] = columns.positions[i];
                    continue;
                }
                let y = columns.positions[i] + columns.velocities[i] * h + config.gravity * (h * h);
                targets[i] = y;
                columns.positions[i] = y;
            }
        }

        for _ in 0..config.iterations.max(1) {
            match coloring {
                Some(coloring) => {
                    self.sweep_colored(particles, springs, adjacency, coloring, h, targets);
                }
                None => self.sweep(particles, springs, adjacency, h, targets),
            }
        }

        let columns = particles.columns_mut();
        let velocity_scale = (1.0 - config.damping * h).max(0.0);
        let inv_h = 1.0 / h;
        for i in 0..columns.positions.len() {
            if columns.inverse_masses[i] == 0.0 {
                columns.velocities[i] = Vec3::ZERO;
                continue;
            }
            let v = (columns.positions[i] - columns.prev_positions[i]) * inv_h;
            columns.velocities[i] = v * velocity_scale;
        }
    }

    /// One Gauss-Seidel sweep over every dynamic vertex, in natural (ascending
    /// index) order.
    fn sweep(
        &self,
        particles: &mut ParticleStorage,
        springs: &SpringSet,
        adjacency: &Adjacency,
        h: Real,
        targets: &[Vec3],
    ) {
        for i in 0..particles.len() {
            self.relax_vertex(i, particles, springs, adjacency, h, targets);
        }
    }

    /// One sweep over every dynamic vertex in colour-major order: Gauss-Seidel
    /// across colours, Jacobi within a colour.
    ///
    /// Because the [`VbdColoring`] guarantees no two vertices in a colour share
    /// a spring, relaxing a colour's vertices in any order yields the same
    /// per-vertex result as relaxing them one at a time. Applying colours in
    /// turn therefore reproduces a valid Gauss-Seidel sweep. This is the exact
    /// schedule the GPU twin dispatches (one dispatch per colour), so the two
    /// agree bit-for-bit.
    fn sweep_colored(
        &self,
        particles: &mut ParticleStorage,
        springs: &SpringSet,
        adjacency: &Adjacency,
        coloring: &VbdColoring,
        h: Real,
        targets: &[Vec3],
    ) {
        let count = particles.len();
        for &v in coloring.order() {
            let i = v as usize;
            if i < count {
                self.relax_vertex(i, particles, springs, adjacency, h, targets);
            }
        }
    }

    /// Takes one exact per-vertex block-descent step for vertex `i`: assembles
    /// the local `3x3` [`VertexSystem`] from the inertial term and the vertex's
    /// incident springs, solves `H dx = f`, and applies `dx` in place.
    ///
    /// Pinned (zero-inverse-mass) vertices are left untouched. The accumulation
    /// order over a vertex's incident springs follows `adjacency.incident(i)`
    /// (ascending spring index), which the GPU CSR upload preserves, so the
    /// summed force and Hessian match across CPU and GPU.
    fn relax_vertex(
        &self,
        i: usize,
        particles: &mut ParticleStorage,
        springs: &SpringSet,
        adjacency: &Adjacency,
        h: Real,
        targets: &[Vec3],
    ) {
        let inverse_mass = particles.inverse_masses()[i];
        if inverse_mass == 0.0 {
            return;
        }
        let handle = adjacency.handle(i);
        let positions = particles.positions();
        let mut system = VertexSystem::new();
        system.add_inertia(1.0 / inverse_mass, h, positions[i], targets[i]);
        for &spring_index in adjacency.incident(i) {
            let contribution = springs.springs[spring_index].contribution(handle, positions);
            system.add_spring(contribution);
        }
        let dx = system.solve();
        particles.positions_mut()[i] += dx;
    }
}

/// Vertex → incident-spring adjacency, built once per [`VbdSolver::step`].
///
/// `incident[i]` lists the indices (into [`SpringSet::springs`]) of every
/// spring touching vertex `i`, so a sweep can gather a vertex's elastic
/// contributions without scanning the whole spring list.
struct Adjacency {
    /// Flattened per-vertex incident-spring index lists.
    entries: Vec<usize>,
    /// `offsets[i]..offsets[i + 1]` is vertex `i`'s slice of `entries`.
    offsets: Vec<usize>,
}

impl Adjacency {
    /// Builds the adjacency for `vertex_count` vertices from `springs`.
    fn build(vertex_count: usize, springs: &SpringSet) -> Adjacency {
        let mut counts = vec![0usize; vertex_count];
        for spring in &springs.springs {
            counts[spring.a.index()] += 1;
            counts[spring.b.index()] += 1;
        }
        let mut offsets = vec![0usize; vertex_count + 1];
        for i in 0..vertex_count {
            offsets[i + 1] = offsets[i] + counts[i];
        }
        let mut cursor = offsets.clone();
        let mut entries = vec![0usize; offsets[vertex_count]];
        for (spring_index, spring) in springs.springs.iter().enumerate() {
            let a = spring.a.index();
            let b = spring.b.index();
            entries[cursor[a]] = spring_index;
            cursor[a] += 1;
            entries[cursor[b]] = spring_index;
            cursor[b] += 1;
        }
        Adjacency { entries, offsets }
    }

    /// Returns the incident-spring indices for vertex `i`.
    fn incident(&self, i: usize) -> &[usize] {
        &self.entries[self.offsets[i]..self.offsets[i + 1]]
    }

    /// Returns the [`ParticleHandle`](crate::soft::particle::ParticleHandle) for
    /// vertex index `i`.
    fn handle(&self, i: usize) -> crate::soft::particle::ParticleHandle {
        // The adjacency indices are dense particle indices, matching handles.
        crate::soft::particle::ParticleHandle::from_index(i as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soft::particle::ParticleHandle;
    use crate::vbd::element::SpringElement;

    fn handle(i: u32) -> ParticleHandle {
        ParticleHandle::from_index(i)
    }

    #[test]
    fn empty_particles_is_a_no_op() {
        let solver = VbdSolver::new();
        let mut particles = ParticleStorage::new();
        let springs = SpringSet::new();
        solver.step(&mut particles, &springs, &VbdConfig::default(), 1.0 / 60.0);
        assert!(particles.is_empty());
        assert_eq!(solver.name(), "vertex-block-descent");
    }

    #[test]
    fn non_positive_dt_is_a_no_op() {
        let solver = VbdSolver::new();
        let mut particles = ParticleStorage::new();
        let p = particles.spawn(Vec3::ZERO, 1.0);
        let springs = SpringSet::new();
        solver.step(&mut particles, &springs, &VbdConfig::default(), 0.0);
        assert_eq!(particles.position(p), Some(Vec3::ZERO));
    }

    #[test]
    fn free_particle_matches_inertial_fall() {
        // With no springs a dynamic vertex should land exactly on the inertial
        // target y = x + h v + h^2 g each substep.
        let solver = VbdSolver::new();
        let mut particles = ParticleStorage::new();
        let p = particles.spawn(Vec3::ZERO, 1.0);
        let springs = SpringSet::new();
        let config = VbdConfig {
            damping: 0.0,
            substeps: 1,
            iterations: 4,
            ..VbdConfig::default()
        };
        let dt = 1.0 / 60.0;
        solver.step(&mut particles, &springs, &config, dt);
        let expected_y = config.gravity.y * dt * dt;
        let pos = particles.position(p).unwrap();
        assert!((pos.y - expected_y).abs() < 1e-4, "y = {}", pos.y);
        assert!(pos.x.abs() < 1e-6 && pos.z.abs() < 1e-6);
    }

    #[test]
    fn pinned_particle_never_moves() {
        let solver = VbdSolver::new();
        let mut particles = ParticleStorage::new();
        let top = particles.spawn_pinned(Vec3::ZERO);
        let _bottom = particles.spawn(Vec3::new(0.0, -1.0, 0.0), 1.0);
        let mut springs = SpringSet::new();
        springs.push(SpringElement::new(top, handle(1), 1.0, 1000.0));
        for _ in 0..120 {
            solver.step(&mut particles, &springs, &VbdConfig::default(), 1.0 / 60.0);
        }
        assert_eq!(particles.position(top), Some(Vec3::ZERO));
    }

    #[test]
    fn stiff_rope_hangs_near_rest_length_and_is_stable() {
        // A pinned-top / dynamic-bottom rope with a very stiff spring should
        // settle close to the rest length without exploding — the hallmark of
        // VBD's unconditional stability.
        let solver = VbdSolver::new();
        let mut particles = ParticleStorage::new();
        let top = particles.spawn_pinned(Vec3::ZERO);
        let bottom = particles.spawn(Vec3::new(0.0, -1.0, 0.0), 1.0);
        let mut springs = SpringSet::new();
        springs.push(SpringElement::new(top, bottom, 1.0, 1.0e6));
        let config = VbdConfig::default();
        for _ in 0..300 {
            solver.step(&mut particles, &springs, &config, 1.0 / 60.0);
        }
        let length =
            (particles.position(top).unwrap() - particles.position(bottom).unwrap()).length();
        assert!(length.is_finite());
        assert!((length - 1.0).abs() < 0.05, "rope settled at {length}");
    }

    #[test]
    fn stepping_is_deterministic() {
        let run = || {
            let solver = VbdSolver::new();
            let mut particles = ParticleStorage::new();
            let top = particles.spawn_pinned(Vec3::ZERO);
            let bottom = particles.spawn(Vec3::new(0.0, -1.0, 0.0), 1.0);
            let mut springs = SpringSet::new();
            springs.push(SpringElement::new(top, bottom, 1.0, 5000.0));
            let config = VbdConfig::default();
            for _ in 0..60 {
                solver.step(&mut particles, &springs, &config, 1.0 / 60.0);
            }
            particles.position(bottom).unwrap()
        };
        assert_eq!(run(), run());
    }

    /// Builds a `w x h` grid of particles (top row pinned) linked by structural
    /// (horizontal + vertical) and shear (diagonal) springs.
    fn grid_body(w: u32, h: u32) -> (ParticleStorage, SpringSet) {
        use crate::vbd::element::SpringElement;
        let mut particles = ParticleStorage::new();
        let idx = |r: u32, c: u32| r * w + c;
        for r in 0..h {
            for c in 0..w {
                let pos = Vec3::new(c as f32 * 0.1, -(r as f32) * 0.1, 0.0);
                if r == 0 {
                    particles.spawn_pinned(pos);
                } else {
                    particles.spawn(pos, 1.0);
                }
            }
        }
        let mut springs = SpringSet::new();
        let mut link = |a: u32, b: u32| {
            let pa = particles.positions()[a as usize];
            let pb = particles.positions()[b as usize];
            springs.push(SpringElement::new(
                handle(a),
                handle(b),
                (pa - pb).length(),
                2000.0,
            ));
        };
        for r in 0..h {
            for c in 0..w {
                if c + 1 < w {
                    link(idx(r, c), idx(r, c + 1));
                }
                if r + 1 < h {
                    link(idx(r, c), idx(r + 1, c));
                }
                if c + 1 < w && r + 1 < h {
                    link(idx(r, c), idx(r + 1, c + 1));
                    link(idx(r, c + 1), idx(r + 1, c));
                }
            }
        }
        (particles, springs)
    }

    #[test]
    fn colored_sweep_holds_pinned_top_row() {
        use crate::vbd::coloring::color_springs;
        let solver = VbdSolver::new();
        let (mut particles, springs) = grid_body(4, 4);
        let coloring = color_springs(&springs, particles.len());
        let config = VbdConfig::default();
        for _ in 0..30 {
            solver.step_colored(&mut particles, &springs, &config, &coloring, 1.0 / 60.0);
        }
        for c in 0..4 {
            let p = particles.position(handle(c)).unwrap();
            assert!(p.y.abs() < 1e-6, "pinned top row drifted: {p:?}");
            assert!(p.is_finite());
        }
    }

    #[test]
    fn colored_sweep_matches_natural_order_closely() {
        // A proper colouring reproduces a valid Gauss-Seidel sweep, so the
        // colour-ordered solve stays physically equivalent to the natural-order
        // solve: the two settle to the same steady state. (Per-step iterates
        // differ because the vertex visitation order differs.)
        use crate::vbd::coloring::color_springs;
        let solver = VbdSolver::new();
        let (mut natural, springs) = grid_body(5, 5);
        let mut colored = natural.clone();
        let coloring = color_springs(&springs, natural.len());
        let config = VbdConfig::default();
        for _ in 0..300 {
            solver.step(&mut natural, &springs, &config, 1.0 / 60.0);
            solver.step_colored(&mut colored, &springs, &config, &coloring, 1.0 / 60.0);
        }
        for i in 0..natural.len() {
            let a = natural.positions()[i];
            let b = colored.positions()[i];
            assert!((a - b).length() < 1e-2, "vertex {i} diverged: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn colored_sweep_matches_natural_order_without_springs() {
        // With no springs every vertex is its own colour-0 block, so the two
        // sweep orders are identical and must agree bit-for-bit.
        use crate::vbd::coloring::color_springs;
        let solver = VbdSolver::new();
        let mut natural = ParticleStorage::new();
        for i in 0..6 {
            natural.spawn(Vec3::new(i as f32, 0.0, 0.0), 1.0);
        }
        let mut colored = natural.clone();
        let springs = SpringSet::new();
        let coloring = color_springs(&springs, natural.len());
        let config = VbdConfig::default();
        for _ in 0..20 {
            solver.step(&mut natural, &springs, &config, 1.0 / 60.0);
            solver.step_colored(&mut colored, &springs, &config, &coloring, 1.0 / 60.0);
        }
        for i in 0..natural.len() {
            assert_eq!(natural.positions()[i], colored.positions()[i]);
        }
    }
}
