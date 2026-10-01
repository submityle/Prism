//! Host-side deterministic preparation for the Vertex Block Descent (VBD) GPU
//! kernel.
//!
//! The GPU kernel does the floating-point block-descent arithmetic, but the
//! topology it folds those contributions in — the per-vertex incident-spring
//! `CSR` list and the colour-major sweep order — is built here on the host so it
//! is bit-identical to the [`prism_physics_core`] golden
//! ([`VbdSolver::step_colored`](prism_physics_core::VbdSolver::step_colored)):
//! the same `Adjacency` layout (ascending-by-spring-index incidence, both
//! endpoints of every spring pushed), and the same
//! [`VbdColoring`](prism_physics_core::vbd::VbdColoring) colour-major `order`
//! and prefix `offsets`.
//!
//! Performing the topology flattening on the host keeps the only GPU
//! floating-point work the per-vertex `3x3` solve itself, which the parity
//! suite checks within a tight tolerance; the integer `CSR` and colour ordering
//! never diverge.
//!
//! # Provenance
//!
//! The VBD energy formulation and per-colour Gauss-Seidel schedule follow Chen
//! et al., "Vertex Block Descent" (SIGGRAPH 2024); greedy graph colouring for
//! parallel Gauss-Seidel is a standard, publicly documented technique. No
//! Unreal Engine source or derived code.

use alloc::vec::Vec;

use prism_physics_core::soft::particle::ParticleStorage;
use prism_physics_core::vbd::{SpringSet, VbdColoring};

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// A spring in the flat, upload-ready layout the kernel binds (16 bytes).
///
/// The field order mirrors the `GpuSpring` struct in `shaders/vbd_sweep.wgsl`
/// exactly; reordering silently corrupts the solve.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSpring {
    /// First endpoint vertex index.
    pub a: u32,
    /// Second endpoint vertex index.
    pub b: u32,
    /// Rest length (metres).
    pub rest_length: f32,
    /// Hookean stiffness (energy per squared metre of stretch).
    pub stiffness: f32,
}

/// A fully flattened, upload-ready snapshot of one VBD step.
///
/// Every array is plain-old-data laid out for direct upload to a storage
/// buffer; the four-wide packing (`[_; 4]`) matches the `vec4` strides the
/// kernel binds. See [`build`] for the construction contract.
#[derive(Clone, Debug, PartialEq)]
pub struct VbdPrep {
    /// Number of addressable particles.
    pub particle_count: u32,
    /// Number of distinct colours (serial per-iteration GPU passes).
    pub color_count: u32,
    /// Start-of-step positions, padded to `vec4` (`w` unused).
    pub positions: Vec<[f32; 4]>,
    /// Start-of-step velocities, padded to `vec4` (`w` unused).
    pub velocities: Vec<[f32; 4]>,
    /// Index-aligned inverse masses (`0` marks a pinned particle).
    pub inverse_masses: Vec<Real>,
    /// Springs in ascending set order.
    pub springs: Vec<GpuSpring>,
    /// `CSR` offsets into [`vert_entries`](Self::vert_entries), length
    /// `particle_count + 1`.
    pub vert_offsets: Vec<u32>,
    /// Flattened per-vertex incident-spring indices, ascending by spring index,
    /// matching the golden `Adjacency` reduction order.
    pub vert_entries: Vec<u32>,
    /// Colour-major vertex order; colour `c` occupies
    /// `order[color_offsets[c]..color_offsets[c + 1]]`. Length `particle_count`.
    pub order: Vec<u32>,
    /// Prefix offsets into [`order`](Self::order), length `color_count + 1`.
    pub color_offsets: Vec<u32>,
}

/// Flattens a [`ParticleStorage`] / [`SpringSet`] / [`VbdColoring`] triple into
/// an upload-ready [`VbdPrep`], or [`None`] when there are no particles.
///
/// # Contract
///
/// * `positions` / `velocities` are padded `[x, y, z, 0]`.
/// * The `CSR` incidence replicates the golden `Adjacency`: every spring pushes
///   *both* endpoints, listed ascending by spring index. The one divergence is
///   defensive: an endpoint whose index is `>= particle_count` is skipped
///   rather than panicking, so a malformed spring can never make the kernel read
///   out of bounds. Well-formed inputs (which the solver assumes) never hit
///   this guard, so the `CSR` is identical to the golden's.
/// * `order` and `color_offsets` are copied verbatim from `coloring`.
#[must_use]
pub fn build(
    particles: &ParticleStorage,
    springs: &SpringSet,
    coloring: &VbdColoring,
) -> Option<VbdPrep> {
    let particle_count = particles.len();
    if particle_count == 0 {
        return None;
    }

    let positions = particles
        .positions()
        .iter()
        .map(|p| [p.x, p.y, p.z, 0.0])
        .collect();
    let velocities = particles
        .velocities()
        .iter()
        .map(|v| [v.x, v.y, v.z, 0.0])
        .collect();
    let inverse_masses = particles.inverse_masses().to_vec();

    let gpu_springs: Vec<GpuSpring> = springs
        .springs
        .iter()
        .map(|s| GpuSpring {
            a: s.a.index() as u32,
            b: s.b.index() as u32,
            rest_length: s.rest_length,
            stiffness: s.stiffness,
        })
        .collect();

    let (vert_offsets, vert_entries) = build_csr(particle_count, springs);

    Some(VbdPrep {
        particle_count: particle_count as u32,
        color_count: coloring.color_count(),
        positions,
        velocities,
        inverse_masses,
        springs: gpu_springs,
        vert_offsets,
        vert_entries,
        order: coloring.order().to_vec(),
        color_offsets: coloring.offsets().to_vec(),
    })
}

/// Builds the per-vertex incident-spring `CSR` arrays, replicating the golden
/// `Adjacency` ascending-by-spring-index layout with a defensive out-of-range
/// endpoint skip (see [`build`]).
fn build_csr(particle_count: usize, springs: &SpringSet) -> (Vec<u32>, Vec<u32>) {
    let mut counts = alloc::vec![0u32; particle_count];
    for spring in &springs.springs {
        let a = spring.a.index();
        let b = spring.b.index();
        if a < particle_count {
            counts[a] += 1;
        }
        if b < particle_count {
            counts[b] += 1;
        }
    }

    let mut offsets = alloc::vec![0u32; particle_count + 1];
    for i in 0..particle_count {
        offsets[i + 1] = offsets[i] + counts[i];
    }

    let total = offsets[particle_count] as usize;
    let mut cursor = offsets.clone();
    let mut entries = alloc::vec![0u32; total];
    for (spring_index, spring) in springs.springs.iter().enumerate() {
        let a = spring.a.index();
        let b = spring.b.index();
        if a < particle_count {
            entries[cursor[a] as usize] = spring_index as u32;
            cursor[a] += 1;
        }
        if b < particle_count {
            entries[cursor[b] as usize] = spring_index as u32;
            cursor[b] += 1;
        }
    }

    (offsets, entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use prism_physics_core::vbd::{color_springs, SpringElement};

    /// Spawns a 3-vertex closed triangle (springs 0:(0,1) 1:(1,2) 2:(2,0)) and
    /// returns the storage, spring set, and colouring.
    fn triangle() -> (ParticleStorage, SpringSet) {
        let mut particles = ParticleStorage::new();
        let a = particles.spawn(Vec3::new(0.0, 0.0, 0.0), 1.0);
        let b = particles.spawn(Vec3::new(1.0, 0.0, 0.0), 1.0);
        let c = particles.spawn(Vec3::new(0.0, 1.0, 0.0), 1.0);
        let mut springs = SpringSet::new();
        springs.push(SpringElement::new(a, b, 1.0, 100.0));
        springs.push(SpringElement::new(b, c, 1.0, 100.0));
        springs.push(SpringElement::new(c, a, 1.0, 100.0));
        (particles, springs)
    }

    #[test]
    fn empty_storage_yields_none() {
        let particles = ParticleStorage::new();
        let springs = SpringSet::new();
        let coloring = color_springs(&springs, 0);
        assert!(build(&particles, &springs, &coloring).is_none());
    }

    #[test]
    fn csr_matches_engine_adjacency_order() {
        let (particles, springs) = triangle();
        let coloring = color_springs(&springs, particles.len());
        let prep = build(&particles, &springs, &coloring).expect("non-empty");

        // Every vertex touches two springs: offsets step by two.
        assert_eq!(prep.vert_offsets, alloc::vec![0, 2, 4, 6]);
        // Vertex 0 is endpoint `a` of spring 0 and endpoint `b` of spring 2,
        // listed ascending by spring index.
        assert_eq!(&prep.vert_entries[0..2], &[0, 2]);
        assert_eq!(&prep.vert_entries[2..4], &[0, 1]);
        assert_eq!(&prep.vert_entries[4..6], &[1, 2]);
        assert_eq!(prep.particle_count, 3);
    }

    #[test]
    fn pinned_vertex_carries_zero_inverse_mass() {
        let mut particles = ParticleStorage::new();
        let _pinned = particles.spawn_pinned(Vec3::ZERO);
        let _free = particles.spawn(Vec3::new(0.0, -1.0, 0.0), 2.0);
        let springs = SpringSet::new();
        let coloring = color_springs(&springs, particles.len());
        let prep = build(&particles, &springs, &coloring).expect("non-empty");
        assert_eq!(prep.inverse_masses[0], 0.0);
        assert!((prep.inverse_masses[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn out_of_range_endpoint_is_skipped() {
        // Spawn one particle, then forge a spring referencing a vertex index
        // beyond the storage. The golden `Adjacency` would panic; the GPU prep
        // defensively drops the stray endpoint so the kernel never reads OOB.
        let mut particles = ParticleStorage::new();
        let a = particles.spawn(Vec3::ZERO, 1.0);
        let b = particles.spawn(Vec3::new(1.0, 0.0, 0.0), 1.0);
        let mut springs = SpringSet::new();
        springs.push(SpringElement::new(a, b, 1.0, 100.0));
        // Drop the second particle's addressability by building for only one.
        let coloring = color_springs(&springs, 1);
        // Build the CSR for a single addressable particle: spring endpoint `b`
        // (index 1) is now out of range and must be skipped.
        let (offsets, entries) = build_csr(1, &springs);
        assert_eq!(offsets, alloc::vec![0, 1]);
        assert_eq!(entries, alloc::vec![0]);
        let _ = coloring;
    }
}
