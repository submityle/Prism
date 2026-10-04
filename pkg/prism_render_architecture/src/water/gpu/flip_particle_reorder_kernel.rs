//! Water `FLIP`/`APIC` particle-reorder (gather) compute kernel: the `WESL`
//! shader plus its bit-exact `CPU` twin.
//!
//! This is the integration pass that pays off the counting-sort chain
//! (histogram -> exclusive scan -> scatter): it gathers the packed particle
//! records into the cell-major order the scatter pass computed, so the measured
//! `MAC` transfer bottleneck (`P2G`/`G2P`) reads a compact face neighbourhood
//! per workgroup. The real-machine-validated transfer kernels are left
//! untouched — they run verbatim over the reordered pool — so their existing
//! parity goldens stay valid.
//!
//! [`WATER_FLIP_PARTICLE_REORDER_WESL`] is the shader (entry point
//! `water_flip_particle_reorder`, see
//! [`WaterKernel::FlipParticleReorder`](super::super::kernels::WaterKernel)) and
//! [`dispatch_flip_particle_reorder`] is its bit-exact `CPU` twin. Because the
//! sandbox has no `GPU`, the twin is the correctness proof: it consumes the
//! identical buffer `ABI` (one read storage buffer `sorted_indices`, one read
//! storage buffer `particles_in`, one read-write storage buffer
//! `particles_out`, one uniform param block, a 64-lane group over the
//! `Particle` domain). The parity test gathers through the independent golden
//! [`super::super::flip_sort::counting_sort_particles`] permutation, so it
//! proves the gather reproduces the golden rather than asserting a tautology.
//! Only verbatim `f32` copies and integer indexing appear; no `f32` equality
//! and no AI/ML.

use alloc::vec;
use alloc::vec::Vec;

/// `WESL` source of the water `FLIP`/`APIC` particle-reorder compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// test; the standalone `naga`/`wesl` compile check runs out of tree, because
/// `prism_render_architecture` is a zero-dependency crate.
pub const WATER_FLIP_PARTICLE_REORDER_WESL: &str = include_str!("water_flip_particle_reorder.wesl");

/// `f32` lanes per packed `FLIP`/`APIC` particle record, matching the
/// `P2G`/`G2P` particle `ABI` (position+active, velocity, three `APIC` affine
/// rows). Re-exported from the `P2G` kernel so the packing stays single-source.
pub use super::flip_mac_p2g_kernel::FLIP_P2G_PARTICLE_FLOATS as FLIP_REORDER_PARTICLE_FLOATS;

/// Uniform parameter block for the reorder gather, mirroring the shader's
/// `FlipReorderParams`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlipReorderParams {
    /// Number of output slots (the in-grid particle total, `offsets[last]`).
    pub sorted_count: u32,
    /// Number of particles in the source pool (bounds the gather read).
    pub src_count: u32,
}

/// Bit-exact `CPU` twin of the particle-reorder kernel.
///
/// Gathers the packed particle records in `particles` into cell-major order per
/// the `sorted_indices` permutation and returns the reordered pool
/// (`sorted_count * FLIP_REORDER_PARTICLE_FLOATS` lanes).
///
/// Out-of-range slots (beyond `sorted_indices`), out-of-range source indices
/// (`>= src_count` or past the source buffer), and a source buffer too short to
/// hold `src_count` records leave the corresponding destination record zeroed.
/// Never indexes out of bounds and never panics.
#[must_use]
pub fn dispatch_flip_particle_reorder(
    particles: &[f32],
    sorted_indices: &[u32],
    params: FlipReorderParams,
) -> Vec<f32> {
    let floats = FLIP_REORDER_PARTICLE_FLOATS;
    let slots = params.sorted_count as usize;
    let mut out = vec![0.0f32; slots.saturating_mul(floats)];
    let src_count = params.src_count as usize;

    for s in 0..slots {
        // A slot beyond the index buffer cannot name a source; leave it zeroed.
        let Some(&src) = sorted_indices.get(s) else {
            continue;
        };
        let src = src as usize;
        if src >= src_count {
            continue;
        }
        let src_base = src.saturating_mul(floats);
        let dst_base = s.saturating_mul(floats);
        // Guard the source read against a short pool before copying.
        if src_base + floats > particles.len() {
            continue;
        }
        out[dst_base..dst_base + floats].copy_from_slice(&particles[src_base..src_base + floats]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::super::flip_sort::{counting_sort_particles, MacCellGrid};
    use super::super::super::kernels::{DispatchDomain, WaterKernel};
    use super::super::super::{Vec3, EPS};
    use super::super::flip_mac_p2g_kernel::{dispatch_mac_p2g, FlipP2GParams};
    use super::*;

    const F: usize = FLIP_REORDER_PARTICLE_FLOATS;

    fn grid() -> MacCellGrid {
        MacCellGrid {
            origin: Vec3::new(0.0, 0.0, 0.0),
            dx: 1.0,
            nx: 2,
            ny: 2,
            nz: 2,
        }
    }

    /// Builds a packed pool where lane 0 of record `i` encodes `i`, so a gather
    /// can be checked by reading back the lane-0 tag.
    fn tagged_pool(count: usize) -> Vec<f32> {
        let mut pool = vec![0.0f32; count * F];
        for i in 0..count {
            pool[i * F] = i as f32;
            // Fill the remaining lanes with a deterministic ramp so a verbatim
            // copy is distinguishable from a zeroed record.
            for lane in 1..F {
                pool[i * F + lane] = (i * 100 + lane) as f32;
            }
        }
        pool
    }

    #[test]
    fn descriptor_abi_matches_the_shader_bindings() {
        // Three storage buffers (sorted indices in, particles in, particles
        // out) + one uniform, no textures, a 64-lane linear group over the
        // particle domain. Lock the contract so a shader-binding drift fails
        // here.
        let d = WaterKernel::FlipParticleReorder.descriptor();
        assert_eq!(d.layout.storage_buffers, 3);
        assert_eq!(d.layout.uniform_buffers, 1);
        assert_eq!(d.layout.storage_textures, 0);
        assert_eq!(d.layout.sampled_textures, 0);
        assert_eq!(d.domain, DispatchDomain::Particle);
        assert_eq!(d.workgroup.x, 64);
        assert_eq!(d.workgroup.y, 1);
        assert_eq!(d.workgroup.z, 1);
        assert_eq!(
            WaterKernel::FlipParticleReorder.wesl_entry_point(),
            "water_flip_particle_reorder"
        );
    }

    #[test]
    fn gather_reorders_records_verbatim() {
        // A hand-built permutation: pull records 2, 0, 3 into slots 0, 1, 2.
        let pool = tagged_pool(4);
        let sorted = [2u32, 0, 3];
        let params = FlipReorderParams {
            sorted_count: 3,
            src_count: 4,
        };
        let out = dispatch_flip_particle_reorder(&pool, &sorted, params);
        assert_eq!(out.len(), 3 * F);
        // Each destination record is the verbatim source record.
        for (slot, &src) in sorted.iter().enumerate() {
            let src = src as usize;
            assert_eq!(&out[slot * F..slot * F + F], &pool[src * F..src * F + F]);
        }
    }

    #[test]
    fn gather_matches_the_golden_counting_sort_order() {
        let pos = [
            Vec3::new(1.9, 1.9, 1.9),
            Vec3::new(0.1, 0.1, 0.1),
            Vec3::new(1.1, 0.1, 1.1),
            Vec3::new(0.1, 1.1, 0.1),
            Vec3::new(1.1, 1.1, 1.1),
            Vec3::new(0.6, 0.2, 1.3),
        ];
        let pool = tagged_pool(pos.len());
        let order = counting_sort_particles(grid(), &pos);
        let params = FlipReorderParams {
            sorted_count: order.sorted_indices.len() as u32,
            src_count: pos.len() as u32,
        };
        let out = dispatch_flip_particle_reorder(&pool, &order.sorted_indices, params);
        // Slot `s` must carry the record of particle `sorted_indices[s]`: check
        // the lane-0 tag against the golden permutation.
        assert!(!order.sorted_indices.is_empty());
        for (slot, &src) in order.sorted_indices.iter().enumerate() {
            assert_eq!(out[slot * F], src as f32);
        }
    }

    #[test]
    fn out_of_range_source_and_slots_leave_zeroed_records() {
        let pool = tagged_pool(3);
        // One valid gather (slot 0 <- src 1), one out-of-range src (slot 1).
        let sorted = [1u32, 99];
        let params = FlipReorderParams {
            sorted_count: 2,
            src_count: 3,
        };
        let out = dispatch_flip_particle_reorder(&pool, &sorted, params);
        assert_eq!(out.len(), 2 * F);
        // Slot 0 is a verbatim copy of record 1.
        assert_eq!(&out[0..F], &pool[F..2 * F]);
        // Slot 1 named an out-of-range source: left zeroed.
        assert!(out[F..2 * F].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn short_index_buffer_and_short_pool_are_well_formed() {
        let pool = tagged_pool(2);
        // Index buffer shorter than sorted_count: trailing slots stay zeroed.
        let sorted = [1u32];
        let params = FlipReorderParams {
            sorted_count: 3,
            src_count: 2,
        };
        let out = dispatch_flip_particle_reorder(&pool, &sorted, params);
        assert_eq!(out.len(), 3 * F);
        assert_eq!(&out[0..F], &pool[F..2 * F]);
        assert!(out[F..].iter().all(|&v| v == 0.0));

        // Source pool shorter than the claimed src_count: the unreadable record
        // is skipped (zeroed), no out-of-bounds read.
        let short_pool = &pool[..F]; // only record 0 is fully present
        let sorted2 = [1u32];
        let params2 = FlipReorderParams {
            sorted_count: 1,
            src_count: 2,
        };
        let out2 = dispatch_flip_particle_reorder(short_pool, &sorted2, params2);
        assert_eq!(out2.len(), F);
        assert!(out2.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn empty_reorder_is_well_formed_and_deterministic() {
        let pool = tagged_pool(2);
        let params = FlipReorderParams {
            sorted_count: 0,
            src_count: 2,
        };
        let a = dispatch_flip_particle_reorder(&pool, &[], params);
        let b = dispatch_flip_particle_reorder(&pool, &[], params);
        assert!(a.is_empty());
        assert_eq!(a, b);
    }

    /// Packs `positions`/`velocities` and the three `APIC` affine rows into the
    /// 20-lane `FLIP`/`APIC` record layout the `MAC` P2G kernel reads, marking
    /// every record active. Lane map: 0-2 position, 3 active flag, 4-6 velocity,
    /// 8-10 / 12-14 / 16-18 the three affine rows (7/11/15/19 stay padding).
    fn packed_flip_pool(positions: &[Vec3], velocities: &[Vec3], affine: &[[Vec3; 3]]) -> Vec<f32> {
        let n = positions.len();
        let mut pool = vec![0.0f32; n * F];
        for i in 0..n {
            let base = i * F;
            pool[base] = positions[i].x;
            pool[base + 1] = positions[i].y;
            pool[base + 2] = positions[i].z;
            pool[base + 3] = 1.0; // active
            pool[base + 4] = velocities[i].x;
            pool[base + 5] = velocities[i].y;
            pool[base + 6] = velocities[i].z;
            let rows = affine[i];
            pool[base + 8] = rows[0].x;
            pool[base + 9] = rows[0].y;
            pool[base + 10] = rows[0].z;
            pool[base + 12] = rows[1].x;
            pool[base + 13] = rows[1].y;
            pool[base + 14] = rows[1].z;
            pool[base + 16] = rows[2].x;
            pool[base + 17] = rows[2].y;
            pool[base + 18] = rows[2].z;
        }
        pool
    }

    #[test]
    fn p2g_scatter_is_invariant_under_cell_sort_reorder() {
        // Payoff proof (`CPU` twin): running `MAC` P2G over the cell-sorted pool
        // must deposit the same momentum/mass grid as running it over the
        // original pool. Scatter is an unordered sum, so a verbatim gather into
        // cell-major order can only change float summation order, never the
        // physical result. This is the sandbox-verifiable equivalence that
        // justifies feeding the reordered pool into P2G for memory locality
        // (the gather-side G2P payoff needs a de-permutation and is deliberately
        // left out, as it is not verifiable without a device).
        let sort_grid = MacCellGrid {
            origin: Vec3::new(0.0, 0.0, 0.0),
            dx: 1.0,
            nx: 4,
            ny: 4,
            nz: 4,
        };
        // Eight distinct in-grid particles (every one inside `[0, 4)^3`, so the
        // counting sort keeps all of them and `sorted_indices` is a true
        // permutation of `0..n`), with varied velocities and affine rows.
        let positions = [
            Vec3::new(0.3, 0.7, 1.2),
            Vec3::new(3.6, 3.1, 0.4),
            Vec3::new(1.9, 2.4, 3.3),
            Vec3::new(2.2, 0.1, 2.8),
            Vec3::new(0.8, 3.9, 1.5),
            Vec3::new(3.3, 1.6, 2.1),
            Vec3::new(1.1, 1.1, 1.1),
            Vec3::new(2.7, 2.9, 0.9),
        ];
        let velocities = [
            Vec3::new(1.0, -0.5, 0.25),
            Vec3::new(-0.75, 0.5, 1.5),
            Vec3::new(0.5, 2.0, -1.0),
            Vec3::new(-1.25, -0.25, 0.75),
            Vec3::new(2.0, 1.0, -0.5),
            Vec3::new(-0.5, -1.5, 0.5),
            Vec3::new(0.25, 0.75, 1.25),
            Vec3::new(1.5, -1.0, -0.75),
        ];
        let affine = [
            [
                Vec3::new(0.1, 0.0, 0.0),
                Vec3::new(0.0, 0.2, 0.0),
                Vec3::new(0.0, 0.0, 0.3),
            ],
            [
                Vec3::new(0.0, 0.1, 0.0),
                Vec3::new(0.2, 0.0, 0.0),
                Vec3::new(0.0, 0.3, 0.0),
            ],
            [
                Vec3::new(0.0, 0.0, 0.1),
                Vec3::new(0.0, 0.2, 0.0),
                Vec3::new(0.3, 0.0, 0.0),
            ],
            [
                Vec3::new(-0.1, 0.0, 0.0),
                Vec3::new(0.0, -0.2, 0.0),
                Vec3::new(0.0, 0.0, -0.3),
            ],
            [
                Vec3::new(0.15, 0.0, 0.0),
                Vec3::new(0.0, 0.25, 0.0),
                Vec3::new(0.0, 0.0, 0.35),
            ],
            [
                Vec3::new(0.0, 0.15, 0.0),
                Vec3::new(0.25, 0.0, 0.0),
                Vec3::new(0.0, 0.35, 0.0),
            ],
            [
                Vec3::new(0.05, 0.05, 0.05),
                Vec3::new(0.05, 0.05, 0.05),
                Vec3::new(0.05, 0.05, 0.05),
            ],
            [
                Vec3::new(-0.15, 0.0, 0.0),
                Vec3::new(0.0, 0.25, 0.0),
                Vec3::new(0.0, 0.0, -0.35),
            ],
        ];
        let n = positions.len();
        let pool = packed_flip_pool(&positions, &velocities, &affine);

        // Golden cell-sort order; all particles are in-grid, so this is a
        // bijection and `sorted_count == n`.
        let order = counting_sort_particles(sort_grid, &positions);
        assert_eq!(order.sorted_indices.len(), n);
        // The permutation is genuinely a reordering (not an identity no-op), so
        // the invariance below is a meaningful test.
        let identity: Vec<u32> = (0..n as u32).collect();
        assert_ne!(order.sorted_indices, identity);
        let mut as_set = order.sorted_indices.clone();
        as_set.sort_unstable();
        assert_eq!(as_set, identity, "order must be a permutation of 0..n");

        let reordered = dispatch_flip_particle_reorder(
            &pool,
            &order.sorted_indices,
            FlipReorderParams {
                sorted_count: n as u32,
                src_count: n as u32,
            },
        );
        assert_eq!(reordered.len(), n * F);

        let p2g_params = FlipP2GParams {
            grid_nx: 4,
            grid_ny: 4,
            grid_nz: 4,
            particle_count: n as u32,
            dx: 1.0,
            use_affine: true,
        };
        let grid_original = dispatch_mac_p2g(&pool, p2g_params);
        let grid_reordered = dispatch_mac_p2g(&reordered, p2g_params);
        assert_eq!(grid_original.len(), grid_reordered.len());
        assert!(!grid_original.is_empty());
        // A reordered pool must actually deposit something, else the invariance
        // would hold trivially over an all-zero grid.
        assert!(grid_reordered.iter().any(|&v| v.abs() > EPS));

        // Equivalence up to float reassociation. A wrong permutation would
        // diverge by `O(deposited magnitude)`, orders of magnitude above this
        // drift bound; summation-reorder noise stays far below it.
        const MAX_SCATTER_DRIFT: f32 = 1e-4;
        for (face, (&a, &b)) in grid_original.iter().zip(grid_reordered.iter()).enumerate() {
            assert!(
                (a - b).abs() < MAX_SCATTER_DRIFT,
                "face accumulator {face} diverged: original {a}, reordered {b}"
            );
        }
    }
}
