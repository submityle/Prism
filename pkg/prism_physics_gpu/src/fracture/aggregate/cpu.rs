//! `CPU` golden twin for the `GPU` per-fragment aggregator.
//!
//! [`cpu_aggregate_fragments`] runs the identical fixed-point accumulation the
//! `WGSL` kernel does, one point at a time, then de-quantises and forms each
//! fragment's rigid-body seed. Because the integer accumulation is exact and
//! order independent, a passing real-device parity test is direct evidence that
//! the ported kernel produced the same mass, centroid, and inertia as this
//! reference — not merely that the shader compiled.
//!
//! # What each fragment reports
//!
//! For a fragment cell `c` gathering the points assigned to it:
//!
//! - `mass` is the sum of the point masses.
//! - `centroid` is the mass-weighted mean position (the centre of mass).
//! - `inertia` is the inertia tensor taken about that centroid, built from the
//!   accumulated second moment via `I = trace(M2) * E - M2`, where `M2` is the
//!   second moment about the centroid and `E` is the identity.
//!
//! A fragment that gathered no mass reports a zero centroid and zero inertia so
//! callers can skip it without a separate flag.
//!
//! # Provenance
//!
//! The rigid-body mass, centroid, and inertia formulas are textbook mechanics;
//! fixed-point atomic accumulation is a standard `GPU` reduction. No Unreal
//! Engine source or derived code.

use glam::{Mat3, Vec3};

use super::super::config::NO_CELL;
use super::config::{dequantise, quantise, AggregateConfig};

/// The rigid-body seed aggregated for one fragment cell.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FragmentAggregate {
    /// Total mass gathered into the fragment.
    pub mass: f32,
    /// Centre of mass, or the origin when the fragment gathered no mass.
    pub centroid: Vec3,
    /// Inertia tensor about the centroid (symmetric), or all zeros when the
    /// fragment gathered no mass.
    pub inertia: Mat3,
}

impl FragmentAggregate {
    /// The seed reported for a fragment that gathered no mass.
    #[must_use]
    pub fn empty() -> FragmentAggregate {
        FragmentAggregate {
            mass: 0.0,
            centroid: Vec3::ZERO,
            inertia: Mat3::ZERO,
        }
    }
}

/// Integer accumulators for one fragment, laid out exactly like the device
/// atomics: one mass slot, three first-moment slots, six second-moment slots.
#[derive(Clone, Copy)]
struct Accumulator {
    mass: i32,
    moment: [i32; 3],
    second: [i32; 6],
}

impl Accumulator {
    const ZERO: Accumulator = Accumulator {
        mass: 0,
        moment: [0; 3],
        second: [0; 6],
    };
}

/// Aggregates the per-fragment mass, centroid, and inertia for `n_cells`
/// fragments from `points`, their `masses`, and their `cells` assignment.
///
/// The three slices must share the same length (one entry per point). A point
/// whose cell is [`NO_CELL`] or is out of range for `n_cells` is skipped, so a
/// classifier's unassigned sentinel passes through harmlessly. The returned
/// vector has exactly `n_cells` entries in cell-index order.
///
/// # Panics
///
/// Panics if `points`, `masses`, and `cells` do not all have the same length.
#[must_use]
pub fn cpu_aggregate_fragments(
    n_cells: usize,
    points: &[Vec3],
    masses: &[f32],
    cells: &[u32],
    config: &AggregateConfig,
) -> Vec<FragmentAggregate> {
    assert!(
        points.len() == masses.len() && masses.len() == cells.len(),
        "points, masses, and cells must have equal length"
    );

    let mut acc = vec![Accumulator::ZERO; n_cells];
    for ((&p, &m), &cell) in points.iter().zip(masses).zip(cells) {
        if cell == NO_CELL {
            continue;
        }
        let c = cell as usize;
        if c >= n_cells {
            continue;
        }
        let a = &mut acc[c];
        a.mass += quantise(m, config.mass_scale);
        a.moment[0] += quantise(m * p.x, config.moment_scale);
        a.moment[1] += quantise(m * p.y, config.moment_scale);
        a.moment[2] += quantise(m * p.z, config.moment_scale);
        a.second[0] += quantise(m * p.x * p.x, config.second_moment_scale);
        a.second[1] += quantise(m * p.y * p.y, config.second_moment_scale);
        a.second[2] += quantise(m * p.z * p.z, config.second_moment_scale);
        a.second[3] += quantise(m * p.x * p.y, config.second_moment_scale);
        a.second[4] += quantise(m * p.x * p.z, config.second_moment_scale);
        a.second[5] += quantise(m * p.y * p.z, config.second_moment_scale);
    }

    acc.iter().map(|a| finish(a, config)).collect()
}

/// Turns one fragment's integer accumulators into its rigid-body seed, matching
/// the de-quantisation and inertia algebra the device kernel performs on
/// read-back.
fn finish(a: &Accumulator, config: &AggregateConfig) -> FragmentAggregate {
    let mass = dequantise(a.mass, config.mass_scale);
    if mass <= 0.0 {
        return FragmentAggregate::empty();
    }

    let first = Vec3::new(
        dequantise(a.moment[0], config.moment_scale),
        dequantise(a.moment[1], config.moment_scale),
        dequantise(a.moment[2], config.moment_scale),
    );
    let centroid = first / mass;

    // Second moment about the origin, then shifted to the centroid via the
    // parallel-axis identity `M2 = S2 - mass * (c (x) c)`.
    let s_xx = dequantise(a.second[0], config.second_moment_scale);
    let s_yy = dequantise(a.second[1], config.second_moment_scale);
    let s_zz = dequantise(a.second[2], config.second_moment_scale);
    let s_xy = dequantise(a.second[3], config.second_moment_scale);
    let s_xz = dequantise(a.second[4], config.second_moment_scale);
    let s_yz = dequantise(a.second[5], config.second_moment_scale);

    let m_xx = s_xx - mass * centroid.x * centroid.x;
    let m_yy = s_yy - mass * centroid.y * centroid.y;
    let m_zz = s_zz - mass * centroid.z * centroid.z;
    let m_xy = s_xy - mass * centroid.x * centroid.y;
    let m_xz = s_xz - mass * centroid.x * centroid.z;
    let m_yz = s_yz - mass * centroid.y * centroid.z;

    let inertia = inertia_from_second_moment(m_xx, m_yy, m_zz, m_xy, m_xz, m_yz);
    FragmentAggregate {
        mass,
        centroid,
        inertia,
    }
}

/// Builds the symmetric inertia tensor `I = trace(M2) * E - M2` from the six
/// unique components of the centroid-relative second moment `M2`.
fn inertia_from_second_moment(
    m_xx: f32,
    m_yy: f32,
    m_zz: f32,
    m_xy: f32,
    m_xz: f32,
    m_yz: f32,
) -> Mat3 {
    let i_xx = m_yy + m_zz;
    let i_yy = m_xx + m_zz;
    let i_zz = m_xx + m_yy;
    // `glam::Mat3` is column-major; the tensor is symmetric so the off-diagonal
    // products are shared between the mirrored slots.
    Mat3::from_cols(
        Vec3::new(i_xx, -m_xy, -m_xz),
        Vec3::new(-m_xy, i_yy, -m_yz),
        Vec3::new(-m_xz, -m_yz, i_zz),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for the analytic checks. Positions and masses are
    /// chosen as small dyadic values so the fixed-point round-trip is effectively
    /// exact and only the final float algebra contributes error.
    const TOL: f32 = 1.0e-3;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= TOL
    }

    #[test]
    fn empty_cell_reports_zero_seed() {
        let out = cpu_aggregate_fragments(2, &[], &[], &[], &AggregateConfig::default());
        assert_eq!(out.len(), 2);
        for frag in out {
            assert_eq!(frag, FragmentAggregate::empty());
        }
    }

    #[test]
    fn zero_cells_returns_empty() {
        let out = cpu_aggregate_fragments(
            0,
            &[Vec3::new(1.0, 2.0, 3.0)],
            &[1.0],
            &[0],
            &AggregateConfig::default(),
        );
        assert!(out.is_empty());
    }

    #[test]
    fn single_point_has_its_position_as_centroid_and_no_inertia() {
        let out = cpu_aggregate_fragments(
            1,
            &[Vec3::new(2.0, -1.0, 0.5)],
            &[3.0],
            &[0],
            &AggregateConfig::default(),
        );
        let frag = out[0];
        assert!(close(frag.mass, 3.0));
        assert!(close(frag.centroid.x, 2.0));
        assert!(close(frag.centroid.y, -1.0));
        assert!(close(frag.centroid.z, 0.5));
        // A single point mass has zero inertia about its own centroid.
        for col in 0..3 {
            for row in 0..3 {
                assert!(close(frag.inertia.col(col)[row], 0.0));
            }
        }
    }

    #[test]
    fn symmetric_pair_matches_analytic_inertia() {
        // Two unit masses at (+2,0,0) and (-2,0,0): centre of mass at the
        // origin, and the inertia about it is diag(0, 2 m d^2, 2 m d^2) with
        // m = 1 and d = 2, so Iyy = Izz = 8 and Ixx = 0.
        let out = cpu_aggregate_fragments(
            1,
            &[Vec3::new(2.0, 0.0, 0.0), Vec3::new(-2.0, 0.0, 0.0)],
            &[1.0, 1.0],
            &[0, 0],
            &AggregateConfig::default(),
        );
        let frag = out[0];
        assert!(close(frag.mass, 2.0));
        assert!(close(frag.centroid.x, 0.0));
        assert!(close(frag.centroid.y, 0.0));
        assert!(close(frag.centroid.z, 0.0));
        assert!(close(frag.inertia.col(0)[0], 0.0), "Ixx should vanish");
        assert!(close(frag.inertia.col(1)[1], 8.0), "Iyy should be 8");
        assert!(close(frag.inertia.col(2)[2], 8.0), "Izz should be 8");
        assert!(close(frag.inertia.col(0)[1], 0.0), "Ixy should vanish");
    }

    #[test]
    fn unassigned_and_out_of_range_points_are_skipped() {
        let out = cpu_aggregate_fragments(
            1,
            &[
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(9.0, 9.0, 9.0),
                Vec3::new(9.0, 9.0, 9.0),
            ],
            &[1.0, 5.0, 5.0],
            &[0, NO_CELL, 7],
            &AggregateConfig::default(),
        );
        // Only the first point counts; the sentinel and the out-of-range index
        // are dropped, so the fragment is a single unit mass at (1,0,0).
        let frag = out[0];
        assert!(close(frag.mass, 1.0));
        assert!(close(frag.centroid.x, 1.0));
    }
}
