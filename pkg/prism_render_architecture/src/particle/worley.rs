//! Worley / cellular (Voronoi) noise: the second, orthogonal branch of the
//! procedural-noise stack (design §8.3, layered on the noise primitives of
//! §8.2).
//!
//! [`super::noise`] is the *gradient/value* branch: it hashes a lattice into
//! smooth `Perlin`-style value and gradient fields, sums them into `fBm`, and
//! takes an analytic curl for divergence-free turbulence. That family answers
//! "how does a continuous scalar/vector field vary smoothly across space?".
//!
//! This module is the *cellular* branch, and it answers a different question:
//! "how far is this point from the nearest of a set of scattered feature
//! points?". Instead of interpolating hashed lattice corners it scatters one
//! jittered feature point per integer cell and measures distances to the
//! nearest (`F1`) and next-nearest (`F2`) of them. The two branches are fully
//! orthogonal and compose: gradient `fBm` gives billowing continuity while
//! Worley `F1`/`F2` give the hard cellular structure of cracks, foam cells,
//! caustic webs, scales, and bubble packing.
//!
//! It is the `CPU`-verifiable contract behind the cellular-noise nodes shipped
//! by Unreal `Niagara`, Unity's `VFX Graph`, `Houdini` (its "Worley/Cellular
//! Noise" `VOP`), and `EmberGen`: `F1` for pitting and bubbles, `F2 - F1` for
//! the Voronoi edge/crack map, and a per-cell id for random flat-shaded cells.
//!
//! Determinism matches [`super::noise`]: all randomness flows through the same
//! integer [`hash_lattice`] avalanche, and the only floating-point primitives
//! beyond ordinary arithmetic are `f32::floor` (integer lattice location),
//! `f32::abs` (the Manhattan / Chebyshev metrics), and `sqrt` (through
//! [`Vec3`]'s Euclidean [`Vec3::distance`]). There are no transcendental calls,
//! so the reference stays bit-reproducible against a future `GPU` kernel that
//! hashes the same cells.

use super::noise::{hash_lattice, FbmParams};
use super::Vec3;

/// Small positive tolerance for the `f32` ordering guards in this module (and
/// the guard against dividing by a fully collapsed `fBm` amplitude sum). Used
/// so the `F1 <= F2` and `F2 - F1 >= 0` invariants are asserted with slack
/// rather than with a forbidden bare `==` comparison.
pub const EPS: f32 = 1.0e-6;

/// Odd-integer salt mixed into the seed for the X component of a cell's jittered
/// feature point, so the three jitter axes are statistically independent.
const SALT_X: u32 = 0x68E3_1DA4;

/// Odd-integer salt for the Y jitter axis (distinct from [`SALT_X`]).
const SALT_Y: u32 = 0xB529_7A4D;

/// Odd-integer salt for the Z jitter axis (distinct from [`SALT_X`]/[`SALT_Y`]).
const SALT_Z: u32 = 0x1B56_C4E9;

/// Salt mixed into the seed when deriving a cell's stable colour id, so the id
/// is decorrelated from the jitter position hashes of the same cell.
const SALT_ID: u32 = 0xA341_316C;

/// Per-octave seed increment for [`worley_fbm`], so successive octaves draw
/// from an independent scatter of feature points rather than a rescaling of the
/// same one.
const SEED_STEP: u32 = 0x1656_67B1;

/// Scale that turns a 16-bit hash segment into the half-open range `[0, 1)`
/// (a pure multiply, so it introduces no transcendental call).
const INV_2POW16: f32 = 1.0 / 65_536.0;

/// The distance metric used to measure the gap between a sample point and a
/// cell feature point.
///
/// The choice reshapes the cells: `Euclidean` gives round Voronoi cells,
/// `Manhattan` (`L1`) gives diamond/axis-aligned cells, and `Chebyshev`
/// (`L∞`) gives square/box cells. Because the enum carries no floating-point
/// field it can derive `Eq` and `Hash`, so a metric can key a cache or a
/// dispatch table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CellularMetric {
    /// Straight-line `L2` distance (round cells); the only variant that uses
    /// `sqrt`, through [`Vec3::distance`].
    Euclidean,
    /// Sum-of-absolute-components `L1` distance (diamond cells).
    Manhattan,
    /// Max-absolute-component `L∞` distance (square/box cells).
    Chebyshev,
}

impl CellularMetric {
    /// Distance from `a` to `b` under this metric.
    ///
    /// For any fixed pair the metrics are ordered `Manhattan >= Euclidean >=
    /// Chebyshev`; because `F1` is a minimum over the *same* candidate set for
    /// every metric, that ordering is inherited by the `F1` fields (see the
    /// tests).
    #[must_use]
    pub fn distance(self, a: Vec3, b: Vec3) -> f32 {
        match self {
            Self::Euclidean => a.distance(b),
            Self::Manhattan => {
                let d = a.sub(b);
                d.x.abs() + d.y.abs() + d.z.abs()
            }
            Self::Chebyshev => {
                let d = a.sub(b);
                d.x.abs().max(d.y.abs()).max(d.z.abs())
            }
        }
    }
}

/// Maps a 32-bit hash to the half-open unit interval `[0, 1)` using its low
/// 16-bit segment (a pure multiply by [`INV_2POW16`]).
#[must_use]
fn unit01(h: u32) -> f32 {
    ((h & 0xFFFF) as f32) * INV_2POW16
}

/// The jittered feature point of integer cell `(i, j, k)`.
///
/// The cell's integer corner is offset by a jitter in `[0, 1)^3`, so exactly
/// one feature point lives inside each unit cell. Each jitter axis is hashed
/// with its own salt (`SALT_X`/`SALT_Y`/`SALT_Z`) so the three coordinates are
/// independent and the scatter looks isotropic rather than diagonally banded.
#[must_use]
fn feature_point(i: i32, j: i32, k: i32, seed: u32) -> Vec3 {
    let jx = unit01(hash_lattice(i, j, k, seed ^ SALT_X));
    let jy = unit01(hash_lattice(i, j, k, seed ^ SALT_Y));
    let jz = unit01(hash_lattice(i, j, k, seed ^ SALT_Z));
    Vec3::new(i as f32 + jx, j as f32 + jy, k as f32 + jz)
}

/// Splits a coordinate into its floor cell index (`f32::floor` then `as i32`).
#[must_use]
fn floor_cell(x: f32) -> i32 {
    x.floor() as i32
}

/// The outcome of searching the `3x3x3` neighbourhood: the nearest (`F1`) and
/// next-nearest (`F2`) distances, plus the integer cell that owns the `F1`
/// feature point (for [`worley_cells`]).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Nearest {
    /// Distance to the closest feature point (`F1`).
    f1: f32,
    /// Distance to the second-closest feature point (`F2`), `>= f1`.
    f2: f32,
    /// Integer cell coordinates owning the `F1` feature point.
    cell: (i32, i32, i32),
}

/// Searches the sample point's own cell plus its 26 neighbours (the `3x3x3`
/// block, 27 candidate feature points) and returns the nearest two distances
/// and the `F1` cell under `metric`.
///
/// Because every one of the 27 cells contributes a feature point, both `F1`
/// and `F2` are always finite. Distances are compared with `<` (never a
/// forbidden bare `==`), keeping the smallest two while recording the `F1`
/// cell.
#[must_use]
fn search(pos: Vec3, seed: u32, metric: CellularMetric) -> Nearest {
    let bi = floor_cell(pos.x);
    let bj = floor_cell(pos.y);
    let bk = floor_cell(pos.z);

    let mut f1 = f32::INFINITY;
    let mut f2 = f32::INFINITY;
    let mut cell = (bi, bj, bk);

    for di in -1..=1 {
        for dj in -1..=1 {
            for dk in -1..=1 {
                let (ci, cj, ck) = (bi + di, bj + dj, bk + dk);
                let fp = feature_point(ci, cj, ck, seed);
                let d = metric.distance(pos, fp);
                if d < f1 {
                    f2 = f1;
                    f1 = d;
                    cell = (ci, cj, ck);
                } else if d < f2 {
                    f2 = d;
                }
            }
        }
    }

    Nearest { f1, f2, cell }
}

/// Worley `F1`: the distance from `pos` to the nearest feature point under
/// `metric`.
///
/// This is the classic cellular-noise value. It is `0` exactly at a feature
/// point and grows toward cell boundaries, so mapping it to opacity/height
/// gives pitting, dents, and packed-sphere looks. It is non-negative and, for
/// the `3x3x3` search, bounded by roughly `sqrt(3)` (Euclidean).
#[must_use]
pub fn worley_f1(pos: Vec3, seed: u32, metric: CellularMetric) -> f32 {
    search(pos, seed, metric).f1
}

/// Worley `(F1, F2)`: the nearest and next-nearest feature-point distances
/// under `metric`.
///
/// `F2 >= F1` always. `F1` alone gives blobs; the pair unlocks the derived
/// fields below — most usefully `F2 - F1` for Voronoi edges (see
/// [`worley_edges`]).
#[must_use]
pub fn worley_f2(pos: Vec3, seed: u32, metric: CellularMetric) -> (f32, f32) {
    let n = search(pos, seed, metric);
    (n.f1, n.f2)
}

/// Worley edge / crack map `F2 - F1` under `metric`.
///
/// The difference is near `0` on the equidistant boundary between two cells
/// (where `F1 ≈ F2`) and rises toward cell interiors, so it draws the Voronoi
/// cell *walls* as thin valleys: cracked mud, dried paint, stained glass
/// leading, cell membranes, and lightning-like fracture webs. It is
/// non-negative by construction (`F2 >= F1`).
#[must_use]
pub fn worley_edges(pos: Vec3, seed: u32, metric: CellularMetric) -> f32 {
    let (f1, f2) = worley_f2(pos, seed, metric);
    f2 - f1
}

/// The stable id of the cell that owns the nearest (`Euclidean` `F1`) feature
/// point at `pos`.
///
/// Every sample inside one Voronoi cell returns the same id, so mapping the id
/// through a palette flat-shades each cell a random constant colour (the
/// classic "random per-cell tint" look). The id is a hash of the winning cell's
/// integer coordinates, decorrelated from the jitter hashes by [`SALT_ID`], so
/// it is stable across runs and independent of the jitter positions.
#[must_use]
pub fn worley_cells(pos: Vec3, seed: u32) -> u32 {
    let (ci, cj, ck) = search(pos, seed, CellularMetric::Euclidean).cell;
    hash_lattice(ci, cj, ck, seed ^ SALT_ID)
}

/// Inverted Worley `F1`: `1.0 - worley_f1`, the "bubble / spot" companion of
/// [`worley_f1`].
///
/// Where `F1` is dark at feature points and bright at cell edges, the inverse
/// is bright *at* the feature points and falls off outward, giving rounded
/// bubbles, blisters, or spots. Note the value can go slightly negative in the
/// corners farthest from any feature point (where `F1 > 1`); callers that need
/// a strict `[0, 1]` mask should saturate the result themselves.
#[must_use]
pub fn worley_inverted(pos: Vec3, seed: u32, metric: CellularMetric) -> f32 {
    1.0 - worley_f1(pos, seed, metric)
}

/// Fractal Worley: an amplitude-normalised sum of `worley_f1` octaves.
///
/// This reuses [`FbmParams`] from [`super::noise`] (its `octaves` /
/// `lacunarity` / `gain`) exactly like [`super::noise::fbm`] does for gradient
/// noise: each octave scales the sample position by `lacunarity`, scales the
/// contribution by `gain`, and advances the seed by [`SEED_STEP`] so octaves
/// scatter independent feature points. Dividing by the total amplitude keeps
/// the output in the bounded range of a single `F1` octave regardless of octave
/// count, so stacked cellular detail (fine foam over coarse cells) stays
/// well-scaled. Zero octaves (or a fully collapsed amplitude) yield `0.0`
/// rather than a `NaN`.
#[must_use]
pub fn worley_fbm(pos: Vec3, params: FbmParams, seed: u32, metric: CellularMetric) -> f32 {
    let mut freq = 1.0_f32;
    let mut amp = 1.0_f32;
    let mut sum = 0.0_f32;
    let mut norm = 0.0_f32;
    let mut octave_seed = seed;

    for _ in 0..params.octaves {
        sum += worley_f1(pos.scale(freq), octave_seed, metric) * amp;
        norm += amp;
        freq *= params.lacunarity;
        amp *= params.gain;
        octave_seed = octave_seed.wrapping_add(SEED_STEP);
    }

    if norm.abs() > EPS {
        sum / norm
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Loose tolerance for value-equality assertions (bit-exact reproducibility
    /// is checked separately with `assert_eq!`).
    const TOL: f32 = 1.0e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= TOL
    }

    const METRICS: [CellularMetric; 3] = [
        CellularMetric::Euclidean,
        CellularMetric::Manhattan,
        CellularMetric::Chebyshev,
    ];

    const SAMPLES: [Vec3; 5] = [
        Vec3::new(0.37, 1.11, -0.62),
        Vec3::new(-1.27, 0.48, 2.15),
        Vec3::new(3.04, -2.11, 0.57),
        Vec3::new(-0.86, -1.53, -2.42),
        Vec3::new(12.5, -7.25, 4.0),
    ];

    #[test]
    fn f1_is_never_greater_than_f2() {
        for &metric in &METRICS {
            for &p in &SAMPLES {
                let (f1, f2) = worley_f2(p, 7, metric);
                assert!(f1 <= f2 + EPS, "F1 {f1} must not exceed F2 {f2}");
            }
        }
    }

    #[test]
    fn f1_and_f2_are_non_negative_and_finite() {
        for &metric in &METRICS {
            for &p in &SAMPLES {
                let (f1, f2) = worley_f2(p, 3, metric);
                assert!(f1 >= -EPS && f1.is_finite(), "F1 {f1}");
                assert!(f2 >= -EPS && f2.is_finite(), "F2 {f2}");
            }
        }
    }

    #[test]
    fn results_are_deterministic_bit_for_bit() {
        for &metric in &METRICS {
            for &p in &SAMPLES {
                assert_eq!(worley_f1(p, 21, metric), worley_f1(p, 21, metric));
                assert_eq!(worley_f2(p, 21, metric), worley_f2(p, 21, metric));
                assert_eq!(worley_edges(p, 21, metric), worley_edges(p, 21, metric));
            }
        }
        let p = Vec3::new(0.37, 1.11, -0.62);
        assert_eq!(worley_cells(p, 21), worley_cells(p, 21));
    }

    #[test]
    fn f1_is_zero_at_a_feature_point() {
        // A point placed exactly on a cell's feature point has F1 == 0 (that
        // point is inside the searched 3x3x3 block).
        for &metric in &METRICS {
            let fp = feature_point(0, 0, 0, 55);
            let f1 = worley_f1(fp, 55, metric);
            assert!(
                approx(f1, 0.0),
                "F1 at a feature point should be ~0, got {f1}"
            );
        }
    }

    #[test]
    fn metric_ordering_manhattan_ge_euclidean_ge_chebyshev() {
        // F1 is a minimum over the same candidate set for every metric, so the
        // per-pair ordering L1 >= L2 >= Linf is inherited by the F1 values.
        for &p in &SAMPLES {
            let man = worley_f1(p, 9, CellularMetric::Manhattan);
            let euc = worley_f1(p, 9, CellularMetric::Euclidean);
            let che = worley_f1(p, 9, CellularMetric::Chebyshev);
            assert!(man >= euc - EPS, "Manhattan {man} < Euclidean {euc}");
            assert!(euc >= che - EPS, "Euclidean {euc} < Chebyshev {che}");
        }
    }

    #[test]
    fn edges_are_non_negative() {
        for &metric in &METRICS {
            for &p in &SAMPLES {
                let e = worley_edges(p, 13, metric);
                assert!(e >= -EPS, "edge value {e} must be non-negative");
            }
        }
    }

    #[test]
    fn edges_equal_f2_minus_f1() {
        for &metric in &METRICS {
            for &p in &SAMPLES {
                let (f1, f2) = worley_f2(p, 44, metric);
                assert!(approx(worley_edges(p, 44, metric), f2 - f1));
            }
        }
    }

    #[test]
    fn cell_id_is_stable_within_a_cell() {
        // Two nearby points that resolve to the same nearest feature point
        // share a cell id; the id is also perfectly reproducible.
        let a = Vec3::new(0.30, 0.30, 0.30);
        let b = Vec3::new(0.32, 0.29, 0.31);
        let id_a = worley_cells(a, 6);
        assert_eq!(id_a, worley_cells(a, 6));
        assert_eq!(id_a, worley_cells(b, 6));
    }

    #[test]
    fn cell_ids_differ_between_distant_cells() {
        // Far-apart samples land in different cells and (with overwhelming
        // probability for this fixed seed) get different ids.
        let far_a = Vec3::new(0.5, 0.5, 0.5);
        let far_b = Vec3::new(40.5, -30.5, 17.5);
        assert!(worley_cells(far_a, 6) != worley_cells(far_b, 6));
    }

    #[test]
    fn fbm_is_bounded_and_finite() {
        for &metric in &METRICS {
            for &p in &SAMPLES {
                let v = worley_fbm(p, FbmParams::DEFAULT, 17, metric);
                assert!(v.is_finite(), "fBm must be finite, got {v}");
                // Normalised sum of non-negative F1 octaves stays within a
                // single octave's bound (well under 3 for the 3x3x3 search).
                assert!((0.0 - EPS..=3.0).contains(&v), "fBm {v} out of range");
            }
        }
    }

    #[test]
    fn fbm_single_octave_equals_f1() {
        for &metric in &METRICS {
            for &p in &SAMPLES {
                let single = worley_fbm(p, FbmParams::SINGLE_OCTAVE, 31, metric);
                assert!(approx(single, worley_f1(p, 31, metric)));
            }
        }
    }

    #[test]
    fn fbm_zero_octaves_is_zero_not_nan() {
        let p = Vec3::new(0.4, -1.7, 2.9);
        let n = worley_fbm(
            p,
            FbmParams::new(0, 2.0, 0.5),
            13,
            CellularMetric::Euclidean,
        );
        assert!(approx(n, 0.0) && n.is_finite());
    }

    #[test]
    fn fbm_is_deterministic() {
        let p = Vec3::new(2.1, 0.5, -3.3);
        assert_eq!(
            worley_fbm(p, FbmParams::DEFAULT, 99, CellularMetric::Manhattan),
            worley_fbm(p, FbmParams::DEFAULT, 99, CellularMetric::Manhattan)
        );
    }

    #[test]
    fn inverted_is_one_minus_f1() {
        for &metric in &METRICS {
            for &p in &SAMPLES {
                let inv = worley_inverted(p, 8, metric);
                assert!(approx(inv, 1.0 - worley_f1(p, 8, metric)));
            }
        }
    }

    #[test]
    fn metric_distance_matches_definitions() {
        let a = Vec3::new(1.0, 2.0, 2.0);
        let b = Vec3::ZERO;
        assert!(approx(CellularMetric::Euclidean.distance(a, b), 3.0));
        assert!(approx(CellularMetric::Manhattan.distance(a, b), 5.0));
        assert!(approx(CellularMetric::Chebyshev.distance(a, b), 2.0));
    }
}
