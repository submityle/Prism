//! The `CPU` golden twin of the free-surface velocity extrapolation that runs
//! after the pressure projection in the full fluid step.
//!
//! After projection only the faces touched by particles during the scatter
//! carry a physically meaningful velocity; the air faces around the free
//! surface are undefined. Advecting markers that straddle the surface samples
//! those air faces, so they must be filled with plausible values first. This
//! module grows the *known* set outward by repeated nearest-neighbour
//! averaging: each sweep fills every still-unknown face from the average of its
//! already-known 6-neighbours, exactly mirroring the reference in
//! [`prism_physics_core`](prism_physics_core::fluid::mac_grid).
//!
//! # Jacobi sweep order
//!
//! Each sweep reads a *snapshot* of the field and known-mask taken before the
//! sweep and writes into fresh state, so no face relaxed earlier in the same
//! sweep feeds a later one. That Jacobi structure is order-independent, so the
//! device kernel — which ping-pongs two buffers, one dispatch per sweep —
//! reproduces the twin's result bit-for-bit apart from the single division per
//! filled face, which the parity test bounds with a tight tolerance.
//!
//! # Provenance
//!
//! Iterative velocity extrapolation from the known band into the air region is
//! a standard free-surface technique (Bridson, *Fluid Simulation for Computer
//! Graphics*; Zhu and Bridson 2005). No Unreal Engine source or derived code.

/// The dimensions of one staggered face field: the sample counts along each
/// axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AxisDims {
    /// Sample count along `x`.
    pub dx: u32,
    /// Sample count along `y`.
    pub dy: u32,
    /// Sample count along `z`.
    pub dz: u32,
}

impl AxisDims {
    /// Creates an axis-dimension triple.
    #[must_use]
    pub fn new(dx: u32, dy: u32, dz: u32) -> AxisDims {
        AxisDims { dx, dy, dz }
    }

    /// Total number of samples, `dx * dy * dz`.
    #[must_use]
    pub fn count(&self) -> usize {
        (self.dx * self.dy * self.dz) as usize
    }

    /// Flat index of sample `(i, j, k)`.
    #[must_use]
    pub fn flat(&self, i: u32, j: u32, k: u32) -> usize {
        (i + self.dx * (j + self.dy * k)) as usize
    }
}

/// Extrapolates one staggered face field in place.
///
/// Faces with positive `weights` are the initial known band; `iterations`
/// sweeps then grow that band outward, filling each unknown face from the mean
/// of its known 6-neighbours. This is the exact per-sweep Jacobi update the
/// device kernel performs.
///
/// # Panics
///
/// Panics if `field` and `weights` are not both [`AxisDims::count`] long.
pub fn extrapolate_axis(field: &mut [f32], weights: &[f32], dims: AxisDims, iterations: u32) {
    assert_eq!(field.len(), dims.count(), "field must match the axis count");
    assert_eq!(
        weights.len(),
        dims.count(),
        "weights must match the axis count"
    );
    let mut known: Vec<bool> = weights.iter().map(|&w| w > 0.0).collect();
    for _ in 0..iterations {
        let prev = known.clone();
        let src = field.to_vec();
        for k in 0..dims.dz {
            for j in 0..dims.dy {
                for i in 0..dims.dx {
                    let id = dims.flat(i, j, k);
                    if prev[id] {
                        continue;
                    }
                    let (acc, cnt) = gather_known(&src, &prev, dims, i, j, k);
                    if cnt > 0.0 {
                        field[id] = acc / cnt;
                        known[id] = true;
                    }
                }
            }
        }
    }
}

/// Sums the values of the already-known 6-neighbours of `(i, j, k)`, returning
/// the accumulated value and the neighbour count.
///
/// The neighbour order (`-x, +x, -y, +y, -z, +z`) matches the device kernel so
/// the accumulation associates face-for-face.
fn gather_known(src: &[f32], prev: &[bool], dims: AxisDims, i: u32, j: u32, k: u32) -> (f32, f32) {
    let mut acc = 0.0f32;
    let mut cnt = 0.0f32;
    let mut take = |ti: u32, tj: u32, tk: u32| {
        let id = dims.flat(ti, tj, tk);
        if prev[id] {
            acc += src[id];
            cnt += 1.0;
        }
    };
    if i > 0 {
        take(i - 1, j, k);
    }
    if i + 1 < dims.dx {
        take(i + 1, j, k);
    }
    if j > 0 {
        take(i, j - 1, k);
    }
    if j + 1 < dims.dy {
        take(i, j + 1, k);
    }
    if k > 0 {
        take(i, j, k - 1);
    }
    if k + 1 < dims.dz {
        take(i, j, k + 1);
    }
    (acc, cnt)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single known face at the centre of a line spreads one cell per sweep.
    #[test]
    fn known_band_grows_one_cell_per_sweep() {
        let dims = AxisDims::new(5, 1, 1);
        let mut field = vec![0.0f32; dims.count()];
        let mut weights = vec![0.0f32; dims.count()];
        field[2] = 4.0;
        weights[2] = 1.0;

        extrapolate_axis(&mut field, &weights, dims, 1);
        // After one sweep the two immediate neighbours copy the centre value.
        assert!((field[1] - 4.0).abs() < 1e-6);
        assert!((field[3] - 4.0).abs() < 1e-6);
        // The far ends are still unknown after a single sweep.
        assert_eq!(field[0], 0.0);
        assert_eq!(field[4], 0.0);
    }

    /// Enough sweeps fill the whole field with the single seed value.
    #[test]
    fn full_extrapolation_fills_every_face() {
        let dims = AxisDims::new(5, 1, 1);
        let mut field = vec![0.0f32; dims.count()];
        let mut weights = vec![0.0f32; dims.count()];
        field[2] = 4.0;
        weights[2] = 1.0;

        extrapolate_axis(&mut field, &weights, dims, 4);
        for (idx, v) in field.iter().enumerate() {
            assert!((v - 4.0).abs() < 1e-6, "face {idx}: {v}");
        }
    }

    /// Averaging of two known neighbours produces their mean.
    #[test]
    fn unknown_between_two_known_averages() {
        let dims = AxisDims::new(3, 1, 1);
        let mut field = vec![0.0f32; dims.count()];
        let mut weights = vec![0.0f32; dims.count()];
        field[0] = 2.0;
        field[2] = 6.0;
        weights[0] = 1.0;
        weights[2] = 1.0;

        extrapolate_axis(&mut field, &weights, dims, 1);
        // The middle face averages its two known neighbours.
        assert!((field[1] - 4.0).abs() < 1e-6, "{}", field[1]);
    }
}
