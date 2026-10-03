//! RGB endpoint fitting for the single-partition LDR encoder.
//!
//! Mirrors the BC7 colour-axis fit (`encode::bc7::principal_axis3`): find the
//! principal RGB axis of the sixteen texels by power iteration on the 3x3
//! covariance, project every texel onto it, and take the extreme projections
//! as the two endpoints. The endpoints are then ordered so that
//! `hadd(e0) <= hadd(e1)` (`hadd = r + g + b`), which is the orientation the
//! CEM-8 decoder reproduces without triggering its blue-contraction swap.
//!
//! Pure analytic `f64` arithmetic -- no AI/ML path.

/// Fit two RGB endpoints for a single-partition CEM-8 block.
///
/// Returns `(e0, e1)` as 8-bit RGB with `hadd(e0) <= hadd(e1)` so the decoder
/// (`cem::decode`, CEM 8) reads them back directly. Alpha is implicitly 255 for
/// CEM 8, so the input alpha channel is ignored.
pub(super) fn fit_rgb_endpoints(texels: &[[u8; 4]; 16]) -> ([u8; 3], [u8; 3]) {
    let points: [[f64; 3]; 16] =
        core::array::from_fn(|t| core::array::from_fn(|c| f64::from(texels[t][c])));

    // Mean colour.
    let mut mean = [0.0f64; 3];
    for p in &points {
        for c in 0..3 {
            mean[c] += p[c];
        }
    }
    for m in &mut mean {
        *m /= 16.0;
    }

    // 3x3 covariance.
    let mut cov = [[0.0f64; 3]; 3];
    for p in &points {
        let d: [f64; 3] = core::array::from_fn(|c| p[c] - mean[c]);
        for (i, row) in cov.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell += d[i] * d[j];
            }
        }
    }

    // Dominant eigenvector via power iteration; a degenerate (constant) block
    // yields a near-zero `next`, so we fall back to the current axis.
    let mut axis = [1.0f64, 1.0, 1.0];
    for _ in 0..24 {
        let next: [f64; 3] =
            core::array::from_fn(|i| (0..3).map(|j| cov[i][j] * axis[j]).sum::<f64>());
        let norm = (next.iter().map(|v| v * v).sum::<f64>()).sqrt();
        if norm < 1e-9 {
            break;
        }
        axis = core::array::from_fn(|c| next[c] / norm);
    }

    // Project onto the axis and take the extreme texels as endpoints.
    let mut min_proj = f64::INFINITY;
    let mut max_proj = f64::NEG_INFINITY;
    let mut lo = points[0];
    let mut hi = points[0];
    for p in &points {
        let proj = (0..3).map(|c| (p[c] - mean[c]) * axis[c]).sum::<f64>();
        if proj < min_proj {
            min_proj = proj;
            lo = *p;
        }
        if proj > max_proj {
            max_proj = proj;
            hi = *p;
        }
    }

    let a = round_rgb(lo);
    let b = round_rgb(hi);

    // Orient so hadd(e0) <= hadd(e1): decode uncontracts/swaps otherwise.
    if hadd(a) > hadd(b) {
        (b, a)
    } else {
        (a, b)
    }
}

/// Round and clamp an `f64` RGB triple to 8-bit UNORM.
fn round_rgb(c: [f64; 3]) -> [u8; 3] {
    core::array::from_fn(|i| {
        let v = c[i].round();
        if v <= 0.0 {
            0
        } else if v >= 255.0 {
            255
        } else {
            v as u8
        }
    })
}

/// `r + g + b`, the ordering key the CEM-8 decoder uses for blue contraction.
fn hadd(c: [u8; 3]) -> u32 {
    u32::from(c[0]) + u32::from(c[1]) + u32::from(c[2])
}
