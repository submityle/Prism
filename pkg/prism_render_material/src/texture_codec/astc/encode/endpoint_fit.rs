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

/// Fit two RGB endpoints for a single-partition CEM-8 block over any
/// footprint of `N` texels (4x4 = 16, 5x5 = 25, ...).
///
/// Returns `(e0, e1)` as 8-bit RGB with `hadd(e0) <= hadd(e1)` so the decoder
/// (`cem::decode`, CEM 8) reads them back directly. Alpha is implicitly 255 for
/// CEM 8, so the input alpha channel is ignored.
pub(super) fn fit_rgb_endpoints<const N: usize>(texels: &[[u8; 4]; N]) -> ([u8; 3], [u8; 3]) {
    let points: [[f64; 3]; N] =
        core::array::from_fn(|t| core::array::from_fn(|c| f64::from(texels[t][c])));

    // Mean colour.
    let mut mean = [0.0f64; 3];
    for p in &points {
        for c in 0..3 {
            mean[c] += p[c];
        }
    }
    for m in &mut mean {
        *m /= N as f64;
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

    // Dominant eigenvector via power iteration. Seed with the covariance column
    // of largest norm rather than a fixed [1,1,1]: a fixed seed that happens to
    // be orthogonal to the dominant eigenvector collapses `cov * seed` to zero
    // and loses the axis. The classic failure is a pure two-colour split whose
    // variance axis is orthogonal to (1,1,1) -- e.g. red (255,0,0) vs blue
    // (0,0,255), whose axis is (1,0,-1) with (1,0,-1)·(1,1,1) = 0. The
    // largest-norm column of a symmetric PSD covariance always carries a
    // non-zero component along the dominant eigenvector, so it is a safe seed.
    let mut axis = seed_axis3(&cov);
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

/// Fit two RGBA endpoints for a single-partition CEM-12 (RGBA direct) block.
///
/// Mirrors [`fit_rgb_endpoints`] but on the full 4D RGBA point cloud: power
/// iteration on the 4x4 covariance finds the principal axis, every texel is
/// projected onto it, and the extreme projections become the endpoints. The
/// pair is ordered so `hadd_rgb(e0) <= hadd_rgb(e1)` (`hadd_rgb = r + g + b`,
/// alpha excluded), the orientation the CEM-12 decoder (`cem::rgba_unpack`)
/// reads back directly without its blue-contraction swap. Alpha is carried
/// through unquantized -- CEM 12 stores a real per-endpoint alpha.
///
/// Pure analytic `f64` arithmetic -- no AI/ML path.
pub(super) fn fit_rgba_endpoints(texels: &[[u8; 4]; 16]) -> ([u8; 4], [u8; 4]) {
    let points: [[f64; 4]; 16] =
        core::array::from_fn(|t| core::array::from_fn(|c| f64::from(texels[t][c])));

    // Mean colour.
    let mut mean = [0.0f64; 4];
    for p in &points {
        for c in 0..4 {
            mean[c] += p[c];
        }
    }
    for m in &mut mean {
        *m /= 16.0;
    }

    // 4x4 covariance.
    let mut cov = [[0.0f64; 4]; 4];
    for p in &points {
        let d: [f64; 4] = core::array::from_fn(|c| p[c] - mean[c]);
        for (i, row) in cov.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell += d[i] * d[j];
            }
        }
    }

    // Dominant eigenvector via power iteration, seeded with the largest-norm
    // covariance column (see [`seed_axis3`] for why a fixed [1,1,1,1] seed can
    // be orthogonal to the variance axis and collapse to zero).
    let mut axis = seed_axis4(&cov);
    for _ in 0..24 {
        let next: [f64; 4] =
            core::array::from_fn(|i| (0..4).map(|j| cov[i][j] * axis[j]).sum::<f64>());
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
        let proj = (0..4).map(|c| (p[c] - mean[c]) * axis[c]).sum::<f64>();
        if proj < min_proj {
            min_proj = proj;
            lo = *p;
        }
        if proj > max_proj {
            max_proj = proj;
            hi = *p;
        }
    }

    let a = round_rgba(lo);
    let b = round_rgba(hi);

    // Orient so hadd_rgb(e0) <= hadd_rgb(e1): the CEM-12 decoder uncontracts
    // and swaps otherwise. Alpha does not participate in the ordering key.
    if hadd_rgba(a) > hadd_rgba(b) {
        (b, a)
    } else {
        (a, b)
    }
}

/// Round and clamp an `f64` RGBA quad to 8-bit UNORM.
fn round_rgba(c: [f64; 4]) -> [u8; 4] {
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

/// `r + g + b` of an RGBA endpoint (alpha excluded), the ordering key the
/// CEM-12 decoder uses for its blue-contraction swap.
fn hadd_rgba(c: [u8; 4]) -> u32 {
    u32::from(c[0]) + u32::from(c[1]) + u32::from(c[2])
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

/// Seed the 3x3 power iteration with the covariance column of largest norm.
///
/// A fixed `[1, 1, 1]` seed collapses to zero whenever it is orthogonal to the
/// dominant eigenvector (e.g. a red/blue split whose variance axis is
/// `(1, 0, -1)`). For a symmetric PSD covariance the column of greatest norm
/// always carries a non-zero component along the dominant eigenvector, so it is
/// a robust starting vector. Degenerate (near-constant) blocks fall back to the
/// classic `[1, 1, 1]` seed; their projections are ~0 regardless, so the
/// endpoints collapse to the constant colour either way.
fn seed_axis3(cov: &[[f64; 3]; 3]) -> [f64; 3] {
    let mut best_col = 0usize;
    let mut best_norm2 = f64::NEG_INFINITY;
    for j in 0..3 {
        let norm2 = (0..3).map(|i| cov[i][j] * cov[i][j]).sum::<f64>();
        if norm2 > best_norm2 {
            best_norm2 = norm2;
            best_col = j;
        }
    }
    if best_norm2 < 1e-9 {
        return [1.0, 1.0, 1.0];
    }
    let norm = best_norm2.sqrt();
    core::array::from_fn(|i| cov[i][best_col] / norm)
}

/// Seed the 4x4 power iteration with the covariance column of largest norm.
///
/// See [`seed_axis3`] for the rationale; the 4D fallback is `[1, 1, 1, 1]`.
fn seed_axis4(cov: &[[f64; 4]; 4]) -> [f64; 4] {
    let mut best_col = 0usize;
    let mut best_norm2 = f64::NEG_INFINITY;
    for j in 0..4 {
        let norm2 = (0..4).map(|i| cov[i][j] * cov[i][j]).sum::<f64>();
        if norm2 > best_norm2 {
            best_norm2 = norm2;
            best_col = j;
        }
    }
    if best_norm2 < 1e-9 {
        return [1.0, 1.0, 1.0, 1.0];
    }
    let norm = best_norm2.sqrt();
    core::array::from_fn(|i| cov[i][best_col] / norm)
}
