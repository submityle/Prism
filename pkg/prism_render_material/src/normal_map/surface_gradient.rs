//! **Surface-gradient (slope-space) normal blending** for an arbitrary number
//! of tangent-space layers (Mikkelsen, "Surface Gradient-Based Bump Mapping
//! Framework", 2020).
//!
//! The ad-hoc tangent-space blends in [`blend`](super::blend) (linear, UDN,
//! whiteout, RNM) combine exactly **two** normals and give subtly different,
//! order-dependent answers. The surface-gradient framework is the modern AAA
//! standard for stacking a base normal with *several* detail / decal / wrinkle
//! layers consistently: each tangent-space unit normal `n` is converted to its
//! **surface gradient** (height-field slope) `g = (-n.x / n.z, -n.y / n.z)`,
//! the per-layer gradients are summed (optionally each scaled by a strength),
//! and the single accumulated gradient is resolved back to a unit normal
//! `normalize(-g.x, -g.y, 1)`. Because the gradients add linearly, the blend is
//! **commutative and associative**, so layer order never changes the result and
//! any number of layers compose in one resolve.
//!
//! A per-layer `strength` scales that layer's gradient: `strength = 1` keeps the
//! layer, `0` drops it (a flat layer contributes a zero gradient and is the
//! identity), `>1` steepens and `<1` flattens the relief, and a negative value
//! mirrors it. Scaling a tangent-space normal with
//! [`scale_strength`](super::scale_strength) multiplies all three components by
//! a common renormalisation factor and so preserves the `-x/z` ratio exactly;
//! hence a `strength` here equals pre-scaling the layer normal with
//! `scale_strength`, which the oracles pin down against that independent code.
//!
//! This reuses the verified [`normal_to_slope`](super::normal_to_slope) /
//! [`slope_to_normal`](super::slope_to_normal) conversions (identical slope
//! convention, including the `nz` floor that keeps a near-tangent normal
//! finite), so the accumulator here only owns the *summation*, not the
//! conversion math.
//!
//! Everything is deterministic analytic `f32` arithmetic -- no AI/ML -- so a CPU
//! golden matches a GPU twin to floating-point tolerance.
//!
//! # References
//! * Mikkelsen, "Surface Gradient-Based Bump Mapping Framework" (2020).
//! * Mikkelsen, "Bump Mapping Unparametrized Surfaces on the GPU" (2010).

use super::strength::{normal_to_slope, slope_to_normal};

/// A tangent-space normal with a strength multiplier for gradient-domain
/// accumulation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalLayer {
    /// Tangent-space unit normal (upper `z >= 0` hemisphere).
    pub normal: [f32; 3],
    /// Gradient scale for this layer (`1.0` keeps it, `0.0` drops it).
    pub strength: f32,
}

impl NormalLayer {
    /// A layer at unit strength.
    #[must_use]
    pub fn new(normal: [f32; 3]) -> Self {
        Self {
            normal,
            strength: 1.0,
        }
    }

    /// A layer with an explicit strength.
    #[must_use]
    pub fn with_strength(normal: [f32; 3], strength: f32) -> Self {
        Self { normal, strength }
    }
}

/// Blend any number of tangent-space normal `layers` in the surface-gradient
/// domain and resolve a single unit normal.
///
/// Each layer's surface gradient (scaled by its `strength`) is summed, then
/// resolved with [`slope_to_normal`](super::slope_to_normal). An empty slice
/// resolves to the flat normal `(0, 0, 1)`.
#[must_use]
pub fn blend_surface_gradient(layers: &[NormalLayer]) -> [f32; 3] {
    let mut g = [0.0f32, 0.0f32];
    for layer in layers {
        let s = normal_to_slope(layer.normal);
        g[0] += layer.strength * s[0];
        g[1] += layer.strength * s[1];
    }
    slope_to_normal(g)
}

/// Convenience base + detail pair blend at unit strength (the common case),
/// equivalent to `blend_surface_gradient(&[base, detail])`.
#[must_use]
pub fn blend_surface_gradient_pair(base: [f32; 3], detail: [f32; 3]) -> [f32; 3] {
    blend_surface_gradient(&[NormalLayer::new(base), NormalLayer::new(detail)])
}

/// Resolve a unit normal directly from an accumulated surface gradient (slope),
/// `normalize(-g.x, -g.y, 1)`.
#[must_use]
pub fn resolve_surface_gradient(gradient: [f32; 2]) -> [f32; 3] {
    slope_to_normal(gradient)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normal_map::{normal_to_slope, scale_strength, slope_to_normal};
    use alloc::vec::Vec;

    const FLAT: [f32; 3] = [0.0, 0.0, 1.0];

    fn unit(v: [f32; 3]) -> [f32; 3] {
        let inv = 1.0 / (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    }

    fn close3(a: [f32; 3], b: [f32; 3]) {
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < 1.0e-5, "a={a:?} b={b:?}");
        }
    }

    fn is_unit(n: [f32; 3]) {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((len - 1.0).abs() < 1.0e-5, "len={len} n={n:?}");
    }

    fn sample_normals() -> Vec<[f32; 3]> {
        [
            unit([0.2, -0.1, 0.95]),
            unit([-0.4, 0.3, 0.8]),
            unit([0.05, 0.6, 0.7]),
            unit([-0.15, -0.5, 0.85]),
        ]
        .to_vec()
    }

    #[test]
    fn single_unit_layer_is_renormalised_input() {
        for n in sample_normals() {
            let out = blend_surface_gradient(&[NormalLayer::new(n)]);
            close3(out, unit(n));
            is_unit(out);
        }
    }

    #[test]
    fn flat_layer_is_identity() {
        for n in sample_normals() {
            // A flat detail contributes a zero gradient.
            let out = blend_surface_gradient_pair(n, FLAT);
            close3(out, unit(n));
            // Zero strength also drops a layer.
            let out2 = blend_surface_gradient(&[
                NormalLayer::new(n),
                NormalLayer::with_strength(unit([0.5, -0.3, 0.8]), 0.0),
            ]);
            close3(out2, unit(n));
        }
    }

    #[test]
    fn empty_resolves_flat() {
        close3(blend_surface_gradient(&[]), FLAT);
    }

    #[test]
    fn pair_matches_independent_slope_sum() {
        // Anti-fake oracle: tie the pair blend to the independently verified
        // normal_to_slope / slope_to_normal conversions summed by separate code.
        let ns = sample_normals();
        for &a in &ns {
            for &b in &ns {
                let want = {
                    let sa = normal_to_slope(a);
                    let sb = normal_to_slope(b);
                    slope_to_normal([sa[0] + sb[0], sa[1] + sb[1]])
                };
                close3(blend_surface_gradient_pair(a, b), want);
            }
        }
    }

    #[test]
    fn strength_equals_scale_strength_prescale() {
        // A per-layer strength must equal pre-scaling the normal with the
        // independently verified scale_strength (which preserves the -x/z slope
        // ratio), summed in the gradient domain.
        let ns = sample_normals();
        for &a in &ns {
            for &b in &ns {
                for s in [0.25f32, 0.5, 1.5, 2.0, -1.0] {
                    let via_strength = blend_surface_gradient(&[
                        NormalLayer::new(a),
                        NormalLayer::with_strength(b, s),
                    ]);
                    let via_prescale = blend_surface_gradient_pair(a, scale_strength(b, s));
                    close3(via_strength, via_prescale);
                }
            }
        }
    }

    #[test]
    fn order_independent() {
        let ns = sample_normals();
        let layers: Vec<NormalLayer> = ns.iter().map(|&n| NormalLayer::new(n)).collect();
        let forward = blend_surface_gradient(&layers);
        let mut rev = layers.clone();
        rev.reverse();
        let backward = blend_surface_gradient(&rev);
        close3(forward, backward);
        is_unit(forward);
    }

    #[test]
    fn associative_accumulation() {
        // Blending all layers at once equals folding the running gradient:
        // ((a) + b) + c ... since gradients add linearly.
        let ns = sample_normals();
        let layers: Vec<NormalLayer> = ns.iter().map(|&n| NormalLayer::new(n)).collect();
        let all = blend_surface_gradient(&layers);

        let folded = {
            let mut acc = normal_to_slope(ns[0]);
            for &n in &ns[1..] {
                let s = normal_to_slope(n);
                acc = [acc[0] + s[0], acc[1] + s[1]];
            }
            slope_to_normal(acc)
        };
        close3(all, folded);
    }

    #[test]
    fn resolve_matches_slope_to_normal() {
        for g in [[0.0f32, 0.0], [0.3, -0.7], [-1.2, 0.4], [2.0, 2.0]] {
            close3(resolve_surface_gradient(g), slope_to_normal(g));
            is_unit(resolve_surface_gradient(g));
        }
    }
}
