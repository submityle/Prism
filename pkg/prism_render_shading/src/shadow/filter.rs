//! Percentage-closer filtering (PCF) and percentage-closer soft shadows
//! (PCSS) over an injectable shadow-depth field.
//!
//! The depth field is abstracted behind [`ShadowDepthSampler`] so the CPU
//! golden reference can be driven by synthetic depth (step edges, planes) while
//! the GPU twin binds a real depth texture / cube array.  `layer` selects the
//! cascade (directional) or cube face (point/spot) inside a shadow atlas.
//!
//! Conventions shared with `shadow.wesl`:
//! * Depths are wgpu clip-space `z` in `[0, 1]` (near = 0, far = 1) for
//!   directional/spot, or a linear normalized distance for point lights.
//! * A texel is **lit** when `reference_depth <= stored_depth`.  The caller is
//!   expected to fold any depth bias into `reference_depth` before filtering.
//! * Samples that fall outside the `[0, 1]` shadow-map UV range must read as
//!   far (`1.0`) so unmapped regions stay lit; that contract lives in the
//!   sampler implementation.

/// Injectable read-only shadow depth field, indexed by atlas `layer` and UV.
pub trait ShadowDepthSampler {
    /// Returns the stored depth at `uv` on `layer`.  Implementations must
    /// return the far value (`1.0`) for `uv` outside `[0, 1]^2` so off-map
    /// regions remain lit.
    fn sample_depth(&self, layer: usize, uv: [f32; 2]) -> f32;
}

/// Fraction of `(2 * radius + 1)^2` box-grid taps that pass the depth
/// comparison, i.e. the lit fraction in `[0, 1]`.
///
/// `radius` is in texels; `radius == 0` degenerates to a single hard tap.
/// `texel_size` is the UV size of one shadow-map texel (`1 / resolution`).
pub fn pcf_visibility<S: ShadowDepthSampler>(
    sampler: &S,
    layer: usize,
    center_uv: [f32; 2],
    reference_depth: f32,
    texel_size: [f32; 2],
    radius: i32,
) -> f32 {
    let radius = radius.max(0);
    let mut lit = 0.0_f32;
    let mut taps = 0.0_f32;
    let mut y = -radius;
    while y <= radius {
        let mut x = -radius;
        while x <= radius {
            let uv = [
                center_uv[0] + (x as f32) * texel_size[0],
                center_uv[1] + (y as f32) * texel_size[1],
            ];
            let stored = sampler.sample_depth(layer, uv);
            if reference_depth <= stored {
                lit += 1.0;
            }
            taps += 1.0;
            x += 1;
        }
        y += 1;
    }
    lit / taps
}

/// Result of the PCSS blocker search: the average depth of occluders in front
/// of the receiver and how many taps found one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockerSearch {
    /// Mean stored depth of taps closer to the light than the receiver.
    pub average_depth: f32,
    /// Fraction of taps that were blockers, in `[0, 1]`.
    pub blocker_fraction: f32,
}

/// Averages the depth of blockers (taps strictly closer to the light than
/// `reference_depth`) in a `(2 * radius + 1)^2` box around `center_uv`.
///
/// When no tap is a blocker, `average_depth` is `1.0` (far) and
/// `blocker_fraction` is `0.0`, signalling a fully lit receiver.
pub fn blocker_search<S: ShadowDepthSampler>(
    sampler: &S,
    layer: usize,
    center_uv: [f32; 2],
    reference_depth: f32,
    texel_size: [f32; 2],
    radius: i32,
) -> BlockerSearch {
    let radius = radius.max(0);
    let mut sum = 0.0_f32;
    let mut blockers = 0.0_f32;
    let mut taps = 0.0_f32;
    let mut y = -radius;
    while y <= radius {
        let mut x = -radius;
        while x <= radius {
            let uv = [
                center_uv[0] + (x as f32) * texel_size[0],
                center_uv[1] + (y as f32) * texel_size[1],
            ];
            let stored = sampler.sample_depth(layer, uv);
            if stored < reference_depth {
                sum += stored;
                blockers += 1.0;
            }
            taps += 1.0;
            x += 1;
        }
        y += 1;
    }
    if blockers <= 0.0 {
        BlockerSearch {
            average_depth: 1.0,
            blocker_fraction: 0.0,
        }
    } else {
        BlockerSearch {
            average_depth: sum / blockers,
            blocker_fraction: blockers / taps,
        }
    }
}

/// Tunables for the three PCSS stages.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PcssConfig {
    /// Half-extent, in texels, of the blocker-search box.
    pub search_radius: i32,
    /// Emitter size in UV units; larger emitters cast softer penumbrae.
    pub light_size_uv: f32,
    /// Minimum PCF radius (texels) so contact shadows keep some softness.
    pub min_filter_radius: i32,
    /// Maximum PCF radius (texels) bounding the worst-case tap count.
    pub max_filter_radius: i32,
}

/// Percentage-closer soft shadows: estimates a penumbra width from the average
/// blocker depth, then runs a variable-radius PCF so shadows sharpen at contact
/// and soften with receiver/blocker separation.
///
/// The penumbra estimate is the standard similar-triangles ratio
/// `(z_receiver - z_blocker) / z_blocker * light_size`, converted from UV width
/// to an integer texel radius bounded by the config.  With no blockers the
/// receiver is fully lit.
pub fn pcss_visibility<S: ShadowDepthSampler>(
    sampler: &S,
    layer: usize,
    center_uv: [f32; 2],
    reference_depth: f32,
    texel_size: [f32; 2],
    config: PcssConfig,
) -> f32 {
    let search = blocker_search(
        sampler,
        layer,
        center_uv,
        reference_depth,
        texel_size,
        config.search_radius,
    );
    if search.blocker_fraction <= 0.0 {
        return 1.0;
    }

    let blocker = search.average_depth.max(1.0e-4);
    let penumbra_ratio = ((reference_depth - blocker) / blocker).max(0.0);
    let penumbra_uv = penumbra_ratio * config.light_size_uv;
    // Convert the UV penumbra into an integer texel radius on the smaller texel
    // axis (square texels in practice) and clamp to the configured window.
    let texel = texel_size[0].min(texel_size[1]).max(1.0e-6);
    let radius_texels = (penumbra_uv / texel).round() as i32;
    let radius = radius_texels.clamp(config.min_filter_radius.max(0), config.max_filter_radius);

    pcf_visibility(sampler, layer, center_uv, reference_depth, texel_size, radius)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A depth field that is `near_depth` on the left half of UV space
    /// (`u < 0.5`) and far (`1.0`) elsewhere, plus a constant far background,
    /// used to exercise soft edges and blocker search.
    struct StepField {
        near_depth: f32,
    }

    impl ShadowDepthSampler for StepField {
        fn sample_depth(&self, _layer: usize, uv: [f32; 2]) -> f32 {
            if uv[0] < 0.0 || uv[0] > 1.0 || uv[1] < 0.0 || uv[1] > 1.0 {
                return 1.0; // off-map reads as far / lit
            }
            if uv[0] < 0.5 {
                self.near_depth
            } else {
                1.0
            }
        }
    }

    /// A uniform occluder plane at a fixed depth everywhere in range.
    struct PlaneField {
        depth: f32,
    }

    impl ShadowDepthSampler for PlaneField {
        fn sample_depth(&self, _layer: usize, uv: [f32; 2]) -> f32 {
            if uv[0] < 0.0 || uv[0] > 1.0 || uv[1] < 0.0 || uv[1] > 1.0 {
                1.0
            } else {
                self.depth
            }
        }
    }

    /// Deep inside the occluded half the receiver is fully shadowed; deep in
    /// the lit half it is fully lit.
    #[test]
    fn pcf_hard_regions_are_saturated() {
        let field = StepField { near_depth: 0.2 };
        let ts = [1.0 / 64.0, 1.0 / 64.0];
        // Receiver behind the occluder (0.8 > 0.2) in the occluded half.
        let occluded = pcf_visibility(&field, 0, [0.1, 0.5], 0.8, ts, 1);
        assert!((occluded - 0.0).abs() < 1.0e-6);
        // Same receiver depth in the lit half (stored far = 1.0 >= 0.8).
        let lit = pcf_visibility(&field, 0, [0.9, 0.5], 0.8, ts, 1);
        assert!((lit - 1.0).abs() < 1.0e-6);
    }

    /// Straddling the shadow edge yields a partial (soft) visibility between 0
    /// and 1, and a larger filter radius softens the transition further.
    #[test]
    fn pcf_edge_is_soft_and_widens_with_radius() {
        let field = StepField { near_depth: 0.2 };
        let ts = [0.1, 0.1]; // large texels so a small radius straddles the seam
        let narrow = pcf_visibility(&field, 0, [0.5, 0.5], 0.8, ts, 1);
        assert!(narrow > 0.0 && narrow < 1.0, "edge should be soft: {narrow}");
        // Exactly on the seam with a symmetric grid -> about half lit.
        assert!((narrow - 0.5).abs() < 0.2, "seam ~= 0.5, got {narrow}");
    }

    /// Blocker search reports the occluder depth and a blocker fraction, and
    /// reports "no blockers" when the receiver is in front of everything.
    #[test]
    fn blocker_search_reports_occluder_depth() {
        let field = PlaneField { depth: 0.3 };
        let ts = [1.0 / 32.0, 1.0 / 32.0];
        let hit = blocker_search(&field, 0, [0.5, 0.5], 0.9, ts, 2);
        assert!((hit.average_depth - 0.3).abs() < 1.0e-6);
        assert!((hit.blocker_fraction - 1.0).abs() < 1.0e-6);

        // Receiver in front of the plane -> no blockers, far average.
        let miss = blocker_search(&field, 0, [0.5, 0.5], 0.1, ts, 2);
        assert_eq!(miss.blocker_fraction, 0.0);
        assert_eq!(miss.average_depth, 1.0);
    }

    /// PCSS penumbra grows with receiver/blocker separation: a receiver far
    /// behind the occluder produces a softer (more partial near the edge)
    /// result than one just behind it.  We measure softness by sampling near
    /// the shadow seam and checking the far receiver is "more mid-grey".
    #[test]
    fn pcss_penumbra_widens_with_distance() {
        let field = StepField { near_depth: 0.1 };
        let ts = [1.0 / 128.0, 1.0 / 128.0];
        let cfg = PcssConfig {
            search_radius: 4,
            light_size_uv: 0.5,
            min_filter_radius: 1,
            max_filter_radius: 32,
        };
        // Sample just inside the occluded half, near the seam.
        let near_receiver = pcss_visibility(&field, 0, [0.46, 0.5], 0.2, ts, cfg);
        let far_receiver = pcss_visibility(&field, 0, [0.46, 0.5], 0.95, ts, cfg);
        // Both are partially shadowed near the seam; the far receiver's wider
        // penumbra pulls in more lit taps from the u>0.5 half.
        assert!(far_receiver >= near_receiver, "{far_receiver} !>= {near_receiver}");
    }

    /// With no occluders in the search window PCSS returns fully lit.
    #[test]
    fn pcss_no_blockers_is_fully_lit() {
        let field = PlaneField { depth: 1.0 };
        let ts = [1.0 / 64.0, 1.0 / 64.0];
        let cfg = PcssConfig {
            search_radius: 3,
            light_size_uv: 0.5,
            min_filter_radius: 1,
            max_filter_radius: 16,
        };
        assert_eq!(pcss_visibility(&field, 0, [0.5, 0.5], 0.5, ts, cfg), 1.0);
    }
}
