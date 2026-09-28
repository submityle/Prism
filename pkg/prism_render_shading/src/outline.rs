//! Backend-neutral CPU golden for NPR silhouette/crease outlines.
//!
//! A stylized "ink" outline is a *screen-space* decision: for each shaded
//! pixel, inspect a small cross of neighbours in the geometry buffer and raise
//! an edge wherever the surface is discontinuous. Three independent
//! discontinuities feed the outline, mirroring the classic UE post-process
//! outline / Freistil edge stack:
//!
//! * **Material-id / outline-id boundaries** — any neighbour carrying a
//!   different outline id is a hard silhouette between two authored objects or
//!   material zones (the vis-buffer material id doubles as this key on device).
//! * **Depth discontinuities** — a large *relative* linear-depth jump is a
//!   silhouette against the background or a self-occluding fold; measuring the
//!   jump relative to the centre depth keeps the outline a stable screen-space
//!   width regardless of camera distance.
//! * **Normal discontinuities** — a sharp normal turn with no depth jump is an
//!   interior crease (the corner of a cube, the seam of a sleeve).
//!
//! The three edges combine as a union (the strongest wins), so the routine
//! returns a single `[0, 1]` outline coverage the resolve/composite pass can
//! `mix` toward the authored line colour. This module owns only the math; the
//! render-world pass that samples the cross and applies the colour lives in
//! `prism_render_scene` and its GPU twin in `shaders/outline.wesl` mirrors this
//! file arm-for-arm.
//!
//! The kernel is a fixed four-neighbour cross (left/right/up/down) so the CPU
//! golden and the WESL twin evaluate the *same* reduction; a Roberts diagonal
//! variant can be layered later without changing this contract.

/// One neighbour (or centre) tap of the geometry buffer the outline reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutlineGeometry {
    /// Authored outline / material id. Equal ids never raise an id edge; any
    /// difference is a hard silhouette when [`OutlineParams::id_edges`] is on.
    pub outline_id: u32,
    /// View-space linear depth (positive, growing away from the camera). Use a
    /// non-positive value for "no surface" taps (sky / cleared background); the
    /// depth term treats those as a maximal discontinuity so silhouettes against
    /// the sky are drawn.
    pub linear_depth: f32,
    /// World- or view-space unit surface normal at the tap.
    pub normal: [f32; 3],
}

impl Default for OutlineGeometry {
    fn default() -> Self {
        Self {
            outline_id: 0,
            linear_depth: 1.0,
            normal: [0.0, 0.0, 1.0],
        }
    }
}

/// Tunable thresholds for the three outline discontinuities.
///
/// The [`Default`] value is a sensible general-purpose detector: 5% relative
/// depth jumps and ~37 degree normal turns begin an edge, and material-id
/// boundaries draw a hard line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutlineParams {
    /// Relative depth jump (`|d - d_n| / max(d, eps)`) where the depth edge
    /// begins to open.
    pub depth_threshold: f32,
    /// Half-width of the depth-edge transition; `0` gives a hard step.
    pub depth_softness: f32,
    /// Normal-turn threshold expressed as `1 - dot(n, n_n)` where the crease
    /// edge begins (`0` = coplanar, `1` = perpendicular, `2` = opposed).
    pub normal_threshold: f32,
    /// Half-width of the crease-edge transition; `0` gives a hard step.
    pub normal_softness: f32,
    /// When `true`, any neighbour with a different [`OutlineGeometry::outline_id`]
    /// contributes a full-strength edge.
    pub id_edges: bool,
}

impl Default for OutlineParams {
    fn default() -> Self {
        Self {
            depth_threshold: 0.05,
            depth_softness: 0.05,
            normal_threshold: 0.2,
            normal_softness: 0.2,
            id_edges: true,
        }
    }
}

/// Guards depth normalisation against a zero/near-zero centre depth.
const DEPTH_EPSILON: f32 = 1.0e-4;

/// Self-contained `smoothstep` matching the WESL twin bit-for-bit.
///
/// On a collapsed or inverted window (`edge1 <= edge0`) it degrades to a hard
/// step at `edge0` rather than dividing by zero, so `softness == 0` yields a
/// crisp edge in both the CPU golden and the shader.
fn outline_smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge1 <= edge0 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Material-id boundary edge over the four-neighbour cross.
///
/// Returns `1.0` if id edges are enabled and any neighbour carries a different
/// outline id, otherwise `0.0`.
#[must_use]
pub fn outline_id_edge(center: u32, neighbors: [u32; 4], id_edges: bool) -> f32 {
    if !id_edges {
        return 0.0;
    }
    let mut differs = false;
    let mut i = 0;
    while i < 4 {
        differs |= neighbors[i] != center;
        i += 1;
    }
    if differs {
        1.0
    } else {
        0.0
    }
}

/// Relative-depth discontinuity edge over the four-neighbour cross.
///
/// The largest `|d - d_n| / max(d, eps)` across the cross is re-shaped through
/// `smoothstep(threshold, threshold + softness, .)`. Neighbours with a
/// non-positive depth (no surface / sky) are treated as a maximal jump so the
/// silhouette against the background is always drawn; a non-positive *centre*
/// depth is likewise a background pixel and never draws its own outline.
#[must_use]
pub fn outline_depth_edge(center: f32, neighbors: [f32; 4], params: &OutlineParams) -> f32 {
    if center <= 0.0 {
        return 0.0;
    }
    let denom = center.max(DEPTH_EPSILON);
    let mut max_rel = 0.0f32;
    let mut i = 0;
    while i < 4 {
        let n = neighbors[i];
        let rel = if n <= 0.0 {
            // Sky / cleared neighbour: force the edge fully open.
            f32::INFINITY
        } else {
            (center - n).abs() / denom
        };
        if rel > max_rel {
            max_rel = rel;
        }
        i += 1;
    }
    if max_rel.is_infinite() {
        return 1.0;
    }
    let half = params.depth_softness.max(0.0);
    outline_smoothstep(
        params.depth_threshold,
        params.depth_threshold + half,
        max_rel,
    )
}

/// Normal-turn (interior crease) edge over the four-neighbour cross.
///
/// The sharpest turn `1 - dot(n, n_n)` across the cross is re-shaped through
/// `smoothstep(threshold, threshold + softness, .)`. Inputs are normalised
/// defensively so unnormalised taps cannot push the cosine outside `[-1, 1]`.
#[must_use]
pub fn outline_normal_edge(
    center: [f32; 3],
    neighbors: [[f32; 3]; 4],
    params: &OutlineParams,
) -> f32 {
    let cn = normalize_or_z(center);
    let mut max_turn = 0.0f32;
    let mut i = 0;
    while i < 4 {
        let nn = normalize_or_z(neighbors[i]);
        let turn = 1.0 - dot3(cn, nn).clamp(-1.0, 1.0);
        if turn > max_turn {
            max_turn = turn;
        }
        i += 1;
    }
    let half = params.normal_softness.max(0.0);
    outline_smoothstep(
        params.normal_threshold,
        params.normal_threshold + half,
        max_turn,
    )
}

fn normalize_or_z(v: [f32; 3]) -> [f32; 3] {
    let len2 = dot3(v, v);
    if len2 > 1.0e-12 {
        let inv = 1.0 / len2.sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Evaluates the combined outline coverage for one pixel.
///
/// Samples the id, depth and normal edges over the four-neighbour cross and
/// returns their union (the strongest edge wins) clamped to `[0, 1]`. A result
/// of `0` leaves the pixel untouched; `1` fully replaces it with the authored
/// line colour. Background (non-positive centre depth) pixels never outline
/// themselves, but their neighbours will outline against them.
#[must_use]
pub fn evaluate_outline(
    center: OutlineGeometry,
    neighbors: [OutlineGeometry; 4],
    params: &OutlineParams,
) -> f32 {
    let ids = [
        neighbors[0].outline_id,
        neighbors[1].outline_id,
        neighbors[2].outline_id,
        neighbors[3].outline_id,
    ];
    let depths = [
        neighbors[0].linear_depth,
        neighbors[1].linear_depth,
        neighbors[2].linear_depth,
        neighbors[3].linear_depth,
    ];
    let normals = [
        neighbors[0].normal,
        neighbors[1].normal,
        neighbors[2].normal,
        neighbors[3].normal,
    ];

    // Background centre: no self outline, but still let an id/depth edge form
    // so a foreground silhouette sampled from a background centre is not lost.
    let id_edge = outline_id_edge(center.outline_id, ids, params.id_edges);
    let depth_edge = outline_depth_edge(center.linear_depth, depths, params);
    let normal_edge = outline_normal_edge(center.normal, normals, params);

    id_edge.max(depth_edge).max(normal_edge).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLAT: [f32; 3] = [0.0, 0.0, 1.0];

    fn flat(id: u32, depth: f32) -> OutlineGeometry {
        OutlineGeometry {
            outline_id: id,
            linear_depth: depth,
            normal: FLAT,
        }
    }

    #[test]
    fn uniform_neighbourhood_has_no_edge() {
        let c = flat(7, 5.0);
        let n = [c, c, c, c];
        assert_eq!(evaluate_outline(c, n, &OutlineParams::default()), 0.0);
    }

    #[test]
    fn differing_id_draws_a_hard_edge() {
        let c = flat(7, 5.0);
        let n = [flat(9, 5.0), c, c, c];
        assert_eq!(evaluate_outline(c, n, &OutlineParams::default()), 1.0);
    }

    #[test]
    fn id_edges_can_be_disabled() {
        let p = OutlineParams { id_edges: false, ..Default::default() };
        assert_eq!(outline_id_edge(7, [9, 9, 9, 9], p.id_edges), 0.0);
        assert_eq!(outline_id_edge(7, [9, 9, 9, 9], true), 1.0);
    }

    #[test]
    fn large_relative_depth_jump_opens_the_depth_edge() {
        let p = OutlineParams::default();
        // Centre at 5.0, one neighbour at 10.0 => rel = 1.0 >> threshold+soft.
        let e = outline_depth_edge(5.0, [10.0, 5.0, 5.0, 5.0], &p);
        assert!((e - 1.0).abs() < 1.0e-6, "far jump must fully open: {e}");
        // A jump below threshold stays closed.
        let small = outline_depth_edge(5.0, [5.0 * 1.01, 5.0, 5.0, 5.0], &p);
        assert_eq!(small, 0.0);
    }

    #[test]
    fn depth_edge_is_relative_so_it_is_distance_stable() {
        let p = OutlineParams::default();
        // Same 20% relative jump near and far => identical edge strength.
        let near = outline_depth_edge(2.0, [2.4, 2.0, 2.0, 2.0], &p);
        let far = outline_depth_edge(200.0, [240.0, 200.0, 200.0, 200.0], &p);
        assert!((near - far).abs() < 1.0e-6, "near {near} vs far {far}");
        assert!(near > 0.0);
    }

    #[test]
    fn sky_neighbour_is_a_full_silhouette() {
        let p = OutlineParams::default();
        // Non-positive neighbour depth (sky) forces the edge fully open.
        assert_eq!(outline_depth_edge(5.0, [0.0, 5.0, 5.0, 5.0], &p), 1.0);
        assert_eq!(outline_depth_edge(5.0, [-1.0, 5.0, 5.0, 5.0], &p), 1.0);
    }

    #[test]
    fn background_centre_never_self_outlines() {
        let p = OutlineParams::default();
        // Centre is sky: even wild neighbours give no depth self-edge.
        assert_eq!(outline_depth_edge(0.0, [1.0, 100.0, 5.0, 9.0], &p), 0.0);
        let c = OutlineGeometry {
            outline_id: 0,
            linear_depth: 0.0,
            normal: FLAT,
        };
        // Same id as neighbours, depth term suppressed => no outline on the sky.
        let n = [flat(0, 5.0), flat(0, 5.0), flat(0, 5.0), flat(0, 5.0)];
        assert_eq!(evaluate_outline(c, n, &p), 0.0);
    }

    #[test]
    fn sharp_normal_turn_opens_the_crease_edge() {
        let p = OutlineParams::default();
        // 90-degree turn => 1 - dot = 1.0, well past threshold+soft.
        let e = outline_normal_edge(FLAT, [[1.0, 0.0, 0.0], FLAT, FLAT, FLAT], &p);
        assert!((e - 1.0).abs() < 1.0e-6, "perp normals must fully open: {e}");
        // A tiny tilt below threshold stays closed.
        let tilt = normalize_or_z([0.02, 0.0, 1.0]);
        let small = outline_normal_edge(FLAT, [tilt, FLAT, FLAT, FLAT], &p);
        assert_eq!(small, 0.0);
    }

    #[test]
    fn crease_survives_without_a_depth_jump() {
        // Cube corner: same id and depth, but a hard normal turn.
        let p = OutlineParams::default();
        let c = OutlineGeometry {
            outline_id: 3,
            linear_depth: 4.0,
            normal: FLAT,
        };
        let side = OutlineGeometry {
            outline_id: 3,
            linear_depth: 4.0,
            normal: [1.0, 0.0, 0.0],
        };
        assert!((evaluate_outline(c, [side, c, c, c], &p) - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn union_takes_the_strongest_edge() {
        let p = OutlineParams::default();
        // Weak depth edge alone.
        let weak_depth = outline_depth_edge(5.0, [5.0 * 1.075, 5.0, 5.0, 5.0], &p);
        assert!(weak_depth > 0.0 && weak_depth < 1.0, "weak: {weak_depth}");
        // Combined with a hard id edge => union pins to 1.0.
        let c = flat(1, 5.0);
        let n = [
            OutlineGeometry {
                outline_id: 2,
                linear_depth: 5.0 * 1.075,
                normal: FLAT,
            },
            c,
            c,
            c,
        ];
        assert_eq!(evaluate_outline(c, n, &p), 1.0);
    }

    #[test]
    fn soft_edge_is_monotonic_across_the_terminator() {
        let p = OutlineParams { depth_softness: 0.2, ..Default::default() };
        let mut prev = -1.0f32;
        for i in 0..=20 {
            let rel = i as f32 / 20.0; // 0..1 relative jump
            let e = outline_depth_edge(1.0, [1.0 + rel, 1.0, 1.0, 1.0], &p);
            assert!(e >= prev - 1.0e-6, "non-monotonic at {rel}: {prev} -> {e}");
            prev = e;
        }
        assert!((prev - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn hard_depth_edge_when_softness_zero() {
        let p = OutlineParams { depth_softness: 0.0, ..Default::default() };
        // Just below threshold => closed; just above => open.
        assert_eq!(outline_depth_edge(1.0, [1.049, 1.0, 1.0, 1.0], &p), 0.0);
        assert_eq!(outline_depth_edge(1.0, [1.051, 1.0, 1.0, 1.0], &p), 1.0);
    }
}
