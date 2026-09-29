//! Garment asset model: 2D panels stitched into a 3D garment, woven material
//! parameters, and per-vertex paint constraints.
//!
//! This is the authoring layer of the cloth subsystem, mirroring how a
//! production garment is built in `Marvelous` Designer / CLO and then handed to
//! a solver such as UE5 `Chaos` Cloth: flat *panels* are cut, their edges are
//! *seamed* together, a woven *fabric* assigns warp/weft/bend stiffness, and an
//! artist *paints* per-vertex constraints (max-distance, backstop, blend, and
//! anim-drive) onto the mesh.
//!
//! Everything here is pure data plus validation and unit conversion. There is
//! no simulation and no allocation: stiffness is converted to XPBD
//! [`Compliance`] for the constraint graph, and every accessor is total —
//! out-of-range vertex spans yield empty ranges, negative or `NaN` inputs are
//! clamped to safe values, and nothing panics or propagates `NaN`.

use super::Compliance;

/// Stiffness at or below this value is treated as "no stiffness" when
/// converting to compliance, so the reciprocal never divides by (near) zero and
/// never yields an infinite or `NaN` compliance.
const MIN_STIFFNESS: f32 = 1.0e-6;

/// Stable identity of one 2D pattern panel within a garment.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PanelId(pub u32);

/// One flat pattern panel: a contiguous run of sim-mesh vertices.
///
/// A garment is cut from panels; when triangulated into the simulation mesh,
/// each panel owns a contiguous block of sim vertices. Storing that block as a
/// `[start, start + count)` interval keeps panel membership a cheap range test
/// and lets a seam reference whole panels without a per-vertex table.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Panel {
    /// Stable identity of this panel.
    pub id: PanelId,
    /// Index of the panel's first sim-mesh vertex.
    pub sim_vertex_start: u32,
    /// Number of contiguous sim-mesh vertices owned by this panel.
    pub sim_vertex_count: u32,
}

impl Panel {
    /// Builds a panel owning `count` sim vertices starting at `start`.
    #[must_use]
    pub const fn new(id: PanelId, sim_vertex_start: u32, sim_vertex_count: u32) -> Self {
        Self {
            id,
            sim_vertex_start,
            sim_vertex_count,
        }
    }

    /// The half-open sim-vertex range `start..start + count` as `usize`, ready
    /// for slicing the sim-mesh vertex array. The addition is saturating so a
    /// pathological `start + count` overflow yields a clamped, non-panicking
    /// range rather than wrapping.
    #[must_use]
    pub fn vertex_range(self) -> core::ops::Range<usize> {
        let start = self.sim_vertex_start as usize;
        let end = start.saturating_add(self.sim_vertex_count as usize);
        start..end
    }

    /// Returns `true` when the panel owns no vertices.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.sim_vertex_count == 0
    }

    /// Returns `true` when `vertex` falls inside this panel's sim-vertex range.
    #[must_use]
    pub fn contains(self, vertex: u32) -> bool {
        self.vertex_range().contains(&(vertex as usize))
    }
}

/// A stitched seam joining two panel edges into one garment.
///
/// A seam sews an edge of `panel_a` to an edge of `panel_b`; `stitch_count`
/// records how many discrete stitches (paired vertices) the seam contributes,
/// which becomes that many seam constraints in the constraint graph. Panels are
/// referenced by [`PanelId`] rather than by vertex index so the seam survives
/// re-triangulation of either panel.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Seam {
    /// The first panel joined by this seam.
    pub panel_a: PanelId,
    /// The second panel joined by this seam.
    pub panel_b: PanelId,
    /// Number of paired stitches along the seam.
    pub stitch_count: u32,
}

impl Seam {
    /// Builds a seam between two panels with the given stitch count.
    #[must_use]
    pub const fn new(panel_a: PanelId, panel_b: PanelId, stitch_count: u32) -> Self {
        Self {
            panel_a,
            panel_b,
            stitch_count,
        }
    }

    /// Returns `true` when the seam joins a panel to itself (a dart or a folded
    /// closure) rather than two distinct panels.
    #[must_use]
    pub fn is_self_seam(self) -> bool {
        self.panel_a == self.panel_b
    }

    /// Returns `true` when the seam has no stitches and therefore contributes
    /// no constraints.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.stitch_count == 0
    }
}

/// Woven fabric parameters for one garment (or one material zone of it).
///
/// A woven cloth is anisotropic: it resists stretching along the warp and weft
/// yarn directions differently, bends far more easily than it stretches, and
/// has an areal density plus surface friction and aerodynamic drag. The solver
/// consumes stiffness as XPBD [`Compliance`] (`compliance = 1 / stiffness`), so
/// this type both stores the authored stiffness and performs that conversion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FabricMaterial {
    /// Stretch stiffness along the warp (lengthwise) yarns.
    pub warp_stiffness: f32,
    /// Stretch stiffness along the weft (crosswise) yarns.
    pub weft_stiffness: f32,
    /// Bending stiffness resisting out-of-plane folding.
    pub bend_stiffness: f32,
    /// Areal density in kilograms per square metre.
    pub density: f32,
    /// Surface friction coefficient against colliders, in `0..=1`.
    pub friction: f32,
    /// Aerodynamic drag coefficient against wind, non-negative.
    pub drag: f32,
}

impl Default for FabricMaterial {
    /// A plausible mid-weight woven default (a cotton-like fabric): stiff along
    /// both yarn axes, floppy in bending, moderate density and friction.
    fn default() -> Self {
        Self {
            warp_stiffness: 1.0e3,
            weft_stiffness: 1.0e3,
            bend_stiffness: 1.0,
            density: 0.2,
            friction: 0.4,
            drag: 0.1,
        }
    }
}

impl FabricMaterial {
    /// Converts a raw stiffness to XPBD compliance (`1 / stiffness`).
    ///
    /// Stiffness at or below [`MIN_STIFFNESS`] (including negative or `NaN`
    /// inputs, which clamp up to the floor) maps to `1 / MIN_STIFFNESS`, a very
    /// soft but finite value, so the conversion is total and never produces an
    /// infinite or `NaN` compliance.
    #[must_use]
    pub fn stiffness_to_compliance(stiffness: f32) -> Compliance {
        // `max` with a NaN operand returns the non-NaN operand, so this also
        // sanitizes NaN to the stiffness floor before taking the reciprocal.
        let clamped = stiffness.max(MIN_STIFFNESS);
        Compliance(1.0 / clamped)
    }

    /// XPBD compliance for warp-direction stretch constraints.
    #[must_use]
    pub fn warp_compliance(self) -> Compliance {
        Self::stiffness_to_compliance(self.warp_stiffness)
    }

    /// XPBD compliance for weft-direction stretch constraints.
    #[must_use]
    pub fn weft_compliance(self) -> Compliance {
        Self::stiffness_to_compliance(self.weft_stiffness)
    }

    /// XPBD compliance for bending constraints.
    #[must_use]
    pub fn bend_compliance(self) -> Compliance {
        Self::stiffness_to_compliance(self.bend_stiffness)
    }

    /// Returns a copy with every parameter forced into its physically valid
    /// range: stiffnesses, density, and drag are made non-negative and
    /// friction is clamped to `0..=1`. Because `f32::max`/`f32::min` propagate
    /// the non-`NaN` operand, any `NaN` input collapses to the relevant bound,
    /// so the result is always finite and [`FabricMaterial::is_valid`].
    #[must_use]
    #[expect(
        clippy::manual_clamp,
        reason = "The max/min chain is deliberate: unlike f32::clamp, it collapses a NaN input to a bound instead of propagating NaN, which is what sanitization requires."
    )]
    pub fn sanitized(self) -> Self {
        Self {
            warp_stiffness: self.warp_stiffness.max(0.0),
            weft_stiffness: self.weft_stiffness.max(0.0),
            bend_stiffness: self.bend_stiffness.max(0.0),
            density: self.density.max(0.0),
            friction: self.friction.max(0.0).min(1.0),
            drag: self.drag.max(0.0),
        }
    }

    /// Returns `true` when every parameter is finite, all magnitudes are
    /// non-negative, and friction lies in `0..=1`.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.warp_stiffness.is_finite()
            && self.weft_stiffness.is_finite()
            && self.bend_stiffness.is_finite()
            && self.density.is_finite()
            && self.friction.is_finite()
            && self.drag.is_finite()
            && self.warp_stiffness >= 0.0
            && self.weft_stiffness >= 0.0
            && self.bend_stiffness >= 0.0
            && self.density >= 0.0
            && self.drag >= 0.0
            && (0.0..=1.0).contains(&self.friction)
    }
}

/// Per-vertex artist-painted simulation constraints.
///
/// Following UE5 `Chaos` Cloth, an artist paints scalar weight maps over the
/// garment that locally steer the solve without editing the mesh:
///
/// * `max_distance` caps how far a vertex may drift from its skinned pose
///   (`0` pins it to the skin, larger values free it to simulate);
/// * `backstop` pushes a vertex out along its normal to keep cloth off the body;
/// * `blend_weight` mixes the simulated position with the skinned position
///   (`1` fully simulated, `0` fully skinned);
/// * `anim_drive` pulls a vertex toward its animated target position.
///
/// All four are bounded scalars; [`PaintedConstraint::clamped`] enforces the
/// bounds and [`PaintedConstraint::is_valid`] checks them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PaintedConstraint {
    /// Maximum allowed drift from the skinned pose, in metres, non-negative.
    pub max_distance: f32,
    /// Backstop push-out distance keeping cloth off the body, non-negative.
    pub backstop: f32,
    /// Simulated-vs-skinned blend weight, in `0..=1`.
    pub blend_weight: f32,
    /// Strength of the pull toward the animated target, in `0..=1`.
    pub anim_drive: f32,
}

impl Default for PaintedConstraint {
    /// A fully free, fully simulated vertex with no anim drive and no backstop.
    fn default() -> Self {
        Self {
            max_distance: f32::INFINITY,
            backstop: 0.0,
            blend_weight: 1.0,
            anim_drive: 0.0,
        }
    }
}

impl PaintedConstraint {
    /// Builds a painted constraint from raw painted weights (unclamped); call
    /// [`PaintedConstraint::clamped`] before handing it to the solver.
    #[must_use]
    pub const fn new(max_distance: f32, backstop: f32, blend_weight: f32, anim_drive: f32) -> Self {
        Self {
            max_distance,
            backstop,
            blend_weight,
            anim_drive,
        }
    }

    /// Returns a copy with every weight forced into its valid range:
    /// `max_distance` and `backstop` made non-negative, and `blend_weight` and
    /// `anim_drive` clamped to `0..=1`. As with [`FabricMaterial::sanitized`],
    /// `NaN` inputs collapse to a bound, so the result is finite and
    /// [`PaintedConstraint::is_valid`].
    ///
    /// A non-finite `max_distance` (the "no cap" default of `+inf`) is preserved
    /// as `+inf` rather than being forced finite, so an unpainted vertex stays
    /// unconstrained; only a `NaN` collapses to `0`.
    #[must_use]
    #[expect(
        clippy::manual_clamp,
        reason = "The max/min chain is deliberate: unlike f32::clamp, it collapses a NaN input to a bound instead of propagating NaN, which is what clamping painted weights requires."
    )]
    pub fn clamped(self) -> Self {
        let max_distance = if self.max_distance.is_nan() {
            0.0
        } else {
            self.max_distance.max(0.0)
        };
        Self {
            max_distance,
            backstop: self.backstop.max(0.0),
            blend_weight: self.blend_weight.max(0.0).min(1.0),
            anim_drive: self.anim_drive.max(0.0).min(1.0),
        }
    }

    /// Returns `true` when `blend_weight`, `anim_drive`, `backstop`, and a
    /// finite `max_distance` are all present and within their bounds.
    ///
    /// `max_distance` may be `+inf` (an uncapped vertex) and still be valid; any
    /// `NaN`, negative distance, or out-of-`0..=1` weight is rejected.
    #[must_use]
    pub fn is_valid(self) -> bool {
        !self.max_distance.is_nan()
            && self.max_distance >= 0.0
            && self.backstop.is_finite()
            && self.backstop >= 0.0
            && (0.0..=1.0).contains(&self.blend_weight)
            && (0.0..=1.0).contains(&self.anim_drive)
    }

    /// Returns `true` when the vertex is effectively pinned to the skinned pose:
    /// it may not drift at all (`max_distance == 0`) and is not simulated.
    #[must_use]
    pub fn is_pinned_to_skin(self) -> bool {
        self.blend_weight <= 0.0 || self.max_distance <= 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compliance and stiffness compare equal only through their reciprocals;
    /// this epsilon keeps the float assertions off exact `==`.
    const EPS: f32 = 1.0e-4;

    #[test]
    fn panel_vertex_range_and_membership() {
        let panel = Panel::new(PanelId(7), 10, 4);
        assert_eq!(panel.vertex_range(), 10..14);
        assert!(!panel.is_empty());
        assert!(panel.contains(10));
        assert!(panel.contains(13));
        assert!(!panel.contains(14));
        assert!(!panel.contains(9));
    }

    #[test]
    fn panel_empty_and_overflow_are_bounds_safe() {
        let empty = Panel::new(PanelId(0), 5, 0);
        assert!(empty.is_empty());
        assert!(empty.vertex_range().is_empty());
        assert!(!empty.contains(5));

        // A pathological span saturates instead of wrapping or panicking.
        let huge = Panel::new(PanelId(1), u32::MAX, u32::MAX);
        let range = huge.vertex_range();
        assert!(range.start <= range.end);
    }

    #[test]
    fn seam_self_and_empty_detection() {
        let cross = Seam::new(PanelId(1), PanelId(2), 8);
        assert!(!cross.is_self_seam());
        assert!(!cross.is_empty());

        let dart = Seam::new(PanelId(3), PanelId(3), 4);
        assert!(dart.is_self_seam());

        let slack = Seam::new(PanelId(1), PanelId(2), 0);
        assert!(slack.is_empty());
    }

    #[test]
    fn stiffness_maps_to_reciprocal_compliance() {
        let stiff = FabricMaterial::stiffness_to_compliance(1000.0);
        assert!((stiff.value() - 0.001).abs() < EPS);

        // Zero / negative / NaN stiffness all clamp to the soft ceiling.
        let ceiling = 1.0 / MIN_STIFFNESS;
        let soft = FabricMaterial::stiffness_to_compliance(0.0);
        assert!((soft.value() - ceiling).abs() < 1.0);
        let neg = FabricMaterial::stiffness_to_compliance(-5.0);
        assert!((neg.value() - ceiling).abs() < 1.0);
        let nan = FabricMaterial::stiffness_to_compliance(f32::NAN);
        assert!(nan.value().is_finite());
    }

    #[test]
    fn fabric_axis_compliance_uses_each_stiffness() {
        let fabric = FabricMaterial {
            warp_stiffness: 2000.0,
            weft_stiffness: 500.0,
            bend_stiffness: 10.0,
            ..FabricMaterial::default()
        };
        assert!((fabric.warp_compliance().value() - 0.0005).abs() < EPS);
        assert!((fabric.weft_compliance().value() - 0.002).abs() < EPS);
        assert!((fabric.bend_compliance().value() - 0.1).abs() < EPS);
    }

    #[test]
    fn fabric_sanitized_clamps_negatives_and_nan() {
        let dirty = FabricMaterial {
            warp_stiffness: -100.0,
            weft_stiffness: f32::NAN,
            bend_stiffness: -1.0,
            density: -0.5,
            friction: 3.0,
            drag: -2.0,
        };
        let clean = dirty.sanitized();
        assert!(clean.is_valid());
        assert!((clean.warp_stiffness - 0.0).abs() < EPS);
        assert!((clean.weft_stiffness - 0.0).abs() < EPS);
        assert!((clean.density - 0.0).abs() < EPS);
        assert!((clean.friction - 1.0).abs() < EPS);
        assert!((clean.drag - 0.0).abs() < EPS);
    }

    #[test]
    fn fabric_validity_rejects_nan_and_out_of_range() {
        assert!(FabricMaterial::default().is_valid());

        let bad = FabricMaterial {
            density: f32::NAN,
            ..FabricMaterial::default()
        };
        assert!(!bad.is_valid());

        let hot = FabricMaterial {
            friction: 2.0,
            ..FabricMaterial::default()
        };
        assert!(!hot.is_valid());
    }

    #[test]
    fn painted_clamped_enforces_bounds() {
        let dirty = PaintedConstraint::new(-1.0, -2.0, 5.0, -0.5);
        let clean = dirty.clamped();
        assert!(clean.is_valid());
        assert!((clean.max_distance - 0.0).abs() < EPS);
        assert!((clean.backstop - 0.0).abs() < EPS);
        assert!((clean.blend_weight - 1.0).abs() < EPS);
        assert!((clean.anim_drive - 0.0).abs() < EPS);
    }

    #[test]
    fn painted_clamped_preserves_infinite_cap_but_kills_nan() {
        let uncapped = PaintedConstraint::default().clamped();
        assert!(uncapped.max_distance.is_infinite());
        assert!(uncapped.is_valid());

        let nan_cap = PaintedConstraint::new(f32::NAN, 0.0, 0.5, 0.5).clamped();
        assert!((nan_cap.max_distance - 0.0).abs() < EPS);
        assert!(nan_cap.is_valid());
    }

    #[test]
    fn painted_validity_and_pinning() {
        assert!(PaintedConstraint::default().is_valid());

        let pinned = PaintedConstraint::new(0.0, 0.0, 0.0, 0.0);
        assert!(pinned.is_valid());
        assert!(pinned.is_pinned_to_skin());

        let free = PaintedConstraint::new(0.05, 0.0, 1.0, 0.0);
        assert!(!free.is_pinned_to_skin());

        let bad = PaintedConstraint::new(f32::NAN, 0.0, 0.5, 0.5);
        assert!(!bad.is_valid());
    }
}
