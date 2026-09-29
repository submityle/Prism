//! Cross-stage integration tests for the `motion` subsystem.
//!
//! The per-module unit tests already pin each stage's behavior in isolation.
//! These tests instead exercise the *contracts between* stages the way a real
//! temporal frame does, so a future refactor that keeps every module green on
//! its own but breaks an inter-stage assumption still fails here:
//!
//! * reprojection -> encoding (velocity survives the quantized texel round-trip),
//! * dilation -> tile-max -> classification (a thin fast silhouette dilates,
//!   reduces, and classifies as fast motion end to end),
//! * disocclusion -> encoding (a rejected verdict lights the history flag that
//!   the packed masks carry to the `GPU`),
//! * determinism (the same inputs reproduce bit-identical tile classifications),
//! * degenerate inputs (`NaN` velocities and empty fields resolve to safe
//!   sentinels rather than propagating garbage).

use super::reproject::ReprojectionContext;
use super::{Mat4, MotionSample, ScreenDims, Vec2, Vec4};

use super::dilation::{dilate_closest_depth, tile_max, DepthField, DepthOrder, VelocityField};
use super::disocclusion::{classify, history_flags, DisocclusionParams, SurfacePoint};
use super::encode::{encode_sample, flags, VelocityEncoding};
use super::tiles::{classify_field, MotionTileClass, TileClassifierParams};

/// Tolerance for comparing derived pixel coordinates (identity transforms are
/// exact in principle, but we stay off exact `f32` equality on principle).
const PIXEL_EPS: f32 = 1e-3;

fn identity_context() -> ReprojectionContext {
    ReprojectionContext::new(Mat4::IDENTITY, Mat4::IDENTITY, ScreenDims::new(100, 100))
}

/// A previous world point one-fifth of the way to the right edge in `NDC`,
/// reprojected under identity transforms, must land at pixel x = 60 (the
/// current point sits at the center, x = 50), yielding a +10px horizontal
/// velocity that survives the quantized encode/decode round-trip.
#[test]
fn reprojection_velocity_survives_encoding() {
    let ctx = identity_context();
    let reprojected =
        ctx.reproject_world_point(Vec4::point(0.2, 0.0, 0.0), Vec4::point(0.0, 0.0, 0.0));

    assert!(
        (reprojected.prev_pixel.x - 60.0).abs() <= PIXEL_EPS,
        "prev pixel x should map ndc 0.2 -> 60, got {}",
        reprojected.prev_pixel.x
    );
    assert!(
        (reprojected.velocity_pixels.x - 10.0).abs() <= PIXEL_EPS,
        "horizontal velocity should be +10px, got {}",
        reprojected.velocity_pixels.x
    );
    assert!(
        reprojected.velocity_pixels.y.abs() <= PIXEL_EPS,
        "vertical velocity should be ~0, got {}",
        reprojected.velocity_pixels.y
    );
    assert!(
        reprojected.confidence > 0.0,
        "in-frame history is confident"
    );

    let sample = MotionSample {
        velocity_pixels: [reprojected.velocity_pixels.x, reprojected.velocity_pixels.y],
        reprojection_confidence: reprojected.confidence,
        reactive: 0.0,
        transparency: 0.0,
        surface_id: 1,
    };

    let encoding = VelocityEncoding::new(540.0);
    let encoded = encode_sample(sample, encoding, 0);
    let decoded = encoding.decode(encoded.velocity);
    let step = encoding.quantization_step_pixels();

    assert!(
        (decoded.x - reprojected.velocity_pixels.x).abs() <= step,
        "decoded x within one quantization step"
    );
    assert!(
        (decoded.y - reprojected.velocity_pixels.y).abs() <= step,
        "decoded y within one quantization step"
    );
    // A confident, opaque, non-disoccluded sample carries no flag bits.
    assert_eq!(encoded.masks.flags(), 0);
}

/// A single fast foreground texel (20px/frame) sitting in front (nearest depth)
/// of a static background must dilate into its neighbor, reduce through the
/// 1x1 tile-max unchanged, and classify as two fast tiles with the default
/// classifier thresholds.
#[test]
fn fast_silhouette_dilates_reduces_and_classifies_fast() {
    let velocity = VelocityField::from_pixels(
        4,
        1,
        alloc::vec![Vec2::new(20.0, 0.0), Vec2::ZERO, Vec2::ZERO, Vec2::ZERO,],
    )
    .expect("4x1 velocity field");
    let depth =
        DepthField::from_depths(4, 1, alloc::vec![0.1, 0.9, 0.9, 0.9]).expect("4x1 depth field");

    let dilated = dilate_closest_depth(&velocity, &depth, 1, DepthOrder::SmallerIsCloser)
        .expect("dilation over a non-empty field");

    // The fast texel (nearest depth 0.1) wins its own cell and bleeds one pixel
    // right; the remaining background cells keep zero motion.
    assert_eq!(dilated.get(0, 0), Some(Vec2::new(20.0, 0.0)));
    assert_eq!(dilated.get(1, 0), Some(Vec2::new(20.0, 0.0)));
    assert_eq!(dilated.get(2, 0), Some(Vec2::ZERO));
    assert_eq!(dilated.get(3, 0), Some(Vec2::ZERO));

    let tiles = tile_max(&dilated, 1).expect("tile-max over a non-empty field");
    let classification = classify_field(&tiles, TileClassifierParams::default());

    assert_eq!(classification.len(), 4);
    assert_eq!(classification.fast_count(), 2, "two dilated fast tiles");
    assert_eq!(
        classification.static_count(),
        2,
        "two static background tiles"
    );
    assert_eq!(classification.get(0, 0), Some(MotionTileClass::Fast));
    assert_eq!(classification.get(2, 0), Some(MotionTileClass::Static));
}

/// A rejected disocclusion verdict must set the `DISOCCLUDED` history flag, and
/// that flag must survive `encode_sample` into the packed mask byte.
#[test]
fn rejected_history_lights_disoccluded_flag_through_encode() {
    // Different surface ids plus a large depth gap: a hard rejection.
    let current = SurfacePoint::new(1.0, [0.0, 0.0, 1.0], 1);
    let history = SurfacePoint::new(5.0, [0.0, 0.0, 1.0], 2);

    let verdict = classify(current, history, DisocclusionParams::default());
    assert!(!verdict.accepted, "mismatched surface must reject");

    let flag_bits = history_flags(verdict);
    assert_ne!(flag_bits & flags::DISOCCLUDED, 0);

    let sample = MotionSample {
        velocity_pixels: [1.0, 2.0],
        reprojection_confidence: verdict.confidence,
        reactive: 0.0,
        transparency: 0.0,
        surface_id: 1,
    };
    let encoded = encode_sample(sample, VelocityEncoding::new(540.0), flag_bits);
    assert_ne!(
        encoded.masks.flags() & flags::DISOCCLUDED,
        0,
        "disocclusion flag must reach the packed masks"
    );
}

/// An accepted verdict (identical surfaces) leaves the history flag clear.
#[test]
fn accepted_history_leaves_flag_clear() {
    let point = SurfacePoint::new(1.0, [0.0, 0.0, 1.0], 7);
    let verdict = classify(point, point, DisocclusionParams::default());
    assert!(verdict.accepted, "identical surface must be accepted");
    assert_eq!(history_flags(verdict), 0);
}

/// The dilate -> tile-max -> classify pipeline is a pure function of its
/// inputs: running it twice yields structurally identical classifications.
#[test]
fn pipeline_is_deterministic() {
    fn run() -> super::tiles::TileClassification {
        let velocity = VelocityField::from_pixels(
            4,
            2,
            alloc::vec![
                Vec2::new(12.0, 3.0),
                Vec2::new(1.0, 0.0),
                Vec2::ZERO,
                Vec2::new(0.2, 0.1),
                Vec2::ZERO,
                Vec2::new(30.0, -5.0),
                Vec2::new(2.0, 2.0),
                Vec2::ZERO,
            ],
        )
        .expect("4x2 velocity field");
        let depth =
            DepthField::from_depths(4, 2, alloc::vec![0.2, 0.5, 0.9, 0.9, 0.9, 0.1, 0.5, 0.9])
                .expect("4x2 depth field");
        let dilated = dilate_closest_depth(&velocity, &depth, 1, DepthOrder::SmallerIsCloser)
            .expect("dilation");
        let tiles = tile_max(&dilated, 2).expect("tile-max");
        classify_field(&tiles, TileClassifierParams::default())
    }

    assert_eq!(
        run(),
        run(),
        "same inputs must reproduce the classification"
    );
}

/// Empty fields and `NaN` velocities resolve to safe sentinels: no motion, no
/// panic, and `NaN` classified as fast (worst case) so it never masquerades as
/// static history that would be blended in.
#[test]
fn degenerate_inputs_resolve_safely() {
    // Empty velocity field -> no tiles.
    assert!(tile_max(&VelocityField::zeroed(0, 0), 1).is_none());

    // A `NaN` velocity is neither <= static nor <= slow, so it lands in `Fast`.
    let params = TileClassifierParams::default();
    assert_eq!(
        super::tiles::classify_tile(Vec2::new(f32::NAN, 0.0), params),
        MotionTileClass::Fast
    );

    // Encoding a `NaN` velocity clamps to the zero texel rather than emitting
    // an out-of-range or `NaN`-derived code.
    let encoding = VelocityEncoding::new(540.0);
    assert_eq!(encoding.encode(Vec2::new(f32::NAN, f32::NAN)), [0, 0]);
}
