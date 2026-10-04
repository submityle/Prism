//! End-to-end integration tests for the object-based rendering chain.
//!
//! These exercise the full public pipeline of `prism_audio_object` as an
//! external consumer would: author an [`ObjectScene`] (an immersive bed plus
//! static and keyframe-animated objects), advance its metadata timeline, and
//! render it through [`render_scene`] into every delivery format
//! ([`OutputFormat::Bed`], [`OutputFormat::Binaural`],
//! [`OutputFormat::Ambisonic`]). The assertions pin the cross-module
//! invariants that no single unit test covers: that the budget ->
//! clustering -> fold/encode chain conserves acoustic energy, that the
//! hardware object budget deterministically bounds the cluster count, that
//! the clustering stage is format-independent, and that advancing the
//! animation timeline actually changes the rendered output.
//!
//! # Provenance
//! Original work. Contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, Dolby, Google Resonance Audio, MPEG, or Web Audio source or derived
//! code, and no AI/ML. The bed-plus-objects delivery model, hardware object
//! budget, and perceptual clustering concepts are drawn only as ideas from
//! public descriptions of object-based audio; the implementation under test
//! and these tests are independent.
//!
//! # Relationship
//! Covers §16 (object audio / Ambisonics), §33 (source clustering under an
//! object budget), and the `prism_audio_object` crate's `scene` -> `render`
//! -> `clustering`/`fold`/`ambisonics` chain as a black-box public-API
//! contract. Complements the per-module unit tests, which test each stage in
//! isolation with flat object lists rather than an animated scene end to end.

use bevy_math::Vec3;

use prism_audio_core::math::Sample;

use prism_audio_object::ambisonics::{channel_count, MAX_ORDER};
use prism_audio_object::bed::{direction_from_angles, BedLayout};
use prism_audio_object::budget::ObjectBudget;
use prism_audio_object::clustering::total_energy;
use prism_audio_object::metadata::stream::MetadataStream;
use prism_audio_object::metadata::ObjectMetadata;
use prism_audio_object::object::{AudioObject, ObjectId};
use prism_audio_object::render::{render_scene, OutputFormat, RenderOutput, RenderPayload};
use prism_audio_object::scene::ObjectScene;

/// Absolute tolerance for energy sums that survive the clustering reduction.
const EPS: Sample = 1e-4;

/// Builds an [`AudioObject`] at the given azimuth/elevation (degrees) and gain.
fn obj(id: u32, az: Sample, el: Sample, gain: Sample) -> AudioObject {
    AudioObject::new(ObjectId(id), direction_from_angles(az, el), gain)
}

/// Returns the summed `gain^2` energy of a raw object slice.
fn object_energy(objects: &[AudioObject]) -> Sample {
    objects.iter().map(AudioObject::energy).sum()
}

/// Builds a representative 7.1.4 scene: a bed layout plus eight discrete
/// objects spread around the horizon and overhead, two of them clustered
/// tightly so an under-budget render has something to merge.
fn authored_scene() -> ObjectScene {
    let mut scene = ObjectScene::new(BedLayout::Surround7_1_4);
    scene.add_object(obj(0, 0.0, 0.0, 1.0));
    scene.add_object(obj(1, 30.0, 0.0, 0.9));
    scene.add_object(obj(2, 33.0, 0.0, 0.8)); // near object 1 -> likely merge
    scene.add_object(obj(3, 90.0, 0.0, 0.7));
    scene.add_object(obj(4, 150.0, 0.0, 0.6));
    scene.add_object(obj(5, -120.0, 0.0, 0.5));
    scene.add_object(obj(6, -60.0, 20.0, 0.4));
    scene.add_object(obj(7, 10.0, 60.0, 0.3)); // overhead
    scene
}

#[test]
fn within_budget_renders_every_object_as_its_own_cluster() {
    let scene = authored_scene();
    let objects = scene.current_objects();
    let budget = ObjectBudget::new(16); // comfortably above the 8 objects

    let out = render_scene(&scene, budget, OutputFormat::Bed);

    assert_eq!(
        out.clusters.len(),
        objects.len(),
        "an under-budget scene must pass every object through 1:1"
    );
    match &out.payload {
        RenderPayload::Bed(rows) => {
            assert_eq!(rows.len(), objects.len());
            for row in rows {
                assert_eq!(row.len(), BedLayout::Surround7_1_4.channel_count());
                assert!(row.iter().all(|g| g.is_finite() && *g >= 0.0));
            }
        }
        other => panic!("expected a bed payload, got {other:?}"),
    }
}

#[test]
fn over_budget_reduces_cluster_count_but_conserves_energy() {
    let scene = authored_scene();
    let objects = scene.current_objects();
    let budget = ObjectBudget::new(4); // fewer than the 8 authored objects

    let out = render_scene(&scene, budget, OutputFormat::Bed);

    assert_eq!(
        out.clusters.len(),
        4,
        "over-budget renders must collapse to exactly the budget count"
    );
    let before = object_energy(&objects);
    let after = total_energy(&out.clusters);
    assert!(
        (before - after).abs() <= EPS,
        "clustering must conserve total acoustic energy: {before} vs {after}"
    );
    match &out.payload {
        RenderPayload::Bed(rows) => assert_eq!(rows.len(), 4),
        other => panic!("expected a bed payload, got {other:?}"),
    }
}

#[test]
fn zero_budget_folds_every_object_directly_without_clusters() {
    let scene = authored_scene();
    let objects = scene.current_objects();
    let budget = ObjectBudget::new(0);

    let out = render_scene(&scene, budget, OutputFormat::Bed);

    assert!(
        out.clusters.is_empty(),
        "a zero budget must bypass clustering entirely"
    );
    match &out.payload {
        RenderPayload::Bed(rows) => assert_eq!(
            rows.len(),
            objects.len(),
            "the direct-fold path must emit one bed row per raw object"
        ),
        other => panic!("expected a bed payload, got {other:?}"),
    }
}

#[test]
fn ambisonic_render_width_matches_order_and_stays_finite() {
    let scene = authored_scene();
    let budget = ObjectBudget::new(4);
    let order = 3;

    let out = render_scene(&scene, budget, OutputFormat::Ambisonic { order });

    match &out.payload {
        RenderPayload::Ambisonic {
            order: encoded,
            rows,
        } => {
            assert_eq!(*encoded, order);
            assert_eq!(rows.len(), out.clusters.len());
            for row in rows {
                assert_eq!(row.len(), channel_count(order));
                assert!(row.iter().all(|c| c.is_finite()));
            }
        }
        other => panic!("expected an ambisonic payload, got {other:?}"),
    }
}

#[test]
fn ambisonic_order_is_clamped_to_the_supported_maximum() {
    let scene = authored_scene();
    let budget = ObjectBudget::new(2);

    let out = render_scene(
        &scene,
        budget,
        OutputFormat::Ambisonic {
            order: MAX_ORDER + 5,
        },
    );

    match &out.payload {
        RenderPayload::Ambisonic { order, rows } => {
            assert_eq!(*order, MAX_ORDER, "excessive orders must clamp");
            for row in rows {
                assert_eq!(row.len(), channel_count(MAX_ORDER));
            }
        }
        other => panic!("expected an ambisonic payload, got {other:?}"),
    }
}

#[test]
fn binaural_render_yields_one_finite_direction_per_cluster() {
    let scene = authored_scene();
    let budget = ObjectBudget::new(4);

    let out = render_scene(&scene, budget, OutputFormat::Binaural);

    match &out.payload {
        RenderPayload::Binaural(dirs) => {
            assert_eq!(dirs.len(), out.clusters.len());
            for d in dirs {
                assert!(d.azimuth.is_finite() && d.elevation.is_finite());
                assert!(d.gain.is_finite() && d.gain >= 0.0);
                assert!((0.0..=1.0).contains(&d.spread));
            }
        }
        other => panic!("expected a binaural payload, got {other:?}"),
    }
}

#[test]
fn clustering_is_identical_across_output_formats() {
    // The budget -> clustering stage runs before the format split, so the
    // representative clusters must not depend on the chosen delivery format.
    let scene = authored_scene();
    let budget = ObjectBudget::new(3);

    let bed = render_scene(&scene, budget, OutputFormat::Bed);
    let bin = render_scene(&scene, budget, OutputFormat::Binaural);
    let amb = render_scene(&scene, budget, OutputFormat::Ambisonic { order: 2 });

    assert_eq!(bed.clusters, bin.clusters);
    assert_eq!(bed.clusters, amb.clusters);
}

#[test]
fn advancing_the_animation_timeline_changes_the_render() {
    // An object that sweeps from front to hard-left between t=0 and t=1 must
    // produce a different bed downmix at the two instants.
    let mut scene = ObjectScene::new(BedLayout::Surround5_1_4);
    scene.add_object(obj(0, 0.0, 0.0, 0.8)); // a static anchor

    let mut stream = MetadataStream::new();
    stream.push(
        0.0,
        ObjectMetadata {
            position: direction_from_angles(0.0, 0.0),
            gain: 0.9,
            spread: 0.0,
            priority: 1.0,
        },
    );
    stream.push(
        1.0,
        ObjectMetadata {
            position: direction_from_angles(-90.0, 0.0),
            gain: 0.9,
            spread: 0.0,
            priority: 1.0,
        },
    );
    scene.add_animated_object(obj(1, 0.0, 0.0, 0.9), stream);

    let budget = ObjectBudget::new(8);

    scene.advance_to(0.0);
    let start = render_scene(&scene, budget, OutputFormat::Bed);

    scene.advance_to(1.0);
    let end = render_scene(&scene, budget, OutputFormat::Bed);

    assert_ne!(
        bed_rows(&start),
        bed_rows(&end),
        "moving an animated object must change the bed downmix"
    );
}

#[test]
fn midpoint_animation_interpolates_between_keyframes() {
    // Sampling the sweep at t=0.5 must land strictly between the endpoints,
    // proving the scene feeds interpolated metadata into the render.
    let mut scene = ObjectScene::new(BedLayout::Surround5_1_4);
    let mut stream = MetadataStream::new();
    stream.push(
        0.0,
        ObjectMetadata {
            position: Vec3::new(1.0, 0.0, 0.0),
            gain: 0.4,
            spread: 0.0,
            priority: 1.0,
        },
    );
    stream.push(
        1.0,
        ObjectMetadata {
            position: Vec3::new(-1.0, 0.0, 0.0),
            gain: 0.8,
            spread: 0.0,
            priority: 1.0,
        },
    );
    scene.add_animated_object(obj(0, 1.0, 0.0, 0.4), stream);

    scene.advance_to(0.5);
    let mid = scene.current_objects();
    assert_eq!(mid.len(), 1);
    // Position x interpolates 1 -> -1, so the midpoint sits near the origin on
    // the x axis; gain interpolates 0.4 -> 0.8.
    assert!(mid[0].position.x.abs() <= EPS, "x should cross zero at t=0.5");
    assert!((mid[0].gain - 0.6).abs() <= 1e-3, "gain should be the mean");
}

/// Extracts the bed gain rows from a render output, panicking on any other
/// payload so a format mix-up fails loudly rather than silently comparing
/// unrelated shapes.
fn bed_rows(out: &RenderOutput) -> &[Vec<Sample>] {
    match &out.payload {
        RenderPayload::Bed(rows) => rows,
        other => panic!("expected a bed payload, got {other:?}"),
    }
}
