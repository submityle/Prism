//! §24.3 tests: a reader always sees a tear-free snapshot (read-old / write-new),
//! `publish` makes the new frame visible, triple buffering lets a held snapshot
//! survive the sim writing the next frame, plus boundary cases.

use crate::double_buffer::{MultiBuffer, TransformDoubleBuffer, TransformTripleBuffer};
use crate::{GlobalTransform, Transform};
use prism_math::vec3;

fn gt(x: f32) -> GlobalTransform {
    GlobalTransform::from_transform(&Transform::from_xyz(x, 0.0, 0.0))
}

fn tx(g: GlobalTransform) -> f32 {
    g.translation().x
}

// ---- read-old / write-new (no tearing) -------------------------------------

#[test]
fn writing_the_back_buffer_does_not_change_what_read_returns() {
    let mut buf = TransformDoubleBuffer::double(3, GlobalTransform::IDENTITY);
    // Publish an initial frame of zeros so `read` is a real snapshot.
    buf.publish_from(&[gt(0.0), gt(0.0), gt(0.0)]);
    assert_eq!(buf.version(), 1);

    // Simulation writes the NEXT frame into the back buffer.
    buf.write_all(&[gt(1.0), gt(2.0), gt(3.0)]);

    // Render still reads the published (old) snapshot: no tearing.
    let snap: Vec<f32> = buf.read().iter().map(|&g| tx(g)).collect();
    assert_eq!(snap, alloc::vec![0.0, 0.0, 0.0]);
}

#[test]
fn publish_makes_the_new_frame_visible() {
    let mut buf = TransformDoubleBuffer::double(2, GlobalTransform::IDENTITY);
    buf.publish_from(&[gt(0.0), gt(0.0)]);

    buf.write_at(0, gt(5.0));
    buf.write_at(1, gt(6.0));
    // Not yet visible.
    assert_eq!(tx(buf.read()[0]), 0.0);

    buf.publish();
    // Now visible, and the version advanced.
    assert_eq!(tx(buf.read()[0]), 5.0);
    assert_eq!(tx(buf.read()[1]), 6.0);
    assert_eq!(buf.version(), 2);
}

#[test]
fn partial_edits_carry_the_rest_forward_via_prime() {
    let mut buf = TransformDoubleBuffer::double(3, GlobalTransform::IDENTITY);
    buf.publish_from(&[gt(1.0), gt(2.0), gt(3.0)]);

    // New frame only moves node 1; prime the back buffer from the last snapshot.
    buf.prime_write_from_read();
    buf.write_at(1, gt(20.0));
    buf.publish();

    let snap: Vec<f32> = buf.read().iter().map(|&g| tx(g)).collect();
    assert_eq!(snap, alloc::vec![1.0, 20.0, 3.0]);
}

// ---- triple buffering: a held snapshot survives one extra publish ----------

#[test]
fn triple_buffer_held_snapshot_survives_next_frame_write() {
    let mut buf = TransformTripleBuffer::triple(2, GlobalTransform::IDENTITY);

    // Frame A published and captured by a (slow) reader.
    buf.publish_from(&[gt(10.0), gt(11.0)]);
    let token = buf.read_token();
    assert!(buf.is_token_live(token));
    let a_snapshot: Vec<f32> = buf.columns(token).iter().map(|&g| tx(g)).collect();
    assert_eq!(a_snapshot, alloc::vec![10.0, 11.0]);

    // Sim produces frame B *while* the reader still holds A. With triple
    // buffering the write lands on a third column, not A's.
    buf.publish_from(&[gt(20.0), gt(21.0)]);

    // The reader's captured snapshot is still intact and still live.
    assert!(buf.is_token_live(token));
    let a_again: Vec<f32> = buf.columns(token).iter().map(|&g| tx(g)).collect();
    assert_eq!(a_again, alloc::vec![10.0, 11.0], "frame A was not overwritten");

    // The newest read is frame B.
    let b: Vec<f32> = buf.read().iter().map(|&g| tx(g)).collect();
    assert_eq!(b, alloc::vec![20.0, 21.0]);
}

#[test]
fn triple_buffer_snapshot_dies_after_full_rotation() {
    let mut buf = TransformTripleBuffer::triple(1, GlobalTransform::IDENTITY);
    buf.publish_from(&[gt(1.0)]);
    let token = buf.read_token();

    // Two further publishes rotate the write column back onto the token buffer.
    buf.publish_from(&[gt(2.0)]);
    assert!(buf.is_token_live(token), "survives one publish");
    buf.publish_from(&[gt(3.0)]);
    assert!(
        !buf.is_token_live(token),
        "no longer live after a full rotation (3 columns, 2 publishes)"
    );
}

#[test]
fn double_buffer_snapshot_dies_on_the_very_next_publish() {
    let mut buf = TransformDoubleBuffer::double(1, GlobalTransform::IDENTITY);
    buf.publish_from(&[gt(1.0)]);
    let token = buf.read_token();
    assert!(buf.is_token_live(token));
    buf.publish_from(&[gt(2.0)]);
    assert!(
        !buf.is_token_live(token),
        "double buffering only guarantees the current frame's snapshot"
    );
}

// ---- write/read never alias before first publish ---------------------------

#[test]
fn write_and_read_columns_start_disjoint() {
    let mut buf = TransformDoubleBuffer::double(1, GlobalTransform::IDENTITY);
    // Before any publish, writing must not change the initial read snapshot.
    buf.write_at(0, gt(99.0));
    assert_eq!(tx(buf.read()[0]), 0.0, "initial fill, untouched by back-buffer write");
    assert_eq!(buf.version(), 0);
}

// ---- generic payload + multi-buffer rotation -------------------------------

#[test]
fn generic_multibuffer_rotates_through_every_column() {
    // Four buffers of plain integers: exercise the generic T path and rotation.
    let mut buf: MultiBuffer<i32> = MultiBuffer::new(4, 2, 0);
    assert_eq!(buf.buffer_count(), 4);

    for frame in 1..=6 {
        buf.write_all(&[frame, frame * 10]);
        buf.publish();
        assert_eq!(buf.read(), &[frame, frame * 10]);
        assert_eq!(buf.version() as i32, frame);
    }
    // A snapshot taken now survives buffer_count - 2 == 2 further publishes.
    let token = buf.read_token();
    buf.write_all(&[7, 70]);
    buf.publish();
    buf.write_all(&[8, 80]);
    buf.publish();
    assert!(buf.is_token_live(token));
    assert_eq!(buf.columns(token), &[6, 60]);
    buf.write_all(&[9, 90]);
    buf.publish();
    assert!(!buf.is_token_live(token));
}

// ---- boundaries ------------------------------------------------------------

#[test]
fn buffer_count_is_clamped_to_at_least_two() {
    let buf: MultiBuffer<i32> = MultiBuffer::new(0, 1, 0);
    assert_eq!(buf.buffer_count(), 2);
    let buf: MultiBuffer<i32> = MultiBuffer::new(1, 1, 0);
    assert_eq!(buf.buffer_count(), 2);
}

#[test]
fn resize_grows_all_columns_and_preserves_rotation() {
    let mut buf: MultiBuffer<i32> = MultiBuffer::triple(1, -1);
    buf.publish_from(&[5]);
    let version_before = buf.version();
    buf.resize(3, -1);
    assert_eq!(buf.len(), 3);
    assert_eq!(buf.version(), version_before, "resize does not publish");
    // Existing published entry preserved, new slots filled.
    assert_eq!(buf.read()[0], 5);
    // The back column is also widened and writable.
    buf.write_all(&[1, 2, 3]);
    buf.publish();
    assert_eq!(buf.read(), &[1, 2, 3]);
}

#[test]
fn empty_buffer_is_empty_but_publishes_versions() {
    let mut buf: MultiBuffer<i32> = MultiBuffer::double(0, 0);
    assert!(buf.is_empty());
    buf.publish();
    assert_eq!(buf.version(), 1);
    assert!(buf.read().is_empty());
}

#[test]
fn vec3_payload_roundtrips_through_publish() {
    // Guard that a non-Copy-looking math type (it is Copy) flows through.
    let mut buf: MultiBuffer<prism_math::Vec3> = MultiBuffer::double(2, vec3(0.0, 0.0, 0.0));
    buf.publish_from(&[vec3(1.0, 2.0, 3.0), vec3(4.0, 5.0, 6.0)]);
    assert_eq!(buf.read()[1], vec3(4.0, 5.0, 6.0));
}
