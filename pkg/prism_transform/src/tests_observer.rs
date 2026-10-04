//! §24.4 tests: deterministic transform-change observation.
//!
//! Covers in-frame merging, net-no-op suppression, ascending-order dispatch,
//! baseline priming (no spurious events), newly-appearing nodes, and the
//! change-delta helpers.

use crate::observer::{ChangeMask, TransformObserver};
use crate::{GlobalTransform, Transform};
use prism_math::{vec3, Quat, Vec3};

fn at(x: f32) -> GlobalTransform {
    GlobalTransform::from_transform(&Transform::from_xyz(x, 0.0, 0.0))
}

fn node(i: u32) -> crate::hierarchy::NodeId {
    crate::hierarchy::NodeId::new(i)
}

fn primed(globals: &[GlobalTransform]) -> TransformObserver {
    let mut obs = TransformObserver::new();
    obs.prime(globals);
    obs
}

// ---- baseline priming emits nothing ----------------------------------------

#[test]
fn prime_then_flush_emits_nothing() {
    let mut obs = primed(&[at(0.0), at(0.0)]);
    assert!(obs.flush().is_empty());
    // Flush still advances the epoch even with no changes.
    assert_eq!(obs.epoch(), 1);
    assert_eq!(obs.len(), 2);
}

// ---- a single net change ---------------------------------------------------

#[test]
fn single_move_emits_one_change() {
    let mut obs = primed(&[at(0.0), at(0.0)]);
    let globals = [at(5.0), at(0.0)];
    obs.observe(&[node(0)], &globals);
    let changes = obs.flush();
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0].node, node(0));
    assert_eq!(changes[0].old, at(0.0));
    assert_eq!(changes[0].new, at(5.0));
}

#[test]
fn baseline_advances_so_a_stable_node_stops_reporting() {
    let mut obs = primed(&[at(0.0)]);
    let g1 = [at(5.0)];
    obs.observe(&[node(0)], &g1);
    assert_eq!(obs.flush().len(), 1);
    // Observing the same (now-baseline) pose again yields nothing.
    obs.observe(&[node(0)], &g1);
    assert!(obs.flush().is_empty());
}

// ---- in-frame merging ------------------------------------------------------

#[test]
fn repeated_touches_merge_to_latest_pose() {
    let mut obs = primed(&[at(0.0)]);
    // Three observations in one frame; only the last pose survives.
    obs.observe(&[node(0)], &[at(1.0)]);
    obs.observe(&[node(0)], &[at(2.0)]);
    obs.observe(&[node(0)], &[at(9.0)]);
    let changes = obs.flush();
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0].old, at(0.0));
    assert_eq!(changes[0].new, at(9.0));
}

#[test]
fn move_and_move_back_within_frame_is_a_no_op() {
    let mut obs = primed(&[at(0.0)]);
    obs.observe(&[node(0)], &[at(7.0)]);
    obs.observe(&[node(0)], &[at(0.0)]); // back to baseline
    assert!(obs.flush().is_empty());
    // ...and the baseline is unchanged, so a later real move still fires.
    obs.observe(&[node(0)], &[at(3.0)]);
    assert_eq!(obs.flush().len(), 1);
}

// ---- deterministic ascending order -----------------------------------------

#[test]
fn changes_are_emitted_in_ascending_node_order() {
    let mut obs = primed(&[at(0.0), at(0.0), at(0.0)]);
    let globals = [at(1.0), at(2.0), at(3.0)];
    // Observe out of order; flush must sort by node index.
    obs.observe(&[node(2), node(0), node(1)], &globals);
    let changes = obs.flush();
    let order: Vec<u32> = changes.iter().map(|c| c.node.index() as u32).collect();
    assert_eq!(order, alloc::vec![0, 1, 2]);
}

// ---- newly-appearing node --------------------------------------------------

#[test]
fn new_node_reports_change_from_identity_baseline() {
    // Observer starts tracking a single node; a dirty index past the end grows
    // the tracker with an identity baseline and reports the new pose.
    let mut obs = TransformObserver::new();
    obs.push(at(0.0));
    assert_eq!(obs.len(), 1);
    let globals = [at(0.0), GlobalTransform::IDENTITY, at(4.0)];
    obs.observe(&[node(2)], &globals);
    assert_eq!(obs.len(), 3);
    let changes = obs.flush();
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0].node, node(2));
    assert_eq!(changes[0].old, GlobalTransform::IDENTITY);
    assert_eq!(changes[0].new, at(4.0));
}

#[test]
fn identity_baseline_node_with_identity_pose_is_no_op() {
    let mut obs = TransformObserver::new();
    obs.push(at(0.0));
    // Node 1 appears with the identity pose == its identity baseline: no change.
    let globals = [at(0.0), GlobalTransform::IDENTITY];
    obs.observe(&[node(1)], &globals);
    assert!(obs.flush().is_empty());
}

// ---- dispatch callback -----------------------------------------------------

#[test]
fn dispatch_invokes_callback_per_change_in_order() {
    let mut obs = primed(&[at(0.0), at(0.0)]);
    obs.observe(&[node(1), node(0)], &[at(1.0), at(2.0)]);
    let mut seen: Vec<(u32, f32)> = Vec::new();
    let n = obs.dispatch(|c| seen.push((c.node.index() as u32, c.new.translation().x)));
    assert_eq!(n, 2);
    assert_eq!(seen, alloc::vec![(0, 1.0), (1, 2.0)]);
}

// ---- change-delta helpers --------------------------------------------------

#[test]
fn translation_delta_and_moved_mask() {
    let mut obs = primed(&[at(0.0)]);
    obs.observe(&[node(0)], &[at(3.0)]);
    let changes = obs.flush();
    let c = changes[0];
    assert_eq!(c.translation_delta(), vec3(3.0, 0.0, 0.0));
    assert!(c.moved(1.0e-3));
    assert!(!c.basis_changed(1.0e-3));
    assert_eq!(
        c.mask(1.0e-3),
        ChangeMask {
            translation: true,
            basis: false,
        }
    );
    assert!(c.mask(1.0e-3).any());
}

#[test]
fn basis_change_is_detected_for_pure_rotation() {
    let base = GlobalTransform::IDENTITY;
    let rotated = GlobalTransform::from_transform(&Transform {
        translation: Vec3::ZERO,
        rotation: Quat::from_rotation_z(core::f32::consts::FRAC_PI_2),
        scale: Vec3::ONE,
    });
    let mut obs = TransformObserver::new();
    obs.push(base);
    obs.observe(&[node(0)], &[rotated]);
    let changes = obs.flush();
    let c = changes[0];
    assert!(c.basis_changed(1.0e-3));
    assert!(!c.moved(1.0e-3));
    let mask = c.mask(1.0e-3);
    assert!(mask.basis && !mask.translation);
}

#[test]
fn with_capacity_tracks_nothing_until_pushed() {
    let mut obs = TransformObserver::with_capacity(8);
    assert!(obs.is_empty());
    obs.push(at(0.0));
    assert_eq!(obs.len(), 1);
}
