//! §24.5 tests: deterministic transform → acceleration-structure sync.
//!
//! Covers the world-box refit math (`transform_aabb` under translation and
//! rotation against hand-computed oracles), insert/update/remove command
//! generation, exact- vs fat-margin refit policy, ascending-order determinism,
//! net-no-op suppression, and the `SyncStats` counters.

use crate::hierarchy::NodeId;
use crate::spatial_sync::{transform_aabb, world_aabb, SpatialCommand, SpatialSync};
use crate::{GlobalTransform, Transform};
use prism_math::{Aabb3, Quat, Vec3};

fn node(i: u32) -> NodeId {
    NodeId::new(i)
}

fn aabb(min: [f32; 3], max: [f32; 3]) -> Aabb3 {
    Aabb3::new(
        Vec3::new(min[0], min[1], min[2]),
        Vec3::new(max[0], max[1], max[2]),
    )
}

fn translated(x: f32, y: f32, z: f32) -> GlobalTransform {
    GlobalTransform::from_transform(&Transform::from_xyz(x, y, z))
}

fn approx(a: Vec3, b: Vec3, eps: f32) -> bool {
    (a - b).length() <= eps
}

fn box_approx(a: Aabb3, b: Aabb3, eps: f32) -> bool {
    approx(a.min, b.min, eps) && approx(a.max, b.max, eps)
}

// ---- transform_aabb oracles -------------------------------------------------

#[test]
fn transform_aabb_identity_is_passthrough() {
    let local = aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    let world = world_aabb(&GlobalTransform::IDENTITY, local);
    assert!(box_approx(world, local, 1e-6));
}

#[test]
fn transform_aabb_translation_shifts_box() {
    let local = aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    let world = world_aabb(&translated(10.0, 0.0, 0.0), local);
    assert!(box_approx(
        world,
        aabb([9.0, -1.0, -1.0], [11.0, 1.0, 1.0]),
        1e-6
    ));
}

#[test]
fn transform_aabb_rotation_refits_asymmetric_box() {
    // Local box on +X/+Y, rotate +90 deg about Z: (x, y) -> (-y, x).
    // Corners (0,0),(2,0),(0,1),(2,1) -> (0,0),(0,2),(-1,0),(-1,2).
    // => x in [-1, 0], y in [0, 2], z unchanged in [0, 1].
    let local = aabb([0.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
    let rot = Quat::from_rotation_z(core::f32::consts::FRAC_PI_2);
    let gt = GlobalTransform::from_transform(&Transform::from_rotation(rot));
    let world = world_aabb(&gt, local);
    assert!(box_approx(
        world,
        aabb([-1.0, 0.0, 0.0], [0.0, 2.0, 1.0]),
        1e-5
    ));
}

#[test]
fn transform_aabb_free_fn_matches_world_aabb() {
    let local = aabb([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]);
    let gt = translated(3.0, -2.0, 1.0);
    let affine = gt.affine();
    let via_fn = transform_aabb(affine.matrix3, affine.translation, local);
    assert!(box_approx(via_fn, world_aabb(&gt, local), 1e-6));
}

// ---- insert / update / remove ----------------------------------------------

#[test]
fn register_then_observe_emits_insert() {
    let mut sync = SpatialSync::new();
    let local = aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    sync.register(node(0), local);
    sync.observe(&[node(0)], &[translated(5.0, 0.0, 0.0)]);
    let (cmds, stats) = sync.drain();
    assert_eq!(stats.inserts, 1);
    assert_eq!(stats.updates, 0);
    assert_eq!(stats.removes, 0);
    assert_eq!(cmds.len(), 1);
    match cmds[0] {
        SpatialCommand::Insert { node: n, bounds } => {
            assert_eq!(n, node(0));
            assert!(box_approx(
                bounds,
                aabb([4.0, -1.0, -1.0], [6.0, 1.0, 1.0]),
                1e-6
            ));
        }
        other => panic!("expected insert, got {other:?}"),
    }
    assert_eq!(sync.live(), 1);
}

#[test]
fn move_after_insert_emits_update_with_old_and_new() {
    let mut sync = SpatialSync::new();
    let local = aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    sync.register(node(0), local);
    sync.observe(&[node(0)], &[translated(0.0, 0.0, 0.0)]);
    let _ = sync.drain();

    sync.observe(&[node(0)], &[translated(10.0, 0.0, 0.0)]);
    let (cmds, stats) = sync.drain();
    assert_eq!(stats.updates, 1);
    assert_eq!(cmds.len(), 1);
    match cmds[0] {
        SpatialCommand::Update { node: n, old, new } => {
            assert_eq!(n, node(0));
            assert!(box_approx(old, aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]), 1e-6));
            assert!(box_approx(new, aabb([9.0, -1.0, -1.0], [11.0, 1.0, 1.0]), 1e-6));
        }
        other => panic!("expected update, got {other:?}"),
    }
}

#[test]
fn unchanged_transform_emits_no_update() {
    let mut sync = SpatialSync::new();
    let local = aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    sync.register(node(0), local);
    sync.observe(&[node(0)], &[translated(2.0, 0.0, 0.0)]);
    let _ = sync.drain();

    // Observe the same pose again: net box identical, so no command.
    sync.observe(&[node(0)], &[translated(2.0, 0.0, 0.0)]);
    let (cmds, stats) = sync.drain();
    assert!(cmds.is_empty());
    assert_eq!(stats.total(), 0);
}

#[test]
fn remove_emits_remove_then_forgets_proxy() {
    let mut sync = SpatialSync::new();
    let local = aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    sync.register(node(0), local);
    sync.observe(&[node(0)], &[translated(0.0, 0.0, 0.0)]);
    let _ = sync.drain();
    assert_eq!(sync.live(), 1);

    sync.remove(node(0));
    let (cmds, stats) = sync.drain();
    assert_eq!(stats.removes, 1);
    assert_eq!(cmds.len(), 1);
    match cmds[0] {
        SpatialCommand::Remove { node: n, old } => {
            assert_eq!(n, node(0));
            assert!(box_approx(old, aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]), 1e-6));
        }
        other => panic!("expected remove, got {other:?}"),
    }
    assert_eq!(sync.live(), 0);
    assert!(!sync.contains(node(0)));
    assert!(sync.is_empty());
}

#[test]
fn remove_before_insert_emits_nothing() {
    let mut sync = SpatialSync::new();
    let local = aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    sync.register(node(0), local);
    // Removed before it ever went live (no observe/drain cycle in between).
    sync.remove(node(0));
    let (cmds, stats) = sync.drain();
    assert!(cmds.is_empty());
    assert_eq!(stats.total(), 0);
    assert!(sync.is_empty());
}

// ---- margin / refit policy --------------------------------------------------

#[test]
fn fat_margin_suppresses_small_moves_until_escape() {
    let mut sync = SpatialSync::with_margin(1.0);
    assert_eq!(sync.margin(), 1.0);
    let local = aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    sync.register(node(0), local);
    sync.observe(&[node(0)], &[translated(0.0, 0.0, 0.0)]);
    let (cmds, _) = sync.drain();
    // Insert stores the fattened box: tight [-1,1] expanded by 1 => [-2,2].
    match cmds[0] {
        SpatialCommand::Insert { bounds, .. } => {
            assert!(box_approx(bounds, aabb([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]), 1e-6));
        }
        other => panic!("expected insert, got {other:?}"),
    }

    // Small move: tight box [-0.5, 1.5] still inside the fat box [-2, 2] -> no refit.
    sync.observe(&[node(0)], &[translated(0.5, 0.0, 0.0)]);
    let (cmds, stats) = sync.drain();
    assert!(cmds.is_empty(), "small move inside margin must not refit");
    assert_eq!(stats.total(), 0);

    // Large move: tight box [4, 6] escapes the fat box -> refit.
    sync.observe(&[node(0)], &[translated(5.0, 0.0, 0.0)]);
    let (cmds, stats) = sync.drain();
    assert_eq!(stats.updates, 1);
    match cmds[0] {
        SpatialCommand::Update { new, .. } => {
            assert!(box_approx(new, aabb([3.0, -2.0, -2.0], [7.0, 2.0, 2.0]), 1e-6));
        }
        other => panic!("expected update, got {other:?}"),
    }
}

#[test]
fn exact_mode_reports_shrink() {
    // margin 0 reports any change, including a box that shrinks within the old.
    let mut sync = SpatialSync::new();
    sync.register(node(0), aabb([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]));
    sync.observe(&[node(0)], &[GlobalTransform::IDENTITY]);
    let _ = sync.drain();

    // Shrink the proxy by re-registering a smaller local box.
    sync.register(node(0), aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]));
    sync.observe(&[node(0)], &[GlobalTransform::IDENTITY]);
    let (cmds, stats) = sync.drain();
    assert_eq!(stats.updates, 1, "exact mode must report a shrink");
    match cmds[0] {
        SpatialCommand::Update { old, new, .. } => {
            assert!(box_approx(old, aabb([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]), 1e-6));
            assert!(box_approx(new, aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]), 1e-6));
        }
        other => panic!("expected update, got {other:?}"),
    }
}

// ---- determinism ------------------------------------------------------------

#[test]
fn commands_are_sorted_ascending_regardless_of_touch_order() {
    let mut sync = SpatialSync::new();
    let unit = aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    // Register out of order.
    for i in [3u32, 0, 2, 1] {
        sync.register(node(i), unit);
    }
    let globals = [
        translated(0.0, 0.0, 0.0),
        translated(1.0, 0.0, 0.0),
        translated(2.0, 0.0, 0.0),
        translated(3.0, 0.0, 0.0),
    ];
    // Observe dirty set out of order too.
    sync.observe(&[node(2), node(0), node(3), node(1)], &globals);
    let (cmds, stats) = sync.drain();
    assert_eq!(stats.inserts, 4);
    let order: Vec<u32> = cmds.iter().map(|c| c.node().index() as u32).collect();
    assert_eq!(order, Vec::from([0u32, 1, 2, 3]));
}

#[test]
fn untracked_dirty_nodes_are_ignored() {
    let mut sync = SpatialSync::new();
    sync.register(node(1), aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]));
    let globals = [
        translated(0.0, 0.0, 0.0),
        translated(1.0, 0.0, 0.0),
        translated(2.0, 0.0, 0.0),
    ];
    // Nodes 0 and 2 are dirty but not tracked; only node 1 produces a command.
    sync.observe(&[node(0), node(1), node(2)], &globals);
    let (cmds, stats) = sync.drain();
    assert_eq!(stats.inserts, 1);
    assert_eq!(cmds.len(), 1);
    assert_eq!(cmds[0].node(), node(1));
}

#[test]
fn repeated_observe_in_frame_keeps_latest_pose() {
    let mut sync = SpatialSync::new();
    sync.register(node(0), aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]));
    // Several observations in one frame; only the last survives to drain.
    sync.observe(&[node(0)], &[translated(1.0, 0.0, 0.0)]);
    sync.observe(&[node(0)], &[translated(2.0, 0.0, 0.0)]);
    sync.observe(&[node(0)], &[translated(7.0, 0.0, 0.0)]);
    let (cmds, _) = sync.drain();
    assert_eq!(cmds.len(), 1);
    match cmds[0] {
        SpatialCommand::Insert { bounds, .. } => {
            assert!(box_approx(bounds, aabb([7.0, 0.0, 0.0], [8.0, 1.0, 1.0]), 1e-6));
        }
        other => panic!("expected insert, got {other:?}"),
    }
}

#[test]
fn drain_into_appends_without_realloc_churn() {
    let mut sync = SpatialSync::new();
    sync.register(node(0), aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]));
    sync.observe(&[node(0)], &[translated(0.0, 0.0, 0.0)]);
    let mut buf = Vec::new();
    let stats = sync.drain_into(&mut buf);
    assert_eq!(stats.inserts, 1);
    assert_eq!(buf.len(), 1);
    assert!(buf[0].is_insert());
}

#[test]
fn stored_bounds_tracks_live_box() {
    let mut sync = SpatialSync::with_margin(0.5);
    sync.register(node(0), aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]));
    assert!(sync.stored_bounds(node(0)).is_none(), "not live before drain");
    sync.observe(&[node(0)], &[GlobalTransform::IDENTITY]);
    let _ = sync.drain();
    let stored = sync.stored_bounds(node(0)).expect("live after drain");
    assert!(box_approx(stored, aabb([-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]), 1e-6));
}
