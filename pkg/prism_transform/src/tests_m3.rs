//! M3 tests: multi-threaded level-parallel propagation is bit-for-bit identical
//! to the serial full pass, across large random forests and edge cases.

use crate::hierarchy::{Hierarchy, HierarchyError, NodeId};
use crate::parallel::{LevelPlan, PARALLEL_THRESHOLD, propagate_parallel, propagate_parallel_with_plan};
use crate::propagation::{identity_globals, propagate};
use crate::{GlobalTransform, Transform};
use prism_math::{Quat, Vec3, vec3};
use prism_tasks::TaskPool;

/// Deterministic xorshift64* PRNG so the differential test is reproducible
/// without a dev-dependency.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
    /// A transform that stays numerically finite when composed over a very deep
    /// chain: unit scale (no multiplicative blow-up) with a small rotation and
    /// translation. Used only by the deep-chain parity test.
    fn transform_mild(&mut self) -> Transform {
        let axis = vec3(
            self.range(-1.0, 1.0),
            self.range(-1.0, 1.0),
            self.range(-1.0, 1.0),
        );
        let axis = if axis.length() < 1e-3 { Vec3::Y } else { axis };
        Transform {
            translation: vec3(
                self.range(-0.5, 0.5),
                self.range(-0.5, 0.5),
                self.range(-0.5, 0.5),
            ),
            rotation: Quat::from_axis_angle(axis, self.range(-0.2, 0.2)),
            scale: Vec3::ONE,
        }
    }
    fn transform(&mut self) -> Transform {
        let axis = vec3(
            self.range(-1.0, 1.0),
            self.range(-1.0, 1.0),
            self.range(-1.0, 1.0),
        );
        // Avoid a zero-length axis; fall back to +Y.
        let axis = if axis.length() < 1e-3 { Vec3::Y } else { axis };
        Transform {
            translation: vec3(
                self.range(-50.0, 50.0),
                self.range(-50.0, 50.0),
                self.range(-50.0, 50.0),
            ),
            // Non-uniform scale so hierarchical shear actually exercises Affine3.
            rotation: Quat::from_axis_angle(axis, self.range(-3.14, 3.14)),
            scale: vec3(
                self.range(0.3, 2.5),
                self.range(0.3, 2.5),
                self.range(0.3, 2.5),
            ),
        }
    }
}

/// Build a random forest of `n` nodes with several roots and a mix of depths,
/// returning the hierarchy and a parallel `locals` buffer.
fn random_forest(n: usize, seed: u64) -> (Hierarchy, Vec<Transform>) {
    let mut rng = Rng::new(seed);
    let mut h = Hierarchy::new();
    let mut locals = Vec::with_capacity(n);
    let mut ids: Vec<NodeId> = Vec::with_capacity(n);
    for i in 0..n {
        // ~1-in-8 nodes is a fresh root; the rest attach to an existing node so
        // depth grows and levels get wide.
        let node = if ids.is_empty() || rng.below(8) == 0 {
            h.spawn_root()
        } else {
            let parent = ids[rng.below(ids.len())];
            h.spawn_child(parent)
        };
        debug_assert_eq!(node.index(), i);
        ids.push(node);
        locals.push(rng.transform());
    }
    (h, locals)
}

/// Serial reference and both parallel entry points must agree node-for-node,
/// exactly (same operand order ⇒ identical floating-point result).
fn assert_parallel_equals_serial(pool: &TaskPool, h: &Hierarchy, locals: &[Transform]) {
    let mut serial = identity_globals(h);
    propagate(h, locals, &mut serial).expect("serial propagate");

    let mut par = identity_globals(h);
    propagate_parallel(pool, h, locals, &mut par).expect("parallel propagate");

    let mut par_plan_globals = identity_globals(h);
    let plan = LevelPlan::build(h).expect("plan build");
    propagate_parallel_with_plan(pool, &plan, h, locals, &mut par_plan_globals);

    for i in 0..h.len() {
        let s = serial[i].affine();
        let p = par[i].affine();
        let pp = par_plan_globals[i].affine();
        assert_eq!(s.matrix3.x_axis, p.matrix3.x_axis, "node {i} x_axis (fresh plan)");
        assert_eq!(s.matrix3.y_axis, p.matrix3.y_axis, "node {i} y_axis (fresh plan)");
        assert_eq!(s.matrix3.z_axis, p.matrix3.z_axis, "node {i} z_axis (fresh plan)");
        assert_eq!(s.translation, p.translation, "node {i} translation (fresh plan)");
        assert_eq!(s.matrix3.x_axis, pp.matrix3.x_axis, "node {i} x_axis (cached plan)");
        assert_eq!(s.matrix3.y_axis, pp.matrix3.y_axis, "node {i} y_axis (cached plan)");
        assert_eq!(s.matrix3.z_axis, pp.matrix3.z_axis, "node {i} z_axis (cached plan)");
        assert_eq!(s.translation, pp.translation, "node {i} translation (cached plan)");
    }
}

#[test]
fn parallel_equals_serial_on_large_forest() {
    let pool = TaskPool::new();
    // Comfortably above PARALLEL_THRESHOLD so several levels take the task path.
    let n = PARALLEL_THRESHOLD * 12 + 37;
    let (h, locals) = random_forest(n, 0xC0FFEE_1234);
    assert!(h.len() >= n);
    assert_parallel_equals_serial(&pool, &h, &locals);
}

#[test]
fn parallel_equals_serial_across_many_seeds() {
    let pool = TaskPool::new();
    for seed in 0..8u64 {
        let n = PARALLEL_THRESHOLD * 2 + (seed as usize) * 17 + 5;
        let (h, locals) = random_forest(n, 0xABCD_0000 ^ (seed.wrapping_mul(0x9E37_79B9)));
        assert_parallel_equals_serial(&pool, &h, &locals);
    }
}

#[test]
fn level_plan_buckets_cover_every_node_once() {
    let (h, _locals) = random_forest(PARALLEL_THRESHOLD * 3, 0x5EED);
    let plan = LevelPlan::build(&h).expect("plan build");
    assert_eq!(plan.len(), h.len());
    assert!(!plan.is_empty());
    assert!(plan.level_count() >= 1);
}

#[test]
fn empty_hierarchy_is_a_noop() {
    let pool = TaskPool::new();
    let h = Hierarchy::new();
    let locals: Vec<Transform> = Vec::new();
    let mut globals: Vec<GlobalTransform> = Vec::new();
    propagate_parallel(&pool, &h, &locals, &mut globals).expect("empty ok");
    assert!(globals.is_empty());
}

#[test]
fn single_node_matches_serial() {
    let pool = TaskPool::new();
    let mut h = Hierarchy::new();
    h.spawn_root();
    let locals = vec![Transform::from_xyz(3.0, -2.0, 7.0)];
    assert_parallel_equals_serial(&pool, &h, &locals);
}

#[test]
fn deep_chain_matches_serial() {
    // A pathological single chain: every level has exactly one node, so the
    // parallel path degrades to the serial inline sweep but must still agree.
    let pool = TaskPool::new();
    let mut h = Hierarchy::new();
    let mut rng = Rng::new(0x1357);
    let mut locals = Vec::new();
    let root = h.spawn_root();
    locals.push(rng.transform_mild());
    let mut parent = root;
    for _ in 0..500 {
        parent = h.spawn_child(parent);
        locals.push(rng.transform_mild());
    }
    assert_parallel_equals_serial(&pool, &h, &locals);
}

#[test]
fn length_mismatch_is_rejected() {
    let pool = TaskPool::new();
    let (h, locals) = random_forest(64, 0x42);
    let mut too_short = identity_globals(&h);
    too_short.pop();
    assert_eq!(
        propagate_parallel(&pool, &h, &locals, &mut too_short),
        Err(HierarchyError::LengthMismatch),
    );

    let mut globals = identity_globals(&h);
    let short_locals = &locals[..locals.len() - 1];
    assert_eq!(
        propagate_parallel(&pool, &h, short_locals, &mut globals),
        Err(HierarchyError::LengthMismatch),
    );
}
