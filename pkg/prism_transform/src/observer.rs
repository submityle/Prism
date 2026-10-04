//! §24.4 Transform change hooks (`Observer`) — dirty / change notification.
//!
//! Systems downstream of the transform hierarchy — spatial acceleration
//! structures, audio emitters, physics proxies, networking replicators — do not
//! want to re-read every world pose each frame; they want to be told *which*
//! nodes actually moved and by how much. [`TransformObserver`] is the pure data
//! structure that produces that signal:
//!
//! - A per-node **baseline** ([`TransformObserver::prime`]) records the last
//!   world pose that was already reported to subscribers.
//! - During a frame, one or more [`TransformObserver::observe`] calls fold the
//!   dirty nodes and their freshly propagated world poses into a pending set,
//!   **merging** repeated touches of the same node within the frame.
//! - [`TransformObserver::flush`] emits one [`TransformChange`] per node whose
//!   net world pose differs from its baseline, in ascending [`NodeId`] order,
//!   updates the baseline, and advances the [`TransformObserver::epoch`].
//!
//! The propagation is **deterministic**: the emitted order depends only on the
//! node indices, a node that is dirtied several times in a frame produces a
//! single net change, and a node that moves and then moves back to its baseline
//! within the same frame produces no change at all. The structure is pure
//! `alloc` state with no threads, no clock, and no ECS binding — the caller
//! supplies the dirty set (e.g. from [`crate::dirty`]) and the world poses, and
//! consumes the returned changes however it likes (direct callbacks via
//! [`TransformObserver::dispatch`] or an event queue via
//! [`TransformObserver::flush`]). The honest boundary in the design doc §24.9
//! records what is intentionally left to the caller.

use alloc::vec::Vec;

use prism_math::Vec3;

use crate::hierarchy::NodeId;
use crate::GlobalTransform;

/// Which parts of a world pose changed between the baseline and the new pose.
///
/// `basis` covers the whole linear `3x3` part (rotation **and** scale/shear
/// combined); splitting it into independent rotation / scale deltas would
/// require decomposing the affine, which the caller can do from the `old` / `new`
/// poses carried by the [`TransformChange`] if it needs that resolution.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct ChangeMask {
    /// The world translation changed by more than the queried epsilon.
    pub translation: bool,
    /// The world linear basis (`matrix3`) changed by more than the queried
    /// epsilon.
    pub basis: bool,
}

impl ChangeMask {
    /// Whether any tracked part changed.
    #[inline]
    pub const fn any(self) -> bool {
        self.translation || self.basis
    }
}

/// A net world-pose change for a single node over one observed frame.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TransformChange {
    /// The node whose world pose changed.
    pub node: NodeId,
    /// The world pose previously reported to subscribers (the baseline).
    pub old: GlobalTransform,
    /// The current world pose.
    pub new: GlobalTransform,
}

impl TransformChange {
    /// The world-space translation delta (`new - old`).
    #[inline]
    pub fn translation_delta(&self) -> Vec3 {
        self.new.translation() - self.old.translation()
    }

    /// Whether the translation moved by more than `eps` (compared on squared
    /// length, so `eps` is a distance threshold).
    #[inline]
    pub fn moved(&self, eps: f32) -> bool {
        self.translation_delta().length_squared() > eps * eps
    }

    /// Whether the linear basis (`matrix3`) changed by more than `eps` in any
    /// of its nine entries.
    #[inline]
    pub fn basis_changed(&self, eps: f32) -> bool {
        let a = self.old.0.matrix3;
        let b = self.new.0.matrix3;
        let dx = a.x_axis - b.x_axis;
        let dy = a.y_axis - b.y_axis;
        let dz = a.z_axis - b.z_axis;
        let worst = max6(
            abs_f32(dx.x),
            abs_f32(dx.y),
            abs_f32(dx.z),
            abs_f32(dy.x),
            abs_f32(dy.y),
            abs_f32(dy.z),
        )
        .max(abs_f32(dz.x))
        .max(abs_f32(dz.y))
        .max(abs_f32(dz.z));
        worst > eps
    }

    /// Summarize which parts changed, using `eps` as the threshold for both the
    /// translation distance and each basis entry.
    #[inline]
    pub fn mask(&self, eps: f32) -> ChangeMask {
        ChangeMask {
            translation: self.moved(eps),
            basis: self.basis_changed(eps),
        }
    }
}

/// Deterministic transform-change tracker.
///
/// Keeps a per-node baseline world pose and accumulates dirty observations for
/// one frame; [`TransformObserver::flush`] turns the accumulated set into an
/// ordered list of net [`TransformChange`]s and advances the baseline.
#[derive(Clone, Debug, Default)]
pub struct TransformObserver {
    /// Last world pose already reported to subscribers, per node.
    prev: Vec<GlobalTransform>,
    /// Latest observed world pose this frame, per node (`None` = untouched).
    pending: Vec<Option<GlobalTransform>>,
    /// Nodes touched this frame, in first-touch order (sorted on flush).
    touched: Vec<NodeId>,
    /// Monotonic flush counter; advances once per [`TransformObserver::flush`].
    epoch: u64,
}

impl TransformObserver {
    /// Create an empty observer tracking no nodes.
    #[inline]
    pub const fn new() -> Self {
        Self {
            prev: Vec::new(),
            pending: Vec::new(),
            touched: Vec::new(),
            epoch: 0,
        }
    }

    /// Create an empty observer pre-allocating room for `capacity` nodes.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            prev: Vec::with_capacity(capacity),
            pending: Vec::with_capacity(capacity),
            touched: Vec::new(),
            epoch: 0,
        }
    }

    /// Number of nodes the observer currently tracks.
    #[inline]
    pub fn len(&self) -> usize {
        self.prev.len()
    }

    /// Whether the observer tracks no nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.prev.is_empty()
    }

    /// The current flush epoch (number of completed [`TransformObserver::flush`]
    /// calls).
    #[inline]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Append a tracked node with the given baseline world pose.
    #[inline]
    pub fn push(&mut self, baseline: GlobalTransform) {
        self.prev.push(baseline);
        self.pending.push(None);
    }

    /// Grow the tracker to at least `len` nodes, using
    /// [`GlobalTransform::IDENTITY`] as the baseline for any new slot. A node
    /// that first appears this way reports a change from identity to its real
    /// pose on the next flush, which marks it as newly visible to subscribers.
    #[inline]
    pub fn ensure_len(&mut self, len: usize) {
        if self.prev.len() < len {
            self.prev.resize(len, GlobalTransform::IDENTITY);
            self.pending.resize(len, None);
        }
    }

    /// Reset the baseline of every node to `globals` **without** emitting any
    /// change. Use this to seed the observer after the first propagation pass
    /// so that only subsequent movement is reported.
    ///
    /// Any pending (unflushed) observations are discarded.
    pub fn prime(&mut self, globals: &[GlobalTransform]) {
        self.prev.clear();
        self.prev.extend_from_slice(globals);
        self.pending.clear();
        self.pending.resize(globals.len(), None);
        self.touched.clear();
    }

    /// Fold a dirty set and its freshly propagated world poses into the pending
    /// change set for this frame.
    ///
    /// May be called multiple times per frame; repeated touches of the same
    /// node keep the latest pose, so the eventual [`TransformChange`] always
    /// reflects the net move from the baseline.
    ///
    /// # Panics
    /// Panics if any `dirty` node index is out of bounds for `globals`.
    pub fn observe(&mut self, dirty: &[NodeId], globals: &[GlobalTransform]) {
        for &node in dirty {
            let i = node.index();
            let new_pose = globals[i];
            self.ensure_len(i + 1);
            if self.pending[i].is_none() {
                self.touched.push(node);
            }
            self.pending[i] = Some(new_pose);
        }
    }

    /// Emit one [`TransformChange`] per node whose net world pose differs from
    /// its baseline, in ascending [`NodeId`] order, then advance the baseline of
    /// every emitted node, clear the pending set, and bump the epoch.
    ///
    /// A node that was dirtied but whose net pose equals its baseline (moved and
    /// moved back within the frame) produces no change.
    pub fn flush(&mut self) -> Vec<TransformChange> {
        self.touched.sort_unstable_by_key(|n| n.index());
        let mut out = Vec::new();
        for &node in &self.touched {
            let i = node.index();
            let Some(new_pose) = self.pending[i].take() else {
                continue;
            };
            let old_pose = self.prev[i];
            if old_pose != new_pose {
                out.push(TransformChange {
                    node,
                    old: old_pose,
                    new: new_pose,
                });
                self.prev[i] = new_pose;
            }
        }
        self.touched.clear();
        self.epoch += 1;
        out
    }

    /// Flush the pending changes and invoke `callback` once per emitted
    /// [`TransformChange`], in ascending [`NodeId`] order. Returns the number of
    /// changes dispatched.
    pub fn dispatch<F>(&mut self, mut callback: F) -> usize
    where
        F: FnMut(&TransformChange),
    {
        let changes = self.flush();
        for change in &changes {
            callback(change);
        }
        changes.len()
    }
}

#[inline]
fn abs_f32(x: f32) -> f32 {
    libm::fabsf(x)
}

#[inline]
fn max6(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32) -> f32 {
    a.max(b).max(c).max(d).max(e).max(f)
}
