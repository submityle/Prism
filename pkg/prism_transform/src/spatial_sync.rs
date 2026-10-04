//! §24.5 Spatial acceleration structure incremental sync (`BVH` / grid hash).
//!
//! Spatial queries — picking, range search, broad-phase collision, culling —
//! all lean on an acceleration structure (a `BVH`, a grid hash, …). When an
//! entity's transform changes, that structure must be updated **incrementally**
//! (refit / re-insert the one moved proxy) rather than rebuilt from scratch
//! every frame. [`SpatialSync`] is the pure, deterministic bridge that turns a
//! stream of transform changes into the minimal list of structure edits:
//!
//! - Each tracked entity owns an **object-space** (local) [`Aabb3`] registered
//!   with [`SpatialSync::register`].
//! - [`SpatialSync::observe`] consumes a dirty [`NodeId`] set plus the freshly
//!   propagated world [`GlobalTransform`]s, recomputes each dirty proxy's
//!   **world-space** box (by transforming the local box's eight corners and
//!   refitting), and records it as pending.
//! - [`SpatialSync::drain`] emits an ordered [`SpatialCommand`] stream —
//!   `Insert` for newly-tracked proxies, `Update` for proxies that moved enough
//!   to need a refit, `Remove` for untracked proxies — sorted ascending by
//!   [`NodeId`] so the output is fully deterministic.
//!
//! ## Refit policy (the `margin`)
//! A dynamic-tree acceleration structure usually stores a **fattened** box so a
//! proxy that jitters within its margin needs no refit. [`SpatialSync`] encodes
//! the same discipline:
//!
//! - `margin == 0.0` (exact mode): any change to the tight world box emits an
//!   `Update` (grow **or** shrink).
//! - `margin > 0.0` (fat mode): the stored box is the tight box grown by
//!   `margin`; an `Update` is emitted only when the new tight box **escapes**
//!   the stored fat box, at which point the proxy is re-fattened. Small moves
//!   that stay inside the margin produce no command.
//!
//! This module is `no_std` + `alloc`, pure math with no threads and no ECS
//! binding: the caller supplies the dirty set and world poses (e.g. from
//! [`crate::dirty`] or [`crate::observer`]) and feeds the emitted commands into
//! whatever real acceleration structure lives in the spatial-index crate. The
//! honest boundary in design-doc §24.9 records what is intentionally left to
//! the consumer (physics / culling / navigation).

use alloc::vec::Vec;

use prism_math::{Aabb3, Mat3, Vec3};

use crate::hierarchy::NodeId;
use crate::GlobalTransform;

/// Transform the eight corners of a local-space [`Aabb3`] by `linear` + a
/// translation and refit the tightest world-space [`Aabb3`] around them.
///
/// Because an [`Aabb3`] is convex and its corners are the extreme points under
/// any affine map, the box of the transformed corners is the exact tight world
/// bound of the transformed box (no slack, no missed volume).
#[inline]
#[must_use]
pub fn transform_aabb(linear: Mat3, translation: Vec3, local: Aabb3) -> Aabb3 {
    let corners = local.corners();
    let mut world = linear.mul_vec3(corners[0]) + translation;
    let mut bb = Aabb3 {
        min: world,
        max: world,
    };
    for &corner in &corners[1..] {
        world = linear.mul_vec3(corner) + translation;
        bb = bb.expand_to_include(world);
    }
    bb
}

/// Compute the world-space [`Aabb3`] of a proxy whose object-space box is
/// `local`, placed by the world transform `world`.
#[inline]
#[must_use]
pub fn world_aabb(world: &GlobalTransform, local: Aabb3) -> Aabb3 {
    let affine = world.affine();
    transform_aabb(affine.matrix3, affine.translation, local)
}

/// One edit the acceleration structure must apply to stay in sync with the
/// transform hierarchy.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum SpatialCommand {
    /// A newly-tracked proxy enters the structure with this world box.
    Insert {
        /// The proxy's node.
        node: NodeId,
        /// The (fattened, if `margin > 0`) world box to insert.
        bounds: Aabb3,
    },
    /// An existing proxy moved enough to require a refit.
    Update {
        /// The proxy's node.
        node: NodeId,
        /// The box currently stored in the structure.
        old: Aabb3,
        /// The box the structure should hold after the refit.
        new: Aabb3,
    },
    /// A proxy left the structure (was removed from tracking).
    Remove {
        /// The proxy's node.
        node: NodeId,
        /// The box that was stored in the structure before removal.
        old: Aabb3,
    },
}

impl SpatialCommand {
    /// The node this command targets.
    #[inline]
    #[must_use]
    pub fn node(&self) -> NodeId {
        match *self {
            SpatialCommand::Insert { node, .. }
            | SpatialCommand::Update { node, .. }
            | SpatialCommand::Remove { node, .. } => node,
        }
    }

    /// Whether this command inserts a proxy.
    #[inline]
    #[must_use]
    pub fn is_insert(&self) -> bool {
        matches!(self, SpatialCommand::Insert { .. })
    }

    /// Whether this command updates a proxy.
    #[inline]
    #[must_use]
    pub fn is_update(&self) -> bool {
        matches!(self, SpatialCommand::Update { .. })
    }

    /// Whether this command removes a proxy.
    #[inline]
    #[must_use]
    pub fn is_remove(&self) -> bool {
        matches!(self, SpatialCommand::Remove { .. })
    }
}

/// Per-proxy tracking state inside a [`SpatialSync`].
#[derive(Clone, Copy, Debug)]
struct Proxy {
    /// Object-space bounds registered by the caller.
    local: Aabb3,
    /// The world box currently stored in the acceleration structure (the
    /// fattened box in fat mode). Only meaningful when `live` is `true`.
    stored: Aabb3,
    /// Whether an `Insert` has been emitted and no matching `Remove` yet.
    live: bool,
    /// The tight world box observed this frame, if the proxy was touched.
    pending: Option<Aabb3>,
    /// Whether the proxy is scheduled to be removed on the next drain.
    removing: bool,
}

/// Counts summarizing one [`SpatialSync::drain`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct SyncStats {
    /// Number of `Insert` commands emitted.
    pub inserts: usize,
    /// Number of `Update` commands emitted.
    pub updates: usize,
    /// Number of `Remove` commands emitted.
    pub removes: usize,
}

impl SyncStats {
    /// Total number of commands emitted.
    #[inline]
    #[must_use]
    pub fn total(self) -> usize {
        self.inserts + self.updates + self.removes
    }
}

/// Deterministic bridge from transform changes to acceleration-structure edits.
///
/// See the [module docs](crate::spatial_sync) for the data model and refit
/// policy.
#[derive(Clone, Debug, Default)]
pub struct SpatialSync {
    /// Per-node proxy slots (dense, `None` where no proxy is tracked).
    entries: Vec<Option<Proxy>>,
    /// Nodes mutated since the last drain, used to bound the drain scan.
    touched: Vec<usize>,
    /// Refit margin; see the module docs.
    margin: f32,
    /// Number of currently-live proxies.
    live: usize,
}

impl SpatialSync {
    /// Create an empty sync with no refit margin (exact mode).
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
            touched: Vec::new(),
            margin: 0.0,
            live: 0,
        }
    }

    /// Create an empty sync whose stored boxes are fattened by `margin` on
    /// every axis. A negative `margin` is clamped to `0.0`.
    #[inline]
    #[must_use]
    pub fn with_margin(margin: f32) -> Self {
        Self {
            entries: Vec::new(),
            touched: Vec::new(),
            margin: if margin > 0.0 { margin } else { 0.0 },
            live: 0,
        }
    }

    /// The refit margin in use.
    #[inline]
    #[must_use]
    pub fn margin(&self) -> f32 {
        self.margin
    }

    /// Number of proxies currently tracked (registered and not yet removed).
    #[inline]
    #[must_use]
    pub fn tracked(&self) -> usize {
        self.entries.iter().filter(|e| e.is_some()).count()
    }

    /// Number of proxies currently live in the structure (an `Insert` has been
    /// drained and no `Remove` has been drained since).
    #[inline]
    #[must_use]
    pub fn live(&self) -> usize {
        self.live
    }

    /// Whether no proxies are tracked.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.iter().all(Option::is_none)
    }

    /// Whether `node` is currently tracked.
    #[inline]
    #[must_use]
    pub fn contains(&self, node: NodeId) -> bool {
        self.entries
            .get(node.index())
            .is_some_and(|slot| slot.as_ref().is_some_and(|p| !p.removing))
    }

    /// The world box currently stored in the structure for `node`, if the proxy
    /// is live.
    #[inline]
    #[must_use]
    pub fn stored_bounds(&self, node: NodeId) -> Option<Aabb3> {
        self.entries
            .get(node.index())
            .and_then(|slot| slot.as_ref())
            .filter(|p| p.live)
            .map(|p| p.stored)
    }

    #[inline]
    fn ensure_slot(&mut self, index: usize) {
        if self.entries.len() <= index {
            self.entries.resize(index + 1, None);
        }
    }

    #[inline]
    fn mark_touched(&mut self, index: usize) {
        self.touched.push(index);
    }

    /// Register (or re-register) `node` with the given object-space bounds.
    ///
    /// Registering a new proxy makes it eligible for an `Insert` on the next
    /// drain **after** its world box has been observed via
    /// [`SpatialSync::observe`]. Re-registering an existing proxy replaces its
    /// local bounds; the change is reflected on the next observe + drain.
    pub fn register(&mut self, node: NodeId, local: Aabb3) {
        let index = node.index();
        self.ensure_slot(index);
        match &mut self.entries[index] {
            Some(proxy) => {
                proxy.local = local;
                proxy.removing = false;
            }
            slot @ None => {
                *slot = Some(Proxy {
                    local,
                    stored: local,
                    live: false,
                    pending: None,
                    removing: false,
                });
            }
        }
        self.mark_touched(index);
    }

    /// Schedule `node` for removal. The matching `Remove` command (carrying the
    /// box last stored in the structure) is emitted on the next drain, after
    /// which the proxy is forgotten. Removing an untracked node is a no-op.
    pub fn remove(&mut self, node: NodeId) {
        let index = node.index();
        if let Some(Some(proxy)) = self.entries.get_mut(index) {
            proxy.removing = true;
            proxy.pending = None;
            self.mark_touched(index);
        }
    }

    /// Recompute world boxes for the dirty proxies.
    ///
    /// For every `node` in `dirty` that is a tracked, non-removing proxy, the
    /// proxy's world box is recomputed from `globals[node]` and its local
    /// bounds and stored as pending for the next drain. Dirty nodes that are
    /// not tracked proxies are ignored, so the caller can feed the whole
    /// hierarchy dirty set without pre-filtering.
    ///
    /// # Panics
    /// Panics if a tracked dirty `node` index is out of bounds for `globals`.
    pub fn observe(&mut self, dirty: &[NodeId], globals: &[GlobalTransform]) {
        for &node in dirty {
            let index = node.index();
            let Some(Some(proxy)) = self.entries.get(index) else {
                continue;
            };
            if proxy.removing {
                continue;
            }
            let local = proxy.local;
            let world = world_aabb(&globals[index], local);
            // Re-borrow mutably now that the immutable read is done.
            if let Some(Some(proxy)) = self.entries.get_mut(index) {
                if proxy.pending.is_none() {
                    self.touched.push(index);
                }
                proxy.pending = Some(world);
            }
        }
    }

    /// Emit the ordered command stream for everything that changed since the
    /// last drain, advancing the stored state, and return per-kind counts.
    ///
    /// Commands are sorted ascending by [`NodeId`], so the output is a
    /// deterministic function of the observed changes regardless of the order
    /// in which [`SpatialSync::register`] / [`SpatialSync::observe`] /
    /// [`SpatialSync::remove`] were called.
    pub fn drain(&mut self) -> (Vec<SpatialCommand>, SyncStats) {
        let mut out = Vec::new();
        let stats = self.drain_into(&mut out);
        (out, stats)
    }

    /// Like [`SpatialSync::drain`] but appends into a caller-owned buffer,
    /// avoiding an allocation when draining every frame.
    pub fn drain_into(&mut self, out: &mut Vec<SpatialCommand>) -> SyncStats {
        self.touched.sort_unstable();
        self.touched.dedup();
        let touched = core::mem::take(&mut self.touched);
        let margin = self.margin;
        let mut stats = SyncStats::default();

        for index in touched {
            let Some(slot) = self.entries.get_mut(index) else {
                continue;
            };
            let Some(proxy) = slot.as_mut() else {
                continue;
            };
            let node = NodeId::new(index as u32);

            if proxy.removing {
                if proxy.live {
                    out.push(SpatialCommand::Remove {
                        node,
                        old: proxy.stored,
                    });
                    stats.removes += 1;
                    self.live -= 1;
                }
                *slot = None;
                continue;
            }

            let Some(tight) = proxy.pending.take() else {
                continue;
            };

            if proxy.live {
                // Fat mode refits only when the tight box escapes the stored
                // fat box; exact mode (margin 0) refits on any change.
                let needs_refit = if margin > 0.0 {
                    !proxy.stored.contains_aabb(tight)
                } else {
                    proxy.stored != tight
                };
                if needs_refit {
                    let old = proxy.stored;
                    let new = if margin > 0.0 { tight.expand(margin) } else { tight };
                    proxy.stored = new;
                    out.push(SpatialCommand::Update { node, old, new });
                    stats.updates += 1;
                }
            } else {
                let stored = if margin > 0.0 { tight.expand(margin) } else { tight };
                proxy.stored = stored;
                proxy.live = true;
                self.live += 1;
                out.push(SpatialCommand::Insert {
                    node,
                    bounds: stored,
                });
                stats.inserts += 1;
            }
        }

        stats
    }
}
