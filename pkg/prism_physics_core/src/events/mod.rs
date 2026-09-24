//! Physics events and observers.
//!
//! This module turns raw per-frame contact manifolds into higher-level
//! gameplay events and delivers them to interested code:
//!
//! - [`event`] defines the [`PhysicsEvent`] value types and the [`ContactPair`]
//!   key.
//! - [`tracker`] holds [`ContactEventTracker`], which diffs successive frames'
//!   manifolds to emit collision start/end and trigger enter/exit events.
//! - [`observer`] holds the [`Observer`] trait and the [`ObserverRegistry`]
//!   fan-out/queue.
//!
//! The whole module is engine-agnostic: observers are plain callbacks and the
//! tracker is plain data, so this works identically under any front end.
//!
//! # Provenance
//!
//! Contact event derivation and the observer pattern are standard, publicly
//! documented techniques. This module contains no Unreal Engine source or
//! derived code.

pub mod event;
pub mod observer;
pub mod tracker;

pub use event::{ContactPair, PhysicsEvent};
pub use observer::{Observer, ObserverRegistry};
pub use tracker::ContactEventTracker;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collide::ContactManifold;
    use crate::state::body::BodyDesc;
    use crate::state::handle::BodyHandle;
    use crate::state::storage::BodyStorage;
    use glam::{Vec3, Vec3 as V};

    fn manifold(a: BodyHandle, b: BodyHandle) -> ContactManifold {
        ContactManifold::new(a, b, V::Y)
    }

    #[test]
    fn contact_pair_is_order_independent() {
        let mut s = BodyStorage::new();
        let a = s.insert(BodyDesc::dynamic_at(Vec3::ZERO));
        let b = s.insert(BodyDesc::dynamic_at(Vec3::X));
        let ab = ContactPair::new(a, b);
        let ba = ContactPair::new(b, a);
        assert_eq!(ab, ba);
        assert_eq!(ab.other(a), Some(b));
        assert_eq!(ab.other(b), Some(a));
    }

    #[test]
    fn observer_registry_dispatches_and_queues() {
        use core::sync::atomic::{AtomicUsize, Ordering};

        // A process-local counter the closure observer bumps on each event.
        static SEEN: AtomicUsize = AtomicUsize::new(0);
        SEEN.store(0, Ordering::SeqCst);

        let mut registry = ObserverRegistry::new();
        registry.register(|_event: &PhysicsEvent| {
            SEEN.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(registry.observer_count(), 1);

        let mut s = BodyStorage::new();
        let a = s.insert(BodyDesc::dynamic_at(Vec3::ZERO));
        let b = s.insert(BodyDesc::dynamic_at(Vec3::X));
        let events = [PhysicsEvent::CollisionStarted(ContactPair::new(a, b))];
        registry.dispatch(&events);

        // The observer callback fired, and the same event was queued.
        assert_eq!(SEEN.load(Ordering::SeqCst), 1);
        assert_eq!(registry.queued().len(), 1);
        let drained = registry.drain_queue();
        assert_eq!(drained.len(), 1);
        assert!(registry.queued().is_empty());
    }

    #[test]
    fn tracker_emits_collision_start_then_end() {
        let mut s = BodyStorage::new();
        let a = s.insert(BodyDesc::dynamic_at(Vec3::ZERO));
        let b = s.insert(BodyDesc::dynamic_at(Vec3::X));
        let mut tracker = ContactEventTracker::new();

        // Frame 1: they touch -> one start.
        let started = tracker.record(&[manifold(a, b)], |h| s.is_sensor(h).unwrap_or(false));
        assert_eq!(started.len(), 1);
        assert!(matches!(started[0], PhysicsEvent::CollisionStarted(_)));
        assert_eq!(tracker.active_collision_count(), 1);

        // Frame 2: still touching -> no event.
        let steady = tracker.record(&[manifold(a, b)], |h| s.is_sensor(h).unwrap_or(false));
        assert!(steady.is_empty());

        // Frame 3: separated -> one end.
        let ended = tracker.record(&[], |h| s.is_sensor(h).unwrap_or(false));
        assert_eq!(ended.len(), 1);
        assert!(matches!(ended[0], PhysicsEvent::CollisionEnded(_)));
        assert_eq!(tracker.active_collision_count(), 0);
    }

    #[test]
    fn tracker_emits_trigger_enter_and_exit_with_roles() {
        let mut s = BodyStorage::new();
        let sensor = s.insert(BodyDesc::dynamic_at(Vec3::ZERO).with_sensor(true));
        let other = s.insert(BodyDesc::dynamic_at(Vec3::X));
        let mut tracker = ContactEventTracker::new();

        let enter = tracker.record(&[manifold(sensor, other)], |h| {
            s.is_sensor(h).unwrap_or(false)
        });
        assert_eq!(enter.len(), 1);
        match enter[0] {
            PhysicsEvent::TriggerEntered {
                sensor: se,
                other: ot,
            } => {
                assert_eq!(se, sensor);
                assert_eq!(ot, other);
            }
            other => panic!("expected TriggerEntered, got {other:?}"),
        }
        assert_eq!(tracker.active_trigger_count(), 1);
        assert_eq!(tracker.active_collision_count(), 0);

        let exit = tracker.record(&[], |h| s.is_sensor(h).unwrap_or(false));
        assert_eq!(exit.len(), 1);
        match exit[0] {
            PhysicsEvent::TriggerExited {
                sensor: se,
                other: ot,
            } => {
                assert_eq!(se, sensor);
                assert_eq!(ot, other);
            }
            other => panic!("expected TriggerExited, got {other:?}"),
        }
    }

    #[test]
    fn reset_forgets_history() {
        let mut s = BodyStorage::new();
        let a = s.insert(BodyDesc::dynamic_at(Vec3::ZERO));
        let b = s.insert(BodyDesc::dynamic_at(Vec3::X));
        let mut tracker = ContactEventTracker::new();
        let _ = tracker.record(&[manifold(a, b)], |h| s.is_sensor(h).unwrap_or(false));
        assert_eq!(tracker.active_collision_count(), 1);
        tracker.reset();
        assert_eq!(tracker.active_collision_count(), 0);
        // After reset the same overlap counts as a fresh start again.
        let again = tracker.record(&[manifold(a, b)], |h| s.is_sensor(h).unwrap_or(false));
        assert_eq!(again.len(), 1);
        assert!(matches!(again[0], PhysicsEvent::CollisionStarted(_)));
    }
}
