//! Island-level sleep bookkeeping for the XPBD solver.
//!
//! Sleeping lets settled islands stop consuming solver time: once every dynamic
//! body in an island has stayed below the configured velocity thresholds for
//! [`SleepConfig::time_to_sleep`](crate::sleep::SleepConfig) seconds, the whole
//! island is put to sleep and skipped by prediction and the constraint solve
//! until something disturbs it.
//!
//! This module exposes the two hooks the solver calls each sub-step:
//!
//! - [`classify_and_wake`] runs before the solve. It reports which islands are
//!   active (must be solved) and wakes any sleeping body that shares an active
//!   island, re-stamping its previous pose so velocity recovery stays sane.
//! - [`update_after_solve`] runs after the velocity solve. It advances or resets
//!   each active island's idle timers and sleeps islands that have settled.
//!
//! The per-body sleep flags and idle timers live as columns on
//! [`BodyStorage`](crate::state::storage::BodyStorage) and are reached here
//! through the mutable [`BodySolverView`].
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! island-sleep policy is built from Prism's own union-find islands and the
//! pure timing helpers in [`crate::sleep`].

use crate::sleep::{advance_sleep_timer, ready_to_sleep, SleepConfig};
use crate::solver::xpbd::island_solve::SolveIslands;
use crate::state::view::BodySolverView;
use glam::Vec3;

/// Classifies every island as *frozen* (all dynamic members sleeping) or
/// *active*, and wakes the sleeping members of active islands.
///
/// Returns the indices of the islands that must be solved this sub-step. An
/// island is frozen only when sleeping is enabled and it has at least one
/// member and *every* member is asleep; such islands are skipped entirely so
/// their bodies neither move nor cost solver time. Any island that still has an
/// awake member is active: its sleeping members (for example a resting box that
/// a moving neighbour just contacted) are woken, and their previous pose is
/// re-stamped to the current pose so [`recover_velocities`] does not read a
/// stale finite difference.
///
/// [`recover_velocities`]: super::integrate::recover_velocities
#[must_use]
pub fn classify_and_wake(
    view: &mut BodySolverView<'_>,
    islands: &SolveIslands,
    config: &SleepConfig,
) -> Vec<usize> {
    let mut active = Vec::new();
    for island in 0..islands.island_count() {
        let members = islands.members(island);
        let frozen = config.enabled
            && !members.is_empty()
            && members.iter().all(|&slot| view.is_sleeping_slot(slot));
        if frozen {
            continue;
        }
        for &slot in members {
            if view.is_sleeping_slot(slot) {
                wake_slot(view, slot);
            }
        }
        active.push(island);
    }
    active
}

/// Advances idle timers for the active islands and sleeps those that settled.
///
/// For each active island, if every dynamic member is below the sleep velocity
/// thresholds this sub-step, its members' idle timers grow by `h` and the whole
/// island sleeps once the smallest member timer reaches
/// [`SleepConfig::time_to_sleep`]. If any member is still moving, every member's
/// timer resets to zero. When sleeping is disabled this is a no-op.
pub fn update_after_solve(
    view: &mut BodySolverView<'_>,
    islands: &SolveIslands,
    active: &[usize],
    config: &SleepConfig,
    h: f32,
) {
    if !config.enabled {
        return;
    }
    for &island in active {
        let members = islands.members(island);
        if members.is_empty() {
            continue;
        }

        let all_idle = members.iter().all(|&slot| {
            config.is_below_thresholds(view.linear_velocities[slot], view.angular_velocities[slot])
        });

        if !all_idle {
            for &slot in members {
                view.sleep_timers[slot] = 0.0;
            }
            continue;
        }

        let mut min_timer = f32::INFINITY;
        for &slot in members {
            let timer = advance_sleep_timer(view.sleep_timers[slot], h, true);
            view.sleep_timers[slot] = timer;
            min_timer = min_timer.min(timer);
        }

        if ready_to_sleep(min_timer, config) {
            for &slot in members {
                view.sleeping[slot] = true;
                view.linear_velocities[slot] = Vec3::ZERO;
                view.angular_velocities[slot] = Vec3::ZERO;
            }
        }
    }
}

/// Wakes a single sleeping slot: clears its flag, resets its idle timer, and
/// re-stamps its previous pose to the current pose.
///
/// Re-stamping matters because a body woken during the solve was skipped by
/// [`predict`](super::integrate::predict) and never had its previous pose set
/// this sub-step; without this the subsequent velocity recovery would divide a
/// stale pose delta by `h` and launch the body.
fn wake_slot(view: &mut BodySolverView<'_>, slot: usize) {
    view.sleeping[slot] = false;
    view.sleep_timers[slot] = 0.0;
    view.prev_positions[slot] = view.positions[slot];
    view.prev_orientations[slot] = view.orientations[slot];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collide::contact::{ContactManifold, ContactPoint};
    use crate::solver::xpbd::contact_constraint::ContactConstraint;
    use crate::state::body::{BodyDesc, BodyKind};
    use crate::state::handle::BodyHandle;
    use crate::state::storage::BodyStorage;

    fn storage_with_kinds(kinds: &[BodyKind]) -> BodyStorage {
        let mut storage = BodyStorage::new();
        for kind in kinds {
            let desc = match kind {
                BodyKind::Dynamic => BodyDesc::dynamic_at(Vec3::ZERO),
                BodyKind::Static => BodyDesc::static_at(Vec3::ZERO),
                BodyKind::Kinematic => {
                    let mut d = BodyDesc::dynamic_at(Vec3::ZERO);
                    d.kind = BodyKind::Kinematic;
                    d
                }
            };
            storage.insert(desc);
        }
        storage
    }

    fn manifold(a: BodyHandle, b: BodyHandle) -> ContactManifold {
        let mut m = ContactManifold::new(a, b, Vec3::Y);
        m.push(ContactPoint::new(Vec3::ZERO, Vec3::ZERO, 0.01));
        m
    }

    /// Builds a storage, contacts and island partition for the given kinds/pairs.
    fn setup(kinds: &[BodyKind], pairs: &[(usize, usize)]) -> (BodyStorage, SolveIslands) {
        let mut storage = storage_with_kinds(kinds);
        let handles: Vec<BodyHandle> = (0..kinds.len())
            .map(|slot| storage.handle_at_slot(slot).expect("live slot"))
            .collect();
        let manifolds: Vec<ContactManifold> = pairs
            .iter()
            .map(|&(a, b)| manifold(handles[a], handles[b]))
            .collect();
        let islands = {
            let view = storage.solver_view_mut();
            let constraints = ContactConstraint::build(&view, &manifolds);
            SolveIslands::build(&view, &constraints, &[])
        };
        (storage, islands)
    }

    #[test]
    fn fully_sleeping_island_is_frozen() {
        let (mut storage, islands) = setup(&[BodyKind::Dynamic; 2], &[(0, 1)]);
        // Put both members to sleep.
        {
            let view = storage.solver_view_mut();
            view.sleeping[0] = true;
            view.sleeping[1] = true;
        }
        let cfg = SleepConfig::default();
        let mut view = storage.solver_view_mut();
        let active = classify_and_wake(&mut view, &islands, &cfg);
        assert!(active.is_empty(), "sleeping island must be skipped");
        assert!(view.is_sleeping_slot(0) && view.is_sleeping_slot(1));
    }

    #[test]
    fn active_island_wakes_sleeping_members() {
        let (mut storage, islands) = setup(&[BodyKind::Dynamic; 2], &[(0, 1)]);
        // Only body 1 is asleep; body 0 is awake and in the same island.
        {
            let view = storage.solver_view_mut();
            view.sleeping[1] = true;
            view.positions[1] = Vec3::new(1.0, 2.0, 3.0);
        }
        let cfg = SleepConfig::default();
        let mut view = storage.solver_view_mut();
        let active = classify_and_wake(&mut view, &islands, &cfg);
        assert_eq!(active, vec![0]);
        assert!(!view.is_sleeping_slot(1), "member must be woken");
        // Previous pose was re-stamped to the current pose.
        assert_eq!(view.prev_positions[1], Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn idle_island_sleeps_after_dwell() {
        let (mut storage, islands) = setup(&[BodyKind::Dynamic; 2], &[(0, 1)]);
        let cfg = SleepConfig::default();
        let active = vec![0];
        let steps = (cfg.time_to_sleep / 0.1).ceil() as usize + 1;
        for _ in 0..steps {
            let mut view = storage.solver_view_mut();
            // Velocities stay at zero (below thresholds).
            update_after_solve(&mut view, &islands, &active, &cfg, 0.1);
        }
        let view = storage.solver_view_mut();
        assert!(view.is_sleeping_slot(0) && view.is_sleeping_slot(1));
    }

    #[test]
    fn moving_island_never_sleeps_and_resets_timer() {
        let (mut storage, islands) = setup(&[BodyKind::Dynamic; 2], &[(0, 1)]);
        let cfg = SleepConfig::default();
        let active = vec![0];
        {
            // Body 0 is moving well above the linear threshold.
            let view = storage.solver_view_mut();
            view.linear_velocities[0] = Vec3::new(5.0, 0.0, 0.0);
            view.sleep_timers[0] = 0.4;
            view.sleep_timers[1] = 0.4;
        }
        let mut view = storage.solver_view_mut();
        update_after_solve(&mut view, &islands, &active, &cfg, 0.1);
        assert_eq!(view.sleep_timers[0], 0.0);
        assert_eq!(view.sleep_timers[1], 0.0);
        assert!(!view.is_sleeping_slot(0) && !view.is_sleeping_slot(1));
    }

    #[test]
    fn disabled_config_never_sleeps() {
        let (mut storage, islands) = setup(&[BodyKind::Dynamic; 2], &[(0, 1)]);
        let cfg = SleepConfig {
            enabled: false,
            ..SleepConfig::default()
        };
        let active = vec![0];
        for _ in 0..100 {
            let mut view = storage.solver_view_mut();
            update_after_solve(&mut view, &islands, &active, &cfg, 0.1);
        }
        let view = storage.solver_view_mut();
        assert!(!view.is_sleeping_slot(0) && !view.is_sleeping_slot(1));
    }
}
