//! Per-frame event coalescing (§10.3).
//!
//! High-frequency, state-overwriting events — resize, move, cursor motion, and
//! scale changes — are collapsed to their last occurrence per window within a
//! frame, because only the final value matters to the window model. Discrete
//! events (close, focus, theme, occlusion, cursor enter/leave) are never
//! merged and keep their exact order relative to everything else.
//!
//! Coalescing is deterministic: the output preserves input order, dropping
//! only the superseded earlier copies of each mergeable `(window, kind)`.

use prism_window::WindowEventEnvelope;
use prism_window::event::WindowEvent;

/// The class a window event falls into for coalescing. `None` means the event
/// is discrete and must never be merged.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum CoalesceKind {
    Resized,
    Moved,
    CursorMoved,
    ScaleFactor,
}

fn coalesce_kind(event: &WindowEvent) -> Option<CoalesceKind> {
    match event {
        WindowEvent::Resized(_) => Some(CoalesceKind::Resized),
        WindowEvent::Moved(_) => Some(CoalesceKind::Moved),
        WindowEvent::CursorMoved { .. } => Some(CoalesceKind::CursorMoved),
        WindowEvent::ScaleFactorChanged { .. } => Some(CoalesceKind::ScaleFactor),
        _ => None,
    }
}

/// Collapses mergeable events in `events` to their last occurrence per
/// `(window, kind)`, preserving the order of everything that remains.
///
/// Operates in place: superseded earlier copies are removed and the surviving
/// envelopes keep their original relative order and stamps.
pub fn coalesce(events: &mut Vec<WindowEventEnvelope>) {
    use prism_window::WindowId;
    use std::collections::HashMap;

    // Record, for each coalescable (window, kind), the index of its LAST entry.
    let mut last_index: HashMap<(WindowId, CoalesceKind), usize> = HashMap::new();
    for (i, env) in events.iter().enumerate() {
        if let Some(kind) = coalesce_kind(&env.event) {
            last_index.insert((env.window, kind), i);
        }
    }

    let mut keep = Vec::with_capacity(events.len());
    for (i, env) in events.iter().enumerate() {
        let survives = match coalesce_kind(&env.event) {
            Some(kind) => last_index.get(&(env.window, kind)) == Some(&i),
            None => true,
        };
        if survives {
            keep.push(*env);
        }
    }
    *events = keep;
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_window::envelope::{
        EventSource, MonotonicTimestamp, PlatformEventSequence, PlatformEventStamp,
    };
    use prism_window::geometry::{PhysicalPosition, PhysicalSize};
    use prism_window::WindowId;

    fn env(seq: u64, window: u64, event: WindowEvent) -> WindowEventEnvelope {
        let stamp = PlatformEventStamp::new(
            MonotonicTimestamp::from_nanos(seq),
            PlatformEventSequence(seq),
            EventSource::Window,
        );
        WindowEventEnvelope::new(stamp, WindowId(window), event)
    }

    #[test]
    fn collapses_repeated_resizes_to_last() {
        let mut events = vec![
            env(0, 1, WindowEvent::Resized(PhysicalSize::new(100, 100))),
            env(1, 1, WindowEvent::Resized(PhysicalSize::new(200, 200))),
            env(2, 1, WindowEvent::Resized(PhysicalSize::new(300, 300))),
        ];
        coalesce(&mut events);
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].event,
            WindowEvent::Resized(PhysicalSize::new(300, 300))
        );
        assert_eq!(events[0].stamp.sequence, PlatformEventSequence(2));
    }

    #[test]
    fn preserves_discrete_events_and_order() {
        let mut events = vec![
            env(0, 1, WindowEvent::Resized(PhysicalSize::new(100, 100))),
            env(1, 1, WindowEvent::Focused(true)),
            env(2, 1, WindowEvent::Resized(PhysicalSize::new(200, 200))),
            env(3, 1, WindowEvent::CloseRequested),
        ];
        coalesce(&mut events);
        // Focused and CloseRequested survive; only the first Resized is dropped.
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].event, WindowEvent::Focused(true));
        assert_eq!(
            events[1].event,
            WindowEvent::Resized(PhysicalSize::new(200, 200))
        );
        assert_eq!(events[2].event, WindowEvent::CloseRequested);
    }

    #[test]
    fn coalesces_per_window_independently() {
        let mut events = vec![
            env(0, 1, WindowEvent::Moved(PhysicalPosition::new(0, 0))),
            env(1, 2, WindowEvent::Moved(PhysicalPosition::new(5, 5))),
            env(2, 1, WindowEvent::Moved(PhysicalPosition::new(10, 10))),
        ];
        coalesce(&mut events);
        assert_eq!(events.len(), 2);
        // Window 2's single move survives; window 1 keeps only its last move.
        assert_eq!(
            events[0].event,
            WindowEvent::Moved(PhysicalPosition::new(5, 5))
        );
        assert_eq!(events[0].window, WindowId(2));
        assert_eq!(
            events[1].event,
            WindowEvent::Moved(PhysicalPosition::new(10, 10))
        );
        assert_eq!(events[1].window, WindowId(1));
    }

    #[test]
    fn different_mergeable_kinds_do_not_collapse_each_other() {
        let mut events = vec![
            env(0, 1, WindowEvent::Resized(PhysicalSize::new(100, 100))),
            env(1, 1, WindowEvent::Moved(PhysicalPosition::new(1, 1))),
            env(2, 1, WindowEvent::CursorMoved {
                position: PhysicalPosition::new(2, 2),
            }),
        ];
        coalesce(&mut events);
        assert_eq!(events.len(), 3);
    }

    #[test]
    fn empty_is_noop() {
        let mut events: Vec<WindowEventEnvelope> = vec![];
        coalesce(&mut events);
        assert!(events.is_empty());
    }
}
