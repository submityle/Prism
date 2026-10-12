//! GPU-independent drag interaction and drop-zone calculation.
//!
//! The module deliberately stops at an intent. A renderer can use the zone
//! for a preview, while [`map_drop_action`] turns the same zone into the
//! reducer action that is committed on pointer release.

use crate::{Action, PanelId, Placement, Rect, Side, Target};
use alloc::string::ToString;

/// A pointer position in the same logical coordinate space as [`Rect`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PointerPosition {
    /// Horizontal coordinate.
    pub x: i32,
    /// Vertical coordinate.
    pub y: i32,
}

/// The five ways a panel can be dropped on a dock leaf.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DropZone {
    /// Add the dragged panel to the target's tab stack.
    CenterTab {
        /// Existing panel or document under the pointer.
        target: Target,
    },
    /// Split the target on one of its four edges.
    Split {
        /// Existing panel or document under the pointer.
        target: Target,
        /// Edge on which the dragged panel is placed.
        side: Side,
    },
    /// Insert the dragged panel at a tab boundary.
    TabInsert {
        /// Existing panel whose tab stack is under the pointer.
        target: PanelId,
        /// Insertion position in the tab stack, in the inclusive range `0..=tab_count`.
        index: usize,
        /// Number of tabs used to calculate this preview.
        tab_count: usize,
    },
}

/// Errors produced while converting a drop preview into a reducer action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DropMapError {
    /// The dragged panel ID is empty.
    EmptySource,
    /// The drop target ID is empty.
    EmptyTarget,
    /// A panel cannot be dropped onto itself.
    SelfDrop,
    /// A tab insertion index is outside the tab stack used for the preview.
    InvalidTabIndex,
}

/// Calculates the five-way drop zone for a leaf rectangle.
///
/// The central 50% by 50% area is a tab drop. Outside that area the nearest
/// edge wins, with ties resolved in the stable order left, right, top, bottom.
/// The rectangle uses half-open bounds, so its right and bottom edges are not
/// considered inside. An empty rectangle or an outside pointer returns
/// `None`.
pub fn calculate_drop_zone(
    bounds: Rect,
    pointer: PointerPosition,
    target: Target,
) -> Option<DropZone> {
    if bounds.width == 0 || bounds.height == 0 {
        return None;
    }
    let local_x = i64::from(pointer.x) - i64::from(bounds.x);
    let local_y = i64::from(pointer.y) - i64::from(bounds.y);
    let width = i64::from(bounds.width);
    let height = i64::from(bounds.height);
    if local_x < 0 || local_y < 0 || local_x >= width || local_y >= height {
        return None;
    }

    if local_x * 4 >= width
        && local_x * 4 < width * 3
        && local_y * 4 >= height
        && local_y * 4 < height * 3
    {
        return Some(DropZone::CenterTab { target });
    }

    let distances = [
        (local_x, Side::Left),
        (width - 1 - local_x, Side::Right),
        (local_y, Side::Top),
        (height - 1 - local_y, Side::Bottom),
    ];
    let (_, side) = distances
        .into_iter()
        .min_by_key(|(distance, _)| *distance)
        .expect("the four drop-zone edges are always present");
    Some(DropZone::Split { target, side })
}

/// Calculates the tab insertion boundary under a pointer.
///
/// Tabs are represented as equal-width logical slots because this layer has
/// no font or renderer dependency. The returned value is a boundary, not a
/// tab index: for `tab_count == 3`, the result ranges from `0` through `3`.
pub fn tab_insertion_index(
    tab_bar: Rect,
    pointer: PointerPosition,
    tab_count: usize,
) -> Option<usize> {
    if tab_bar.width == 0 || tab_bar.height == 0 {
        return None;
    }
    let local_x = i64::from(pointer.x) - i64::from(tab_bar.x);
    let local_y = i64::from(pointer.y) - i64::from(tab_bar.y);
    let width = i64::from(tab_bar.width);
    let height = i64::from(tab_bar.height);
    if local_x < 0 || local_y < 0 || local_x >= width || local_y >= height {
        return None;
    }
    if tab_count == 0 {
        return Some(0);
    }
    let index = (local_x * tab_count as i64 / width) as usize;
    Some(index.min(tab_count))
}

/// Converts a drop preview into the deterministic action consumed by
/// [`crate::reduce`].
pub fn map_drop_action(source: &str, zone: &DropZone) -> Result<Action, DropMapError> {
    if source.is_empty() {
        return Err(DropMapError::EmptySource);
    }
    let action = match zone {
        DropZone::CenterTab { target } => {
            if target_is_empty(target) {
                return Err(DropMapError::EmptyTarget);
            }
            let Target::Panel(target) = target else {
                return Err(DropMapError::InvalidTabIndex);
            };
            reject_self(source, target)?;
            Action::Move {
                panel: source.to_string(),
                target: Target::Panel(target.clone()),
                placement: Placement::Tab,
            }
        }
        DropZone::Split { target, side } => {
            if target_is_empty(target) {
                return Err(DropMapError::EmptyTarget);
            }
            if let Target::Panel(target) = target {
                reject_self(source, target)?;
            }
            Action::Move {
                panel: source.to_string(),
                target: target.clone(),
                placement: Placement::Side(*side),
            }
        }
        DropZone::TabInsert {
            target,
            index,
            tab_count,
        } => {
            if target.is_empty() {
                return Err(DropMapError::EmptyTarget);
            }
            reject_self(source, target)?;
            if *index > *tab_count {
                return Err(DropMapError::InvalidTabIndex);
            }
            Action::Move {
                panel: source.to_string(),
                target: Target::Panel(target.clone()),
                placement: Placement::TabAt(*index),
            }
        }
    };
    Ok(action)
}

fn target_is_empty(target: &Target) -> bool {
    match target {
        Target::Panel(id) | Target::Document(id) => id.is_empty(),
    }
}

fn reject_self(source: &str, target: &str) -> Result<(), DropMapError> {
    if source == target {
        Err(DropMapError::SelfDrop)
    } else {
        Ok(())
    }
}

/// Events accepted by the drag state machine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DragEvent {
    /// Begin dragging a panel at a pointer position.
    PickUp {
        /// Panel being dragged.
        source: PanelId,
        /// Pointer position at pickup.
        position: PointerPosition,
    },
    /// Update the pointer and the currently highlighted zone.
    Move {
        /// Current pointer position.
        position: PointerPosition,
        /// Zone under the pointer, if any.
        zone: Option<DropZone>,
    },
    /// Commit the currently highlighted zone.
    Drop {
        /// Zone selected on release.
        zone: DropZone,
    },
    /// Abort the drag and restore the source to its pre-drag location.
    Cancel,
    /// Return a terminal state to idle.
    Reset,
}

/// State of one panel drag interaction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DragState {
    /// No active interaction.
    Idle,
    /// The pointer has picked up a panel but has not moved it yet.
    PickUp {
        /// Panel being dragged.
        source: PanelId,
        /// Pointer position at pickup.
        origin: PointerPosition,
    },
    /// The pointer is moving a panel and may have a highlighted zone.
    Dragging {
        /// Panel being dragged.
        source: PanelId,
        /// Pointer position at pickup.
        origin: PointerPosition,
        /// Current pointer position.
        position: PointerPosition,
        /// Current preview zone.
        zone: Option<DropZone>,
    },
    /// A valid drop was selected; the reducer action can now be committed.
    Drop {
        /// Panel being dropped.
        source: PanelId,
        /// Valid destination selected on release.
        zone: DropZone,
    },
    /// The interaction was cancelled without changing the layout.
    Cancel {
        /// Panel whose drag was cancelled.
        source: PanelId,
    },
}

/// Errors for invalid state-machine transitions or invalid drop commits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DragError {
    /// The event is not valid in the current state.
    InvalidTransition,
    /// The requested drop cannot be mapped to a reducer action.
    InvalidDrop(DropMapError),
}

/// Advances a drag interaction by one event.
pub fn transition(state: &DragState, event: DragEvent) -> Result<DragState, DragError> {
    match (state, event) {
        (DragState::Idle, DragEvent::PickUp { source, position }) if !source.is_empty() => {
            Ok(DragState::PickUp {
                source,
                origin: position,
            })
        }
        (DragState::PickUp { source, origin }, DragEvent::Move { position, zone }) => {
            Ok(DragState::Dragging {
                source: source.clone(),
                origin: *origin,
                position,
                zone,
            })
        }
        (
            DragState::Dragging {
                source,
                origin,
                position: _,
                zone: _,
            },
            DragEvent::Move { position, zone },
        ) => Ok(DragState::Dragging {
            source: source.clone(),
            origin: *origin,
            position,
            zone,
        }),
        (
            DragState::PickUp { source, .. } | DragState::Dragging { source, .. },
            DragEvent::Drop { zone },
        ) => {
            map_drop_action(source, &zone).map_err(DragError::InvalidDrop)?;
            Ok(DragState::Drop {
                source: source.clone(),
                zone,
            })
        }
        (
            DragState::PickUp { source, .. } | DragState::Dragging { source, .. },
            DragEvent::Cancel,
        ) => Ok(DragState::Cancel {
            source: source.clone(),
        }),
        (DragState::Drop { .. } | DragState::Cancel { .. }, DragEvent::Reset) => {
            Ok(DragState::Idle)
        }
        _ => Err(DragError::InvalidTransition),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target() -> Target {
        Target::Panel("target".into())
    }

    fn bounds() -> Rect {
        Rect {
            x: 10,
            y: 20,
            width: 100,
            height: 80,
        }
    }

    #[test]
    fn five_way_center_and_edges_are_deterministic() {
        assert!(matches!(
            calculate_drop_zone(bounds(), PointerPosition { x: 60, y: 60 }, target()),
            Some(DropZone::CenterTab { .. })
        ));
        for (position, side) in [
            (PointerPosition { x: 11, y: 60 }, Side::Left),
            (PointerPosition { x: 108, y: 60 }, Side::Right),
            (PointerPosition { x: 60, y: 21 }, Side::Top),
            (PointerPosition { x: 60, y: 98 }, Side::Bottom),
        ] {
            assert!(matches!(
                calculate_drop_zone(bounds(), position, target()),
                Some(DropZone::Split { side: actual, .. }) if actual == side
            ));
        }
    }

    #[test]
    fn zone_boundaries_are_stable_and_outside_is_empty() {
        assert!(matches!(
            calculate_drop_zone(bounds(), PointerPosition { x: 35, y: 40 }, target()),
            Some(DropZone::CenterTab { .. })
        ));
        assert!(matches!(
            calculate_drop_zone(bounds(), PointerPosition { x: 34, y: 60 }, target()),
            Some(DropZone::Split {
                side: Side::Left,
                ..
            })
        ));
        assert!(calculate_drop_zone(bounds(), PointerPosition { x: 9, y: 60 }, target()).is_none());
        assert!(calculate_drop_zone(
            Rect {
                width: 0,
                ..bounds()
            },
            PointerPosition { x: 10, y: 20 },
            target()
        )
        .is_none());
    }

    #[test]
    fn tab_insertion_handles_boundaries_and_empty_stacks() {
        let bar = Rect {
            x: 0,
            y: 0,
            width: 90,
            height: 20,
        };
        assert_eq!(
            tab_insertion_index(bar, PointerPosition { x: 0, y: 10 }, 3),
            Some(0)
        );
        assert_eq!(
            tab_insertion_index(bar, PointerPosition { x: 30, y: 10 }, 3),
            Some(1)
        );
        assert_eq!(
            tab_insertion_index(bar, PointerPosition { x: 89, y: 10 }, 3),
            Some(2)
        );
        assert_eq!(
            tab_insertion_index(bar, PointerPosition { x: 40, y: 10 }, 0),
            Some(0)
        );
        assert_eq!(
            tab_insertion_index(bar, PointerPosition { x: 90, y: 10 }, 3),
            None
        );
    }

    #[test]
    fn action_mapping_covers_center_split_and_tab_insert() {
        assert!(matches!(
            map_drop_action("source", &DropZone::CenterTab { target: target() }),
            Ok(Action::Move {
                placement: Placement::Tab,
                ..
            })
        ));
        assert!(matches!(
            map_drop_action(
                "source",
                &DropZone::Split {
                    target: Target::Document("doc".into()),
                    side: Side::Bottom
                }
            ),
            Ok(Action::Move {
                placement: Placement::Side(Side::Bottom),
                ..
            })
        ));
        assert!(matches!(
            map_drop_action(
                "source",
                &DropZone::TabInsert {
                    target: "target".into(),
                    index: 2,
                    tab_count: 2
                }
            ),
            Ok(Action::Move {
                placement: Placement::TabAt(2),
                ..
            })
        ));
    }

    #[test]
    fn illegal_targets_are_rejected_deterministically() {
        assert_eq!(
            map_drop_action(
                "same",
                &DropZone::CenterTab {
                    target: Target::Panel("same".into())
                }
            ),
            Err(DropMapError::SelfDrop)
        );
        assert_eq!(
            map_drop_action(
                "source",
                &DropZone::TabInsert {
                    target: "target".into(),
                    index: 3,
                    tab_count: 2
                }
            ),
            Err(DropMapError::InvalidTabIndex)
        );
        assert_eq!(
            map_drop_action(
                "source",
                &DropZone::CenterTab {
                    target: Target::Document("doc".into())
                }
            ),
            Err(DropMapError::InvalidTabIndex)
        );
    }

    #[test]
    fn cancellation_reaches_terminal_state_without_a_drop() {
        let state = transition(
            &DragState::Idle,
            DragEvent::PickUp {
                source: "source".into(),
                position: PointerPosition { x: 1, y: 2 },
            },
        )
        .unwrap();
        let state = transition(&state, DragEvent::Cancel).unwrap();
        assert_eq!(
            state,
            DragState::Cancel {
                source: "source".into()
            }
        );
        assert_eq!(
            transition(&state, DragEvent::Reset).unwrap(),
            DragState::Idle
        );
    }

    #[test]
    fn dragging_updates_preview_and_drop_is_terminal() {
        let picked = transition(
            &DragState::Idle,
            DragEvent::PickUp {
                source: "source".into(),
                position: PointerPosition { x: 0, y: 0 },
            },
        )
        .unwrap();
        let dragging = transition(
            &picked,
            DragEvent::Move {
                position: PointerPosition { x: 10, y: 10 },
                zone: Some(DropZone::Split {
                    target: target(),
                    side: Side::Left,
                }),
            },
        )
        .unwrap();
        let dropped = transition(
            &dragging,
            DragEvent::Drop {
                zone: DropZone::Split {
                    target: target(),
                    side: Side::Left,
                },
            },
        )
        .unwrap();
        assert!(matches!(dropped, DragState::Drop { .. }));
        assert!(matches!(
            transition(
                &dropped,
                DragEvent::Move {
                    position: PointerPosition { x: 1, y: 1 },
                    zone: None
                }
            ),
            Err(DragError::InvalidTransition)
        ));
    }

    #[test]
    fn invalid_pickup_and_invalid_drop_do_not_advance() {
        assert_eq!(
            transition(
                &DragState::Idle,
                DragEvent::PickUp {
                    source: String::new(),
                    position: PointerPosition { x: 0, y: 0 }
                }
            ),
            Err(DragError::InvalidTransition)
        );
        let picked = DragState::PickUp {
            source: "source".into(),
            origin: PointerPosition { x: 0, y: 0 },
        };
        assert_eq!(
            transition(
                &picked,
                DragEvent::Drop {
                    zone: DropZone::CenterTab {
                        target: Target::Panel("source".into())
                    }
                }
            ),
            Err(DragError::InvalidDrop(DropMapError::SelfDrop))
        );
    }

    #[test]
    fn invalid_event_in_idle_is_rejected() {
        assert_eq!(
            transition(&DragState::Idle, DragEvent::Cancel),
            Err(DragError::InvalidTransition)
        );
        assert_eq!(
            transition(&DragState::Idle, DragEvent::Reset),
            Err(DragError::InvalidTransition)
        );
    }
}
