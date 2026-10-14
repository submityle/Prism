//! Deterministic input-to-action orchestration for a solved dock plan.
//!
//! The adapter owns only pointer capture state and drag intent. It does not
//! own platform events, widget trees, or renderer resources; callers translate
//! their platform events into [`DockInputEvent`] and consume the returned
//! preview/plan.

use crate::{
    calculate_drop_zone, hit_test, map_drop_action, resolve_drop_preview, Action, DockHit,
    DragError, DragEvent, DragState, DropMapError, DropPreview, DropPreviewError, DropZone,
    PanelId, PointerPosition, RenderPlan, TabId, Target,
};

/// A platform-neutral pointer event accepted by [`crate::DockHost::handle_input`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DockInputEvent {
    /// Starts a drag when the pointer is over a panel tab or active panel.
    PointerDown {
        /// Stable pointer identity used for capture validation.
        pointer_id: u64,
        /// Pointer position in viewport logical pixels.
        position: PointerPosition,
    },
    /// Updates the captured pointer and preview.
    PointerMove {
        /// Captured pointer identity.
        pointer_id: u64,
        /// Pointer position in viewport logical pixels.
        position: PointerPosition,
    },
    /// Releases the captured pointer and commits a valid drop.
    PointerUp {
        /// Captured pointer identity.
        pointer_id: u64,
        /// Pointer position in viewport logical pixels.
        position: PointerPosition,
    },
    /// Cancels the current interaction.
    Cancel {
        /// Pointer identity that owns the capture.
        pointer_id: u64,
    },
    /// Cancels the current interaction when the owning window loses focus.
    WindowLostFocus,
}

/// Failure while converting a platform-neutral input event into a dock action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DockInputError {
    /// An event was delivered by a pointer that does not own the capture.
    PointerCaptureMismatch {
        /// Pointer identity currently holding the capture.
        expected: u64,
        /// Pointer identity that delivered the event.
        actual: u64,
    },
    /// The drag state machine rejected the event.
    Drag(DragError),
    /// The selected drop zone could not become a reducer action.
    DropMap(DropMapError),
    /// The selected drop zone could not become preview geometry.
    Preview(DropPreviewError),
}

impl From<DragError> for DockInputError {
    fn from(error: DragError) -> Self {
        Self::Drag(error)
    }
}

impl From<DropMapError> for DockInputError {
    fn from(error: DropMapError) -> Self {
        Self::DropMap(error)
    }
}

impl From<DropPreviewError> for DockInputError {
    fn from(error: DropPreviewError) -> Self {
        Self::Preview(error)
    }
}

/// Pure output of one input step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DockInputOutput {
    /// Current solved plan after the event.
    pub plan: RenderPlan,
    /// Current drop preview, if a captured drag is over a valid target.
    pub preview: Option<DropPreview>,
    /// Whether the event committed a reducer action.
    pub committed: bool,
    /// Current drag state after the event.
    pub drag: DragState,
}

/// Finds the panel source represented by a hit result.
fn source_from_hit(hit: &DockHit) -> Option<PanelId> {
    match &hit.target {
        crate::HitTarget::Panel(id) | crate::HitTarget::Tab(TabId::Panel(id)) => Some(id.clone()),
        _ => None,
    }
}

fn target_from_hit(hit: &DockHit) -> Option<Target> {
    match &hit.target {
        crate::HitTarget::Panel(id) | crate::HitTarget::Tab(TabId::Panel(id)) => {
            Some(Target::Panel(id.clone()))
        }
        crate::HitTarget::Document(id) | crate::HitTarget::Tab(TabId::Document(id)) => {
            Some(Target::Document(id.clone()))
        }
        _ => None,
    }
}

fn document_drop_zone(
    bounds: crate::Rect,
    position: PointerPosition,
    target: Target,
) -> Option<DropZone> {
    match calculate_drop_zone(bounds, position, target) {
        Some(DropZone::CenterTab { .. }) => None,
        zone => zone,
    }
}

fn tab_count(plan: &RenderPlan, hit: &DockHit) -> usize {
    let Some(tab) = plan.tabs.iter().find(|tab| {
        tab.surface == hit.surface
            && tab.rect == hit.rect
            && matches!(&hit.target, crate::HitTarget::Tab(id) if &tab.id == id)
    }) else {
        return 0;
    };
    plan.tabs
        .iter()
        .filter(|candidate| candidate.surface == hit.surface && candidate.node == tab.node)
        .count()
}

fn zone_for_hit(plan: &RenderPlan, hit: &DockHit, position: PointerPosition) -> Option<DropZone> {
    match &hit.target {
        crate::HitTarget::Tab(TabId::Panel(target)) => {
            let count = tab_count(plan, hit);
            let index = crate::tab_insertion_index(hit.rect, position, count)?;
            Some(DropZone::TabInsert {
                target: target.clone(),
                index,
                tab_count: count,
            })
        }
        crate::HitTarget::Tab(TabId::Document(_)) | crate::HitTarget::Document(_) => {
            target_from_hit(hit).and_then(|target| document_drop_zone(hit.rect, position, target))
        }
        crate::HitTarget::Panel(_) => {
            target_from_hit(hit).and_then(|target| calculate_drop_zone(hit.rect, position, target))
        }
        crate::HitTarget::Splitter { .. } | crate::HitTarget::AutoHide(_) => None,
    }
}

fn current_zone(plan: &RenderPlan, position: PointerPosition) -> Option<DropZone> {
    let hit = hit_test(plan, position.x, position.y)?;
    zone_for_hit(plan, &hit, position)
}

fn pointer_id(event: DockInputEvent) -> Option<u64> {
    match event {
        DockInputEvent::PointerDown { pointer_id, .. }
        | DockInputEvent::PointerMove { pointer_id, .. }
        | DockInputEvent::PointerUp { pointer_id, .. }
        | DockInputEvent::Cancel { pointer_id } => Some(pointer_id),
        DockInputEvent::WindowLostFocus => None,
    }
}

/// Processes one event against a validated layout and solved plan.
pub(crate) fn process_input(
    drag: &mut DragState,
    capture: &mut Option<u64>,
    event: DockInputEvent,
    plan: RenderPlan,
) -> Result<(Option<Action>, Option<DropPreview>, bool), DockInputError> {
    if let (Some(expected), Some(actual)) = (*capture, pointer_id(event))
        && expected != actual
        && !matches!(event, DockInputEvent::PointerDown { .. })
    {
        return Err(DockInputError::PointerCaptureMismatch { expected, actual });
    }

    let mut committed = false;
    let mut preview = None;
    let action = match event {
        DockInputEvent::PointerDown {
            pointer_id,
            position,
        } => {
            let Some(hit) = hit_test(&plan, position.x, position.y) else {
                return Ok((None, None, false));
            };
            let Some(source) = source_from_hit(&hit) else {
                return Ok((None, None, false));
            };
            *drag = crate::transition(&DragState::Idle, DragEvent::PickUp { source, position })?;
            *capture = Some(pointer_id);
            None
        }
        DockInputEvent::PointerMove {
            pointer_id: _,
            position,
        } => {
            let zone = current_zone(&plan, position);
            *drag = crate::transition(
                drag,
                DragEvent::Move {
                    position,
                    zone: zone.clone(),
                },
            )?;
            preview = zone
                .as_ref()
                .map(|value| resolve_drop_preview(&plan, value))
                .transpose()?;
            None
        }
        DockInputEvent::PointerUp {
            pointer_id: _,
            position,
        } => {
            let zone = current_zone(&plan, position);
            let (DragState::Dragging { source, .. } | DragState::PickUp { source, .. }) = &*drag
            else {
                return Err(DockInputError::Drag(DragError::InvalidTransition));
            };
            let Some(zone) = zone else {
                *drag = crate::transition(drag, DragEvent::Cancel)?;
                *capture = None;
                return Ok((None, None, false));
            };
            preview = Some(resolve_drop_preview(&plan, &zone)?);
            let action = map_drop_action(source, &zone)?;
            *drag = crate::transition(drag, DragEvent::Drop { zone })?;
            *capture = None;
            committed = true;
            Some(action)
        }
        DockInputEvent::Cancel { pointer_id: _ } => {
            *drag = crate::transition(drag, DragEvent::Cancel)?;
            *capture = None;
            None
        }
        DockInputEvent::WindowLostFocus => {
            if !matches!(drag, DragState::Idle) {
                *drag = crate::transition(drag, DragEvent::WindowLostFocus)?;
            }
            *capture = None;
            None
        }
    };
    Ok((action, preview, committed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NodePath, Rect, SurfaceId, TabHeader};
    use alloc::vec;

    fn plan_with_tabs() -> RenderPlan {
        RenderPlan {
            viewport: Rect {
                x: 0,
                y: 0,
                width: 400,
                height: 200,
            },
            main_rect: Rect {
                x: 0,
                y: 0,
                width: 400,
                height: 200,
            },
            panels: vec![],
            documents: vec![],
            tabs: vec![
                TabHeader {
                    id: TabId::Panel("a".into()),
                    surface: SurfaceId::Main,
                    node: NodePath { indices: vec![0] },
                    rect: Rect {
                        x: 0,
                        y: 0,
                        width: 50,
                        height: 20,
                    },
                    active: true,
                    focused: false,
                },
                TabHeader {
                    id: TabId::Panel("b".into()),
                    surface: SurfaceId::Main,
                    node: NodePath { indices: vec![0] },
                    rect: Rect {
                        x: 50,
                        y: 0,
                        width: 50,
                        height: 20,
                    },
                    active: false,
                    focused: false,
                },
                TabHeader {
                    id: TabId::Panel("c".into()),
                    surface: SurfaceId::Main,
                    node: NodePath { indices: vec![1] },
                    rect: Rect {
                        x: 100,
                        y: 0,
                        width: 50,
                        height: 20,
                    },
                    active: true,
                    focused: false,
                },
            ],
            splitters: vec![],
            floating: vec![],
            autohide: Default::default(),
            maximized: None,
            focus: crate::FocusMetadata {
                target: None,
                reason: None,
            },
        }
    }

    #[test]
    fn tab_insertion_counts_only_the_hit_leaf() {
        let plan = plan_with_tabs();
        let hit = DockHit {
            target: crate::HitTarget::Tab(TabId::Panel("b".into())),
            surface: SurfaceId::Main,
            rect: plan.tabs[1].rect,
        };
        assert_eq!(
            zone_for_hit(&plan, &hit, PointerPosition { x: 75, y: 10 }),
            Some(DropZone::TabInsert {
                target: "b".into(),
                index: 1,
                tab_count: 2,
            })
        );
    }

    #[test]
    fn document_center_is_not_a_drop_zone() {
        let plan = RenderPlan {
            tabs: vec![],
            ..plan_with_tabs()
        };
        let hit = DockHit {
            target: crate::HitTarget::Document("doc".into()),
            surface: SurfaceId::Main,
            rect: Rect {
                x: 0,
                y: 0,
                width: 100,
                height: 100,
            },
        };
        assert_eq!(
            zone_for_hit(&plan, &hit, PointerPosition { x: 50, y: 50 }),
            None
        );
        assert!(matches!(
            zone_for_hit(&plan, &hit, PointerPosition { x: 1, y: 50 }),
            Some(DropZone::Split { .. })
        ));
    }
}
