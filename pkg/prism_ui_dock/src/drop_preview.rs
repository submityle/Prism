//! Deterministic drop-preview geometry between drag intent and overlay host.

use crate::{DropZone, Rect, RenderPlan, SurfaceId, Target};

/// Geometry a renderer can paint as a drag/drop highlight.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DropPreview {
    /// The intent that produced this preview.
    pub zone: DropZone,
    /// Surface containing the target.
    pub surface: SurfaceId,
    /// Original target leaf rectangle.
    pub target_rect: Rect,
    /// Rectangle to highlight in an overlay layer.
    pub highlight_rect: Rect,
}

/// Failure while resolving a drop intent against a solved render plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DropPreviewError {
    /// The target panel/document is not present in the plan.
    TargetNotFound(Target),
    /// The tab insertion metadata is inconsistent with its target rectangle.
    InvalidTabIndex {
        /// Requested insertion boundary.
        index: usize,
        /// Number of existing tabs used for the preview.
        tab_count: usize,
    },
}

/// Resolves a drag intent into deterministic overlay geometry.
pub fn resolve_drop_preview(
    plan: &RenderPlan,
    zone: &DropZone,
) -> Result<DropPreview, DropPreviewError> {
    let target = match zone {
        DropZone::CenterTab { target } | DropZone::Split { target, .. } => target,
        DropZone::TabInsert { target, .. } => &Target::Panel(target.clone()),
    };
    let (surface, target_rect, tab_rect) = target_rect(plan, target)
        .ok_or_else(|| DropPreviewError::TargetNotFound(target.clone()))?;
    let highlight_rect = match zone {
        DropZone::CenterTab { .. } => target_rect,
        DropZone::Split { side, .. } => split_preview(target_rect, *side),
        DropZone::TabInsert {
            index, tab_count, ..
        } => tab_preview(tab_rect, *index, *tab_count)?,
    };
    Ok(DropPreview {
        zone: zone.clone(),
        surface,
        target_rect,
        highlight_rect,
    })
}

fn target_rect(plan: &RenderPlan, target: &Target) -> Option<(SurfaceId, Rect, Rect)> {
    match target {
        Target::Panel(id) => plan
            .panels
            .iter()
            .find(|panel| panel.id == *id)
            .map(|panel| (panel.surface, panel.rect, panel.tab_rect)),
        Target::Document(id) => plan
            .documents
            .iter()
            .find(|document| document.id == *id)
            .map(|document| (document.surface, document.rect, document.tab_rect)),
    }
}

fn split_preview(rect: Rect, side: crate::Side) -> Rect {
    let half_width = (rect.width / 2).max(1).min(rect.width);
    let half_height = (rect.height / 2).max(1).min(rect.height);
    match side {
        crate::Side::Left => Rect {
            width: half_width,
            ..rect
        },
        crate::Side::Right => Rect {
            x: rect.x.saturating_add((rect.width - half_width) as i32),
            width: half_width,
            ..rect
        },
        crate::Side::Top => Rect {
            height: half_height,
            ..rect
        },
        crate::Side::Bottom => Rect {
            y: rect.y.saturating_add((rect.height - half_height) as i32),
            height: half_height,
            ..rect
        },
    }
}

fn tab_preview(rect: Rect, index: usize, tab_count: usize) -> Result<Rect, DropPreviewError> {
    if index > tab_count {
        return Err(DropPreviewError::InvalidTabIndex { index, tab_count });
    }
    if tab_count == 0 {
        return Ok(rect);
    }
    let start = rect.width.saturating_mul(index as u32) / tab_count as u32;
    let end = rect.width.saturating_mul(index.saturating_add(1) as u32) / tab_count as u32;
    Ok(Rect {
        x: rect.x.saturating_add(start as i32),
        width: end.saturating_sub(start).max(1),
        ..rect
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PanelLayout, PanelVisibility, TabId};
    use alloc::string::String;

    fn plan() -> RenderPlan {
        RenderPlan {
            viewport: Rect {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
            main_rect: Rect {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
            panels: vec![PanelLayout {
                id: String::from("scene"),
                surface: SurfaceId::Main,
                node: Default::default(),
                rect: Rect {
                    x: 0,
                    y: 0,
                    width: 400,
                    height: 600,
                },
                content_rect: None,
                tab_rect: Rect {
                    x: 0,
                    y: 0,
                    width: 400,
                    height: 28,
                },
                active: true,
                focused: false,
                visibility: PanelVisibility::Visible,
            }],
            documents: Vec::new(),
            tabs: vec![crate::TabHeader {
                id: TabId::Panel(String::from("scene")),
                surface: SurfaceId::Main,
                node: Default::default(),
                rect: Rect {
                    x: 0,
                    y: 0,
                    width: 400,
                    height: 28,
                },
                active: true,
                focused: false,
            }],
            splitters: Vec::new(),
            floating: Vec::new(),
            autohide: Default::default(),
            maximized: None,
            focus: crate::FocusMetadata {
                target: None,
                reason: None,
            },
        }
    }

    #[test]
    fn center_and_split_previews_use_target_geometry() {
        let plan = plan();
        let center = resolve_drop_preview(
            &plan,
            &DropZone::CenterTab {
                target: Target::Panel(String::from("scene")),
            },
        )
        .unwrap();
        assert_eq!(center.highlight_rect, center.target_rect);
        let split = resolve_drop_preview(
            &plan,
            &DropZone::Split {
                target: Target::Panel(String::from("scene")),
                side: crate::Side::Right,
            },
        )
        .unwrap();
        assert_eq!(split.highlight_rect.x, 200);
        assert_eq!(split.highlight_rect.width, 200);
    }

    #[test]
    fn tab_preview_validates_index_and_divides_header_space() {
        let plan = plan();
        let preview = resolve_drop_preview(
            &plan,
            &DropZone::TabInsert {
                target: String::from("scene"),
                index: 1,
                tab_count: 2,
            },
        )
        .unwrap();
        // The insertion indicator belongs to the 28px header, not the leaf.
        assert_eq!(preview.highlight_rect.x, 200);
        assert_eq!(preview.highlight_rect.y, 0);
        assert_eq!(preview.highlight_rect.width, 200);
        assert_eq!(preview.highlight_rect.height, 28);
        assert!(matches!(
            resolve_drop_preview(
                &plan,
                &DropZone::TabInsert {
                    target: String::from("scene"),
                    index: 3,
                    tab_count: 2
                }
            ),
            Err(DropPreviewError::InvalidTabIndex {
                index: 3,
                tab_count: 2
            })
        ));
    }

    #[test]
    fn missing_target_is_structured() {
        assert!(matches!(
            resolve_drop_preview(
                &plan(),
                &DropZone::CenterTab {
                    target: Target::Panel(String::from("missing"))
                }
            ),
            Err(DropPreviewError::TargetNotFound(Target::Panel(_)))
        ));
    }
}
