//! Deterministic geometry transitions for dock plans.

use alloc::vec::Vec;

use crate::{Rect, RenderPlan};

/// User-selectable dock motion quality.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MotionMode {
    /// Apply the target plan immediately.
    Disabled,
    /// Apply the target plan immediately for reduced-motion users.
    ReducedMotion,
    /// Use the standard geometry transition duration.
    #[default]
    Normal,
    /// Use the longer, more visible transition duration.
    HighQuality,
}

/// Duration policy for dock geometry transitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MotionPolicy {
    /// Selected motion mode.
    pub mode: MotionMode,
    /// Normal-mode duration in milliseconds.
    pub normal_duration_ms: u32,
    /// High-quality duration in milliseconds.
    pub high_quality_duration_ms: u32,
}

impl Default for MotionPolicy {
    fn default() -> Self {
        Self {
            mode: MotionMode::Normal,
            normal_duration_ms: 120,
            high_quality_duration_ms: 220,
        }
    }
}

impl MotionPolicy {
    /// Returns the effective duration for the selected mode.
    pub const fn duration_ms(self) -> u32 {
        match self.mode {
            MotionMode::Disabled | MotionMode::ReducedMotion => 0,
            MotionMode::Normal => self.normal_duration_ms,
            MotionMode::HighQuality => self.high_quality_duration_ms,
        }
    }
}

/// A retargetable transition between two solved plans.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DockGeometryTransition {
    start: RenderPlan,
    target: RenderPlan,
    duration_ms: u32,
}

impl DockGeometryTransition {
    /// Creates a transition from `start` to `target`.
    pub fn new(start: RenderPlan, target: RenderPlan, policy: MotionPolicy) -> Self {
        Self {
            start,
            target,
            duration_ms: policy.duration_ms(),
        }
    }

    /// Returns the configured duration.
    pub const fn duration_ms(&self) -> u32 {
        self.duration_ms
    }

    /// Samples the transition at a monotonic elapsed time.
    pub fn sample(&self, elapsed_ms: u32) -> RenderPlan {
        if self.duration_ms == 0 || elapsed_ms >= self.duration_ms {
            return self.target.clone();
        }
        if elapsed_ms == 0 {
            return self.start.clone();
        }
        let numerator = elapsed_ms as u64;
        let denominator = self.duration_ms as u64;
        let mut result = self.target.clone();
        result.viewport = lerp_rect(
            self.start.viewport,
            self.target.viewport,
            numerator,
            denominator,
        );
        result.main_rect = lerp_rect(
            self.start.main_rect,
            self.target.main_rect,
            numerator,
            denominator,
        );
        for to in &mut result.panels {
            if let Some(from) = self.start.panels.iter().find(|from| from.id == to.id) {
                to.rect = lerp_rect(from.rect, to.rect, numerator, denominator);
                to.tab_rect = lerp_rect(from.tab_rect, to.tab_rect, numerator, denominator);
                to.content_rect = lerp_content_rect(
                    from.content_rect,
                    to.content_rect,
                    from.rect,
                    to.rect,
                    numerator,
                    denominator,
                );
            }
        }
        for to in &mut result.documents {
            if let Some(from) = self.start.documents.iter().find(|from| from.id == to.id) {
                to.rect = lerp_rect(from.rect, to.rect, numerator, denominator);
                to.tab_rect = lerp_rect(from.tab_rect, to.tab_rect, numerator, denominator);
                to.content_rect = lerp_content_rect(
                    from.content_rect,
                    to.content_rect,
                    from.rect,
                    to.rect,
                    numerator,
                    denominator,
                );
            }
        }
        for to in &mut result.tabs {
            if let Some(from) = self.start.tabs.iter().find(|from| from.id == to.id) {
                to.rect = lerp_rect(from.rect, to.rect, numerator, denominator);
            }
        }
        for to in &mut result.splitters {
            if let Some(from) = self
                .start
                .splitters
                .iter()
                .find(|from| from.surface == to.surface && from.node == to.node)
            {
                to.line_rect = lerp_rect(from.line_rect, to.line_rect, numerator, denominator);
                to.hit_rect = lerp_rect(from.hit_rect, to.hit_rect, numerator, denominator);
            }
        }
        for to in &mut result.floating {
            if let Some(from) = self
                .start
                .floating
                .iter()
                .find(|from| from.index == to.index)
            {
                to.rect = lerp_rect(from.rect, to.rect, numerator, denominator);
            }
        }
        result.autohide = interpolate_autohide(
            &self.start.autohide,
            &self.target.autohide,
            &self.start,
            &self.target,
            numerator,
            denominator,
        );
        result.maximized = interpolate_maximized(
            self.start.maximized.as_ref(),
            self.target.maximized.as_ref(),
            &self.start,
            &self.target,
            numerator,
            denominator,
        );
        result
    }

    /// Retargets from the current interpolated state without accumulating
    /// stale timelines.
    pub fn retarget(&self, elapsed_ms: u32, target: RenderPlan, policy: MotionPolicy) -> Self {
        Self::new(self.sample(elapsed_ms), target, policy)
    }
}

fn lerp_content_rect(
    from: Option<Rect>,
    to: Option<Rect>,
    from_outer: Rect,
    _to_outer: Rect,
    numerator: u64,
    denominator: u64,
) -> Option<Rect> {
    match (from, to) {
        (Some(from), Some(to)) => Some(lerp_rect(from, to, numerator, denominator)),
        (None, Some(to)) => Some(lerp_rect(from_outer, to, numerator, denominator)),
        (Some(_), None) | (None, None) => None,
    }
}

fn interpolate_autohide(
    from: &crate::AutoHideLayout,
    to: &crate::AutoHideLayout,
    from_plan: &RenderPlan,
    to_plan: &RenderPlan,
    numerator: u64,
    denominator: u64,
) -> crate::AutoHideLayout {
    let mut sides = Vec::with_capacity(from.bars.len() + to.bars.len());
    for bar in &to.bars {
        if !sides.contains(&bar.side) {
            sides.push(bar.side);
        }
    }
    for bar in &from.bars {
        if !sides.contains(&bar.side) {
            sides.push(bar.side);
        }
    }

    let bars = sides
        .into_iter()
        .filter_map(|side| {
            let from_bar = from.bars.iter().find(|bar| bar.side == side);
            let to_bar = to.bars.iter().find(|bar| bar.side == side);
            interpolate_bar(from_bar, to_bar, side, numerator, denominator)
        })
        .collect();

    let revealed = match (&from.revealed, &to.revealed) {
        (None, None) => None,
        (Some(from), Some(to)) => Some(interpolate_reveal(from, to, numerator, denominator)),
        (None, Some(to)) => {
            let from = reveal_anchor(from_plan, &to.panel, to.side, to);
            Some(interpolate_reveal(&from, to, numerator, denominator))
        }
        (Some(from), None) => {
            let to = reveal_anchor(to_plan, &from.panel, from.side, from);
            Some(interpolate_reveal(from, &to, numerator, denominator))
        }
    };

    crate::AutoHideLayout { bars, revealed }
}

fn interpolate_bar(
    from: Option<&crate::AutoHideBarLayout>,
    to: Option<&crate::AutoHideBarLayout>,
    side: crate::Side,
    numerator: u64,
    denominator: u64,
) -> Option<crate::AutoHideBarLayout> {
    let (from_rect, to_rect) = match (from, to) {
        (Some(from), Some(to)) => (from.rect, to.rect),
        (None, Some(to)) => (collapse_rect(to.rect, side), to.rect),
        (Some(from), None) => (from.rect, collapse_rect(from.rect, side)),
        (None, None) => return None,
    };
    let from_buttons = from.map(|bar| bar.buttons.as_slice()).unwrap_or(&[]);
    let to_buttons = to.map(|bar| bar.buttons.as_slice()).unwrap_or(&[]);

    let mut panels: Vec<crate::PanelId> = Vec::with_capacity(from_buttons.len() + to_buttons.len());
    for button in to_buttons {
        if !panels.iter().any(|panel| panel == &button.panel) {
            panels.push(button.panel.clone());
        }
    }
    for button in from_buttons {
        if !panels.iter().any(|panel| panel == &button.panel) {
            panels.push(button.panel.clone());
        }
    }

    let buttons = panels
        .into_iter()
        .filter_map(|panel| {
            let from_button = from_buttons.iter().find(|button| button.panel == panel);
            let to_button = to_buttons.iter().find(|button| button.panel == panel);
            let (source, destination) = match (from_button, to_button) {
                (Some(from), Some(to)) => (from.rect, to.rect),
                (None, Some(to)) => (collapse_rect(to.rect, side), to.rect),
                (Some(from), None) => (from.rect, collapse_rect(from.rect, side)),
                (None, None) => return None,
            };
            let template = to_button.or(from_button)?;
            Some(crate::AutoHideButton {
                panel: template.panel.clone(),
                side: template.side,
                rect: lerp_rect(source, destination, numerator, denominator),
                active: to_button.map(|button| button.active).unwrap_or(false),
                focused: to_button.map(|button| button.focused).unwrap_or(false),
            })
        })
        .collect();

    Some(crate::AutoHideBarLayout {
        side,
        rect: lerp_rect(from_rect, to_rect, numerator, denominator),
        buttons,
    })
}

fn interpolate_reveal(
    from: &crate::AutoHideReveal,
    to: &crate::AutoHideReveal,
    numerator: u64,
    denominator: u64,
) -> crate::AutoHideReveal {
    crate::AutoHideReveal {
        panel: to.panel.clone(),
        side: to.side,
        rect: lerp_rect(from.rect, to.rect, numerator, denominator),
        content_rect: lerp_rect(from.content_rect, to.content_rect, numerator, denominator),
        tab_rect: lerp_rect(from.tab_rect, to.tab_rect, numerator, denominator),
    }
}

fn reveal_anchor(
    plan: &RenderPlan,
    panel: &str,
    side: crate::Side,
    template: &crate::AutoHideReveal,
) -> crate::AutoHideReveal {
    let rect = plan
        .autohide
        .bars
        .iter()
        .flat_map(|bar| bar.buttons.iter())
        .find(|button| button.panel == panel)
        .map(|button| button.rect)
        .or_else(|| {
            plan.panels
                .iter()
                .find(|candidate| candidate.id == panel)
                .map(|candidate| candidate.rect)
        })
        .unwrap_or_else(|| collapse_rect(template.rect, side));
    crate::AutoHideReveal {
        panel: panel.into(),
        side,
        rect,
        content_rect: rect,
        tab_rect: rect,
    }
}

fn interpolate_maximized(
    from: Option<&crate::MaximizedLayout>,
    to: Option<&crate::MaximizedLayout>,
    from_plan: &RenderPlan,
    to_plan: &RenderPlan,
    numerator: u64,
    denominator: u64,
) -> Option<crate::MaximizedLayout> {
    match (from, to) {
        (None, None) => None,
        (Some(from), Some(to)) => Some(crate::MaximizedLayout {
            panel: to.panel.clone(),
            rect: lerp_rect(from.rect, to.rect, numerator, denominator),
            can_restore: to.can_restore,
        }),
        (None, Some(to)) => {
            let from_rect = panel_rect(from_plan, &to.panel).unwrap_or(from_plan.main_rect);
            Some(crate::MaximizedLayout {
                panel: to.panel.clone(),
                rect: lerp_rect(from_rect, to.rect, numerator, denominator),
                can_restore: to.can_restore,
            })
        }
        (Some(from), None) => {
            let to_rect = panel_rect(to_plan, &from.panel).unwrap_or(to_plan.main_rect);
            Some(crate::MaximizedLayout {
                panel: from.panel.clone(),
                rect: lerp_rect(from.rect, to_rect, numerator, denominator),
                can_restore: from.can_restore,
            })
        }
    }
}

fn panel_rect(plan: &RenderPlan, panel: &str) -> Option<Rect> {
    plan.panels
        .iter()
        .find(|candidate| candidate.id == panel)
        .map(|candidate| candidate.rect)
}

fn collapse_rect(rect: Rect, side: crate::Side) -> Rect {
    match side {
        crate::Side::Left => Rect { width: 0, ..rect },
        crate::Side::Right => Rect {
            x: edge_coordinate(rect.x, rect.width),
            width: 0,
            ..rect
        },
        crate::Side::Top => Rect { height: 0, ..rect },
        crate::Side::Bottom => Rect {
            y: edge_coordinate(rect.y, rect.height),
            height: 0,
            ..rect
        },
    }
}

fn edge_coordinate(origin: i32, extent: u32) -> i32 {
    (i64::from(origin) + i64::from(extent)).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

fn lerp_rect(from: Rect, to: Rect, numerator: u64, denominator: u64) -> Rect {
    Rect {
        x: lerp_i32(from.x, to.x, numerator, denominator),
        y: lerp_i32(from.y, to.y, numerator, denominator),
        width: lerp_u32(from.width, to.width, numerator, denominator),
        height: lerp_u32(from.height, to.height, numerator, denominator),
    }
}

fn lerp_i32(from: i32, to: i32, numerator: u64, denominator: u64) -> i32 {
    let from = i128::from(from);
    let delta = i128::from(to) - from;
    (from + delta * i128::from(numerator) / i128::from(denominator)) as i32
}

fn lerp_u32(from: u32, to: u32, numerator: u64, denominator: u64) -> u32 {
    let from = i128::from(from);
    let delta = i128::from(to) - from;
    (from + delta * i128::from(numerator) / i128::from(denominator)).max(0) as u32
}

/// Alias kept short for callers building host transition state.
pub type DockPlanTransition = DockGeometryTransition;

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::{
        reduce, solve, Action, Axis, DockLayout, DockNode, Ratio, Side, SplitChild, Viewport,
    };

    #[test]
    fn motion_modes_are_deterministic_and_retargetable() {
        let start = solve(
            &DockLayout {
                root: DockNode::panel("one"),
                ..DockLayout::default()
            },
            Viewport::new(800, 600),
        )
        .unwrap();
        let target = solve(
            &DockLayout {
                root: DockNode::panel("two"),
                ..DockLayout::default()
            },
            Viewport::new(1_000, 700),
        )
        .unwrap();
        let instant = DockGeometryTransition::new(
            start.clone(),
            target.clone(),
            MotionPolicy {
                mode: MotionMode::ReducedMotion,
                ..MotionPolicy::default()
            },
        );
        assert_eq!(instant.duration_ms(), 0);
        assert_eq!(instant.sample(1), target);
        let normal = DockGeometryTransition::new(start, target.clone(), MotionPolicy::default());
        assert_eq!(normal.sample(120), target);
        let retargeted = normal.retarget(20, target.clone(), MotionPolicy::default());
        assert_eq!(retargeted.duration_ms(), 120);
    }

    #[test]
    fn autohide_bar_geometry_enters_and_exits_without_jumping() {
        let start_layout = DockLayout {
            root: DockNode::Split {
                axis: Axis::Horizontal,
                children: vec![
                    SplitChild {
                        ratio: Ratio::HALF,
                        node: DockNode::panel("main"),
                    },
                    SplitChild {
                        ratio: Ratio::HALF,
                        node: DockNode::panel("tools"),
                    },
                ],
            },
            ..DockLayout::default()
        };
        let target_layout = reduce(
            &start_layout,
            Action::AutoHide {
                panel: "tools".into(),
                side: Side::Right,
            },
        )
        .unwrap();
        let start = solve(&start_layout, Viewport::new(800, 600)).unwrap();
        let target = solve(&target_layout, Viewport::new(800, 600)).unwrap();
        let transition =
            DockGeometryTransition::new(start.clone(), target.clone(), MotionPolicy::default());
        let sample = transition.sample(60);
        let target_bar = &target.autohide.bars[0];
        let sample_bar = sample
            .autohide
            .bars
            .iter()
            .find(|bar| bar.side == target_bar.side)
            .expect("the entering auto-hide bar remains represented during motion");
        assert!(sample_bar.rect.width > 0);
        assert!(sample_bar.rect.width < target_bar.rect.width);
        assert_eq!(transition.sample(0), start);
        assert_eq!(transition.sample(120), target);
    }

    #[test]
    fn maximize_geometry_is_interpolated_and_restore_is_retargetable() {
        let start_layout = DockLayout {
            root: DockNode::Split {
                axis: Axis::Horizontal,
                children: vec![
                    SplitChild {
                        ratio: Ratio::HALF,
                        node: DockNode::panel("left"),
                    },
                    SplitChild {
                        ratio: Ratio::HALF,
                        node: DockNode::panel("right"),
                    },
                ],
            },
            ..DockLayout::default()
        };
        let target_layout = reduce(
            &start_layout,
            Action::Maximize {
                panel: "left".into(),
            },
        )
        .unwrap();
        let start = solve(&start_layout, Viewport::new(800, 600)).unwrap();
        let target = solve(&target_layout, Viewport::new(800, 600)).unwrap();
        let transition =
            DockGeometryTransition::new(start.clone(), target.clone(), MotionPolicy::default());
        let sample = transition.sample(60);
        let start_width = start
            .panels
            .iter()
            .find(|panel| panel.id == "left")
            .unwrap()
            .rect
            .width;
        let target_width = target.maximized.as_ref().unwrap().rect.width;
        let sample_width = sample.maximized.as_ref().unwrap().rect.width;
        assert!(sample_width > start_width);
        assert!(sample_width < target_width);
        assert_eq!(transition.sample(0), start);
        assert_eq!(transition.sample(120), target);

        let restore =
            DockGeometryTransition::new(target.clone(), start.clone(), MotionPolicy::default());
        let mid_restore = restore.sample(60);
        assert!(mid_restore.maximized.is_some());
        assert_eq!(restore.sample(120), start);
    }
}
