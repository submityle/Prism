//! Deterministic materialization of a registry into a first-use dock layout.
//!
//! This module owns the policy boundary between panel contributions and the
//! pure [`DockLayout`] model. It creates one default instance per visible
//! descriptor; supporting multiple instances remains an explicit host concern.

use crate::{
    Axis, DefaultSlot, DockLayout, DockNode, FloatingWindow, LayoutError, PanelId, PanelRegistry,
    Ratio, Rect, SplitChild,
};
use alloc::{vec, vec::Vec};

/// Policy used when a host materializes a registry for the first time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DefaultLayoutPolicy {
    /// Weight of the left lane when it is populated.
    pub left_ratio: Ratio,
    /// Weight of the right lane when it is populated.
    pub right_ratio: Ratio,
    /// Fraction of the central lane reserved for the bottom lane.
    pub bottom_ratio: Ratio,
    /// Logical rectangle assigned to each generated floating panel.
    pub floating_rect: Rect,
}

impl Default for DefaultLayoutPolicy {
    fn default() -> Self {
        Self {
            left_ratio: Ratio::from_millionths(200_000),
            right_ratio: Ratio::from_millionths(240_000),
            bottom_ratio: Ratio::from_millionths(250_000),
            floating_rect: Rect {
                x: 64,
                y: 64,
                width: 960,
                height: 640,
            },
        }
    }
}

/// Why a registry could not be materialized into a legal layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DefaultLayoutError {
    /// A policy ratio is outside the fixed-point domain.
    InvalidRatio {
        /// Policy field containing the invalid value.
        field: &'static str,
        /// Raw fixed-point value.
        value: u32,
    },
    /// A populated lane was assigned zero weight.
    ZeroLaneWeight(&'static str),
    /// The central lane could not leave room for its populated children.
    CentralLaneOverflow,
    /// The generated floating rectangle has no area.
    InvalidFloatingRect,
    /// The generated stacking order exceeded the serialized representation.
    FloatingIndexOverflow,
    /// The resulting model violated a dock invariant.
    InvalidLayout(LayoutError),
}

impl PanelRegistry {
    /// Materializes one deterministic default layout from visible registry
    /// contributions, preserving registration order within every tab stack.
    pub fn default_layout(
        &self,
        policy: DefaultLayoutPolicy,
    ) -> Result<DockLayout, DefaultLayoutError> {
        validate_policy(policy)?;

        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut document = Vec::new();
        let mut bottom = Vec::new();
        let mut floating = Vec::new();

        for descriptor in self.iter().filter(|descriptor| descriptor.default_visible) {
            match descriptor.default_slot {
                DefaultSlot::LeftBar => left.push(descriptor.id.clone()),
                DefaultSlot::RightBar => right.push(descriptor.id.clone()),
                DefaultSlot::Document => document.push(descriptor.id.clone()),
                DefaultSlot::Bottom => bottom.push(descriptor.id.clone()),
                DefaultSlot::Floating => floating.push(descriptor.id.clone()),
            }
        }

        let central = central_lane(document, bottom, policy.bottom_ratio)?;
        let mut lanes = Vec::new();
        if !left.is_empty() {
            if policy.left_ratio.millionths() == 0 {
                return Err(DefaultLayoutError::ZeroLaneWeight("left_ratio"));
            }
            lanes.push((policy.left_ratio.millionths(), tab_stack(left)));
        }
        if let Some(node) = central {
            let weight = Ratio::SCALE
                .checked_sub(policy.left_ratio.millionths())
                .and_then(|remaining| remaining.checked_sub(policy.right_ratio.millionths()))
                .ok_or(DefaultLayoutError::CentralLaneOverflow)?;
            if weight == 0 {
                return Err(DefaultLayoutError::CentralLaneOverflow);
            }
            lanes.push((weight, node));
        }
        if !right.is_empty() {
            if policy.right_ratio.millionths() == 0 {
                return Err(DefaultLayoutError::ZeroLaneWeight("right_ratio"));
            }
            lanes.push((policy.right_ratio.millionths(), tab_stack(right)));
        }

        let root = weighted_split(Axis::Horizontal, lanes)?;
        let floating_panels = floating;
        let mut windows = Vec::new();
        for (index, panel) in floating_panels.into_iter().enumerate() {
            let z = u32::try_from(index)
                .ok()
                .and_then(|index| index.checked_add(1))
                .ok_or(DefaultLayoutError::FloatingIndexOverflow)?;
            windows.push(FloatingWindow {
                rect: policy.floating_rect,
                root: DockNode::panel(panel),
                z,
            });
        }

        let layout = DockLayout {
            root,
            floating: windows,
            ..DockLayout::default()
        };
        layout
            .validate()
            .map_err(DefaultLayoutError::InvalidLayout)?;
        Ok(layout)
    }
}

fn validate_policy(policy: DefaultLayoutPolicy) -> Result<(), DefaultLayoutError> {
    for (field, ratio) in [
        ("left_ratio", policy.left_ratio),
        ("right_ratio", policy.right_ratio),
        ("bottom_ratio", policy.bottom_ratio),
    ] {
        if ratio.millionths() > Ratio::SCALE {
            return Err(DefaultLayoutError::InvalidRatio {
                field,
                value: ratio.millionths(),
            });
        }
    }
    if policy.floating_rect.width == 0 || policy.floating_rect.height == 0 {
        return Err(DefaultLayoutError::InvalidFloatingRect);
    }
    Ok(())
}

fn central_lane(
    document: Vec<PanelId>,
    bottom: Vec<PanelId>,
    bottom_ratio: Ratio,
) -> Result<Option<DockNode>, DefaultLayoutError> {
    match (document.is_empty(), bottom.is_empty()) {
        (true, true) => Ok(None),
        (false, true) => Ok(Some(tab_stack(document))),
        (true, false) => Ok(Some(tab_stack(bottom))),
        (false, false) => {
            let bottom_weight = bottom_ratio.millionths();
            if bottom_weight == 0 {
                return Err(DefaultLayoutError::ZeroLaneWeight("bottom_ratio"));
            }
            let top_weight = Ratio::SCALE - bottom_weight;
            Ok(Some(weighted_split(
                Axis::Vertical,
                vec![
                    (top_weight, tab_stack(document)),
                    (bottom_weight, tab_stack(bottom)),
                ],
            )?))
        }
    }
}

fn tab_stack(tabs: Vec<PanelId>) -> DockNode {
    DockNode::TabStack { tabs, active: 0 }
}

fn weighted_split(
    axis: Axis,
    children: Vec<(u32, DockNode)>,
) -> Result<DockNode, DefaultLayoutError> {
    if children.is_empty() {
        return Ok(DockNode::Empty);
    }
    if children.len() == 1 {
        return Ok(children.into_iter().next().expect("one child").1);
    }
    let total: u64 = children.iter().map(|(weight, _)| u64::from(*weight)).sum();
    if total == 0 {
        return Err(DefaultLayoutError::ZeroLaneWeight("split"));
    }
    let mut ratios = Vec::with_capacity(children.len());
    let mut remainders = Vec::with_capacity(children.len());
    let mut assigned = 0u32;
    for (index, (weight, _)) in children.iter().enumerate() {
        if *weight == 0 {
            return Err(DefaultLayoutError::ZeroLaneWeight("split"));
        }
        let numerator = u64::from(*weight) * u64::from(Ratio::SCALE);
        let base = (numerator / total) as u32;
        assigned = assigned
            .checked_add(base)
            .ok_or(DefaultLayoutError::CentralLaneOverflow)?;
        ratios.push(base);
        remainders.push((numerator % total, index));
    }
    let mut leftover = Ratio::SCALE - assigned;
    remainders.sort_by(
        |(left_remainder, left_index), (right_remainder, right_index)| {
            right_remainder
                .cmp(left_remainder)
                .then(left_index.cmp(right_index))
        },
    );
    for (_, index) in remainders {
        if leftover == 0 {
            break;
        }
        ratios[index] += 1;
        leftover -= 1;
    }
    let children = children
        .into_iter()
        .zip(ratios)
        .map(|((_, node), ratio)| SplitChild {
            ratio: Ratio::from_millionths(ratio),
            node,
        })
        .collect();
    Ok(DockNode::Split { axis, children })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{decode_layout, encode_layout, DefaultSlot, PanelDescriptor};

    fn registry() -> PanelRegistry {
        let mut registry = PanelRegistry::new();
        registry
            .register(PanelDescriptor::new("left.a", "A"))
            .unwrap();
        registry
            .register(PanelDescriptor::new("right", "R").with_default_slot(DefaultSlot::RightBar))
            .unwrap();
        registry
            .register(PanelDescriptor::new("doc", "D").with_default_slot(DefaultSlot::Document))
            .unwrap();
        registry
            .register(PanelDescriptor::new("bottom", "B").with_default_slot(DefaultSlot::Bottom))
            .unwrap();
        registry
            .register(
                PanelDescriptor::new("left.b", "B")
                    .with_default_visible(false)
                    .with_default_slot(DefaultSlot::LeftBar),
            )
            .unwrap();
        registry
    }

    #[test]
    fn default_layout_preserves_registration_order_and_slots() {
        let layout = registry()
            .default_layout(DefaultLayoutPolicy::default())
            .unwrap();
        assert!(layout.root.contains_panel("left.a"));
        assert!(!layout.root.contains_panel("left.b"));
        assert!(layout.root.contains_panel("right"));
        assert!(layout.root.contains_panel("doc"));
        assert!(layout.root.contains_panel("bottom"));
        assert!(layout.validate().is_ok());
    }

    #[test]
    fn default_layout_is_empty_for_an_empty_registry() {
        let layout = PanelRegistry::new()
            .default_layout(DefaultLayoutPolicy::default())
            .unwrap();
        assert_eq!(layout, DockLayout::default());
    }

    #[test]
    fn floating_defaults_have_stable_z_and_round_trip() {
        let mut registry = PanelRegistry::new();
        registry
            .register(PanelDescriptor::new("float.a", "A").with_default_slot(DefaultSlot::Floating))
            .unwrap();
        registry
            .register(PanelDescriptor::new("float.b", "B").with_default_slot(DefaultSlot::Floating))
            .unwrap();
        let layout = registry
            .default_layout(DefaultLayoutPolicy::default())
            .unwrap();
        assert_eq!(
            layout
                .floating
                .iter()
                .map(|window| window.z)
                .collect::<Vec<_>>(),
            [1, 2]
        );
        let bytes = encode_layout(&layout).unwrap();
        assert_eq!(decode_layout(&bytes, Default::default()).unwrap(), layout);
    }

    #[test]
    fn invalid_policy_is_rejected_before_building_layout() {
        let policy = DefaultLayoutPolicy {
            left_ratio: Ratio::from_millionths(Ratio::SCALE + 1),
            ..DefaultLayoutPolicy::default()
        };
        assert!(matches!(
            PanelRegistry::new().default_layout(policy),
            Err(DefaultLayoutError::InvalidRatio {
                field: "left_ratio",
                ..
            })
        ));
    }
}
