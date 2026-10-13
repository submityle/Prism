use crate::*;
use alloc::vec;

fn initial() -> DockLayout {
    DockLayout {
        root: DockNode::Split {
            axis: Axis::Horizontal,
            children: vec![
                SplitChild {
                    ratio: Ratio::from_millionths(400_000),
                    node: DockNode::panel("tool.a"),
                },
                SplitChild {
                    ratio: Ratio::from_millionths(600_000),
                    node: DockNode::Split {
                        axis: Axis::Vertical,
                        children: vec![
                            SplitChild {
                                ratio: Ratio::HALF,
                                node: DockNode::document("doc.main"),
                            },
                            SplitChild {
                                ratio: Ratio::HALF,
                                node: DockNode::panel("tool.b"),
                            },
                        ],
                    },
                },
            ],
        },
        floating: vec![],
        autohide: AutoHideBars::default(),
        maximized: None,
    }
}
fn apply(l: &DockLayout, a: Action) -> DockLayout {
    reduce(l, a).expect("valid reducer action")
}
fn count(n: &DockNode, id: &str) -> usize {
    match n {
        DockNode::TabStack { tabs, .. } => tabs.iter().filter(|x| x.as_str() == id).count(),
        DockNode::Split { children, .. } => children.iter().map(|c| count(&c.node, id)).sum(),
        _ => 0,
    }
}
fn has_doc(n: &DockNode, id: &str) -> bool {
    match n {
        DockNode::Document { tabs, .. } => tabs.iter().any(|x| x == id),
        DockNode::Split { children, .. } => children.iter().any(|c| has_doc(&c.node, id)),
        _ => false,
    }
}
fn ratios(n: &DockNode) {
    if let DockNode::Split { children, .. } = n {
        assert_eq!(
            children
                .iter()
                .map(|c| c.ratio.millionths() as u64)
                .sum::<u64>(),
            Ratio::SCALE as u64
        );
        for c in children {
            assert!(c.ratio.millionths() > 0);
            ratios(&c.node);
        }
    }
}

#[test]
fn model_validates_and_owns_every_panel_once() {
    initial().validate().unwrap();
}

#[test]
fn rejects_duplicate_ownership_bad_ratio_and_document_tabify() {
    let mut l = initial();
    l.floating.push(FloatingWindow {
        rect: Rect {
            x: 0,
            y: 0,
            width: 100,
            height: 100,
        },
        root: DockNode::panel("tool.a"),
        z: 1,
    });
    assert!(matches!(
        l.validate().unwrap_err().issue,
        ValidationIssue::DuplicatePanel(_)
    ));
    let l = initial();
    assert!(reduce(
        &l,
        Action::Move {
            panel: "tool.a".into(),
            target: Target::Document("doc.main".into()),
            placement: Placement::Tab
        }
    )
    .is_err());
    l.validate().unwrap();
}

#[test]
fn split_tabify_preserve_document_and_uniqueness() {
    let l = apply(
        &initial(),
        Action::Split {
            panel: "tool.c".into(),
            target: Target::Document("doc.main".into()),
            side: Side::Right,
        },
    );
    assert_eq!(count(&l.root, "tool.c"), 1);
    assert!(has_doc(&l.root, "doc.main"));
    ratios(&l.root);
    l.validate().unwrap();
    let l = apply(
        &l,
        Action::Tabify {
            panel: "tool.a".into(),
            target: Target::Panel("tool.b".into()),
        },
    );
    assert_eq!(count(&l.root, "tool.a"), 1);
    assert_eq!(count(&l.root, "tool.b"), 1);
    assert!(has_doc(&l.root, "doc.main"));
    l.validate().unwrap();
}

#[test]
fn close_repeatedly_collapses() {
    let mut l = initial();
    for p in ["tool.a", "tool.b"] {
        l = apply(&l, Action::Close { panel: p.into() });
        l.validate().unwrap();
    }
    assert_eq!(
        l.root,
        DockNode::Document {
            tabs: vec!["doc.main".into()],
            active: 0
        }
    );
    for _ in 0..20 {
        l = apply(
            &l,
            Action::Split {
                panel: "temporary".into(),
                target: Target::Document("doc.main".into()),
                side: Side::Left,
            },
        );
        l = apply(
            &l,
            Action::Close {
                panel: "temporary".into(),
            },
        );
        l.validate().unwrap();
    }
    assert_eq!(
        l.root,
        DockNode::Document {
            tabs: vec!["doc.main".into()],
            active: 0
        }
    );
}

#[test]
fn float_dock_and_maximize_restore() {
    let l = initial();
    let f = apply(
        &l,
        Action::Float {
            panel: "tool.a".into(),
            rect: Rect {
                x: 5,
                y: 6,
                width: 700,
                height: 500,
            },
        },
    );
    assert_eq!(f.floating.len(), 1);
    f.validate().unwrap();
    let d = apply(
        &f,
        Action::Dock {
            window: 0,
            target: Target::Document("doc.main".into()),
            placement: Placement::Side(Side::Bottom),
        },
    );
    assert!(d.floating.is_empty());
    d.validate().unwrap();
    let m = apply(
        &d,
        Action::Maximize {
            panel: "tool.a".into(),
        },
    );
    assert_eq!(m.root, DockNode::panel("tool.a"));
    m.validate().unwrap();
    assert_eq!(apply(&m, Action::Restore), d);
}

#[test]
fn autohide_resize_are_deterministic_and_normalized() {
    let l = initial();
    let h = apply(
        &l,
        Action::AutoHide {
            panel: "tool.a".into(),
            side: Side::Left,
        },
    );
    assert_eq!(h.autohide.entries.len(), 1);
    h.validate().unwrap();
    let s = apply(
        &h,
        Action::ShowAutoHide {
            panel: Some("tool.a".into()),
        },
    );
    assert_eq!(s.autohide.active.as_deref(), Some("tool.a"));
    let r = apply(
        &l,
        Action::Resize {
            path: vec![],
            ratios: vec![Ratio::from_millionths(2), Ratio::from_millionths(3)],
        },
    );
    ratios(&r.root);
    r.validate().unwrap();
    assert_eq!(
        r,
        apply(
            &l,
            Action::Resize {
                path: vec![],
                ratios: vec![Ratio::from_millionths(2), Ratio::from_millionths(3)]
            }
        )
    );
}

#[test]
fn systematic_sequences_keep_invariants() {
    let ops = vec![
        Action::Split {
            panel: "x".into(),
            target: Target::Document("doc.main".into()),
            side: Side::Left,
        },
        Action::Float {
            panel: "tool.a".into(),
            rect: Rect::default(),
        },
        Action::AutoHide {
            panel: "tool.b".into(),
            side: Side::Bottom,
        },
        Action::Close {
            panel: "tool.a".into(),
        },
    ];
    for a in &ops {
        for b in &ops {
            for c in &ops {
                let mut l = initial();
                for op in [a, b, c] {
                    if let Ok(n) = reduce(&l, op.clone()) {
                        n.validate().unwrap();
                        ratios(&n.root);
                        l = n;
                    } else {
                        l.validate().unwrap();
                    }
                }
            }
        }
    }
}

#[test]
fn invalid_actions_return_structured_errors() {
    let l = initial();
    assert!(reduce(
        &l,
        Action::Resize {
            path: vec![90],
            ratios: vec![Ratio::HALF]
        }
    )
    .is_err());
    assert!(reduce(&l, Action::Restore).is_err());
}

#[test]
fn maximized_layout_is_read_only_until_restore() {
    let layout = apply(
        &initial(),
        Action::Maximize {
            panel: "tool.a".into(),
        },
    );
    assert_eq!(
        reduce(
            &layout,
            Action::Close {
                panel: "tool.a".into()
            }
        )
        .unwrap_err()
        .issue,
        ValidationIssue::LayoutMaximized,
    );
    assert_eq!(apply(&layout, Action::Restore), initial());
}

#[test]
fn reducer_rejects_invalid_input_without_silently_repairing_it() {
    let mut invalid = initial();
    if let DockNode::Split { children, .. } = &mut invalid.root {
        children[0].ratio = Ratio::from_millionths(1);
    }
    let before = invalid.clone();
    assert!(reduce(
        &invalid,
        Action::Close {
            panel: "tool.a".into()
        }
    )
    .is_err());
    assert_eq!(invalid, before);
}

#[test]
fn floating_rect_must_have_nonzero_extent() {
    let mut invalid = initial();
    invalid.floating.push(FloatingWindow {
        rect: Rect {
            x: 0,
            y: 0,
            width: 0,
            height: 300,
        },
        root: DockNode::panel("tool.float"),
        z: 1,
    });
    assert_eq!(
        invalid.validate().unwrap_err().issue,
        ValidationIssue::InvalidFloatingRect
    );
}
