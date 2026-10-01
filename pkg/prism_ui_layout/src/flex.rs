//! The flexbox layout algorithm.
//!
//! This module implements the main-axis and cross-axis sizing and alignment
//! described by the CSS Flexible Box specification, including flex-basis
//! resolution, min/max clamping, line wrapping, grow/shrink distribution,
//! gaps, and the full set of justify/align modes. It is intentionally
//! engine-agnostic and deterministic.

use alloc::vec::Vec;

use crate::geometry::{AvailableSpace, Edges, Point, Size};
use crate::result::Layout;
use crate::style::{
    AlignContent, AlignItems, Display, FlexWrap, JustifyContent, LayoutStyle, Position,
};
use crate::tree::{LayoutTree, NodeId};

const EPS: f32 = 1e-4;

/// Entry point used by [`LayoutTree::compute_layout`]. A definite available
/// space is treated as the root's border-box size.
pub(crate) fn compute_root(tree: &mut LayoutTree, root: NodeId, available: Size<AvailableSpace>) {
    let known = Size {
        width: available.width.into_option(),
        height: available.height.into_option(),
    };
    let size = compute_node(tree, root, known, available, true);
    tree.set_layout(
        root,
        Layout {
            order: 0,
            location: Point::ZERO,
            size,
        },
    );
}

/// Computes the border-box size of `node`, laying out its subtree when
/// `perform_layout` is `true`.
fn compute_node(
    tree: &mut LayoutTree,
    node: NodeId,
    known: Size<Option<f32>>,
    available: Size<AvailableSpace>,
    perform_layout: bool,
) -> Size<f32> {
    let style = tree.style(node).clone();

    if style.display == Display::None {
        if perform_layout {
            zero_subtree(tree, node);
        }
        return Size::ZERO;
    }

    if tree.child_count(node) == 0 {
        return compute_leaf(tree, node, &style, known, available);
    }

    compute_flex(tree, node, &style, known, available, perform_layout)
}

/// Sizes a childless box from its style and optional measure function.
fn compute_leaf(
    tree: &LayoutTree,
    node: NodeId,
    style: &LayoutStyle,
    known: Size<Option<f32>>,
    available: Size<AvailableSpace>,
) -> Size<f32> {
    let basis = Size {
        width: available.width.into_option(),
        height: available.height.into_option(),
    };
    let style_size = style.size.resolve(basis);
    let min = style.min_size.resolve(basis);
    let max = style.max_size.resolve(basis);

    let known_w = known.width.or(style_size.width);
    let known_h = known.height.or(style_size.height);

    let needs_measure = known_w.is_none() || known_h.is_none();
    let measured = if needs_measure {
        tree.measure_of(
            node,
            Size {
                width: known_w,
                height: known_h,
            },
            available,
        )
    } else {
        None
    };

    let width = known_w.or_else(|| measured.map(|m| m.width)).unwrap_or(0.0);
    let height = known_h
        .or_else(|| measured.map(|m| m.height))
        .unwrap_or(0.0);

    Size {
        width: clamp_f(width, min.width, max.width),
        height: clamp_f(height, min.height, max.height),
    }
}

/// Per-item scratch state used while running the flex algorithm.
struct FlexItem {
    node: NodeId,
    order: u32,

    size_cross: Option<f32>,
    known_width: Option<f32>,
    known_height: Option<f32>,

    min_main: Option<f32>,
    max_main: Option<f32>,
    min_cross: Option<f32>,
    max_cross: Option<f32>,

    margin: Edges<f32>,
    margin_main: f32,
    margin_cross: f32,

    flex_grow: f32,
    flex_shrink: f32,
    align: AlignItems,

    flex_basis: f32,
    hyp_main: f32,
    outer_hyp_main: f32,
    target_main: f32,
    frozen: bool,
    violation: f32,

    hyp_cross: f32,
    target_cross: f32,
}

/// Lays out a flex container and returns its border-box size.
fn compute_flex(
    tree: &mut LayoutTree,
    node: NodeId,
    style: &LayoutStyle,
    known: Size<Option<f32>>,
    available: Size<AvailableSpace>,
    perform_layout: bool,
) -> Size<f32> {
    let dir = style.flex_direction;
    let is_row = dir.is_row();

    let avail_opt = Size {
        width: available.width.into_option(),
        height: available.height.into_option(),
    };

    let style_size = style.size.resolve(avail_opt);
    let min_size = style.min_size.resolve(avail_opt);
    let max_size = style.max_size.resolve(avail_opt);

    let container_outer = Size {
        width: clamp_opt(
            known.width.or(style_size.width),
            min_size.width,
            max_size.width,
        ),
        height: clamp_opt(
            known.height.or(style_size.height),
            min_size.height,
            max_size.height,
        ),
    };

    let pct_w = container_outer.width.or(avail_opt.width);
    let pct_h = container_outer.height.or(avail_opt.height);

    let padding = resolve_edges(style.padding, pct_w, pct_h);
    let border = resolve_edges(style.border, pct_w, pct_h);
    let inset = Edges {
        left: padding.left + border.left,
        right: padding.right + border.right,
        top: padding.top + border.top,
        bottom: padding.bottom + border.bottom,
    };
    let inset_w = inset.horizontal();
    let inset_h = inset.vertical();

    let inner_width_avail = inner_axis(container_outer.width, avail_opt.width, inset_w);
    let inner_height_avail = inner_axis(container_outer.height, avail_opt.height, inset_h);
    let child_pct = Size {
        width: inner_width_avail,
        height: inner_height_avail,
    };

    let inner_main_avail = pick(is_row, inner_width_avail, inner_height_avail);
    let inner_cross_avail = pick(is_row, inner_height_avail, inner_width_avail);

    let main_gap = if is_row {
        style.gap.width
    } else {
        style.gap.height
    };
    let cross_gap = if is_row {
        style.gap.height
    } else {
        style.gap.width
    };

    let cross_av = match inner_cross_avail {
        Some(c) => AvailableSpace::Definite(c),
        None => AvailableSpace::MaxContent,
    };

    // Gather flow items and defer absolutely positioned children.
    let children = tree.children(node);
    let mut items: Vec<FlexItem> = Vec::new();
    let mut absolutes: Vec<(NodeId, u32)> = Vec::new();

    for (idx, &child) in children.iter().enumerate() {
        let cs = tree.style(child).clone();
        if cs.display == Display::None {
            if perform_layout {
                zero_subtree(tree, child);
            }
            continue;
        }
        if cs.position == Position::Absolute {
            absolutes.push((child, idx as u32));
            continue;
        }

        let csize = cs.size.resolve(child_pct);
        let cmin = cs.min_size.resolve(child_pct);
        let cmax = cs.max_size.resolve(child_pct);
        let margin = resolve_edges(cs.margin, child_pct.width, child_pct.height);

        let size_main = pick(is_row, csize.width, csize.height);
        let size_cross = pick(is_row, csize.height, csize.width);
        let min_main = pick(is_row, cmin.width, cmin.height);
        let max_main = pick(is_row, cmax.width, cmax.height);
        let min_cross = pick(is_row, cmin.height, cmin.width);
        let max_cross = pick(is_row, cmax.height, cmax.width);

        let margin_main = margin.main_axis(is_row);
        let margin_cross = margin.cross_axis(is_row);

        // Resolve the flex base size.
        let flex_basis = if cs.flex_basis.is_auto() {
            if let Some(main) = size_main {
                main
            } else {
                let child_avail = axis_size(is_row, AvailableSpace::MaxContent, cross_av);
                let content = compute_node(
                    tree,
                    child,
                    Size {
                        width: csize.width,
                        height: csize.height,
                    },
                    child_avail,
                    false,
                );
                pick(is_row, content.width, content.height)
            }
        } else {
            cs.flex_basis.resolve_or_zero(inner_main_avail)
        };

        let hyp_main = clamp_f(flex_basis.max(0.0), min_main, max_main);

        items.push(FlexItem {
            node: child,
            order: idx as u32,
            size_cross,
            known_width: csize.width,
            known_height: csize.height,
            min_main,
            max_main,
            min_cross,
            max_cross,
            margin,
            margin_main,
            margin_cross,
            flex_grow: cs.flex_grow,
            flex_shrink: cs.flex_shrink,
            align: cs.align_self.unwrap_or(style.align_items),
            flex_basis,
            hyp_main,
            outer_hyp_main: hyp_main + margin_main,
            target_main: hyp_main,
            frozen: false,
            violation: 0.0,
            hyp_cross: 0.0,
            target_cross: 0.0,
        });
    }

    // Line breaking.
    let lines = break_lines(&items, style.flex_wrap, inner_main_avail, main_gap);

    // Resolve flexible lengths per line.
    for line in &lines {
        if let Some(container_main) = inner_main_avail {
            resolve_flexible_lengths(&mut items, line, container_main, main_gap);
        } else {
            for &i in line {
                items[i].target_main = items[i].hyp_main;
            }
        }
    }

    // Resolve each item's hypothetical cross size.
    #[expect(
        clippy::needless_range_loop,
        reason = "the loop body mutably borrows the tree while updating items[i], so an index loop is required"
    )]
    for i in 0..items.len() {
        let size_cross = items[i].size_cross;
        let cross = if let Some(c) = size_cross {
            clamp_f(c, items[i].min_cross, items[i].max_cross)
        } else {
            let target_main = items[i].target_main;
            let known_child = if is_row {
                Size {
                    width: Some(target_main),
                    height: items[i].known_height,
                }
            } else {
                Size {
                    width: items[i].known_width,
                    height: Some(target_main),
                }
            };
            let child_avail = axis_size(is_row, AvailableSpace::Definite(target_main), cross_av);
            let content = compute_node(tree, items[i].node, known_child, child_avail, false);
            let cc = pick(is_row, content.height, content.width);
            clamp_f(cc, items[i].min_cross, items[i].max_cross)
        };
        items[i].hyp_cross = cross;
        items[i].target_cross = cross;
    }

    // Cross size of each line.
    let line_count = lines.len();
    let mut line_cross: Vec<f32> = Vec::with_capacity(line_count);
    for line in &lines {
        let mut max_c = 0.0_f32;
        for &i in line {
            max_c = max_c.max(items[i].hyp_cross + items[i].margin_cross);
        }
        line_cross.push(max_c);
    }
    if let (1, Some(c)) = (line_count, inner_cross_avail) {
        line_cross[0] = c;
    }

    let content_cross_sum: f32 = line_cross.iter().sum::<f32>() + cross_gap * gaps(line_count);
    let container_cross_inner = inner_cross_avail.unwrap_or(content_cross_sum);

    // Distribute lines along the cross axis (align-content).
    let free_cross = container_cross_inner - content_cross_sum;
    let (mut cross_lead, mut cross_between) = (0.0_f32, cross_gap);
    if line_count > 0 {
        match style.align_content {
            AlignContent::Start => {}
            AlignContent::End => cross_lead = free_cross,
            AlignContent::Center => cross_lead = free_cross / 2.0,
            AlignContent::Stretch => {
                if free_cross > 0.0 {
                    let add = free_cross / line_count as f32;
                    for l in &mut line_cross {
                        *l += add;
                    }
                }
            }
            AlignContent::SpaceBetween => {
                if free_cross > 0.0 && line_count > 1 {
                    cross_between = cross_gap + free_cross / (line_count as f32 - 1.0);
                }
            }
            AlignContent::SpaceAround => {
                if free_cross > 0.0 {
                    let s = free_cross / line_count as f32;
                    cross_lead = s / 2.0;
                    cross_between = cross_gap + s;
                }
            }
            AlignContent::SpaceEvenly => {
                if free_cross > 0.0 {
                    let s = free_cross / (line_count as f32 + 1.0);
                    cross_lead = s;
                    cross_between = cross_gap + s;
                }
            }
        }
    }

    let mut line_start: Vec<f32> = Vec::with_capacity(line_count);
    let mut cursor = cross_lead;
    for &lc in &line_cross {
        line_start.push(cursor);
        cursor += lc + cross_between;
    }
    if style.flex_wrap == FlexWrap::WrapReverse {
        for l in 0..line_count {
            line_start[l] = container_cross_inner - line_start[l] - line_cross[l];
        }
    }

    // Final sizes of the container.
    let mut content_main = 0.0_f32;
    for line in &lines {
        let used: f32 = line
            .iter()
            .map(|&i| items[i].target_main + items[i].margin_main)
            .sum::<f32>()
            + main_gap * gaps(line.len());
        content_main = content_main.max(used);
    }

    let resolved_main = pick(is_row, container_outer.width, container_outer.height);
    let resolved_cross = pick(is_row, container_outer.height, container_outer.width);
    let min_main_c = pick(is_row, min_size.width, min_size.height);
    let max_main_c = pick(is_row, max_size.width, max_size.height);
    let min_cross_c = pick(is_row, min_size.height, min_size.width);
    let max_cross_c = pick(is_row, max_size.height, max_size.width);
    let inset_main = inset.main_axis(is_row);
    let inset_cross = inset.cross_axis(is_row);

    let final_main =
        resolved_main.unwrap_or_else(|| clamp_f(content_main + inset_main, min_main_c, max_main_c));
    let final_cross = resolved_cross.unwrap_or_else(|| {
        clamp_f(
            container_cross_inner + inset_cross,
            min_cross_c,
            max_cross_c,
        )
    });

    let (final_w, final_h) = if is_row {
        (final_main, final_cross)
    } else {
        (final_cross, final_main)
    };

    if perform_layout {
        place_items(
            tree,
            &mut items,
            &lines,
            &line_cross,
            &line_start,
            PlacementCtx {
                is_row,
                reverse: dir.is_reverse(),
                justify: style.justify_content,
                main_gap,
                inner_main_avail,
                origin_main: inset.main_start(is_row),
                origin_cross: inset.cross_start(is_row),
            },
        );

        let final_inner_w = (final_w - inset_w).max(0.0);
        let final_inner_h = (final_h - inset_h).max(0.0);
        for (child, order) in absolutes {
            place_absolute(
                tree,
                child,
                order,
                final_inner_w,
                final_inner_h,
                inset.left,
                inset.top,
            );
        }
    }

    Size {
        width: final_w,
        height: final_h,
    }
}

/// Parameters threaded into the item placement pass.
struct PlacementCtx {
    is_row: bool,
    reverse: bool,
    justify: JustifyContent,
    main_gap: f32,
    inner_main_avail: Option<f32>,
    origin_main: f32,
    origin_cross: f32,
}

/// Positions every flow item and lays out its subtree.
fn place_items(
    tree: &mut LayoutTree,
    items: &mut [FlexItem],
    lines: &[Vec<usize>],
    line_cross: &[f32],
    line_start: &[f32],
    ctx: PlacementCtx,
) {
    for (l, line) in lines.iter().enumerate() {
        let n = line.len();

        // Apply cross-axis stretch.
        for &i in line {
            if items[i].align == AlignItems::Stretch && items[i].size_cross.is_none() {
                let stretched = clamp_f(
                    (line_cross[l] - items[i].margin_cross).max(0.0),
                    items[i].min_cross,
                    items[i].max_cross,
                );
                items[i].target_cross = stretched;
            } else {
                items[i].target_cross = items[i].hyp_cross;
            }
        }

        let total_main: f32 = line
            .iter()
            .map(|&i| items[i].target_main + items[i].margin_main)
            .sum::<f32>();
        let container_main = ctx
            .inner_main_avail
            .unwrap_or(total_main + ctx.main_gap * gaps(n));
        let free_main = container_main - total_main - ctx.main_gap * gaps(n);

        let (lead, between) = justify_offsets(ctx.justify, free_main, n, ctx.main_gap);

        let mut main_cursor = lead;
        for &i in line {
            let outer_main = items[i].target_main + items[i].margin_main;
            let box_start = if ctx.reverse {
                container_main - main_cursor - outer_main
            } else {
                main_cursor
            };
            let main_pos = box_start + items[i].margin.main_start(ctx.is_row);

            let free_item_cross = line_cross[l] - (items[i].target_cross + items[i].margin_cross);
            let cross_align = match items[i].align {
                AlignItems::Start | AlignItems::Stretch => 0.0,
                AlignItems::End => free_item_cross,
                AlignItems::Center => free_item_cross / 2.0,
            };
            let cross_pos = line_start[l] + cross_align + items[i].margin.cross_start(ctx.is_row);

            let (x, y) = if ctx.is_row {
                (ctx.origin_main + main_pos, ctx.origin_cross + cross_pos)
            } else {
                (ctx.origin_cross + cross_pos, ctx.origin_main + main_pos)
            };
            let (w, h) = if ctx.is_row {
                (items[i].target_main, items[i].target_cross)
            } else {
                (items[i].target_cross, items[i].target_main)
            };

            compute_node(
                tree,
                items[i].node,
                Size {
                    width: Some(w),
                    height: Some(h),
                },
                Size {
                    width: AvailableSpace::Definite(w),
                    height: AvailableSpace::Definite(h),
                },
                true,
            );
            tree.set_layout(
                items[i].node,
                Layout {
                    order: items[i].order,
                    location: Point::new(x, y),
                    size: Size::new(w, h),
                },
            );

            main_cursor += items[i].margin_main + items[i].target_main + between;
        }
    }
}

/// Resolves flexible lengths for a single line per CSS 9.7.
fn resolve_flexible_lengths(
    items: &mut [FlexItem],
    line: &[usize],
    container_main: f32,
    main_gap: f32,
) {
    let total_gap = main_gap * gaps(line.len());
    let sum_outer_hyp: f32 = line.iter().map(|&i| items[i].outer_hyp_main).sum();
    let growing = sum_outer_hyp + total_gap < container_main;

    for &i in line {
        items[i].target_main = items[i].hyp_main;
        items[i].frozen = false;
    }

    // Freeze items that cannot flex in the active direction.
    for &i in line {
        let factor = if growing {
            items[i].flex_grow
        } else {
            items[i].flex_shrink
        };
        let clamped_against_base = (growing && items[i].flex_basis > items[i].hyp_main)
            || (!growing && items[i].flex_basis < items[i].hyp_main);
        if factor == 0.0 || clamped_against_base {
            items[i].frozen = true;
            items[i].target_main = items[i].hyp_main;
        }
    }

    let initial_free_space = {
        let used: f32 = line
            .iter()
            .map(|&i| {
                items[i].margin_main
                    + if items[i].frozen {
                        items[i].target_main
                    } else {
                        items[i].flex_basis
                    }
            })
            .sum::<f32>()
            + total_gap;
        container_main - used
    };

    loop {
        if line.iter().all(|&i| items[i].frozen) {
            break;
        }

        let used: f32 = line
            .iter()
            .map(|&i| {
                items[i].margin_main
                    + if items[i].frozen {
                        items[i].target_main
                    } else {
                        items[i].flex_basis
                    }
            })
            .sum::<f32>()
            + total_gap;
        let remaining = container_main - used;

        let mut sum_grow = 0.0;
        let mut sum_shrink = 0.0;
        let mut sum_scaled = 0.0;
        for &i in line {
            if !items[i].frozen {
                sum_grow += items[i].flex_grow;
                sum_shrink += items[i].flex_shrink;
                sum_scaled += items[i].flex_shrink * items[i].flex_basis;
            }
        }

        let free = if growing && sum_grow < 1.0 {
            min_f(initial_free_space * sum_grow, remaining)
        } else if !growing && sum_shrink < 1.0 {
            max_f(initial_free_space * sum_shrink, remaining)
        } else {
            remaining
        };

        if growing && sum_grow > 0.0 {
            for &i in line {
                if !items[i].frozen {
                    items[i].target_main =
                        items[i].flex_basis + free * (items[i].flex_grow / sum_grow);
                }
            }
        } else if !growing && sum_scaled > 0.0 {
            for &i in line {
                if !items[i].frozen {
                    let scaled = items[i].flex_shrink * items[i].flex_basis;
                    items[i].target_main = items[i].flex_basis + free * (scaled / sum_scaled);
                }
            }
        } else {
            for &i in line {
                if !items[i].frozen {
                    items[i].target_main = items[i].flex_basis;
                    items[i].frozen = true;
                }
            }
            break;
        }

        let mut total_violation = 0.0_f32;
        for &i in line {
            if items[i].frozen {
                continue;
            }
            let unclamped = items[i].target_main;
            let clamped = clamp_f(unclamped.max(0.0), items[i].min_main, items[i].max_main);
            items[i].violation = clamped - unclamped;
            total_violation += items[i].violation;
            items[i].target_main = clamped;
        }

        for &i in line {
            if items[i].frozen {
                continue;
            }
            if total_violation > EPS {
                if items[i].violation > 0.0 {
                    items[i].frozen = true;
                }
            } else if total_violation < -EPS {
                if items[i].violation < 0.0 {
                    items[i].frozen = true;
                }
            } else {
                items[i].frozen = true;
            }
        }
    }
}

/// Positions a single absolutely positioned child.
fn place_absolute(
    tree: &mut LayoutTree,
    child: NodeId,
    order: u32,
    inner_w: f32,
    inner_h: f32,
    origin_x: f32,
    origin_y: f32,
) {
    let cs = tree.style(child).clone();
    let basis = Size {
        width: Some(inner_w),
        height: Some(inner_h),
    };
    let csize = cs.size.resolve(basis);
    let cmin = cs.min_size.resolve(basis);
    let cmax = cs.max_size.resolve(basis);

    let left = cs.inset.left.resolve(Some(inner_w));
    let right = cs.inset.right.resolve(Some(inner_w));
    let top = cs.inset.top.resolve(Some(inner_h));
    let bottom = cs.inset.bottom.resolve(Some(inner_h));

    let width = match csize.width {
        Some(w) => w,
        None => match (left, right) {
            (Some(l), Some(r)) => (inner_w - l - r).max(0.0),
            _ => {
                let m = compute_node(
                    tree,
                    child,
                    Size {
                        width: None,
                        height: csize.height,
                    },
                    Size {
                        width: AvailableSpace::Definite(inner_w),
                        height: AvailableSpace::Definite(inner_h),
                    },
                    false,
                );
                m.width
            }
        },
    };
    let height = match csize.height {
        Some(h) => h,
        None => match (top, bottom) {
            (Some(t), Some(b)) => (inner_h - t - b).max(0.0),
            _ => {
                let m = compute_node(
                    tree,
                    child,
                    Size {
                        width: Some(width),
                        height: None,
                    },
                    Size {
                        width: AvailableSpace::Definite(width),
                        height: AvailableSpace::Definite(inner_h),
                    },
                    false,
                );
                m.height
            }
        },
    };

    let width = clamp_f(width, cmin.width, cmax.width);
    let height = clamp_f(height, cmin.height, cmax.height);

    let x = match (left, right) {
        (Some(l), _) => l,
        (None, Some(r)) => inner_w - width - r,
        (None, None) => 0.0,
    };
    let y = match (top, bottom) {
        (Some(t), _) => t,
        (None, Some(b)) => inner_h - height - b,
        (None, None) => 0.0,
    };

    compute_node(
        tree,
        child,
        Size {
            width: Some(width),
            height: Some(height),
        },
        Size {
            width: AvailableSpace::Definite(width),
            height: AvailableSpace::Definite(height),
        },
        true,
    );
    tree.set_layout(
        child,
        Layout {
            order,
            location: Point::new(origin_x + x, origin_y + y),
            size: Size::new(width, height),
        },
    );
}

/// Breaks items into flex lines.
fn break_lines(
    items: &[FlexItem],
    wrap: FlexWrap,
    inner_main_avail: Option<f32>,
    main_gap: f32,
) -> Vec<Vec<usize>> {
    let mut lines: Vec<Vec<usize>> = Vec::new();
    if items.is_empty() {
        return lines;
    }

    match (wrap, inner_main_avail) {
        (FlexWrap::NoWrap, _) | (_, None) => {
            lines.push((0..items.len()).collect());
        }
        (_, Some(container_main)) => {
            let mut current: Vec<usize> = Vec::new();
            let mut used = 0.0_f32;
            for (i, item) in items.iter().enumerate() {
                let item_main = item.outer_hyp_main;
                let add_gap = if current.is_empty() { 0.0 } else { main_gap };
                if !current.is_empty() && used + add_gap + item_main > container_main + EPS {
                    lines.push(core::mem::take(&mut current));
                    used = 0.0;
                }
                let gap_now = if current.is_empty() { 0.0 } else { main_gap };
                used += gap_now + item_main;
                current.push(i);
            }
            if !current.is_empty() {
                lines.push(current);
            }
        }
    }
    lines
}

/// Computes the leading offset and inter-item spacing for a justify mode.
fn justify_offsets(justify: JustifyContent, free: f32, n: usize, main_gap: f32) -> (f32, f32) {
    if n == 0 {
        return (0.0, main_gap);
    }
    if free <= 0.0 {
        return match justify {
            JustifyContent::End => (free, main_gap),
            JustifyContent::Center => (free / 2.0, main_gap),
            _ => (0.0, main_gap),
        };
    }
    match justify {
        JustifyContent::Start => (0.0, main_gap),
        JustifyContent::End => (free, main_gap),
        JustifyContent::Center => (free / 2.0, main_gap),
        JustifyContent::SpaceBetween => {
            if n > 1 {
                (0.0, main_gap + free / (n as f32 - 1.0))
            } else {
                (0.0, main_gap)
            }
        }
        JustifyContent::SpaceAround => {
            let s = free / n as f32;
            (s / 2.0, main_gap + s)
        }
        JustifyContent::SpaceEvenly => {
            let s = free / (n as f32 + 1.0);
            (s, main_gap + s)
        }
    }
}

/// Recursively zeroes the layout of a subtree (used for `display: none`).
fn zero_subtree(tree: &mut LayoutTree, node: NodeId) {
    tree.set_layout(node, Layout::ZERO);
    for child in tree.children(node) {
        zero_subtree(tree, child);
    }
}

// --- Small numeric helpers. ----------------------------------------------

fn resolve_edges(
    edges: Edges<crate::geometry::Dimension>,
    h_basis: Option<f32>,
    v_basis: Option<f32>,
) -> Edges<f32> {
    Edges {
        left: edges.left.resolve_or_zero(h_basis),
        right: edges.right.resolve_or_zero(h_basis),
        top: edges.top.resolve_or_zero(v_basis),
        bottom: edges.bottom.resolve_or_zero(v_basis),
    }
}

fn inner_axis(outer: Option<f32>, avail: Option<f32>, inset: f32) -> Option<f32> {
    match outer {
        Some(v) => Some((v - inset).max(0.0)),
        None => avail.map(|a| (a - inset).max(0.0)),
    }
}

fn pick<T>(is_row: bool, row_value: T, column_value: T) -> T {
    if is_row {
        row_value
    } else {
        column_value
    }
}

fn axis_size(is_row: bool, main: AvailableSpace, cross: AvailableSpace) -> Size<AvailableSpace> {
    if is_row {
        Size {
            width: main,
            height: cross,
        }
    } else {
        Size {
            width: cross,
            height: main,
        }
    }
}

fn gaps(count: usize) -> f32 {
    if count > 1 {
        count as f32 - 1.0
    } else {
        0.0
    }
}

fn clamp_f(value: f32, min: Option<f32>, max: Option<f32>) -> f32 {
    let mut result = value;
    if let Some(mx) = max {
        result = result.min(mx);
    }
    if let Some(mn) = min {
        result = result.max(mn);
    }
    result
}

fn clamp_opt(value: Option<f32>, min: Option<f32>, max: Option<f32>) -> Option<f32> {
    value.map(|v| clamp_f(v, min, max))
}

fn min_f(a: f32, b: f32) -> f32 {
    if a < b {
        a
    } else {
        b
    }
}

fn max_f(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}
