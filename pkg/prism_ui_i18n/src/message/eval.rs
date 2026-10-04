//! Evaluator for a parsed `ICU`-style message pattern.
//!
//! Evaluation walks the [`Node`] tree produced by [`parser`](super::parser),
//! resolving `select`/`plural`/`selectordinal` arms against runtime [`Args`]
//! and the active locale's plural rules, and rendering the result into a
//! caller-supplied `String`. All integer rendering goes through
//! [`crate::format::push_i64`], so no floating-point math is used.
//!
//! `#` handling follows `ICU` semantics: a `#` renders the nearest enclosing
//! plural's value *minus its offset*, and that active value is threaded through
//! nested `select` arms via the `pound` parameter. At the top level (outside any
//! plural), `#` is a literal `#`.

use alloc::string::String;

use crate::format::{push_i64, Args, Value};
use crate::plural::{PluralCategory, PluralRules};

use super::ast::{Node, PluralSelector};

/// Immutable context shared across a single `format` call.
pub(crate) struct EvalCtx<'a> {
    /// The runtime argument map.
    pub args: &'a Args,
    /// The locale's registered cardinal rules (used by `plural`).
    pub cardinal: PluralRules,
    /// The locale identifier (used to resolve `selectordinal` rules).
    pub locale: &'a str,
}

/// Evaluate `nodes`, appending the rendered output to `out`.
///
/// `pound` carries the active plural value (already offset-adjusted) for `#`
/// substitution, or `None` when no enclosing plural is active.
pub(crate) fn eval(nodes: &[Node], ctx: &EvalCtx<'_>, pound: Option<i64>, out: &mut String) {
    for node in nodes {
        eval_node(node, ctx, pound, out);
    }
}

fn eval_node(node: &Node, ctx: &EvalCtx<'_>, pound: Option<i64>, out: &mut String) {
    match node {
        Node::Text(text) => out.push_str(text),
        Node::Pound => match pound {
            Some(value) => push_i64(out, value),
            None => out.push('#'),
        },
        Node::Arg(name) => match ctx.args.get(name) {
            Some(value) => value.render_into(out),
            None => {
                out.push('{');
                out.push_str(name);
                out.push('}');
            }
        },
        Node::Select { name, arms } => {
            let selector = arg_as_string(ctx.args.get(name));
            let arm = arms
                .iter()
                .find(|arm| arm.key == selector)
                .or_else(|| arms.iter().find(|arm| arm.key == "other"));
            if let Some(arm) = arm {
                // `#` refers to the nearest enclosing plural, so pass it through.
                eval(&arm.body, ctx, pound, out);
            }
        }
        Node::Plural {
            name,
            ordinal,
            offset,
            arms,
        } => {
            let value = arg_as_i64(ctx.args.get(name));
            let adjusted = value.saturating_sub(*offset);

            // Exact `=N` selectors match the original (non-offset) value.
            let exact = arms
                .iter()
                .find(|arm| arm.selector == PluralSelector::Exact(value));
            let arm = exact.or_else(|| {
                let category = if *ordinal {
                    PluralRules::ordinal(ctx.locale).select_i64(adjusted)
                } else {
                    ctx.cardinal.select_i64(adjusted)
                };
                arms.iter()
                    .find(|arm| arm.selector == PluralSelector::Category(category))
                    .or_else(|| {
                        arms.iter().find(|arm| {
                            arm.selector == PluralSelector::Category(PluralCategory::Other)
                        })
                    })
            });

            if let Some(arm) = arm {
                eval(&arm.body, ctx, Some(adjusted), out);
            }
        }
    }
}

/// Render an argument to its string form for a `select` key comparison.
///
/// Missing arguments compare as the empty string, which selects `other`.
fn arg_as_string(value: Option<&Value>) -> String {
    match value {
        Some(Value::Str(s)) => s.clone(),
        Some(Value::Num(n)) => {
            let mut s = String::new();
            push_i64(&mut s, *n);
            s
        }
        None => String::new(),
    }
}

/// Read an integer argument for a `plural`/`selectordinal` switch.
///
/// Non-numeric or missing arguments fall back to `0`.
fn arg_as_i64(value: Option<&Value>) -> i64 {
    match value {
        Some(Value::Num(n)) => *n,
        _ => 0,
    }
}
