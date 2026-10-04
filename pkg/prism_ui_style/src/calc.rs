//! CSS math functions: `calc()`, `min()`, `max()` and `clamp()`.
//!
//! This module provides a pure-data, engine-agnostic representation of the CSS
//! Values and Units Module Level 4 [math functions]. A [`CalcValue`] is an
//! expression tree built from length leaves (`px`), percentage leaves,
//! dimensionless number leaves and the arithmetic/comparison operators CSS
//! allows. It is deliberately free of any renderer or layout types so it can be
//! authored, serialized and hot-reloaded alongside the rest of the style layer.
//!
//! # Type checking
//!
//! CSS math expressions are dimensionally typed. For the units this crate
//! models the relevant categories are:
//!
//! * [`CalcType::Length`] — a `px` or percentage term, or any combination that
//!   resolves to a length. (A percentage resolves against a length reference,
//!   so it is length-compatible for the purpose of `+`/`-`/`min`/`max`.)
//! * [`CalcType::Number`] — a dimensionless factor.
//!
//! The CSS grammar restricts which operands are legal:
//!
//! * `+` and `-` require both sides to share a type.
//! * `*` requires at least one side to be a [`CalcType::Number`].
//! * `/` requires the divisor to be a non-zero [`CalcType::Number`].
//! * `min()`, `max()` and `clamp()` require all of their arguments to share a
//!   type and must be non-empty.
//!
//! [`CalcValue::value_type`] performs this check without a reference and
//! surfaces violations as a [`CalcError`] rather than panicking.
//!
//! # Resolution
//!
//! [`CalcValue::resolve`] evaluates the tree against a length reference (the
//! basis a percentage is taken against, in logical pixels) and returns the
//! numeric result. Resolution never panics: division by a value that evaluates
//! to zero yields `0.0`. Callers that need the dimensional guarantee should
//! call [`CalcValue::value_type`] first (see [`CalcValue::resolve_length`]).
//!
//! Design tokens (`var()`-style references) inside a math function are out of
//! scope: a [`CalcValue`] stores already-resolved literals, mirroring the rest
//! of this crate where token references are a separate [`crate::StyleValue`]
//! variant resolved by the cascade before layout.
//!
//! [math functions]: https://www.w3.org/TR/css-values-4/#math

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};

/// Values smaller than this (in magnitude) are treated as zero when guarding
/// division, matching the precision used elsewhere in the UI stack.
const EPS: f32 = 1.0e-6;

/// The dimensional type of a (sub)expression in a CSS math function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalcType {
    /// A dimensionless number (a legal factor or divisor).
    Number,
    /// A length, including percentages that resolve against a length reference.
    Length,
}

impl CalcType {
    const fn describe(self) -> &'static str {
        match self {
            CalcType::Number => "<number>",
            CalcType::Length => "<length>",
        }
    }
}

/// An error produced while type checking a [`CalcValue`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CalcError {
    /// `+` or `-` combined operands of different types.
    IncompatibleTerms {
        /// The type found on the left-hand side.
        left: CalcType,
        /// The type found on the right-hand side.
        right: CalcType,
    },
    /// `*` was applied without a [`CalcType::Number`] on either side.
    NonNumericProduct {
        /// The type found on the left-hand side.
        left: CalcType,
        /// The type found on the right-hand side.
        right: CalcType,
    },
    /// `/` was applied with a divisor that is not a [`CalcType::Number`].
    NonNumericDivisor {
        /// The type found for the divisor.
        divisor: CalcType,
    },
    /// `/` was applied with a divisor that evaluates to zero.
    DivideByZero,
    /// A `min()`, `max()` or `clamp()` combined arguments of different types.
    MismatchedComparison {
        /// The type of the first argument.
        first: CalcType,
        /// The type of the offending argument.
        found: CalcType,
    },
    /// A `min()` or `max()` was given no arguments.
    EmptyComparison,
}

impl fmt::Display for CalcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CalcError::IncompatibleTerms { left, right } => write!(
                f,
                "cannot add or subtract {} and {}",
                left.describe(),
                right.describe()
            ),
            CalcError::NonNumericProduct { left, right } => write!(
                f,
                "a product requires a <number> operand, found {} and {}",
                left.describe(),
                right.describe()
            ),
            CalcError::NonNumericDivisor { divisor } => write!(
                f,
                "a divisor must be a <number>, found {}",
                divisor.describe()
            ),
            CalcError::DivideByZero => write!(f, "division by zero in a math function"),
            CalcError::MismatchedComparison { first, found } => write!(
                f,
                "comparison arguments must share a type, found {} and {}",
                first.describe(),
                found.describe()
            ),
            CalcError::EmptyComparison => {
                write!(f, "`min()`/`max()` require at least one argument")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CalcError {}

/// A node in a CSS math expression tree.
#[derive(Clone, Debug, PartialEq)]
enum CalcNode {
    /// A length in logical pixels.
    Px(f32),
    /// A percentage of the reference, where `100.0` is 100%.
    Percent(f32),
    /// A dimensionless number.
    Number(f32),
    /// Addition of two sub-expressions.
    Sum(Box<CalcNode>, Box<CalcNode>),
    /// Subtraction of two sub-expressions.
    Diff(Box<CalcNode>, Box<CalcNode>),
    /// Multiplication of two sub-expressions.
    Prod(Box<CalcNode>, Box<CalcNode>),
    /// Division of a sub-expression by another.
    Quot(Box<CalcNode>, Box<CalcNode>),
    /// Negation of a sub-expression.
    Neg(Box<CalcNode>),
    /// The minimum of one or more sub-expressions.
    Min(Vec<CalcNode>),
    /// The maximum of one or more sub-expressions.
    Max(Vec<CalcNode>),
    /// `clamp(min, value, max)`.
    Clamp(Box<CalcNode>, Box<CalcNode>, Box<CalcNode>),
}

impl CalcNode {
    fn value_type(&self) -> Result<CalcType, CalcError> {
        match self {
            CalcNode::Px(_) | CalcNode::Percent(_) => Ok(CalcType::Length),
            CalcNode::Number(_) => Ok(CalcType::Number),
            CalcNode::Neg(inner) => inner.value_type(),
            CalcNode::Sum(a, b) | CalcNode::Diff(a, b) => {
                let left = a.value_type()?;
                let right = b.value_type()?;
                if left == right {
                    Ok(left)
                } else {
                    Err(CalcError::IncompatibleTerms { left, right })
                }
            }
            CalcNode::Prod(a, b) => {
                let left = a.value_type()?;
                let right = b.value_type()?;
                match (left, right) {
                    (CalcType::Number, other) | (other, CalcType::Number) => Ok(other),
                    _ => Err(CalcError::NonNumericProduct { left, right }),
                }
            }
            CalcNode::Quot(a, b) => {
                let numerator = a.value_type()?;
                let divisor = b.value_type()?;
                if divisor != CalcType::Number {
                    return Err(CalcError::NonNumericDivisor { divisor });
                }
                // A `<number>` subtree is reference-independent, so it can be
                // evaluated now to reject a statically zero divisor.
                if b.eval(0.0).abs() < EPS {
                    return Err(CalcError::DivideByZero);
                }
                Ok(numerator)
            }
            CalcNode::Min(items) | CalcNode::Max(items) => Self::comparison_type(items),
            CalcNode::Clamp(min, value, max) => {
                let first = min.value_type()?;
                for operand in [value, max] {
                    let found = operand.value_type()?;
                    if found != first {
                        return Err(CalcError::MismatchedComparison { first, found });
                    }
                }
                Ok(first)
            }
        }
    }

    fn comparison_type(items: &[CalcNode]) -> Result<CalcType, CalcError> {
        let mut iter = items.iter();
        let Some(head) = iter.next() else {
            return Err(CalcError::EmptyComparison);
        };
        let first = head.value_type()?;
        for item in iter {
            let found = item.value_type()?;
            if found != first {
                return Err(CalcError::MismatchedComparison { first, found });
            }
        }
        Ok(first)
    }

    fn eval(&self, reference: f32) -> f32 {
        match self {
            CalcNode::Px(px) => *px,
            CalcNode::Percent(pct) => pct / 100.0 * reference,
            CalcNode::Number(n) => *n,
            CalcNode::Neg(inner) => -inner.eval(reference),
            CalcNode::Sum(a, b) => a.eval(reference) + b.eval(reference),
            CalcNode::Diff(a, b) => a.eval(reference) - b.eval(reference),
            CalcNode::Prod(a, b) => a.eval(reference) * b.eval(reference),
            CalcNode::Quot(a, b) => {
                let divisor = b.eval(reference);
                if divisor.abs() < EPS {
                    0.0
                } else {
                    a.eval(reference) / divisor
                }
            }
            CalcNode::Min(items) => items
                .iter()
                .map(|item| item.eval(reference))
                .fold(f32::INFINITY, f32::min),
            CalcNode::Max(items) => items
                .iter()
                .map(|item| item.eval(reference))
                .fold(f32::NEG_INFINITY, f32::max),
            CalcNode::Clamp(min, value, max) => {
                let lo = min.eval(reference);
                let val = value.eval(reference);
                let hi = max.eval(reference);
                // Per spec: clamp(MIN, VAL, MAX) == max(MIN, min(VAL, MAX)).
                lo.max(val.min(hi))
            }
        }
    }
}

/// A CSS math value: `calc()`, `min()`, `max()` or `clamp()`.
///
/// Build one from the leaf constructors ([`CalcValue::px`],
/// [`CalcValue::percent`], [`CalcValue::number`]) and combine them with the
/// arithmetic ([`CalcValue::add`], [`CalcValue::sub`], [`CalcValue::mul`],
/// [`CalcValue::div`], [`CalcValue::neg`]) and comparison ([`CalcValue::min`],
/// [`CalcValue::max`], [`CalcValue::clamp`]) builders.
///
/// # Example
///
/// ```
/// use prism_ui_style::CalcValue;
///
/// // calc(100% - 32px): fill the parent minus a fixed gutter.
/// let width = CalcValue::percent(100.0) - CalcValue::px(32.0);
/// assert_eq!(width.resolve(400.0), 368.0);
///
/// // clamp(200px, 50%, 600px): a responsive but bounded width.
/// let clamped = CalcValue::clamp(
///     CalcValue::px(200.0),
///     CalcValue::percent(50.0),
///     CalcValue::px(600.0),
/// );
/// assert_eq!(clamped.resolve(300.0), 200.0); // 50% == 150px, floored to 200
/// assert_eq!(clamped.resolve(800.0), 400.0); // 50% == 400px, within bounds
/// assert_eq!(clamped.resolve(2000.0), 600.0); // 50% == 1000px, capped at 600
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct CalcValue {
    root: CalcNode,
}

impl CalcValue {
    const fn from_node(root: CalcNode) -> Self {
        Self { root }
    }

    /// A length leaf in logical pixels.
    #[must_use]
    pub const fn px(value: f32) -> Self {
        Self::from_node(CalcNode::Px(value))
    }

    /// A percentage leaf, where `100.0` means 100% of the reference.
    #[must_use]
    pub const fn percent(value: f32) -> Self {
        Self::from_node(CalcNode::Percent(value))
    }

    /// A dimensionless number leaf, for use as a factor or divisor.
    #[must_use]
    pub const fn number(value: f32) -> Self {
        Self::from_node(CalcNode::Number(value))
    }

    /// The minimum of the supplied values (`min(a, b, ...)`).
    ///
    /// All arguments must share a type and the iterator must be non-empty for
    /// the result to type check; see [`CalcValue::value_type`].
    #[must_use]
    pub fn min(values: impl IntoIterator<Item = CalcValue>) -> Self {
        Self::from_node(CalcNode::Min(
            values.into_iter().map(|value| value.root).collect(),
        ))
    }

    /// The maximum of the supplied values (`max(a, b, ...)`).
    ///
    /// All arguments must share a type and the iterator must be non-empty for
    /// the result to type check; see [`CalcValue::value_type`].
    #[must_use]
    pub fn max(values: impl IntoIterator<Item = CalcValue>) -> Self {
        Self::from_node(CalcNode::Max(
            values.into_iter().map(|value| value.root).collect(),
        ))
    }

    /// `clamp(min, value, max)`: `value` restricted to `[min, max]`.
    ///
    /// Per the CSS definition this is exactly `max(min, min(value, max))`, so a
    /// `min` greater than `max` makes `min` win.
    #[must_use]
    pub fn clamp(min: CalcValue, value: CalcValue, max: CalcValue) -> Self {
        Self::from_node(CalcNode::Clamp(
            Box::new(min.root),
            Box::new(value.root),
            Box::new(max.root),
        ))
    }

    /// Type checks the expression, returning its dimensional [`CalcType`].
    ///
    /// # Errors
    ///
    /// Returns a [`CalcError`] when the expression breaks a CSS math rule, for
    /// example adding a length to a number, multiplying two lengths, dividing by
    /// a length, dividing by zero or comparing mismatched types.
    pub fn value_type(&self) -> Result<CalcType, CalcError> {
        self.root.value_type()
    }

    /// Evaluates the expression against a length `reference` (in logical
    /// pixels) and returns the numeric result.
    ///
    /// This never panics. Division by a value that evaluates to zero yields
    /// `0.0`. The result is only dimensionally meaningful when
    /// [`CalcValue::value_type`] succeeds; use [`CalcValue::resolve_length`]
    /// when the length guarantee is required.
    #[must_use]
    pub fn resolve(&self, reference: f32) -> f32 {
        self.root.eval(reference)
    }

    /// Resolves the expression to a length in logical pixels.
    ///
    /// Returns [`None`] when the expression does not type check or does not
    /// resolve to a [`CalcType::Length`] (for example a bare numeric
    /// expression).
    #[must_use]
    pub fn resolve_length(&self, reference: f32) -> Option<f32> {
        match self.value_type() {
            Ok(CalcType::Length) => Some(self.resolve(reference)),
            _ => None,
        }
    }
}

/// `self + rhs`. See [`CalcValue::value_type`] for the typing rule (`+`
/// requires both operands to share a type).
impl Add for CalcValue {
    type Output = CalcValue;

    fn add(self, rhs: CalcValue) -> CalcValue {
        CalcValue::from_node(CalcNode::Sum(Box::new(self.root), Box::new(rhs.root)))
    }
}

/// `self - rhs`. See [`CalcValue::value_type`] for the typing rule (`-`
/// requires both operands to share a type).
impl Sub for CalcValue {
    type Output = CalcValue;

    fn sub(self, rhs: CalcValue) -> CalcValue {
        CalcValue::from_node(CalcNode::Diff(Box::new(self.root), Box::new(rhs.root)))
    }
}

/// `self * rhs`. See [`CalcValue::value_type`] for the typing rule (at least
/// one operand must be a [`CalcValue::number`]).
impl Mul for CalcValue {
    type Output = CalcValue;

    fn mul(self, rhs: CalcValue) -> CalcValue {
        CalcValue::from_node(CalcNode::Prod(Box::new(self.root), Box::new(rhs.root)))
    }
}

/// `self / rhs`. See [`CalcValue::value_type`] for the typing rule (the divisor
/// must be a non-zero [`CalcValue::number`]).
impl Div for CalcValue {
    type Output = CalcValue;

    fn div(self, rhs: CalcValue) -> CalcValue {
        CalcValue::from_node(CalcNode::Quot(Box::new(self.root), Box::new(rhs.root)))
    }
}

/// `-self`.
impl Neg for CalcValue {
    type Output = CalcValue;

    fn neg(self) -> CalcValue {
        CalcValue::from_node(CalcNode::Neg(Box::new(self.root)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn approx(got: f32, want: f32) {
        assert!(
            (got - want).abs() <= 1.0e-3,
            "expected {want}, got {got}"
        );
    }

    #[test]
    fn leaves_resolve_against_reference() {
        approx(CalcValue::px(42.0).resolve(1000.0), 42.0);
        approx(CalcValue::percent(50.0).resolve(400.0), 200.0);
        approx(CalcValue::number(3.0).resolve(999.0), 3.0);
    }

    #[test]
    fn calc_fill_minus_gutter() {
        // calc(100% - 32px)
        let v = CalcValue::percent(100.0) - CalcValue::px(32.0);
        approx(v.resolve(400.0), 368.0);
        approx(v.resolve(100.0), 68.0);
        assert_eq!(v.value_type(), Ok(CalcType::Length));
    }

    #[test]
    fn nested_division_and_grouping() {
        // calc((100% - 20px) / 2)
        let v = (CalcValue::percent(100.0) - CalcValue::px(20.0)) / CalcValue::number(2.0);
        approx(v.resolve(100.0), 40.0);
        approx(v.resolve(220.0), 100.0);
    }

    #[test]
    fn multiplication_is_commutative_in_type_and_value() {
        let left = CalcValue::number(2.0) * CalcValue::px(50.0);
        let right = CalcValue::px(50.0) * CalcValue::number(2.0);
        approx(left.resolve(0.0), 100.0);
        approx(right.resolve(0.0), 100.0);
        assert_eq!(left.value_type(), Ok(CalcType::Length));
        assert_eq!(right.value_type(), Ok(CalcType::Length));
    }

    #[test]
    fn min_max_pick_extremes() {
        let min = CalcValue::min(vec![CalcValue::percent(50.0), CalcValue::px(300.0)]);
        approx(min.resolve(400.0), 200.0); // 50% == 200 < 300
        approx(min.resolve(1000.0), 300.0); // 50% == 500 > 300

        let max = CalcValue::max(vec![CalcValue::percent(50.0), CalcValue::px(300.0)]);
        approx(max.resolve(400.0), 300.0);
        approx(max.resolve(1000.0), 500.0);
    }

    #[test]
    fn clamp_matches_max_of_min() {
        // clamp(200px, 50%, 600px)
        let v = CalcValue::clamp(
            CalcValue::px(200.0),
            CalcValue::percent(50.0),
            CalcValue::px(600.0),
        );
        approx(v.resolve(300.0), 200.0);
        approx(v.resolve(800.0), 400.0);
        approx(v.resolve(2000.0), 600.0);
    }

    #[test]
    fn clamp_with_inverted_bounds_prefers_min() {
        // clamp(600px, 50%, 200px) == max(600, min(50%, 200)) == 600
        let v = CalcValue::clamp(
            CalcValue::px(600.0),
            CalcValue::percent(50.0),
            CalcValue::px(200.0),
        );
        approx(v.resolve(800.0), 600.0);
    }

    #[test]
    fn negation() {
        let v = -CalcValue::px(10.0);
        approx(v.resolve(0.0), -10.0);
        assert_eq!(v.value_type(), Ok(CalcType::Length));
    }

    #[test]
    fn type_errors_follow_css_rules() {
        // length + number is invalid
        assert_eq!(
            (CalcValue::px(1.0) + CalcValue::number(2.0)).value_type(),
            Err(CalcError::IncompatibleTerms {
                left: CalcType::Length,
                right: CalcType::Number,
            })
        );
        // length * length is invalid
        assert_eq!(
            (CalcValue::px(1.0) * CalcValue::percent(2.0)).value_type(),
            Err(CalcError::NonNumericProduct {
                left: CalcType::Length,
                right: CalcType::Length,
            })
        );
        // dividing by a length is invalid
        assert_eq!(
            (CalcValue::px(1.0) / CalcValue::px(2.0)).value_type(),
            Err(CalcError::NonNumericDivisor {
                divisor: CalcType::Length,
            })
        );
        // dividing by zero is invalid
        assert_eq!(
            (CalcValue::px(1.0) / CalcValue::number(0.0)).value_type(),
            Err(CalcError::DivideByZero)
        );
        // mismatched comparison arguments are invalid
        assert_eq!(
            CalcValue::min(vec![CalcValue::px(1.0), CalcValue::number(2.0)]).value_type(),
            Err(CalcError::MismatchedComparison {
                first: CalcType::Length,
                found: CalcType::Number,
            })
        );
        // empty min/max is invalid
        assert_eq!(
            CalcValue::min(Vec::new()).value_type(),
            Err(CalcError::EmptyComparison)
        );
    }

    #[test]
    fn resolve_length_rejects_non_length() {
        let number = CalcValue::number(2.0) * CalcValue::number(3.0);
        assert_eq!(number.resolve_length(100.0), None);
        let invalid = CalcValue::px(1.0) + CalcValue::number(2.0);
        assert_eq!(invalid.resolve_length(100.0), None);
        let length = CalcValue::percent(25.0);
        assert_eq!(length.resolve_length(400.0), Some(100.0));
    }

    #[test]
    fn divide_by_runtime_zero_is_guarded() {
        // The divisor type-checks (a plain number) yet the guard keeps resolve
        // total. Build the zero divisor directly so construction is allowed.
        let v = CalcValue::px(10.0) / CalcValue::number(f32::MIN_POSITIVE / 2.0);
        // MIN_POSITIVE/2 is subnormal and below EPS, so the guard returns 0.
        approx(v.resolve(0.0), 0.0);
    }

    /// Symbolic linear form `(reference_coefficient, constant)` for the
    /// arithmetic subset (no `min`/`max`/`clamp`). Computed by an independent
    /// code path from the numeric evaluator, so it is a genuine oracle: for any
    /// reference `r`, `resolve(r)` must equal `coeff * r + constant`.
    fn linear_form(node: &CalcNode) -> (f64, f64) {
        match node {
            CalcNode::Px(p) => (0.0, f64::from(*p)),
            CalcNode::Percent(f) => (f64::from(*f) / 100.0, 0.0),
            CalcNode::Number(n) => (0.0, f64::from(*n)),
            CalcNode::Neg(a) => {
                let (c, k) = linear_form(a);
                (-c, -k)
            }
            CalcNode::Sum(a, b) => {
                let (ca, ka) = linear_form(a);
                let (cb, kb) = linear_form(b);
                (ca + cb, ka + kb)
            }
            CalcNode::Diff(a, b) => {
                let (ca, ka) = linear_form(a);
                let (cb, kb) = linear_form(b);
                (ca - cb, ka - kb)
            }
            CalcNode::Prod(a, b) => {
                let (ca, ka) = linear_form(a);
                let (cb, kb) = linear_form(b);
                // One side is a pure number (coeff 0) by construction below.
                if ca == 0.0 {
                    (cb * ka, kb * ka)
                } else {
                    (ca * kb, ka * kb)
                }
            }
            CalcNode::Quot(a, b) => {
                let (ca, ka) = linear_form(a);
                let (_, kb) = linear_form(b);
                (ca / kb, ka / kb)
            }
            CalcNode::Min(_) | CalcNode::Max(_) | CalcNode::Clamp(..) => {
                unreachable!("oracle only drives the arithmetic subset")
            }
        }
    }

    // Tiny deterministic PRNG (xorshift32) to keep the test dependency-free.
    struct Rng(u32);
    impl Rng {
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            x
        }

        fn below(&mut self, n: u32) -> u32 {
            self.next_u32() % n
        }

        fn coeff(&mut self) -> f32 {
            // Small non-zero values in [-5, 5] with one decimal place.
            let raw = (self.below(99) as i32) - 49; // -49..=49
            f32::from(raw as i16) / 10.0
        }
    }

    fn random_length_tree(rng: &mut Rng, depth: u32) -> CalcValue {
        if depth == 0 || rng.below(3) == 0 {
            // Leaf: a length (px or percent).
            if rng.below(2) == 0 {
                CalcValue::px(rng.coeff() * 20.0)
            } else {
                CalcValue::percent(rng.coeff() * 20.0)
            }
        } else {
            match rng.below(4) {
                0 => random_length_tree(rng, depth - 1) + random_length_tree(rng, depth - 1),
                1 => random_length_tree(rng, depth - 1) - random_length_tree(rng, depth - 1),
                2 => {
                    // length * number (number on the right).
                    let factor = rng.coeff() + 6.0; // keep away from zero
                    random_length_tree(rng, depth - 1) * CalcValue::number(factor)
                }
                _ => {
                    // length / number (non-zero divisor).
                    let divisor = rng.coeff() + 6.0;
                    random_length_tree(rng, depth - 1) / CalcValue::number(divisor)
                }
            }
        }
    }

    #[test]
    fn resolve_matches_linear_oracle() {
        let mut rng = Rng(0x9E37_79B9);
        for _ in 0..2000 {
            let tree = random_length_tree(&mut rng, 4);
            // Every generated tree is a well-typed length.
            assert_eq!(tree.value_type(), Ok(CalcType::Length));
            let (coeff, constant) = linear_form(&tree.root);
            for reference in [0.0_f32, 1.0, 37.5, 128.0, 1024.0] {
                let expected = coeff * f64::from(reference) + constant;
                let got = f64::from(tree.resolve(reference));
                assert!(
                    (got - expected).abs() <= 1.0e-2 * (1.0 + expected.abs()),
                    "ref {reference}: expected {expected}, got {got}"
                );
            }
        }
    }
}
