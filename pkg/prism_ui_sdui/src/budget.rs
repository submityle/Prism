//! A total-node budget for bounding sanitation on hostile input.
//!
//! The [`Sandbox`](crate::Sandbox) already caps recursion *depth* so a deeply
//! nested document cannot exhaust the stack. Depth alone, however, says nothing
//! about *breadth*: a shallow document with an enormous number of siblings is
//! still cheap to encode on the wire yet expensive to materialize into a
//! sanitized tree and, downstream, into [`prism_ui::Element`](prism_ui::Element)
//! nodes. [`NodeBudget`] closes that gap by capping the total number of nodes
//! the sandbox will emit, mirroring the "amplification" limits that production
//! server-driven UI stacks apply to untrusted payloads.
//!
//! The budget is a simple monotonically draining counter: each node the sandbox
//! commits to its output claims exactly one unit via [`NodeBudget::try_consume`].
//! Once the budget reaches zero, further nodes cannot be emitted and the sandbox
//! truncates the remaining siblings, recording the truncation as a diagnostic
//! rather than dropping content silently.

/// A monotonically draining allowance for the number of nodes a sanitation pass
/// may emit.
///
/// A budget is created with a fixed `limit` and spends one unit per committed
/// node through [`try_consume`](NodeBudget::try_consume). It never refills, so
/// the number of successful consumptions over the budget's lifetime can never
/// exceed `limit`. This gives the sandbox a hard upper bound on output size
/// regardless of how pathological the untrusted input is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeBudget {
    limit: usize,
    remaining: usize,
}

impl NodeBudget {
    /// Creates a budget allowing at most `limit` nodes to be emitted.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_ui_sdui::NodeBudget;
    ///
    /// let budget = NodeBudget::new(3);
    /// assert_eq!(budget.limit(), 3);
    /// assert_eq!(budget.remaining(), 3);
    /// ```
    #[must_use]
    pub const fn new(limit: usize) -> Self {
        Self {
            limit,
            remaining: limit,
        }
    }

    /// The original allowance this budget was created with.
    #[must_use]
    pub const fn limit(&self) -> usize {
        self.limit
    }

    /// How many nodes may still be emitted before the budget is exhausted.
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.remaining
    }

    /// The number of nodes already committed against this budget.
    #[must_use]
    pub const fn spent(&self) -> usize {
        self.limit - self.remaining
    }

    /// Whether the budget has been fully spent.
    #[must_use]
    pub const fn is_exhausted(&self) -> bool {
        self.remaining == 0
    }

    /// Claims one unit of the budget for a node about to be emitted.
    ///
    /// Returns `true` and decrements the remaining allowance when a unit was
    /// available, or `false` without changing state when the budget is already
    /// exhausted. Because the budget never refills, the total number of `true`
    /// results it will ever return is bounded by [`limit`](NodeBudget::limit).
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_ui_sdui::NodeBudget;
    ///
    /// let mut budget = NodeBudget::new(1);
    /// assert!(budget.try_consume());
    /// assert!(budget.is_exhausted());
    /// assert!(!budget.try_consume());
    /// ```
    pub fn try_consume(&mut self) -> bool {
        if self.remaining == 0 {
            return false;
        }
        self.remaining -= 1;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::NodeBudget;

    #[test]
    fn fresh_budget_reports_full_allowance() {
        let budget = NodeBudget::new(5);
        assert_eq!(budget.limit(), 5);
        assert_eq!(budget.remaining(), 5);
        assert_eq!(budget.spent(), 0);
        assert!(!budget.is_exhausted());
    }

    #[test]
    fn consuming_drains_the_allowance_exactly_once() {
        let mut budget = NodeBudget::new(2);
        assert!(budget.try_consume());
        assert_eq!(budget.remaining(), 1);
        assert_eq!(budget.spent(), 1);
        assert!(budget.try_consume());
        assert!(budget.is_exhausted());
        assert_eq!(budget.spent(), 2);
    }

    #[test]
    fn exhausted_budget_refuses_without_underflow() {
        let mut budget = NodeBudget::new(1);
        assert!(budget.try_consume());
        // Repeated attempts past exhaustion must stay false and never wrap.
        for _ in 0..1_000 {
            assert!(!budget.try_consume());
        }
        assert_eq!(budget.remaining(), 0);
        assert_eq!(budget.spent(), 1);
    }

    #[test]
    fn zero_budget_consumes_nothing() {
        let mut budget = NodeBudget::new(0);
        assert!(budget.is_exhausted());
        assert!(!budget.try_consume());
        assert_eq!(budget.spent(), 0);
    }

    #[test]
    fn total_successful_consumptions_never_exceed_limit() {
        // The core budget invariant, checked across a range of limits.
        for limit in 0..64 {
            let mut budget = NodeBudget::new(limit);
            let mut granted = 0usize;
            for _ in 0..256 {
                if budget.try_consume() {
                    granted += 1;
                }
            }
            assert_eq!(granted, limit);
            assert!(granted <= budget.limit());
        }
    }
}
