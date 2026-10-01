//! A cyclic focus trap over an ordered set of focusable keys.

use alloc::vec::Vec;

/// Keeps keyboard focus confined to an ordered set of focusable keys, cycling
/// with wrap-around.
///
/// `FocusTrap` is deliberately generic over the key type `K` so it can trap any
/// identifier a caller uses for focusable targets (for instance
/// [`prism_ui::Key`], an integer index, or a `&'static str`). It stores no
/// elements itself; it only tracks which key currently holds focus and how to
/// advance.
///
/// An empty trap is handled gracefully: [`current`](FocusTrap::current),
/// [`next`](FocusTrap::next) and [`prev`](FocusTrap::prev) all return `None`.
#[derive(Clone, Debug, Default)]
pub struct FocusTrap<K> {
    keys: Vec<K>,
    index: usize,
}

impl<K> FocusTrap<K> {
    /// Creates a trap over `keys`, focusing the first key (if any).
    #[must_use]
    pub fn new(keys: Vec<K>) -> Self {
        Self { keys, index: 0 }
    }

    /// The number of focusable keys in the trap.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether the trap holds no focusable keys.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// The ordered focusable keys.
    #[must_use]
    pub fn keys(&self) -> &[K] {
        &self.keys
    }

    /// The key that currently holds focus, or `None` if the trap is empty.
    #[must_use]
    pub fn current(&self) -> Option<&K> {
        self.keys.get(self.index)
    }

    /// Advances focus to the next key, wrapping around to the first, and
    /// returns the newly focused key. Returns `None` if the trap is empty.
    #[expect(
        clippy::should_implement_trait,
        reason = "cycling focus is the domain API; FocusTrap is not an iterator and never exhausts"
    )]
    pub fn next(&mut self) -> Option<&K> {
        if self.keys.is_empty() {
            return None;
        }
        self.index = (self.index + 1) % self.keys.len();
        self.keys.get(self.index)
    }

    /// Moves focus to the previous key, wrapping around to the last, and
    /// returns the newly focused key. Returns `None` if the trap is empty.
    pub fn prev(&mut self) -> Option<&K> {
        if self.keys.is_empty() {
            return None;
        }
        // Add `len` before subtracting to avoid underflow at index 0.
        self.index = (self.index + self.keys.len() - 1) % self.keys.len();
        self.keys.get(self.index)
    }
}

impl<K: PartialEq> FocusTrap<K> {
    /// Moves focus directly to `key` if it is present, returning whether it was
    /// found.
    pub fn focus(&mut self, key: &K) -> bool {
        if let Some(pos) = self.keys.iter().position(|k| k == key) {
            self.index = pos;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FocusTrap;
    use alloc::vec;

    #[test]
    fn empty_trap_is_graceful() {
        let mut trap: FocusTrap<i32> = FocusTrap::new(vec![]);
        assert!(trap.is_empty());
        assert_eq!(trap.len(), 0);
        assert_eq!(trap.current(), None);
        assert_eq!(trap.next(), None);
        assert_eq!(trap.prev(), None);
    }

    #[test]
    fn next_wraps_around() {
        let mut trap = FocusTrap::new(vec!["a", "b", "c"]);
        assert_eq!(trap.current(), Some(&"a"));
        assert_eq!(trap.next(), Some(&"b"));
        assert_eq!(trap.next(), Some(&"c"));
        assert_eq!(trap.next(), Some(&"a"));
    }

    #[test]
    fn prev_wraps_around() {
        let mut trap = FocusTrap::new(vec!["a", "b", "c"]);
        assert_eq!(trap.prev(), Some(&"c"));
        assert_eq!(trap.prev(), Some(&"b"));
        assert_eq!(trap.prev(), Some(&"a"));
        assert_eq!(trap.prev(), Some(&"c"));
    }

    #[test]
    fn focus_moves_to_present_key_only() {
        let mut trap = FocusTrap::new(vec![10, 20, 30]);
        assert!(trap.focus(&30));
        assert_eq!(trap.current(), Some(&30));
        assert!(!trap.focus(&99));
        assert_eq!(trap.current(), Some(&30));
    }
}
