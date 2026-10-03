//! Scope / span `RAII` timing.
//!
//! A [`Scope`] captures a monotonic start timestamp and the current nesting
//! depth when constructed, and on drop records a completed [`SpanRecord`] with
//! the measured duration into the calling thread's ring buffer. Use the
//! [`span!`](crate::span!) macro for the common "time this block" case, or
//! construct a [`Scope`] directly for builder-style categories and args.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use prism_platform::{now, MonotonicNanos};

use crate::trace::ring::{self, SpanRecord};

/// An `RAII` timing guard. Construct one at the top of a scope with
/// [`Scope::new`] (or the [`span!`](crate::span!) macro); when it drops it
/// records a completed [`SpanRecord`] into the current thread's ring buffer.
#[derive(Debug)]
pub struct Scope {
    name: String,
    category: Option<String>,
    args: Vec<(String, String)>,
    start: MonotonicNanos,
    depth: u32,
    thread_id: u64,
}

impl Scope {
    /// Open a new timing scope named `name`, capturing the start timestamp and
    /// the current nesting depth on this thread.
    pub fn new(name: impl Into<String>) -> Self {
        let (depth, thread_id) = ring::enter();
        Self {
            name: name.into(),
            category: None,
            args: Vec::new(),
            // Read the clock last so setup work is excluded from the measured
            // duration.
            start: now(),
            depth,
            thread_id,
        }
    }

    /// Attach a category/track label (builder style).
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Attach a structured key/value arg (builder style).
    pub fn with_arg(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.args.push((key.into(), value.into()));
        self
    }

    /// This scope's nesting depth at entry (0 is top level).
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// The Prism-assigned thread id that owns this scope.
    pub fn thread_id(&self) -> u64 {
        self.thread_id
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        let duration_nanos = now().saturating_since(self.start);
        let record = SpanRecord {
            name: core::mem::take(&mut self.name),
            category: self.category.take(),
            thread_id: self.thread_id,
            start_nanos: self.start.0,
            duration_nanos,
            depth: self.depth,
            args: core::mem::take(&mut self.args),
        };
        ring::leave(record);
    }
}

/// Open a timing [`Scope`] bound to a hidden local guard that closes at the end
/// of the enclosing block.
///
/// Accepts an optional category: `span!(name)` or `span!(name, category)`.
///
/// # Examples
///
/// ```
/// use prism_diagnostic::span;
///
/// span!("outer");
/// {
///     span!("inner", "physics");
///     // ... timed code ...
/// }
/// ```
#[macro_export]
macro_rules! span {
    ($name:expr) => {
        let _prism_span = $crate::span::Scope::new($name);
    };
    ($name:expr, $category:expr) => {
        let _prism_span = $crate::span::Scope::new($name).with_category($category);
    };
}

#[cfg(test)]
mod tests {
    use super::Scope;
    use crate::trace::ring;

    #[test]
    fn nested_spans_record_depth_and_duration() {
        ring::clear_current_thread();
        {
            let outer = Scope::new("outer");
            assert_eq!(outer.depth(), 0);
            {
                let inner = Scope::new("inner");
                assert_eq!(inner.depth(), 1);
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }
        let spans = ring::current_thread_spans();
        assert_eq!(spans.len(), 2);
        // The inner scope closes first, so it is recorded first.
        assert_eq!(spans[0].name, "inner");
        assert_eq!(spans[0].depth, 1);
        assert_eq!(spans[1].name, "outer");
        assert_eq!(spans[1].depth, 0);
        assert!(spans[0].duration_nanos > 0);
        assert!(spans[1].duration_nanos >= spans[0].duration_nanos);
    }

    #[test]
    fn span_macro_times_block() {
        ring::clear_current_thread();
        {
            span!("macro_scope", "cat_a");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let spans = ring::current_thread_spans();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].name, "macro_scope");
        assert_eq!(spans[0].category.as_deref(), Some("cat_a"));
        assert!(spans[0].duration_nanos > 0);
    }
}
