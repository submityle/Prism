//! Automatic instrumentation for ECS systems (and any per-frame work unit).
//!
//! A [`SystemScope`] is a thin wrapper over a timing [`Scope`] that tags the
//! span with the system track category so every instrumented system shows up on
//! a dedicated row in the flame graph and feeds the
//! [`LoadProfile`](crate::metrics::LoadProfile) aggregation. Open one at the top
//! of a system's `run` and let it close at the end of the call:
//!
//! ```
//! use prism_diagnostic::instrument::instrument_system;
//!
//! fn run_physics() {
//!     let _scope = instrument_system("physics");
//!     // ... system body ...
//! }
//! ```

extern crate alloc;

use alloc::string::String;

use crate::span::Scope;

/// Trace category/track that instrumented systems record under.
pub const SYSTEM_CATEGORY: &str = "system";

/// An `RAII` guard that times one system invocation.
///
/// Construct it at the start of a system's work; on drop it records a completed
/// span (via the inner [`Scope`]) into the calling thread's ring buffer under
/// the [`SYSTEM_CATEGORY`] track (or a caller-chosen category).
#[derive(Debug)]
pub struct SystemScope {
    scope: Scope,
}

impl SystemScope {
    /// Open a scope for the system `name` under the default [`SYSTEM_CATEGORY`].
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            scope: Scope::new(name).with_category(SYSTEM_CATEGORY),
        }
    }

    /// Open a scope for the system `name` under an explicit `category` track.
    pub fn in_category(name: impl Into<String>, category: impl Into<String>) -> Self {
        Self {
            scope: Scope::new(name).with_category(category),
        }
    }

    /// The nesting depth of this scope at entry (0 is top level).
    pub fn depth(&self) -> u32 {
        self.scope.depth()
    }

    /// The Prism-assigned thread id that owns this scope.
    pub fn thread_id(&self) -> u64 {
        self.scope.thread_id()
    }
}

/// Open a [`SystemScope`] for `name` under the default [`SYSTEM_CATEGORY`].
///
/// Free-function form of [`SystemScope::new`] for call sites that read more
/// naturally as a verb.
pub fn instrument_system(name: impl Into<String>) -> SystemScope {
    SystemScope::new(name)
}

/// Open a [`SystemScope`] bound to a hidden local guard that closes at the end
/// of the enclosing block.
///
/// Accepts an optional category: `instrument_system!(name)` or
/// `instrument_system!(name, category)`.
///
/// # Examples
///
/// ```
/// use prism_diagnostic::instrument_system;
///
/// {
///     instrument_system!("ai_tick", "gameplay");
///     // ... system body ...
/// }
/// ```
#[macro_export]
macro_rules! instrument_system {
    ($name:expr) => {
        let _prism_system = $crate::instrument::system::SystemScope::new($name);
    };
    ($name:expr, $category:expr) => {
        let _prism_system = $crate::instrument::system::SystemScope::in_category($name, $category);
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::ring;

    #[test]
    fn system_scope_records_span_under_system_category() {
        ring::clear_current_thread();
        {
            let s = SystemScope::new("render");
            assert_eq!(s.depth(), 0);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let spans = ring::current_thread_spans();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].name, "render");
        assert_eq!(spans[0].category.as_deref(), Some(SYSTEM_CATEGORY));
        assert!(spans[0].duration_nanos > 0);
    }

    #[test]
    fn nested_systems_capture_increasing_depth() {
        ring::clear_current_thread();
        {
            let _outer = SystemScope::new("outer_sys");
            {
                let inner = SystemScope::in_category("inner_sys", "gameplay");
                assert_eq!(inner.depth(), 1);
            }
        }
        let spans = ring::current_thread_spans();
        assert_eq!(spans.len(), 2);
        // Inner closes first.
        assert_eq!(spans[0].name, "inner_sys");
        assert_eq!(spans[0].category.as_deref(), Some("gameplay"));
        assert_eq!(spans[0].depth, 1);
        assert_eq!(spans[1].name, "outer_sys");
        assert_eq!(spans[1].depth, 0);
    }

    #[test]
    fn macro_times_block() {
        ring::clear_current_thread();
        {
            instrument_system!("macro_sys");
        }
        {
            instrument_system!("macro_sys_cat", "io");
        }
        let spans = ring::current_thread_spans();
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].name, "macro_sys");
        assert_eq!(spans[0].category.as_deref(), Some(SYSTEM_CATEGORY));
        assert_eq!(spans[1].category.as_deref(), Some("io"));
    }
}
