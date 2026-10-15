//! Typed GPU-resource replay used by device-loss recovery.
//!
//! A replay entry owns the recipe for one self-contained resource bundle and a
//! typed handle to the bundle's current generation. Recovery first builds every
//! registered bundle against the replacement device/queue. Only after every
//! recipe succeeds are the handles committed, so a failed replay cannot leave
//! half of the renderer pointing at the replacement device and half at the
//! lost one.

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::sync::{Arc, Weak};
use alloc::vec::Vec;
use core::fmt;
use std::sync::RwLock;

use prism_render_driver::Instance;

/// A resource recipe could not rebuild its GPU state.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ReplayError {
    detail: String,
}

impl ReplayError {
    /// Creates an error carrying a backend- or asset-specific diagnostic.
    #[must_use]
    pub fn new(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
        }
    }

    /// The diagnostic supplied by the replay recipe.
    #[must_use]
    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.detail)
    }
}

impl std::error::Error for ReplayError {}

/// Identifies the replay entry that failed and preserves its diagnostic.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ReplayFailure {
    label: String,
    source: ReplayError,
}

impl ReplayFailure {
    /// The stable label supplied when the replay entry was registered.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The recipe's failure diagnostic.
    #[must_use]
    pub fn source_error(&self) -> &ReplayError {
        &self.source
    }
}

impl fmt::Display for ReplayFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "resource replay '{}' failed: {}",
            self.label, self.source
        )
    }
}

impl std::error::Error for ReplayFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// A typed, generation-aware view of a replayed GPU resource bundle.
///
/// Call [`current`](Self::current) when recording a frame instead of caching
/// the returned `Arc` indefinitely. A successful device recovery atomically
/// replaces the value held by this handle.
pub struct ReplayHandle<T> {
    current: Arc<RwLock<Arc<T>>>,
}

impl<T> Clone for ReplayHandle<T> {
    fn clone(&self) -> Self {
        Self {
            current: Arc::clone(&self.current),
        }
    }
}

impl<T> ReplayHandle<T> {
    /// Returns the resource bundle for the runtime's current device generation.
    #[must_use]
    pub fn current(&self) -> Arc<T> {
        Arc::clone(
            &self
                .current
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }
}

type Commit = Box<dyn FnOnce() + Send + 'static>;

trait ErasedReplay<I: Instance>: Send + Sync {
    fn is_live(&self) -> bool;

    fn stage(&self, device: &I::Device, queue: &I::Queue) -> Result<Option<Commit>, ReplayFailure>;
}

type ReplayRecipe<I, T> = dyn Fn(&<I as Instance>::Device, &<I as Instance>::Queue) -> Result<T, ReplayError>
    + Send
    + Sync
    + 'static;

struct ReplayEntry<I: Instance, T> {
    label: String,
    recipe: Box<ReplayRecipe<I, T>>,
    current: Weak<RwLock<Arc<T>>>,
}

impl<I, T> ErasedReplay<I> for ReplayEntry<I, T>
where
    I: Instance,
    T: Send + Sync + 'static,
{
    fn is_live(&self) -> bool {
        self.current.strong_count() != 0
    }

    fn stage(&self, device: &I::Device, queue: &I::Queue) -> Result<Option<Commit>, ReplayFailure> {
        let Some(current) = self.current.upgrade() else {
            return Ok(None);
        };
        let replacement = (self.recipe)(device, queue).map_err(|source| ReplayFailure {
            label: self.label.clone(),
            source,
        })?;
        Ok(Some(Box::new(move || {
            *current
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Arc::new(replacement);
        })))
    }
}

pub(crate) struct StagedReplay {
    commits: Vec<Commit>,
}

impl StagedReplay {
    pub(crate) fn count(&self) -> usize {
        self.commits.len()
    }

    pub(crate) fn commit(self) {
        for commit in self.commits {
            commit();
        }
    }
}

pub(crate) struct ReplayRegistry<I: Instance> {
    entries: Vec<Box<dyn ErasedReplay<I>>>,
}

impl<I: Instance + 'static> ReplayRegistry<I> {
    pub(crate) fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub(crate) fn register<T, F>(
        &mut self,
        label: impl Into<String>,
        device: &I::Device,
        queue: &I::Queue,
        recipe: F,
    ) -> Result<ReplayHandle<T>, ReplayFailure>
    where
        T: Send + Sync + 'static,
        F: Fn(&I::Device, &I::Queue) -> Result<T, ReplayError> + Send + Sync + 'static,
    {
        let label = label.into();
        let initial = recipe(device, queue).map_err(|source| ReplayFailure {
            label: label.clone(),
            source,
        })?;
        let current = Arc::new(RwLock::new(Arc::new(initial)));
        self.entries.push(Box::new(ReplayEntry {
            label,
            recipe: Box::new(recipe),
            current: Arc::downgrade(&current),
        }));
        Ok(ReplayHandle { current })
    }

    pub(crate) fn stage(
        &mut self,
        device: &I::Device,
        queue: &I::Queue,
    ) -> Result<StagedReplay, ReplayFailure> {
        self.entries.retain(|entry| entry.is_live());
        let mut commits = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            if let Some(commit) = entry.stage(device, queue)? {
                commits.push(commit);
            }
        }
        Ok(StagedReplay { commits })
    }
}

impl<I: Instance + 'static> Default for ReplayRegistry<I> {
    fn default() -> Self {
        Self::new()
    }
}

impl From<&str> for ReplayError {
    fn from(value: &str) -> Self {
        Self::new(value.to_string())
    }
}
