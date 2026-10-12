//! Draw functions: the registry of recording closures a phase item invokes.
//!
//! Bevy's render phases reference a `DrawFunctionId` into a per-phase
//! `DrawFunctions` registry; the draw command then records real GPU work. Prism
//! keeps the same shape but stays decoupled from the concrete recorder: a
//! [`DrawFunctions`] is generic over the recording context `C`, so a test can
//! pass a plain buffer and the production path passes the frame graph's pass
//! recorder. That keeps the queue layer free of a dependency on the graph's
//! internals while still recording real commands.

use crate::queue::PhaseItem;

/// A stable handle to a registered draw function.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct DrawFunctionId(pub u32);

/// A registry of draw functions keyed by [`DrawFunctionId`].
///
/// `C` is the recording context a draw function records into (the frame graph's
/// pass recorder in production, any scratch type in tests).
pub struct DrawFunctions<C> {
    functions: Vec<Box<dyn Fn(&PhaseItem, &mut C)>>,
}

impl<C> Default for DrawFunctions<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C> DrawFunctions<C> {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
        }
    }

    /// Registers `function` and returns its id.
    pub fn register(&mut self, function: impl Fn(&PhaseItem, &mut C) + 'static) -> DrawFunctionId {
        let id = DrawFunctionId(self.functions.len() as u32);
        self.functions.push(Box::new(function));
        id
    }

    /// The number of registered draw functions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.functions.len()
    }

    /// Whether no draw functions are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.functions.is_empty()
    }

    /// Records `item` by invoking the draw function `id`.
    ///
    /// A no-op for an unknown id, so a queue entry can never panic the frame.
    pub fn call(&self, id: DrawFunctionId, item: &PhaseItem, context: &mut C) {
        if let Some(function) = self.functions.get(id.0 as usize) {
            function(item, context);
        }
    }
}
