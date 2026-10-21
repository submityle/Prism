//! Optional schema-owned admission rules for complete controller candidates.

use alloc::{boxed::Box, string::String, sync::Arc};
use core::any::Any;

use prism_editor_reflect::FieldPath;
use prism_reflect::{Reflect, TypeData};

type ValidateObject = dyn Fn(&dyn Reflect, &[FieldPath]) -> Result<(), String> + Send + Sync;

/// Registry type data for business-neutral controller admission.
///
/// Attach this to a root's `TypeRegistration`. Registration and document decode
/// validate the complete root with an empty path slice. Edits validate the final
/// isolated candidate once, with all changed paths, before any model or history
/// is published. Thus coupled fields may be changed atomically without admitting
/// an invalid intermediate state. The callback must be pure and bounded; it may
/// inspect only affected fields when the unchanged root was already validated.
/// Registry clones retain the same immutable validator through codec roundtrips.
#[derive(Clone)]
pub struct ControllerObjectValidator {
    validate: Arc<ValidateObject>,
}

impl ControllerObjectValidator {
    /// Creates an explicit schema admission rule, independent of any renderer.
    pub fn new<F>(validate: F) -> Self
    where
        F: Fn(&dyn Reflect, &[FieldPath]) -> Result<(), String> + Send + Sync + 'static,
    {
        Self {
            validate: Arc::new(validate),
        }
    }

    /// Validates a complete root or the final candidate for the supplied paths.
    pub fn validate(&self, root: &dyn Reflect, paths: &[FieldPath]) -> Result<(), String> {
        (self.validate)(root, paths)
    }
}

impl TypeData for ControllerObjectValidator {
    fn clone_type_data(&self) -> Box<dyn TypeData> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
