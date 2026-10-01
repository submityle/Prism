//! The reactive [`Form`]: named fields, two-way bindings, and validation.

use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::{Cell, RefCell};

use prism_ui_reactive::{Memo, Runtime, Signal};

use crate::error::ValidationError;
use crate::field::{parse_i64, Field, FieldId};
use crate::validator::BoxedValidator;

/// A reactive form: a set of named fields, each backed by a reactive text
/// signal and an ordered list of validators.
///
/// A form is a cheap, clonable handle; clones share the same fields and
/// reactive graph. Reads reflect the latest value and writes revalidate
/// through the reactive runtime, so an [`errors_memo`](Form::errors_memo) stays
/// in sync automatically.
#[derive(Clone)]
pub struct Form {
    runtime: Runtime,
    fields: Rc<RefCell<BTreeMap<FieldId, Field>>>,
    revision: Signal<u64>,
    errors_memo: Rc<RefCell<Option<Memo<Vec<ValidationError>>>>>,
}

impl Form {
    /// Create an empty form bound to `runtime`.
    pub fn new(runtime: Runtime) -> Self {
        let revision = runtime.signal(0u64);
        Self {
            runtime,
            fields: Rc::new(RefCell::new(BTreeMap::new())),
            revision,
            errors_memo: Rc::new(RefCell::new(None)),
        }
    }

    /// The reactive runtime backing this form.
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// Register a field with an initial value and its validators.
    ///
    /// Returns the resolved field key for convenient reuse. Re-registering an
    /// existing key replaces it.
    pub fn register(
        &self,
        id: impl Into<FieldId>,
        initial: impl Into<String>,
        validators: Vec<BoxedValidator>,
    ) -> FieldId {
        let id = id.into();
        let field = Field {
            text: self.runtime.signal(initial.into()),
            touched: Cell::new(false),
            dirty: Cell::new(false),
            validators,
        };
        self.fields.borrow_mut().insert(id.clone(), field);
        // Bump the revision so a cached errors memo re-subscribes to the new
        // field on its next recomputation.
        self.revision.update(|revision| *revision += 1);
        id
    }

    /// Read a field's current text, or the empty string if it is not
    /// registered. Recording a dependency when called inside a reactive scope.
    pub fn value(&self, id: impl Into<FieldId>) -> String {
        let id = id.into();
        let signal = self
            .fields
            .borrow()
            .get(&id)
            .map(|field| field.text.clone());
        match signal {
            Some(signal) => signal.get(),
            None => String::new(),
        }
    }

    /// Read a field's text parsed as a signed 64-bit integer.
    pub fn value_as_i64(&self, id: impl Into<FieldId>) -> Option<i64> {
        let id = id.into();
        let signal = self
            .fields
            .borrow()
            .get(&id)
            .map(|field| field.text.clone())?;
        parse_i64(&signal.get()).ok()
    }

    /// The reactive text signal backing `id`, for direct two-way binding.
    ///
    /// Writing through the returned signal updates the same value the form
    /// reads back, but does not mark the field dirty; use [`set`](Form::set)
    /// for the form-tracked mutation.
    pub fn binding(&self, id: impl Into<FieldId>) -> Option<Signal<String>> {
        let id = id.into();
        self.fields
            .borrow()
            .get(&id)
            .map(|field| field.text.clone())
    }

    /// Set a field's value, marking it dirty and triggering revalidation.
    pub fn set(&self, id: impl Into<FieldId>, value: impl Into<String>) {
        let id = id.into();
        let value = value.into();
        let signal = {
            let fields = self.fields.borrow();
            match fields.get(&id) {
                Some(field) => {
                    field.dirty.set(true);
                    Some(field.text.clone())
                }
                None => None,
            }
        };
        if let Some(signal) = signal {
            signal.set_if_changed(value);
        }
    }

    /// Mark a field as touched (for example, after it loses focus).
    pub fn touch(&self, id: impl Into<FieldId>) {
        let id = id.into();
        if let Some(field) = self.fields.borrow().get(&id) {
            field.touched.set(true);
        }
    }

    /// Whether a field has been touched.
    pub fn is_touched(&self, id: impl Into<FieldId>) -> bool {
        let id = id.into();
        self.fields
            .borrow()
            .get(&id)
            .is_some_and(|field| field.touched.get())
    }

    /// Whether a field's value has been changed through the form.
    pub fn is_dirty(&self, id: impl Into<FieldId>) -> bool {
        let id = id.into();
        self.fields
            .borrow()
            .get(&id)
            .is_some_and(|field| field.dirty.get())
    }

    /// Run a single field's validators, tagging each error with the field key.
    ///
    /// When `stop_at_first` is set, validation halts at the first failure.
    fn run_field(&self, id: &FieldId, field: &Field, stop_at_first: bool) -> Vec<ValidationError> {
        let text = field.text.get();
        let mut errors = Vec::new();
        for validator in &field.validators {
            if let Err(mut error) = validator.validate(&text) {
                error.field = id.clone();
                errors.push(error);
                if stop_at_first {
                    break;
                }
            }
        }
        errors
    }

    /// Every validation error for a single field, in validator order.
    pub fn all_errors(&self, id: impl Into<FieldId>) -> Vec<ValidationError> {
        let id = id.into();
        let fields = self.fields.borrow();
        match fields.get(&id) {
            Some(field) => self.run_field(&id, field, false),
            None => Vec::new(),
        }
    }

    /// The first validation error for a single field, if any.
    pub fn first_error(&self, id: impl Into<FieldId>) -> Option<ValidationError> {
        let id = id.into();
        let fields = self.fields.borrow();
        let field = fields.get(&id)?;
        self.run_field(&id, field, true).into_iter().next()
    }

    /// Every validation error for a single field (alias of
    /// [`all_errors`](Form::all_errors)).
    pub fn errors_for(&self, id: impl Into<FieldId>) -> Vec<ValidationError> {
        self.all_errors(id)
    }

    /// Validate every registered field, collecting all errors.
    pub fn validate(&self) -> Vec<ValidationError> {
        let fields = self.fields.borrow();
        let mut errors = Vec::new();
        for (id, field) in fields.iter() {
            errors.append(&mut self.run_field(id, field, false));
        }
        errors
    }

    /// Whether the whole form currently validates, without recording a
    /// dependency on any field.
    pub fn is_valid(&self) -> bool {
        self.runtime.untrack(|| self.validate().is_empty())
    }

    /// A reactive memo of every field's errors that recomputes whenever any
    /// field value (or the set of fields) changes.
    ///
    /// The memo is created on first use and cached, so repeated calls return
    /// the same reactive node.
    pub fn errors_memo(&self) -> Memo<Vec<ValidationError>> {
        if let Some(memo) = self.errors_memo.borrow().as_ref() {
            return memo.clone();
        }
        let fields = self.fields.clone();
        let revision = self.revision.clone();
        let memo = self.runtime.memo(move || {
            // Depend on the field set so new registrations re-subscribe.
            let _ = revision.get();
            let mut errors = Vec::new();
            for (id, field) in fields.borrow().iter() {
                let text = field.text.get();
                for validator in &field.validators {
                    if let Err(mut error) = validator.validate(&text) {
                        error.field = id.clone();
                        errors.push(error);
                    }
                }
            }
            errors
        });
        *self.errors_memo.borrow_mut() = Some(memo.clone());
        memo
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validator::{int_range, min_len, required};
    use alloc::vec;

    fn form() -> Form {
        Form::new(Runtime::new())
    }

    #[test]
    fn binding_round_trip() {
        let form = form();
        form.register("name", "ada", vec![]);
        assert_eq!(form.value("name"), "ada");
        form.set("name", "grace");
        assert_eq!(form.value("name"), "grace");

        let binding = form.binding("name").expect("field exists");
        binding.set(String::from("hopper"));
        assert_eq!(form.value("name"), "hopper");
    }

    #[test]
    fn touched_and_dirty_transitions() {
        let form = form();
        form.register("email", "", vec![]);
        assert!(!form.is_touched("email"));
        assert!(!form.is_dirty("email"));

        form.touch("email");
        assert!(form.is_touched("email"));
        assert!(!form.is_dirty("email"));

        form.set("email", "a@b.c");
        assert!(form.is_dirty("email"));
    }

    #[test]
    fn multi_validator_first_versus_all() {
        let form = form();
        form.register("name", "", vec![required(), min_len(3)]);

        let all = form.all_errors("name");
        assert_eq!(all.len(), 2);
        let first = form.first_error("name").expect("has error");
        assert_eq!(first.message, "This field is required.");
        assert_eq!(first.field.as_str(), "name");
    }

    #[test]
    fn aggregated_validate() {
        let form = form();
        form.register("name", "", vec![required()]);
        form.register("age", "x", vec![int_range(0, 120)]);
        let errors = form.validate();
        assert_eq!(errors.len(), 2);
        assert!(!form.is_valid());

        form.set("name", "ada");
        form.set("age", "40");
        assert!(form.is_valid());
        assert_eq!(form.value_as_i64("age"), Some(40));
    }
}
