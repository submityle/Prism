//! Bind the reactive signal graph to the [`Ui`] view runtime.
//!
//! [`ReactiveView`] closes Loom's most important loop: it turns a plain
//! `Fn() -> Element` *view function* into a live subscription. The view is run
//! once to mount the initial tree, and then **re-run automatically** whenever
//! any [`Signal`](crate::reactive::Signal) or [`Memo`](crate::reactive::Memo)
//! it read changes — reconciling the new [`Element`] against the retained tree
//! so the backend sees only real deltas.
//!
//! This is the same contract as `SolidJS`'s `createEffect` + rendering: the
//! dependency set is discovered by *running* the view, so there is no manual
//! wiring of "which state affects which node".
//!
//! # Ownership model
//!
//! The [`Runtime`] that owns the signals is passed in explicitly and is kept
//! *separate* from the [`Ui`]'s own internal runtime. The view effect is
//! created on that external runtime, while the [`Ui`] is wrapped in an
//! `Rc<RefCell<_>>` so the effect can borrow it mutably each time it fires
//! without conflicting with the runtime's own borrows.
//!
//! # Example
//!
//! ```
//! use prism_ui::{Element, RecordingBackend, Ui};
//! use prism_ui::reactive::Runtime;
//! use prism_ui::reactive_view::ReactiveView;
//!
//! let rt = Runtime::new();
//! let label = rt.signal(String::from("hello"));
//!
//! // The view reads `label`, so it re-runs whenever `label` changes.
//! let view = {
//!     let label = label.clone();
//!     move || Element::box_().child(Element::text(label.get()))
//! };
//!
//! let app = ReactiveView::new(&rt, Ui::new(RecordingBackend::new()), view);
//!
//! // Mounted once: a box and its text child.
//! assert_eq!(app.with_ui(|ui| ui.node_count()), 2);
//!
//! // Changing the signal re-runs the view and emits a `SetText` op.
//! let before = app.with_ui(|ui| ui.backend().len());
//! label.set(String::from("world"));
//! assert!(app.with_ui(|ui| ui.backend().len()) > before);
//! ```

use alloc::rc::Rc;
use core::cell::RefCell;

use prism_ui_reactive::{Effect, Runtime};

use crate::backend::Backend;
use crate::element::Element;
use crate::ui::Ui;

/// A live binding between a view function and a [`Ui`] runtime.
///
/// Construct it with [`ReactiveView::new`]; the view mounts immediately and
/// then re-renders on every reactive dependency change. The handle keeps the
/// underlying [`Effect`] alive, so dropping the [`ReactiveView`] tears the
/// subscription down.
pub struct ReactiveView<B: Backend + 'static> {
    ui: Rc<RefCell<Ui<B>>>,
    // The effect is retained to keep the subscription alive. It is wrapped in an
    // `Option` so [`Drop`] (and [`ReactiveView::dispose`]) can move it out and
    // call [`Effect::dispose`], detaching the binding from the reactive graph.
    effect: Option<Effect>,
}

impl<B: Backend + 'static> ReactiveView<B> {
    /// Mount `ui` with the result of `view`, then keep it in sync.
    ///
    /// `view` is run once immediately (mounting the initial tree) and re-run
    /// whenever any signal or memo it read on `runtime` changes. Each re-run
    /// reconciles the freshly built [`Element`] against the retained tree.
    pub fn new(runtime: &Runtime, ui: Ui<B>, view: impl Fn() -> Element + 'static) -> Self {
        let ui = Rc::new(RefCell::new(ui));
        let first_run = Rc::new(RefCell::new(true));

        let effect = {
            let ui = Rc::clone(&ui);
            let first_run = Rc::clone(&first_run);
            runtime.effect(move || {
                // Build the view *inside* the effect so signal reads are
                // tracked as dependencies of this effect.
                let element = view();
                let mut ui = ui.borrow_mut();
                let mut flag = first_run.borrow_mut();
                if *flag {
                    ui.mount(&element);
                    *flag = false;
                } else {
                    ui.update(&element);
                }
            })
        };

        Self {
            ui,
            effect: Some(effect),
        }
    }

    /// Tear down the reactive binding, stopping all further re-renders.
    ///
    /// This is also performed automatically when the [`ReactiveView`] is
    /// dropped; call it explicitly when you want to end updates while keeping a
    /// separately held [`ui_cell`](Self::ui_cell) alive.
    pub fn dispose(mut self) {
        if let Some(effect) = self.effect.take() {
            effect.dispose();
        }
    }

    /// Borrow the underlying [`Ui`] immutably for inspection.
    pub fn with_ui<R>(&self, f: impl FnOnce(&Ui<B>) -> R) -> R {
        f(&self.ui.borrow())
    }

    /// Borrow the underlying [`Ui`] mutably (e.g. to drive layout).
    pub fn with_ui_mut<R>(&self, f: impl FnOnce(&mut Ui<B>) -> R) -> R {
        f(&mut self.ui.borrow_mut())
    }

    /// Share the underlying [`Ui`] cell (advanced use / interop).
    #[must_use]
    pub fn ui_cell(&self) -> Rc<RefCell<Ui<B>>> {
        Rc::clone(&self.ui)
    }
}

impl<B: Backend + 'static> Drop for ReactiveView<B> {
    fn drop(&mut self) {
        if let Some(effect) = self.effect.take() {
            effect.dispose();
        }
    }
}
