//! Component lifecycle hooks: `on_mount`, `on_update`, `on_unmount` and
//! `on_cleanup`.
//!
//! The base component model in this crate is stateless: [`mount_component`]
//! turns props into an [`Element`] and forgets about them. Real components,
//! however, need to run side effects when they enter the tree, when they are
//! re-rendered, and — crucially — when they leave it, so that subscriptions,
//! timers and reactive effects are released rather than leaked.
//!
//! This module models that with an explicitly-driven set of callbacks,
//! mirroring Solid's `onMount` / `onCleanup` design:
//!
//! * [`on_mount`](LifecycleScope::on_mount) runs once when the component is
//!   mounted.
//! * [`on_update`](LifecycleScope::on_update) runs on every explicit update.
//! * [`on_cleanup`](LifecycleScope::on_cleanup) registers teardown work that
//!   runs — in reverse registration order — when the component unmounts. This
//!   is where reactive subscriptions should be released.
//! * [`on_unmount`](LifecycleScope::on_unmount) runs after the cleanups, once,
//!   when the component unmounts.
//!
//! A [`LifecycleScope`] is the registration handle a render closure captures;
//! [`mount`] drives the whole flow and returns a [`Mounted`] handle whose
//! [`Drop`] (or explicit [`Mounted::unmount`]) fires the teardown exactly once.
//!
//! # Example
//!
//! ```
//! use prism_ui::Element;
//! use prism_ui_component::lifecycle::mount;
//!
//! let mounted = mount(|scope| {
//!     scope.on_mount(|| { /* run entry side effects */ });
//!     scope.on_cleanup(|| { /* release subscriptions here */ });
//!     Element::text("view")
//! });
//!
//! assert_eq!(mounted.element().text_content(), Some("view"));
//! assert!(!mounted.is_unmounted());
//!
//! mounted.unmount();
//! assert!(mounted.is_unmounted());
//! ```

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::mem;

use prism_ui::Element;

/// The collected lifecycle callbacks for a single component instance.
///
/// Callbacks are stored as boxed `FnMut` closures. Mount callbacks run once and
/// are then dropped; update callbacks persist so they can run on every update;
/// cleanup callbacks run in reverse (last-in, first-out) order at unmount; and
/// unmount callbacks run after the cleanups.
#[derive(Default)]
struct LifecycleHooks {
    on_mount: Vec<Box<dyn FnMut()>>,
    on_update: Vec<Box<dyn FnMut()>>,
    on_unmount: Vec<Box<dyn FnMut()>>,
    cleanups: Vec<Box<dyn FnMut()>>,
    disposed: bool,
}

/// A shared, clonable handle used to register lifecycle callbacks.
///
/// A render closure receives a `&LifecycleScope` and calls
/// [`on_mount`](LifecycleScope::on_mount),
/// [`on_update`](LifecycleScope::on_update),
/// [`on_cleanup`](LifecycleScope::on_cleanup) and
/// [`on_unmount`](LifecycleScope::on_unmount) to attach side effects to the
/// component. Clones share the same underlying callback set, so a scope can be
/// handed to nested helpers freely.
#[derive(Clone, Default)]
pub struct LifecycleScope(Rc<RefCell<LifecycleHooks>>);

impl LifecycleScope {
    /// Creates an empty scope with no registered callbacks.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a callback to run once when the component is mounted.
    pub fn on_mount(&self, callback: impl FnMut() + 'static) {
        self.0.borrow_mut().on_mount.push(Box::new(callback));
    }

    /// Registers a callback to run on every call to
    /// [`run_update`](LifecycleScope::run_update).
    pub fn on_update(&self, callback: impl FnMut() + 'static) {
        self.0.borrow_mut().on_update.push(Box::new(callback));
    }

    /// Registers a callback to run once when the component is unmounted, after
    /// all cleanups have run.
    pub fn on_unmount(&self, callback: impl FnMut() + 'static) {
        self.0.borrow_mut().on_unmount.push(Box::new(callback));
    }

    /// Registers teardown work to run when the component unmounts.
    ///
    /// Cleanups run in reverse registration order, so a resource acquired later
    /// is released before one acquired earlier. This is the hook to use for
    /// releasing reactive subscriptions. The callback is `FnOnce` because a
    /// cleanup runs exactly once; it may therefore consume captured resources
    /// (for example by calling `dispose` on a reactive effect).
    pub fn on_cleanup(&self, callback: impl FnOnce() + 'static) {
        let mut callback = Some(callback);
        self.0.borrow_mut().cleanups.push(Box::new(move || {
            if let Some(callback) = callback.take() {
                callback();
            }
        }));
    }

    /// Runs every registered mount callback once, in registration order, then
    /// drops them.
    pub fn run_mount(&self) {
        let mut taken = {
            let mut hooks = self.0.borrow_mut();
            mem::take(&mut hooks.on_mount)
        };
        for callback in &mut taken {
            callback();
        }
    }

    /// Runs every registered update callback, in registration order.
    ///
    /// Update callbacks are retained, so repeated calls re-run them. Callbacks
    /// registered while updating are preserved for the next update.
    pub fn run_update(&self) {
        let mut taken = {
            let mut hooks = self.0.borrow_mut();
            mem::take(&mut hooks.on_update)
        };
        for callback in &mut taken {
            callback();
        }
        let mut hooks = self.0.borrow_mut();
        taken.append(&mut hooks.on_update);
        hooks.on_update = taken;
    }

    /// Runs the teardown sequence once: cleanups in reverse order, then unmount
    /// callbacks in registration order.
    ///
    /// Calling this more than once is a no-op after the first call, so it is
    /// safe to invoke explicitly and also rely on [`Mounted`]'s [`Drop`].
    pub fn run_unmount(&self) {
        let (mut cleanups, mut on_unmount) = {
            let mut hooks = self.0.borrow_mut();
            if hooks.disposed {
                return;
            }
            hooks.disposed = true;
            (
                mem::take(&mut hooks.cleanups),
                mem::take(&mut hooks.on_unmount),
            )
        };
        while let Some(mut cleanup) = cleanups.pop() {
            cleanup();
        }
        for callback in &mut on_unmount {
            callback();
        }
    }

    /// Returns `true` once the unmount sequence has run.
    #[must_use]
    pub fn is_disposed(&self) -> bool {
        self.0.borrow().disposed
    }
}

/// A mounted component: its rendered [`Element`] plus the lifecycle scope that
/// owns its teardown.
///
/// Build one with [`mount`]. The teardown sequence runs exactly once — either
/// via an explicit [`Mounted::unmount`] call or automatically when the handle
/// is dropped.
pub struct Mounted {
    element: Element,
    scope: LifecycleScope,
}

impl Mounted {
    /// Borrows the rendered element.
    #[must_use]
    pub fn element(&self) -> &Element {
        &self.element
    }

    /// Borrows the lifecycle scope, e.g. to register further callbacks.
    #[must_use]
    pub fn scope(&self) -> &LifecycleScope {
        &self.scope
    }

    /// Runs the update callbacks registered on this component.
    pub fn update(&self) {
        self.scope.run_update();
    }

    /// Runs the teardown sequence now (idempotent).
    pub fn unmount(&self) {
        self.scope.run_unmount();
    }

    /// Returns `true` once this component has been unmounted.
    #[must_use]
    pub fn is_unmounted(&self) -> bool {
        self.scope.is_disposed()
    }
}

impl Drop for Mounted {
    fn drop(&mut self) {
        self.scope.run_unmount();
    }
}

/// Mounts a component by running `render` against a fresh [`LifecycleScope`].
///
/// The render closure builds the [`Element`] and registers any lifecycle
/// callbacks it needs. After it returns, the mount callbacks fire immediately,
/// and the returned [`Mounted`] drives updates and the eventual teardown.
#[must_use]
pub fn mount<F>(render: F) -> Mounted
where
    F: FnOnce(&LifecycleScope) -> Element,
{
    let scope = LifecycleScope::new();
    let element = render(&scope);
    scope.run_mount();
    Mounted { element, scope }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std and share its Rc/RefCell"
    )]

    use super::*;
    use alloc::rc::Rc;
    use alloc::string::{String, ToString};
    use alloc::vec;
    use alloc::vec::Vec;
    use core::cell::RefCell;

    fn recorder() -> Rc<RefCell<Vec<String>>> {
        Rc::new(RefCell::new(Vec::new()))
    }

    fn record(log: &Rc<RefCell<Vec<String>>>, tag: &str) -> impl FnMut() + 'static {
        let log = Rc::clone(log);
        let tag = tag.to_string();
        move || log.borrow_mut().push(tag.clone())
    }

    #[test]
    fn full_order_is_mount_update_cleanup_unmount() {
        let log = recorder();
        let mounted = mount(|scope| {
            scope.on_mount(record(&log, "mount"));
            scope.on_update(record(&log, "update"));
            scope.on_cleanup(record(&log, "cleanup"));
            scope.on_unmount(record(&log, "unmount"));
            Element::text("view")
        });

        assert_eq!(mounted.element().text_content(), Some("view"));
        assert_eq!(*log.borrow(), vec!["mount".to_string()]);

        mounted.update();
        assert_eq!(
            *log.borrow(),
            vec!["mount".to_string(), "update".to_string()]
        );

        mounted.unmount();
        assert_eq!(
            *log.borrow(),
            vec![
                "mount".to_string(),
                "update".to_string(),
                "cleanup".to_string(),
                "unmount".to_string(),
            ]
        );
    }

    #[test]
    fn mount_runs_only_once_even_across_updates() {
        let log = recorder();
        let mounted = mount(|scope| {
            scope.on_mount(record(&log, "mount"));
            Element::text("x")
        });
        mounted.update();
        mounted.update();
        assert_eq!(*log.borrow(), vec!["mount".to_string()]);
    }

    #[test]
    fn update_runs_every_time() {
        let log = recorder();
        let mounted = mount(|scope| {
            scope.on_update(record(&log, "u"));
            Element::text("x")
        });
        mounted.update();
        mounted.update();
        mounted.update();
        assert_eq!(log.borrow().len(), 3);
    }

    #[test]
    fn cleanups_run_in_reverse_order() {
        let log = recorder();
        let mounted = mount(|scope| {
            scope.on_cleanup(record(&log, "first"));
            scope.on_cleanup(record(&log, "second"));
            scope.on_cleanup(record(&log, "third"));
            Element::text("x")
        });
        mounted.unmount();
        assert_eq!(
            *log.borrow(),
            vec![
                "third".to_string(),
                "second".to_string(),
                "first".to_string()
            ]
        );
    }

    #[test]
    fn drop_triggers_cleanup_once() {
        let log = recorder();
        {
            let _mounted = mount(|scope| {
                scope.on_cleanup(record(&log, "cleanup"));
                Element::text("x")
            });
            assert!(log.borrow().is_empty());
        }
        assert_eq!(*log.borrow(), vec!["cleanup".to_string()]);
    }

    #[test]
    fn explicit_unmount_then_drop_is_idempotent() {
        let log = recorder();
        let mounted = mount(|scope| {
            scope.on_cleanup(record(&log, "cleanup"));
            scope.on_unmount(record(&log, "unmount"));
            Element::text("x")
        });
        mounted.unmount();
        assert!(mounted.is_unmounted());
        drop(mounted);
        assert_eq!(
            *log.borrow(),
            vec!["cleanup".to_string(), "unmount".to_string()]
        );
    }

    #[test]
    fn cleanup_releases_a_reactive_subscription() {
        use prism_ui_reactive::Runtime;

        let rt = Runtime::new();
        let signal = rt.signal(0i32);
        let seen = recorder();

        let effect = rt.effect({
            let signal = signal.clone();
            let seen = Rc::clone(&seen);
            move || seen.borrow_mut().push(alloc::format!("{}", signal.get()))
        });

        let mounted = mount(move |scope| {
            // The effect is owned by the component; cleanup disposes it on
            // unmount so later signal writes no longer re-run it.
            scope.on_cleanup(move || effect.dispose());
            Element::text("live")
        });

        assert_eq!(*seen.borrow(), vec!["0".to_string()]);
        signal.set(1);
        assert_eq!(*seen.borrow(), vec!["0".to_string(), "1".to_string()]);

        mounted.unmount();
        signal.set(2);
        // The subscription was released, so no new value was recorded.
        assert_eq!(*seen.borrow(), vec!["0".to_string(), "1".to_string()]);
    }

    #[test]
    fn scope_can_be_driven_manually() {
        let log = recorder();
        let scope = LifecycleScope::new();
        scope.on_mount(record(&log, "mount"));
        scope.on_unmount(record(&log, "unmount"));
        scope.run_mount();
        scope.run_unmount();
        scope.run_unmount();
        assert_eq!(
            *log.borrow(),
            vec!["mount".to_string(), "unmount".to_string()]
        );
    }
}
