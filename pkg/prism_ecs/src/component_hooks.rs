//! Component lifecycle **hooks** (design §12): low-level, synchronous callbacks
//! fired by the [`World`] as components are structurally added to, overwritten
//! on, or removed from an entity.
//!
//! Hooks are the foundation of the higher-level reactive layer (observers,
//! relation cleanup, derived-index maintenance, GPU-resource registration). A
//! hook is a plain `fn` pointer registered against a [`ComponentId`]; it runs
//! immediately at the structural-change site with a **fully consistent**
//! [`World`] view.
//!
//! # Trigger model
//!
//! For a single structural operation the engine fires, per affected component:
//!
//! | Transition | Hooks (in order) |
//! |---|---|
//! | component newly added to the entity | [`on_add`](ComponentHooks::on_add) then [`on_insert`](ComponentHooks::on_insert) |
//! | existing component value overwritten | [`on_replace`](ComponentHooks::on_replace) then [`on_insert`](ComponentHooks::on_insert) |
//! | component removed (incl. despawn)    | [`on_replace`](ComponentHooks::on_replace) then [`on_remove`](ComponentHooks::on_remove) |
//!
//! `on_add` / `on_insert` run **after** the new value is in place; `on_replace`
//! / `on_remove` run **before** the old value is overwritten or dropped, so a
//! hook can still read the outgoing value.
//!
//! # Re-entrancy
//!
//! A hook receives `&mut World` and may read or mutate component values and
//! issue structural changes to **other** entities. Restructuring the *same*
//! entity's hooked components from within its own hook is memory-safe (every
//! [`World`] method re-validates state) but is **not recommended** in this
//! revision: the surrounding batch's add/insert/remove ordering is defined
//! against the pre-hook snapshot. Fully deferred, command-buffered re-entrancy
//! is handled by the observer layer (design §12) and is tracked as follow-up.
//!
//! [`World`]: crate::world::World

use crate::component::ComponentId;
use crate::entity::Entity;
use crate::world::World;

/// The context handed to a [`ComponentHook`] when it fires.
///
/// It names the entity and component that triggered the hook and grants
/// mutable access to the owning [`World`] so the hook can inspect the value
/// (e.g. `ctx.world.get::<T>(ctx.entity)`), update derived state, or perform
/// structural changes to other entities.
pub struct HookContext<'w> {
    /// The world in which the triggering structural change occurred. Fully
    /// consistent at the moment the hook runs.
    pub world: &'w mut World,
    /// The entity whose component set changed.
    pub entity: Entity,
    /// The component that triggered this hook.
    pub component: ComponentId,
}

/// A component lifecycle hook: a stateless `fn` pointer invoked with a
/// [`HookContext`] at a structural-change site.
///
/// `fn` pointers (rather than boxed closures) keep [`ComponentHooks`] `Copy`
/// and allocation-free, matching the `no_std` core and letting the registry
/// store them inline in [`ComponentInfo`](crate::component::ComponentInfo).
pub type ComponentHook = for<'w> fn(HookContext<'w>);

/// The set of lifecycle hooks registered for one component type.
///
/// Every field is optional; an unset hook costs nothing at the trigger site.
/// Construct via [`ComponentHooks::new`] and the `with_*` builders, then attach
/// to a component with
/// [`World::register_component_hooks`](crate::world::World::register_component_hooks)
/// or [`Components::set_hooks`](crate::component::Components::set_hooks).
#[derive(Clone, Copy, Default)]
pub struct ComponentHooks {
    on_add: Option<ComponentHook>,
    on_insert: Option<ComponentHook>,
    on_replace: Option<ComponentHook>,
    on_remove: Option<ComponentHook>,
}

impl ComponentHooks {
    /// An empty hook set (no callbacks registered).
    #[inline]
    pub const fn new() -> Self {
        Self {
            on_add: None,
            on_insert: None,
            on_replace: None,
            on_remove: None,
        }
    }

    /// Set the hook fired when the component is **newly added** to an entity
    /// (the entity did not previously have it). Runs after the value is stored.
    #[inline]
    pub fn with_on_add(mut self, hook: ComponentHook) -> Self {
        self.on_add = Some(hook);
        self
    }

    /// Set the hook fired on **every** write of the component value, whether
    /// newly added or overwritten. Runs after the value is stored, and after
    /// [`on_add`](Self::on_add) / [`on_replace`](Self::on_replace).
    #[inline]
    pub fn with_on_insert(mut self, hook: ComponentHook) -> Self {
        self.on_insert = Some(hook);
        self
    }

    /// Set the hook fired just **before** an existing value is overwritten or
    /// removed. The outgoing value is still readable when it runs.
    #[inline]
    pub fn with_on_replace(mut self, hook: ComponentHook) -> Self {
        self.on_replace = Some(hook);
        self
    }

    /// Set the hook fired just **before** the component is removed from the
    /// entity (including on despawn). The outgoing value is still readable.
    #[inline]
    pub fn with_on_remove(mut self, hook: ComponentHook) -> Self {
        self.on_remove = Some(hook);
        self
    }

    /// The registered `on_add` hook, if any.
    #[inline]
    pub fn on_add(&self) -> Option<ComponentHook> {
        self.on_add
    }

    /// The registered `on_insert` hook, if any.
    #[inline]
    pub fn on_insert(&self) -> Option<ComponentHook> {
        self.on_insert
    }

    /// The registered `on_replace` hook, if any.
    #[inline]
    pub fn on_replace(&self) -> Option<ComponentHook> {
        self.on_replace
    }

    /// The registered `on_remove` hook, if any.
    #[inline]
    pub fn on_remove(&self) -> Option<ComponentHook> {
        self.on_remove
    }

    /// Whether no hooks are registered (the common case; lets the engine skip
    /// all hook bookkeeping for a component).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.on_add.is_none()
            && self.on_insert.is_none()
            && self.on_replace.is_none()
            && self.on_remove.is_none()
    }
}
