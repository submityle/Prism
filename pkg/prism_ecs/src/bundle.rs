//! Bundles: static groups of components inserted together.
//!
//! A [`Bundle`] is a set of component values that can be spawned onto an entity
//! in one call, e.g. `world.spawn((Position(..), Velocity(..)))`. Every
//! [`Component`] is a one-element bundle, and tuples of bundles are themselves
//! bundles, so bundles compose and nest.
//!
//! The trait exposes two cooperating halves:
//! - [`Bundle::component_ids`] registers every component type and records its
//!   [`ComponentId`] in a fixed order.
//! - [`Bundle::get_components`] moves each component value out of the bundle,
//!   yielding a pointer per value in that **same order**.
//!
//! The world zips these two orderings to route each value into the correct
//! column, so neither half needs to know the archetype's sorted layout.

use alloc::vec::Vec;
use core::mem::ManuallyDrop;

use crate::component::{Component, ComponentId, Components};

/// A statically-known group of components that can be inserted together.
///
/// # Safety
/// Implementors must guarantee that [`Bundle::get_components`] yields exactly
/// one pointer per id produced by [`Bundle::component_ids`], in the same order,
/// and that each yielded value is a valid, initialized value of the
/// corresponding component type whose ownership is transferred to the callback
/// (the implementor must not also drop it).
pub unsafe trait Bundle: Send + Sync + 'static {
    /// Register every component type in this bundle and append its id to `out`
    /// in a stable order.
    fn component_ids(components: &mut Components, out: &mut Vec<ComponentId>);

    /// Move each component value out of `self`, calling `func` once per value
    /// with a pointer to it, in the same order as [`Bundle::component_ids`].
    ///
    /// # Safety
    /// The callback takes ownership of each value (it will move the bytes out),
    /// so the implementor must not drop the yielded values afterwards.
    unsafe fn get_components(self, func: &mut dyn FnMut(*mut u8));
}

// SAFETY: a single component yields exactly one id and one matching value; the
// value is held in `ManuallyDrop` and handed to `func` by pointer, so it is
// moved out exactly once and never dropped here.
unsafe impl<C: Component> Bundle for C {
    fn component_ids(components: &mut Components, out: &mut Vec<ComponentId>) {
        out.push(components.register::<C>());
    }

    unsafe fn get_components(self, func: &mut dyn FnMut(*mut u8)) {
        let mut value = ManuallyDrop::new(self);
        func((&mut *value as *mut C).cast::<u8>());
    }
}

// SAFETY: the empty tuple yields no ids and no values, trivially consistent.
unsafe impl Bundle for () {
    fn component_ids(_components: &mut Components, _out: &mut Vec<ComponentId>) {}
    unsafe fn get_components(self, _func: &mut dyn FnMut(*mut u8)) {}
}

macro_rules! impl_bundle_for_tuple {
    ($($T:ident),+) => {
        // SAFETY: each element is itself a `Bundle` that upholds the one-id /
        // one-value ordering contract; concatenating them preserves it.
        unsafe impl<$($T: Bundle),+> Bundle for ($($T,)+) {
            fn component_ids(components: &mut Components, out: &mut Vec<ComponentId>) {
                $($T::component_ids(components, out);)+
            }

            #[allow(non_snake_case)]
            unsafe fn get_components(self, func: &mut dyn FnMut(*mut u8)) {
                let ($($T,)+) = self;
                // SAFETY: forwarded — each element moves its own values out once.
                unsafe { $($T.get_components(func);)+ }
            }
        }
    };
}

impl_bundle_for_tuple!(A);
impl_bundle_for_tuple!(A, B);
impl_bundle_for_tuple!(A, B, C);
impl_bundle_for_tuple!(A, B, C, D);
impl_bundle_for_tuple!(A, B, C, D, E);
impl_bundle_for_tuple!(A, B, C, D, E, F);
impl_bundle_for_tuple!(A, B, C, D, E, F, G);
impl_bundle_for_tuple!(A, B, C, D, E, F, G, H);
impl_bundle_for_tuple!(A, B, C, D, E, F, G, H, I);
impl_bundle_for_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_bundle_for_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_bundle_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);
