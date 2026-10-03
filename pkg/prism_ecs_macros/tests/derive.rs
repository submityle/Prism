//! Integration tests that prove the derives expand to compilable, correct
//! code.
//!
//! A proc-macro crate cannot depend on `prism_ecs` without creating a cyclic
//! dependency, so this test defines a local `mod prism_ecs { .. }` whose module
//! paths (`prism_ecs::component::*`, `prism_ecs::bundle::*`) and trait
//! signatures mirror the real kernel exactly. Because the test crate has no
//! extern crate named `prism_ecs`, the absolute paths emitted by the derives
//! resolve to this local shim, letting us exercise the generated impls at
//! runtime.

#![allow(dead_code)]
// The shim below mirrors `prism_ecs`'s genuinely-`unsafe` `Bundle` trait so the
// derive expansions resolve against the real signatures. The proc-macro crate
// itself contains no unsafe; only this fixture does.
#![allow(unsafe_code)]

use std::sync::atomic::{AtomicU32, Ordering};

/// Local stand-in mirroring the real `prism_ecs` public surface the derives
/// target.
mod prism_ecs {
    pub mod component {
        use std::any::TypeId;

        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub enum StorageType {
            Table,
            SparseSet,
        }

        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub struct ComponentId(pub u32);

        pub trait Component: Send + Sync + 'static {
            const STORAGE: StorageType = StorageType::Table;
        }

        /// Minimal registry: assigns a dense id per distinct component type.
        #[derive(Default)]
        pub struct Components {
            types: Vec<TypeId>,
        }

        impl Components {
            pub fn new() -> Self {
                Self { types: Vec::new() }
            }

            pub fn register<C: Component>(&mut self) -> ComponentId {
                let tid = TypeId::of::<C>();
                if let Some(i) = self.types.iter().position(|&t| t == tid) {
                    return ComponentId(i as u32);
                }
                let id = ComponentId(self.types.len() as u32);
                self.types.push(tid);
                id
            }

            pub fn len(&self) -> usize {
                self.types.len()
            }

            pub fn is_empty(&self) -> bool {
                self.types.is_empty()
            }
        }
    }

    pub mod bundle {
        use super::component::{Component, ComponentId, Components};
        use core::mem::ManuallyDrop;

        /// Mirror of `prism_ecs::bundle::Bundle`.
        ///
        /// # Safety
        /// Implementors must yield exactly one pointer per id produced by
        /// `component_ids`, in the same order, transferring ownership of each
        /// value to the callback without also dropping it.
        pub unsafe trait Bundle: Send + Sync + 'static {
            fn component_ids(components: &mut Components, out: &mut Vec<ComponentId>);

            /// Move each component value out of `self`, calling `func` once per
            /// value in `component_ids` order.
            ///
            /// # Safety
            /// The callback takes ownership of each yielded value; the
            /// implementor must not drop those values afterwards.
            unsafe fn get_components(self, func: &mut dyn FnMut(*mut u8));
        }

        // Blanket impl mirroring the real kernel: every component is a
        // one-element bundle.
        unsafe impl<C: Component> Bundle for C {
            fn component_ids(components: &mut Components, out: &mut Vec<ComponentId>) {
                out.push(components.register::<C>());
            }

            unsafe fn get_components(self, func: &mut dyn FnMut(*mut u8)) {
                let mut value = ManuallyDrop::new(self);
                func((&mut *value as *mut C).cast::<u8>());
            }
        }
    }

    pub mod schedule {
        use std::any::{type_name, TypeId};

        /// Mirror of `prism_ecs::schedule::SystemSetId`.
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
        pub struct SystemSetId {
            type_id: TypeId,
            discriminant: u64,
            name: &'static str,
        }

        impl SystemSetId {
            pub fn of<T: 'static>() -> Self {
                Self { type_id: TypeId::of::<T>(), discriminant: 0, name: type_name::<T>() }
            }

            pub fn with<T: 'static>(discriminant: u64) -> Self {
                Self { type_id: TypeId::of::<T>(), discriminant, name: type_name::<T>() }
            }

            pub fn discriminant(&self) -> u64 {
                self.discriminant
            }
        }

        /// Mirror of `prism_ecs::schedule::SystemSet`.
        pub trait SystemSet: Send + Sync + 'static {
            fn set_id(&self) -> SystemSetId;
        }
    }
}

use prism_ecs::bundle::Bundle;
use prism_ecs::component::{Component, Components, StorageType};
use prism_ecs::schedule::{SystemSet, SystemSetId};
use prism_ecs_macros::{Bundle, Component, SystemSet};

// ---- Component derive targets ------------------------------------------------

#[derive(Component)]
struct Position(i32, i32);

#[derive(Component)]
struct Velocity {
    dx: i32,
    dy: i32,
}

#[derive(Component)]
#[component(storage = "SparseSet")]
struct Tag;

#[derive(Component)]
#[component(storage = "Table")]
struct Health(i32);

#[derive(Component)]
struct Wrapper<T: Send + Sync + 'static>(T);

// ---- Bundle derive targets ---------------------------------------------------

#[derive(Bundle)]
struct Physics {
    pos: Position,
    vel: Velocity,
}

#[derive(Bundle)]
struct PhysicsTuple(Position, Velocity);

#[derive(Bundle)]
struct Nested {
    physics: Physics,
    tag: Tag,
}

#[derive(Bundle)]
struct EmptyBundle;

#[derive(Bundle)]
struct Wrap<B> {
    inner: B,
}

// ---- Drop-tracking fixture ---------------------------------------------------

static DROPS: AtomicU32 = AtomicU32::new(0);

#[derive(Component)]
struct Tracked(u32);

impl Drop for Tracked {
    fn drop(&mut self) {
        DROPS.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Bundle)]
struct TwoTracked {
    first: Tracked,
    second: Tracked,
}

// ---- Tests -------------------------------------------------------------------

#[test]
fn component_storage_const_defaults_and_overrides() {
    assert_eq!(<Position as Component>::STORAGE, StorageType::Table);
    assert_eq!(<Velocity as Component>::STORAGE, StorageType::Table);
    assert_eq!(<Health as Component>::STORAGE, StorageType::Table);
    assert_eq!(<Tag as Component>::STORAGE, StorageType::SparseSet);
    // Generic component also implements the trait.
    assert_eq!(<Wrapper<u32> as Component>::STORAGE, StorageType::Table);
}

#[test]
fn component_ids_registers_fields_in_declaration_order() {
    let mut components = Components::new();
    let mut out = Vec::new();
    Physics::component_ids(&mut components, &mut out);
    assert_eq!(out.len(), 2);
    assert_eq!(components.len(), 2);
    assert_eq!(out[0], prism_ecs::component::ComponentId(0)); // Position first
    assert_eq!(out[1], prism_ecs::component::ComponentId(1)); // Velocity second
}

#[test]
fn nested_bundle_flattens_in_order() {
    let mut components = Components::new();
    let mut out = Vec::new();
    Nested::component_ids(&mut components, &mut out);
    // Physics(Position, Velocity) then Tag => three ids.
    assert_eq!(out.len(), 3);
    assert_eq!(components.len(), 3);
}

#[test]
fn empty_bundle_registers_nothing_and_yields_nothing() {
    let mut components = Components::new();
    let mut out = Vec::new();
    EmptyBundle::component_ids(&mut components, &mut out);
    assert!(out.is_empty());
    assert!(components.is_empty());

    // get_components must not invoke the callback.
    unsafe {
        EmptyBundle.get_components(&mut |_ptr| unreachable!("no fields to yield"));
    }
}

#[test]
fn tuple_bundle_yields_values_in_declaration_order() {
    let bundle = PhysicsTuple(Position(1, 2), Velocity { dx: 3, dy: 4 });
    let mut idx = 0usize;
    let mut pos: Option<Position> = None;
    let mut vel: Option<Velocity> = None;
    unsafe {
        bundle.get_components(&mut |ptr| {
            match idx {
                0 => pos = Some(std::ptr::read(ptr.cast::<Position>())),
                1 => vel = Some(std::ptr::read(ptr.cast::<Velocity>())),
                _ => unreachable!("tuple bundle has exactly two fields"),
            }
            idx += 1;
        });
    }
    let pos = pos.expect("position yielded");
    let vel = vel.expect("velocity yielded");
    assert_eq!((pos.0, pos.1), (1, 2));
    assert_eq!((vel.dx, vel.dy), (3, 4));
}

#[test]
fn get_components_moves_each_field_once_without_double_drop() {
    DROPS.store(0, Ordering::SeqCst);
    let bundle = TwoTracked {
        first: Tracked(10),
        second: Tracked(20),
    };
    let mut seen = Vec::new();
    unsafe {
        bundle.get_components(&mut |ptr| {
            // The callback takes ownership of each value exactly once.
            let value = std::ptr::read(ptr.cast::<Tracked>());
            seen.push(value.0);
            drop(value); // dropped here, once
        });
    }
    assert_eq!(seen, vec![10, 20], "declaration order preserved");
    assert_eq!(
        DROPS.load(Ordering::SeqCst),
        2,
        "each field moved out and dropped exactly once"
    );
}

#[test]
fn generic_bundle_wraps_inner_bundle() {
    let mut components = Components::new();
    let mut out = Vec::new();
    Wrap::<Position>::component_ids(&mut components, &mut out);
    assert_eq!(out.len(), 1);
}


// ---- SystemSet derive targets -----------------------------------------------

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PhysicsSet;

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SyncSet {
    Pull,
    Push,
}

#[test]
fn system_set_struct_is_single_set() {
    assert_eq!(PhysicsSet.set_id(), SystemSetId::of::<PhysicsSet>());
    assert_eq!(PhysicsSet.set_id().discriminant(), 0);
}

#[test]
fn system_set_enum_variants_are_distinct() {
    assert_eq!(SyncSet::Pull.set_id().discriminant(), 0);
    assert_eq!(SyncSet::Push.set_id().discriminant(), 1);
    assert_ne!(SyncSet::Pull.set_id(), SyncSet::Push.set_id());
    // Same variant => same id (stable identity).
    assert_eq!(SyncSet::Pull.set_id(), SyncSet::Pull.set_id());
}
