//! Consumer compile/behaviour tests for the `bind!` field-binding macro.
//!
//! The macro generates calls shaped exactly like `prism_ui_ecs::EcsBridge`'s
//! `bind` / `bind_two_way` methods. These tests exercise that generated code
//! against a lightweight structural bridge, so the test stays offline and free
//! of the heavy ECS dependency while still type-checking — and running — the
//! emitted reader and writer closures end to end. The projected `TYPE` is
//! always the field's own type, since the generated reader clones the field.

use core::marker::PhantomData;

use prism_ui_macro::bind;

/// A structural stand-in for `prism_ui_ecs::EcsBridge`.
///
/// It mirrors the exact method shapes `bind!` targets and invokes the generated
/// `reader`/`writer` closures against a sample component, so the test exercises
/// their behaviour rather than only type-checking them.
#[derive(Default)]
struct MockBridge {
    /// Count of one-way bindings registered.
    one_way: u32,
    /// Count of two-way bindings registered.
    two_way: u32,
}

/// A no-op signal placeholder spliced in where a real `Signal<T>` would go.
struct MockSignal<T>(PhantomData<T>);

impl<T> MockSignal<T> {
    /// Builds an empty signal placeholder.
    fn new() -> Self {
        Self(PhantomData)
    }
}

impl MockBridge {
    /// Mirrors `EcsBridge::bind`: register and run a one-way reader.
    fn bind<C, T>(&mut self, entity: Sample<C>, _signal: MockSignal<T>, reader: impl Fn(&C) -> T) {
        let _value = reader(&entity.component);
        self.one_way += 1;
    }

    /// Mirrors `EcsBridge::bind_two_way`: register and run a reader + writer.
    fn bind_two_way<C, T>(
        &mut self,
        mut entity: Sample<C>,
        _signal: MockSignal<T>,
        reader: impl Fn(&C) -> T,
        writer: impl Fn(&mut C, &T),
    ) {
        let value = reader(&entity.component);
        writer(&mut entity.component, &value);
        self.two_way += 1;
    }
}

/// Carries a sample component value in place of a real `Entity` handle so the
/// mock can drive the generated closures.
struct Sample<C> {
    /// The sample component the reader/writer closures act on.
    component: C,
}

/// A component with a named field.
#[derive(Clone)]
struct Health {
    /// Current hit points.
    current: u32,
}

/// A tuple-struct component.
#[derive(Clone)]
struct Name(String);

#[test]
fn two_way_binding_reads_and_writes_named_field() {
    let mut bridge = MockBridge::default();
    let entity = Sample {
        component: Health { current: 42 },
    };
    let signal = MockSignal::<u32>::new();

    bind!(bridge, signal <-> $entity.Health.current : u32);

    assert_eq!(bridge.two_way, 1);
    assert_eq!(bridge.one_way, 0);
}

#[test]
fn one_way_binding_reads_tuple_field() {
    let mut bridge = MockBridge::default();
    let entity = Sample {
        component: Name(String::from("hero")),
    };
    let signal = MockSignal::<String>::new();

    bind!(bridge, signal <- $entity.Name.0 : String);

    assert_eq!(bridge.one_way, 1);
    assert_eq!(bridge.two_way, 0);
}

#[test]
fn dotted_signal_target_binds() {
    // The signal expression may be a field access as long as it has no
    // top-level `<`.
    struct Ctx {
        bridge: MockBridge,
    }
    let mut ctx = Ctx {
        bridge: MockBridge::default(),
    };
    let entity = Sample {
        component: Health { current: 7 },
    };
    let signal = MockSignal::<u32>::new();

    bind!(ctx.bridge, signal <-> $entity.Health.current : u32);

    assert_eq!(ctx.bridge.two_way, 1);
}
