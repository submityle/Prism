//! Value contracts exercised independently of snapshot capture/delta wiring.

use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::hash::{Hash, Hasher};
use core::mem::size_of;
use core::sync::atomic::{AtomicUsize, Ordering};

use crate::change::Tick;
use crate::component::Component;
use crate::resource::Resource;
use crate::world::World;

use super::super::hash::{component_equality, FnvHasher};
use super::super::resource::{capture_resources, clone_resources, restore_resources};
use super::SnapshotColumn;

fn column<T: Component + Clone>(world: &mut World, value: &T) -> SnapshotColumn {
    let id = world.register_snapshot_component::<T>();
    let info = world.components.info(id).unwrap();
    let mut column = SnapshotColumn::new(
        id,
        info.storage(),
        info.layout(),
        info.clone_fn().unwrap(),
        info.drop_fn(),
        info.snapshot_hash_fn(),
    );
    column.equality = component_equality(world, id);
    // SAFETY: value is an initialized T matching the registered clone glue.
    unsafe { column.push_cloned(core::ptr::from_ref(value).cast(), Tick::ZERO, Tick::ZERO, 0) };
    column
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq)]
struct Padded {
    flag: u8,
    count: u64,
}
impl Component for Padded {}

#[test]
fn padded_values_compare_fields_without_hash_registration() {
    assert!(size_of::<Padded>() > 9);
    let mut world = World::new();
    world.register_snapshot_component_equatable::<Padded>();
    let a = column(&mut world, &Padded { flag: 1, count: 2 });
    let b = column(&mut world, &Padded { flag: 1, count: 2 });
    let c = column(&mut world, &Padded { flag: 1, count: 3 });
    assert!(a.hash.is_none());
    assert_eq!(a.value_eq(0, &b, 0), Some(true));
    assert_eq!(a.value_eq(0, &c, 0), Some(false));
}

#[derive(Clone, Debug, PartialEq)]
struct Owned {
    text: String,
    values: Vec<u64>,
}
impl Component for Owned {}
impl Resource for Owned {}

fn owned() -> Owned {
    Owned {
        text: "owned snapshot".to_string(),
        values: vec![3, 5, 8],
    }
}

#[test]
fn string_and_vec_compare_contents_after_independent_clones() {
    let mut world = World::new();
    world.register_snapshot_component_equatable::<Owned>();
    let mut value = owned();
    let a = column(&mut world, &value);
    let b = column(&mut world, &value);
    value.text.push_str(" changed");
    value.values.push(13);
    let changed = column(&mut world, &value);
    drop(value);
    drop(world);
    assert_eq!(a.value_eq(0, &b, 0), Some(true));
    assert_eq!(a.value_eq(0, &changed, 0), Some(false));
}

#[test]
fn missing_comparison_is_unknown_even_for_same_column_and_ticks() {
    let mut world = World::new();
    let a = column(&mut world, &owned());
    let b = column(&mut world, &owned());
    assert_eq!(a.value_eq(0, &a, 0), None);
    assert_eq!(a.value_eq(0, &b, 0), None);
    world.register_snapshot_component_equatable::<Owned>();
    let after_registration = column(&mut world, &owned());
    assert_eq!(a.value_eq(0, &after_registration, 0), None);
}

#[derive(Clone, PartialEq)]
struct FloatValue(f32);
impl Component for FloatValue {}

#[test]
fn floating_point_comparison_uses_partial_eq_instead_of_object_bits() {
    let mut world = World::new();
    world.register_snapshot_component_equatable::<FloatValue>();
    let positive_zero = column(&mut world, &FloatValue(0.0));
    let negative_zero = column(&mut world, &FloatValue(-0.0));
    let nan = column(&mut world, &FloatValue(f32::NAN));
    assert_eq!(positive_zero.value_eq(0, &negative_zero, 0), Some(true));
    assert_eq!(nan.value_eq(0, &nan, 0), Some(false));
}

#[test]
fn coincident_component_ids_from_different_types_never_invoke_wrong_glue() {
    #[derive(Clone, PartialEq)]
    struct Other(u64);
    impl Component for Other {}
    let mut left = World::new();
    let mut right = World::new();
    left.register_snapshot_component_equatable::<Collision>();
    right.register_snapshot_component_equatable::<Other>();
    let a = column(&mut left, &Collision(7));
    let b = column(&mut right, &Other(7));
    assert_eq!(a.component, b.component);
    assert_eq!(a.layout, b.layout);
    assert_eq!(a.value_eq(0, &b, 0), None);
}

#[derive(Clone, PartialEq)]
struct Collision(u64);
impl Component for Collision {}
impl Resource for Collision {}
impl Hash for Collision {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u8(0);
    }
}

#[test]
fn component_hash_collision_does_not_establish_value_equality() {
    let mut world = World::new();
    world.register_snapshot_component_hashable::<Collision>();
    world.register_snapshot_component_equatable::<Collision>();
    let a = column(&mut world, &Collision(1));
    let b = column(&mut world, &Collision(2));
    let (mut ah, mut bh) = (FnvHasher::new(), FnvHasher::new());
    // SAFETY: both columns contain one initialized Collision value.
    unsafe {
        assert!(a.hash_value(0, &mut ah));
        assert!(b.hash_value(0, &mut bh));
    }
    assert_eq!(ah.finish(), bh.finish());
    assert_eq!(a.value_eq(0, &b, 0), Some(false));
}

#[test]
fn resource_hash_collision_does_not_establish_value_equality() {
    let mut world = World::new();
    world.register_snapshot_resource_hashable::<Collision>();
    world.register_snapshot_resource_equatable::<Collision>();
    world.insert_resource(Collision(1));
    let a = capture_resources(&world);
    world.insert_resource(Collision(2));
    let b = capture_resources(&world);
    assert_eq!(a[0].value_hash(), b[0].value_hash());
    assert_eq!(a[0].value_eq(&b[0]), Some(false));
}

#[test]
fn clone_only_resource_capture_restore_and_comparison_remain_distinct() {
    let mut world = World::new();
    world.register_snapshot_resource::<Owned>();
    world.insert_resource(owned());
    let a = capture_resources(&world);
    let copied = clone_resources(&a);
    assert_eq!(a[0].value_eq(&copied[0]), None);
    world.resource_mut::<Owned>().text.clear();
    restore_resources(&mut world, &copied);
    assert_eq!(world.resource::<Owned>(), &owned());
}

#[test]
fn resource_comparison_survives_clone_restore_and_world_drop() {
    let mut world = World::new();
    world.register_snapshot_resource_equatable::<Owned>();
    world.insert_resource(owned());
    let a = capture_resources(&world);
    let copied = clone_resources(&a);
    world.resource_mut::<Owned>().values.clear();
    restore_resources(&mut world, &copied);
    let restored = capture_resources(&world);
    drop(world);
    assert_eq!(a[0].value_eq(&copied[0]), Some(true));
    assert_eq!(copied[0].value_eq(&restored[0]), Some(true));
}

#[derive(Clone, PartialEq)]
struct FixedValue(prism_math::Fixed);
impl Component for FixedValue {}

#[test]
fn fixed_point_values_use_exact_typed_comparison() {
    let mut world = World::new();
    world.register_snapshot_component_equatable::<FixedValue>();
    let a = column(&mut world, &FixedValue(prism_math::Fixed::from_bits(17)));
    let b = column(&mut world, &FixedValue(prism_math::Fixed::from_bits(17)));
    let c = column(&mut world, &FixedValue(prism_math::Fixed::from_bits(18)));
    assert_eq!(a.value_eq(0, &b, 0), Some(true));
    assert_eq!(a.value_eq(0, &c, 0), Some(false));
}

struct Tracked {
    value: Owned,
    live: Arc<AtomicUsize>,
}
impl Component for Tracked {}
impl Clone for Tracked {
    fn clone(&self) -> Self {
        let value = self.value.clone();
        self.live.fetch_add(1, Ordering::SeqCst);
        Self {
            value,
            live: self.live.clone(),
        }
    }
}
impl PartialEq for Tracked {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}
impl Drop for Tracked {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(feature = "std")]
#[test]
fn equality_panic_leaves_owning_columns_intact() {
    struct Panicking(Tracked);
    impl Component for Panicking {}
    impl Clone for Panicking {
        fn clone(&self) -> Self {
            Self(self.0.clone())
        }
    }
    impl PartialEq for Panicking {
        fn eq(&self, _: &Self) -> bool {
            panic!("comparison callback panic")
        }
    }
    let live = Arc::new(AtomicUsize::new(1));
    let value = Panicking(Tracked {
        value: owned(),
        live: live.clone(),
    });
    let mut world = World::new();
    world.register_snapshot_component_equatable::<Panicking>();
    let a = column(&mut world, &value);
    let b = column(&mut world, &value);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a.value_eq(0, &b, 0)));
    assert!(result.is_err());
    assert_eq!(live.load(Ordering::SeqCst), 3);
    assert_eq!(a.len(), 1);
    assert_eq!(b.len(), 1);
    drop((a, b, value, world));
    assert_eq!(live.load(Ordering::SeqCst), 0);
}

#[test]
fn delta_style_column_clones_own_values_and_keep_comparison_glue() {
    let live = Arc::new(AtomicUsize::new(1));
    let value = Tracked {
        value: owned(),
        live: live.clone(),
    };
    let mut world = World::new();
    world.register_snapshot_component_equatable::<Tracked>();
    let captured = column(&mut world, &value);
    let mut fresh = captured.empty_like();
    // SAFETY: matching empty_like column, valid source slot, ascending row.
    unsafe { fresh.push_cloned_from(&captured, 0, 0) };
    let mut rebuilt = fresh.empty_like();
    // SAFETY: matching empty_like column, valid source slot, ascending row.
    unsafe { rebuilt.push_cloned_from(&fresh, 0, 0) };
    assert_eq!(live.load(Ordering::SeqCst), 4);
    drop(value);
    drop(captured);
    drop(world);
    assert_eq!(live.load(Ordering::SeqCst), 2);
    assert_eq!(fresh.value_eq(0, &rebuilt, 0), Some(true));
    drop(fresh);
    let mut restored = core::mem::MaybeUninit::<Tracked>::uninit();
    // SAFETY: valid source slot, matching aligned uninitialized destination.
    unsafe { rebuilt.clone_value_into(0, restored.as_mut_ptr().cast()) };
    // SAFETY: clone_value_into initialized the destination with an owned T.
    let restored = unsafe { restored.assume_init() };
    drop(rebuilt);
    assert_eq!(restored.value, owned());
    assert_eq!(live.load(Ordering::SeqCst), 1);
    drop(restored);
    assert_eq!(live.load(Ordering::SeqCst), 0);
}

#[test]
fn fnv_integer_writes_have_explicit_little_endian_encoding() {
    let mut typed = FnvHasher::new();
    typed.write_u8(1);
    typed.write_u16(0x0203);
    typed.write_u32(0x04050607);
    typed.write_u64(0x08090a0b0c0d0e0f);
    typed.write_u128(0x101112131415161718191a1b1c1d1e1f);
    typed.write_usize(32);
    typed.write_i8(-1);
    typed.write_i16(-2);
    typed.write_i32(-3);
    typed.write_i64(-4);
    typed.write_i128(-5);
    typed.write_isize(-6);

    let mut bytes = FnvHasher::new();
    bytes.write(&[1]);
    bytes.write(&0x0203u16.to_le_bytes());
    bytes.write(&0x04050607u32.to_le_bytes());
    bytes.write(&0x08090a0b0c0d0e0fu64.to_le_bytes());
    bytes.write(&0x101112131415161718191a1b1c1d1e1fu128.to_le_bytes());
    bytes.write(&32u64.to_le_bytes());
    bytes.write(&(-1i8).to_le_bytes());
    bytes.write(&(-2i16).to_le_bytes());
    bytes.write(&(-3i32).to_le_bytes());
    bytes.write(&(-4i64).to_le_bytes());
    bytes.write(&(-5i128).to_le_bytes());
    bytes.write(&(-6i64).to_le_bytes());
    assert_eq!(typed.finish(), bytes.finish());
}
