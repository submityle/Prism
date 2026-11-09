use super::*;
use prism_render_driver::{
    BindGroupDescriptor, DeviceDescriptor, Extent3d, Instance, RequestAdapterOptions,
};
use prism_render_driver_null::{NullDevice, NullInstance};

fn null_device() -> Arc<NullDevice> {
    let instance = NullInstance::new();
    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .unwrap();
    Arc::new(
        instance
            .create_device(&adapter, &DeviceDescriptor::default())
            .unwrap()
            .0,
    )
}

fn bind(device: &NullDevice) -> BindGroupId {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("target-cache-lifecycle".into()),
        layout: None,
        entries: vec![],
    })
}

#[test]
fn target_eviction_is_exact_and_idempotent() {
    let device = null_device();
    let owner: Arc<dyn RenderDevice + Send + Sync> = device.clone();
    let first = RenderTarget::new(&*device, Extent3d::new_2d(4, 4));
    let second = RenderTarget::new(&*device, Extent3d::new_2d(4, 4));
    let baseline = device.live_resource_count();
    let first_group = bind(&device);
    let second_group = bind(&device);
    let mut cache = CompositeTargetCache::default();
    cache.device_generation = Some(3);
    cache.device = Some(owner.clone());
    cache.groups = BTreeMap::from([
        (CompositeTargetKey::from(&first), first_group),
        (CompositeTargetKey::from(&second), second_group),
    ]);
    assert_eq!(device.live_resource_count(), baseline + 2);
    cache.evict(CompositeTargetKey::from(&first), &owner, 3);
    assert_eq!(device.live_resource_count(), baseline + 1);
    assert_eq!(cache.groups.len(), 1);
    assert_eq!(
        cache.groups[&CompositeTargetKey::from(&second)],
        second_group
    );
    cache.evict(CompositeTargetKey::from(&first), &owner, 3);
    assert_eq!(device.live_resource_count(), baseline + 1);
    drop(cache);
    assert_eq!(device.live_resource_count(), baseline);
    first.destroy(&*device);
    second.destroy(&*device);
    assert_eq!(device.live_resource_count(), 0);
}

#[test]
fn target_eviction_requires_creating_device_and_generation() {
    let device = null_device();
    let replacement = null_device();
    let owner: Arc<dyn RenderDevice + Send + Sync> = device.clone();
    let other: Arc<dyn RenderDevice + Send + Sync> = replacement.clone();
    let first = RenderTarget::new(&*device, Extent3d::new_2d(4, 4));
    let second = RenderTarget::new(&*replacement, Extent3d::new_2d(4, 4));
    // Null IDs start at the same slot/generation on independent devices.
    assert_eq!(
        CompositeTargetKey::from(&first),
        CompositeTargetKey::from(&second)
    );
    let first_group = bind(&device);
    let second_group = bind(&replacement);
    assert_eq!(first_group, second_group);
    let baseline = device.live_resource_count();
    let other_baseline = replacement.live_resource_count();
    let mut cache = CompositeTargetCache::default();
    cache.device_generation = Some(3);
    cache.device = Some(owner.clone());
    cache.groups = BTreeMap::from([(CompositeTargetKey::from(&first), first_group)]);
    cache.evict(CompositeTargetKey::from(&first), &owner, 4);
    cache.evict(CompositeTargetKey::from(&second), &other, 3);
    assert_eq!(device.live_resource_count(), baseline);
    assert_eq!(replacement.live_resource_count(), other_baseline);
    assert_eq!(cache.groups.len(), 1);
    cache.evict(CompositeTargetKey::from(&first), &owner, 3);
    assert_eq!(device.live_resource_count(), baseline - 1);
    assert_eq!(replacement.live_resource_count(), other_baseline);
    replacement.destroy_bind_group(second_group);
    first.destroy(&*device);
    second.destroy(&*replacement);
    assert_eq!(device.live_resource_count(), 0);
    assert_eq!(replacement.live_resource_count(), 0);
}
