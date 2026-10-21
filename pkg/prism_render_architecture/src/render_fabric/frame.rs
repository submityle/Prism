//! Shared frame-local items, independent of extraction, ECS and GPU ownership.

use alloc::vec::Vec;
use core::fmt;

use super::{
    DomainPayload, FrameRenderItem, PayloadRole, PayloadValue, RenderDomain, RenderObjectId,
};

/// An owned, validated frame sorted by object, domain, history and payload role.
///
/// This container checks only identity and basic payload semantics. Object
/// liveness belongs to domain owners; resource layouts, physical bindings and
/// access compatibility remain the responsibility of compilation and runtime.
#[derive(Default, Clone, Debug, PartialEq)]
pub struct RenderFabricFrame {
    frame: u64,
    items: Vec<FrameRenderItem>,
}

impl RenderFabricFrame {
    /// Validates and canonicalizes an owned batch without copying its items.
    pub fn from_items(
        frame: u64,
        items: Vec<FrameRenderItem>,
    ) -> Result<Self, RenderFabricFrameError> {
        for item in &items {
            validate_item(item)?;
        }
        let mut output = Self { frame, items };
        output.canonicalize()?;
        Ok(output)
    }

    /// Returns owned items without reallocating; the frame clock is discarded.
    #[must_use]
    pub fn into_items(self) -> Vec<FrameRenderItem> {
        self.items
    }

    #[must_use]
    pub const fn frame(&self) -> u64 {
        self.frame
    }

    #[must_use]
    pub fn items(&self) -> &[FrameRenderItem] {
        &self.items
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Clears published items without releasing their allocation.
    pub fn clear(&mut self, frame: u64) {
        self.frame = frame;
        self.items.clear();
    }

    /// Replaces the frame, reusing capacity and sorting by stable identity.
    ///
    /// On any validation error, both old and partially collected items are
    /// discarded. The requested frame number and allocated capacity remain.
    pub fn replace(
        &mut self,
        frame: u64,
        items: impl IntoIterator<Item = FrameRenderItem>,
    ) -> Result<(), RenderFabricFrameError> {
        self.clear(frame);
        for item in items {
            if let Err(error) = validate_item(&item) {
                self.items.clear();
                return Err(error);
            }
            self.items.push(item);
        }
        self.canonicalize()
    }

    /// Replaces one domain without cloning the other domains or their storage.
    ///
    /// The frame clock is unchanged. Every incoming item must belong to the
    /// requested domain. Errors clear the whole frame, including other domains,
    /// so consumers can never observe a partially refreshed publication.
    pub fn replace_domain(
        &mut self,
        domain: RenderDomain,
        items: impl IntoIterator<Item = FrameRenderItem>,
    ) -> Result<(), RenderFabricFrameError> {
        self.items.retain(|item| item.domain != domain);
        for item in items {
            let result = validate_item(&item).and_then(|()| {
                if item.domain == domain {
                    Ok(())
                } else {
                    Err(RenderFabricFrameError::DomainIdentityMismatch {
                        object: item.object_id,
                        declared: domain,
                    })
                }
            });
            if let Err(error) = result {
                self.items.clear();
                return Err(error);
            }
            self.items.push(item);
        }
        self.canonicalize()
    }

    fn canonicalize(&mut self) -> Result<(), RenderFabricFrameError> {
        // Equal keys are rejected below, so unstable sorting produces the same
        // canonical order as stable sorting without a scratch allocation.
        self.items.sort_unstable_by_key(identity);
        if let Some(pair) = self
            .items
            .windows(2)
            .find(|pair| identity(&pair[0]) == identity(&pair[1]))
        {
            let item = pair[0];
            let error = RenderFabricFrameError::DuplicateItem {
                object: item.object_id,
                domain: item.domain,
                history_key: item.history_key,
                role: item.payload_role(),
            };
            self.items.clear();
            return Err(error);
        }
        Ok(())
    }

    /// Looks up any payload for an object in logarithmic time.
    #[must_use]
    pub fn contains_object(&self, object: RenderObjectId) -> bool {
        self.items
            .binary_search_by_key(&object, |item| item.object_id)
            .is_ok()
    }
}

fn identity(item: &FrameRenderItem) -> (RenderObjectId, u32, u32, PayloadRole) {
    (
        item.object_id,
        item.domain as u32,
        item.history_key,
        item.payload_role(),
    )
}

fn validate_item(item: &FrameRenderItem) -> Result<(), RenderFabricFrameError> {
    let object = item.object_id;
    let role = item.payload_role();
    if object.domain() != item.domain {
        return Err(RenderFabricFrameError::DomainIdentityMismatch {
            object,
            declared: item.domain,
        });
    }
    match item.payload {
        DomainPayload::Value(payload) => {
            let finite = match payload.value {
                PayloadValue::Scalar(value) => value.is_finite(),
                PayloadValue::Vector(values) => values.iter().all(|value| value.is_finite()),
                PayloadValue::Bounds { min, max } => {
                    min.iter().chain(max.iter()).all(|value| value.is_finite())
                }
            };
            if !finite {
                return Err(RenderFabricFrameError::NonFiniteValue { object, role });
            }
            if let PayloadValue::Bounds { min, max } = payload.value {
                if min.iter().zip(max.iter()).any(|(min, max)| min > max) {
                    return Err(RenderFabricFrameError::InvertedBounds { object, role });
                }
            }
        }
        DomainPayload::Resource(payload) => {
            if payload.size == 0 {
                return Err(RenderFabricFrameError::ZeroResourceSize { object, role });
            }
            if !payload.alignment.is_power_of_two() {
                return Err(RenderFabricFrameError::InvalidResourceAlignment {
                    object,
                    role,
                    alignment: payload.alignment,
                });
            }
        }
    }
    Ok(())
}

/// Basic semantic validation failed while replacing a shared frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderFabricFrameError {
    DomainIdentityMismatch {
        object: RenderObjectId,
        declared: RenderDomain,
    },
    DuplicateItem {
        object: RenderObjectId,
        domain: RenderDomain,
        history_key: u32,
        role: PayloadRole,
    },
    NonFiniteValue {
        object: RenderObjectId,
        role: PayloadRole,
    },
    InvertedBounds {
        object: RenderObjectId,
        role: PayloadRole,
    },
    ZeroResourceSize {
        object: RenderObjectId,
        role: PayloadRole,
    },
    InvalidResourceAlignment {
        object: RenderObjectId,
        role: PayloadRole,
        alignment: u64,
    },
}

impl fmt::Display for RenderFabricFrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "render fabric frame error: {self:?}")
    }
}

impl std::error::Error for RenderFabricFrameError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        frame_graph::ResourceLifetime,
        gpu_scene::SceneHandle,
        motion::MotionSource,
        render_fabric::{
            BufferUsages, DomainObjectHandle, ResourceKey, ResourceLayout, ResourcePayload,
            ValuePayload,
        },
    };

    fn objects(index: u32, generation: u32) -> [RenderObjectId; 5] {
        let handle = DomainObjectHandle::new(index, generation);
        [
            RenderObjectId::Surface(SceneHandle::new(index, generation)),
            RenderObjectId::Particle(handle),
            RenderObjectId::Volume(handle),
            RenderObjectId::Hair(handle),
            RenderObjectId::Decal(handle),
        ]
    }

    fn value_item(object: RenderObjectId, history_key: u32, role: u16) -> FrameRenderItem {
        FrameRenderItem {
            object_id: object,
            domain: object.domain(),
            history_key,
            motion_source: MotionSource::Rigid,
            payload: DomainPayload::Value(ValuePayload {
                value: PayloadValue::Scalar(1.0),
                role: PayloadRole(role),
            }),
        }
    }

    fn with_value(mut item: FrameRenderItem, value: PayloadValue) -> FrameRenderItem {
        item.payload = DomainPayload::Value(ValuePayload {
            value,
            role: item.payload_role(),
        });
        item
    }

    fn resource_item(object: RenderObjectId, size: u64, alignment: u64) -> FrameRenderItem {
        FrameRenderItem {
            payload: DomainPayload::Resource(ResourcePayload {
                resource: ResourceKey::engine(0),
                lifetime: ResourceLifetime::Persistent,
                size,
                alignment,
                layout: ResourceLayout::Buffer {
                    bytes: size,
                    usage: BufferUsages::STORAGE,
                },
                role: PayloadRole(2),
            }),
            ..value_item(object, 0, 2)
        }
    }

    /// Every validation failure must drop both a previous frame and a valid
    /// prefix of the requested frame, retain capacity, and allow recovery.
    fn assert_rejected(item: FrameRenderItem, expected: RenderFabricFrameError) {
        let old = value_item(objects(8, 1)[0], 0, 0);
        let prefix = value_item(objects(9, 1)[0], 0, 0);
        let mut frame = RenderFabricFrame::default();
        frame.replace(10, [old, prefix]).unwrap();
        let capacity = frame.items.capacity();
        let allocation = frame.items.as_ptr();
        assert_eq!(frame.replace(11, [prefix, item]), Err(expected));
        assert_eq!(frame.frame(), 11);
        assert!(frame.is_empty());
        assert_eq!(frame.len(), 0);
        assert_eq!(frame.items(), &[]);
        assert!(!frame.contains_object(old.object_id));
        assert!(!frame.contains_object(prefix.object_id));
        assert_eq!(frame.items.capacity(), capacity);
        assert_eq!(frame.items.as_ptr(), allocation);
        frame.replace(12, [old]).unwrap();
        assert_eq!(frame.items(), &[old]);
        assert_eq!(frame.frame(), 12);
    }

    #[test]
    fn default_clone_debug_and_equality() {
        let mut frame = RenderFabricFrame::default();
        assert_eq!(frame.frame(), 0);
        assert_eq!(frame.len(), 0);
        assert!(frame.is_empty());
        assert!(frame.items().is_empty());
        assert!(!frame.contains_object(objects(1, 1)[0]));
        assert_eq!(frame, frame.clone());
        let item = value_item(objects(1, 1)[0], 1, 2);
        frame.replace(5, [item]).unwrap();
        let copy = frame.clone();
        assert_eq!(frame, copy);
        assert!(alloc::format!("{frame:?}").contains("RenderFabricFrame"));
        frame.clear(5);
        assert_ne!(frame, copy);
        assert_eq!(copy.items(), &[item]);
        let mut other_frame = copy.clone();
        other_frame.replace(6, [item]).unwrap();
        assert_ne!(copy, other_frame);
    }

    #[test]
    fn owned_batch_validates_without_reallocation() {
        let first = value_item(objects(1, 1)[0], 0, 0);
        let last = value_item(objects(2, 1)[4], 1, 2);
        let mut items = Vec::with_capacity(8);
        items.extend([last, first]);
        let allocation = items.as_ptr();
        let frame = RenderFabricFrame::from_items(5, items).unwrap();
        assert_eq!(frame.frame(), 5);
        assert_eq!(frame.items(), &[first, last]);
        let items = frame.into_items();
        assert_eq!(items.as_ptr(), allocation);
        assert_eq!(items.capacity(), 8);
        assert!(matches!(
            RenderFabricFrame::from_items(6, alloc::vec![first, first]),
            Err(RenderFabricFrameError::DuplicateItem { .. })
        ));
        assert!(matches!(
            RenderFabricFrame::from_items(
                6,
                alloc::vec![with_value(first, PayloadValue::Scalar(f32::NAN))]
            ),
            Err(RenderFabricFrameError::NonFiniteValue { .. })
        ));
    }

    #[test]
    fn domain_refresh_preserves_other_domains_clock_and_allocation() {
        let surface = value_item(objects(2, 1)[0], 0, 0);
        let particle = value_item(objects(2, 1)[1], 0, 0);
        let decal = value_item(objects(2, 1)[4], 0, 0);
        let replacement = resource_item(objects(1, 2)[1], 64, 16);
        let mut frame = RenderFabricFrame::default();
        frame.replace(11, [surface, particle, decal]).unwrap();
        let allocation = frame.items.as_ptr();
        let capacity = frame.items.capacity();
        frame
            .replace_domain(RenderDomain::Particle, [replacement])
            .unwrap();
        assert_eq!(frame.frame(), 11);
        assert_eq!(frame.items(), &[surface, replacement, decal]);
        assert!(!frame.contains_object(particle.object_id));
        assert_eq!(frame.items.capacity(), capacity);
        assert_eq!(frame.items.as_ptr(), allocation);
        frame.replace_domain(RenderDomain::Particle, []).unwrap();
        assert_eq!(frame.items(), &[surface, decal]);
        assert_eq!(frame.frame(), 11);
        assert_eq!(frame.items.as_ptr(), allocation);
        frame
            .replace_domain(RenderDomain::Particle, [replacement])
            .unwrap();
        assert_eq!(frame.items(), &[surface, replacement, decal]);
    }

    #[test]
    fn rejected_domain_refresh_clears_all_domains_and_allows_recovery() {
        let surface = value_item(objects(2, 1)[0], 0, 0);
        let particle = value_item(objects(2, 1)[1], 0, 0);
        let mut frame = RenderFabricFrame::default();
        for (items, error) in [
            (
                alloc::vec![particle, surface],
                RenderFabricFrameError::DomainIdentityMismatch {
                    object: surface.object_id,
                    declared: RenderDomain::Particle,
                },
            ),
            (
                alloc::vec![particle, particle],
                RenderFabricFrameError::DuplicateItem {
                    object: particle.object_id,
                    domain: RenderDomain::Particle,
                    history_key: 0,
                    role: particle.payload_role(),
                },
            ),
            (
                alloc::vec![
                    particle,
                    with_value(particle, PayloadValue::Scalar(f32::NAN))
                ],
                RenderFabricFrameError::NonFiniteValue {
                    object: particle.object_id,
                    role: particle.payload_role(),
                },
            ),
        ] {
            frame.replace(11, [surface, particle]).unwrap();
            let allocation = frame.items.as_ptr();
            assert_eq!(
                frame.replace_domain(RenderDomain::Particle, items),
                Err(error)
            );
            assert!(frame.is_empty());
            assert_eq!(frame.frame(), 11);
            assert_eq!(frame.items.as_ptr(), allocation);
        }
        frame.replace(12, [surface, particle]).unwrap();
        assert_eq!(frame.len(), 2);
    }

    #[test]
    fn clear_and_replacements_reuse_allocation_and_discard_old_items() {
        let mut frame = RenderFabricFrame::default();
        frame
            .replace(
                1,
                (1..=32).map(|index| value_item(objects(index, 1)[0], 0, 0)),
            )
            .unwrap();
        let capacity = frame.items.capacity();
        let allocation = frame.items.as_ptr();
        let next = value_item(objects(40, 2)[1], 0, 0);
        frame.replace(2, core::iter::once(next)).unwrap();
        assert_eq!(frame.items(), &[next]);
        assert_eq!(frame.items.capacity(), capacity);
        assert_eq!(frame.items.as_ptr(), allocation);
        assert!(!frame.contains_object(objects(1, 1)[0]));
        frame.clear(3);
        assert_eq!(frame.frame(), 3);
        assert!(frame.is_empty());
        assert_eq!(frame.items.capacity(), capacity);
        assert_eq!(frame.items.as_ptr(), allocation);
        frame.replace(4, [next]).unwrap();
        frame.replace(5, core::iter::empty()).unwrap();
        assert_eq!(frame.frame(), 5);
        assert!(frame.is_empty());
        assert_eq!(frame.items.capacity(), capacity);
        assert_eq!(frame.items.as_ptr(), allocation);
    }

    #[test]
    fn sorting_is_canonical_across_domains_objects_histories_and_roles() {
        let mut expected = Vec::new();
        // Explicit expected order, independent of the implementation's key.
        let ordered_objects = [
            objects(1, 2)[0],
            objects(2, 0)[0],
            objects(2, 1)[0],
            objects(3, 0)[0],
            objects(2, 1)[1],
            objects(2, 1)[2],
            objects(2, 1)[3],
            objects(2, 1)[4],
        ];
        for object in ordered_objects {
            for history in [0, 7, u32::MAX] {
                for role in [0, 3, u16::MAX] {
                    expected.push(value_item(object, history, role));
                }
            }
        }
        let mut reversed = RenderFabricFrame::default();
        reversed
            .replace(17, expected.iter().rev().copied())
            .unwrap();
        assert_eq!(reversed.items(), expected.as_slice());
        assert_eq!(reversed.len(), expected.len());
        assert!(!reversed.is_empty());
        let mut rotated = RenderFabricFrame::default();
        let mut input = expected.clone();
        input.rotate_left(11);
        rotated.replace(17, input).unwrap();
        assert_eq!(rotated, reversed);
        assert!(reversed
            .items()
            .windows(2)
            .all(|pair| identity(&pair[0]) < identity(&pair[1])));
    }

    #[test]
    fn duplicate_identity_is_rejected_even_with_different_payload_or_motion() {
        let first = value_item(objects(1, 1)[0], 7, 3);
        let expected = RenderFabricFrameError::DuplicateItem {
            object: first.object_id,
            domain: first.domain,
            history_key: first.history_key,
            role: first.payload_role(),
        };
        for mut duplicate in [
            first,
            with_value(first, PayloadValue::Scalar(2.0)),
            FrameRenderItem {
                payload: resource_item(first.object_id, 4, 4).payload,
                ..first
            },
        ] {
            // Resource and value payloads share one role namespace.
            if let DomainPayload::Resource(ref mut payload) = duplicate.payload {
                payload.role = first.payload_role();
            }
            duplicate.motion_source = MotionSource::Particle;
            let old = value_item(objects(7, 1)[4], 0, 0);
            let separator = value_item(objects(2, 1)[0], 0, 0);
            let mut frame = RenderFabricFrame::default();
            frame.replace(20, [old, separator, first]).unwrap();
            let capacity = frame.items.capacity();
            let allocation = frame.items.as_ptr();
            assert_eq!(
                frame.replace(21, [first, separator, duplicate]),
                Err(expected)
            );
            assert_eq!(frame.frame(), 21);
            assert!(frame.is_empty());
            assert!(!frame.contains_object(old.object_id));
            assert!(!frame.contains_object(first.object_id));
            assert_eq!(frame.items.capacity(), capacity);
            assert_eq!(frame.items.as_ptr(), allocation);
        }
    }

    #[test]
    fn rejects_every_object_domain_mismatch() {
        for object in objects(1, 1) {
            for other in objects(1, 1) {
                if object.domain() == other.domain() {
                    continue;
                }
                let mut item = value_item(object, 0, 0);
                item.domain = other.domain();
                assert_rejected(
                    item,
                    RenderFabricFrameError::DomainIdentityMismatch {
                        object,
                        declared: other.domain(),
                    },
                );
            }
        }
    }

    #[test]
    fn zero_index_and_generation_particle_is_valid() {
        let object = RenderObjectId::Particle(DomainObjectHandle::new(0, 0));
        let item = value_item(object, 0, 0);
        let mut frame = RenderFabricFrame::default();
        frame.replace(0, [item]).unwrap();
        assert_eq!(frame.items(), &[item]);
        assert!(frame.contains_object(object));
        assert!(!frame.contains_object(RenderObjectId::Particle(DomainObjectHandle::new(0, 1))));
    }

    #[test]
    fn accepts_all_finite_value_shapes_including_degenerate_bounds() {
        let object = objects(1, 1)[0];
        let values = [
            PayloadValue::Scalar(-f32::MAX),
            PayloadValue::Scalar(-0.0),
            PayloadValue::Vector([f32::MIN, f32::MAX, f32::MIN_POSITIVE, 0.0]),
            PayloadValue::Bounds {
                min: [-f32::MAX, -1.0, 0.0],
                max: [f32::MAX, 1.0, 0.0],
            },
            PayloadValue::Bounds {
                min: [1.0; 3],
                max: [1.0; 3],
            },
        ];
        let mut frame = RenderFabricFrame::default();
        frame
            .replace(
                u64::MAX,
                values
                    .into_iter()
                    .enumerate()
                    .map(|(role, value)| with_value(value_item(object, 0, role as u16), value)),
            )
            .unwrap();
        assert_eq!(frame.len(), values.len());
        assert_eq!(frame.frame(), u64::MAX);
    }

    #[test]
    fn rejects_nonfinite_scalars() {
        let item = value_item(objects(1, 1)[0], 0, 4);
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_rejected(
                with_value(item, PayloadValue::Scalar(value)),
                RenderFabricFrameError::NonFiniteValue {
                    object: item.object_id,
                    role: item.payload_role(),
                },
            );
        }
    }

    #[test]
    fn rejects_nonfinite_values_in_every_vector_component() {
        let item = value_item(objects(1, 1)[1], 0, 4);
        for component in 0..4 {
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut vector = [0.0; 4];
                vector[component] = value;
                assert_rejected(
                    with_value(item, PayloadValue::Vector(vector)),
                    RenderFabricFrameError::NonFiniteValue {
                        object: item.object_id,
                        role: item.payload_role(),
                    },
                );
            }
        }
    }

    #[test]
    fn rejects_nonfinite_values_in_every_bounds_component() {
        let item = value_item(objects(1, 1)[2], 0, 4);
        for component in 0..3 {
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                for invalid_min in [false, true] {
                    let mut min = [-1.0; 3];
                    let mut max = [1.0; 3];
                    if invalid_min {
                        min[component] = value;
                    } else {
                        max[component] = value;
                    }
                    assert_rejected(
                        with_value(item, PayloadValue::Bounds { min, max }),
                        RenderFabricFrameError::NonFiniteValue {
                            object: item.object_id,
                            role: item.payload_role(),
                        },
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_inverted_bounds_on_every_axis() {
        let item = value_item(objects(1, 1)[3], 0, 4);
        for axis in 0..3 {
            let min = [-1.0; 3];
            let mut max = [1.0; 3];
            max[axis] = -2.0;
            assert_rejected(
                with_value(item, PayloadValue::Bounds { min, max }),
                RenderFabricFrameError::InvertedBounds {
                    object: item.object_id,
                    role: item.payload_role(),
                },
            );
        }
    }

    #[test]
    fn resource_size_must_be_nonzero() {
        let item = resource_item(objects(1, 1)[4], 0, 4);
        assert_rejected(
            item,
            RenderFabricFrameError::ZeroResourceSize {
                object: item.object_id,
                role: item.payload_role(),
            },
        );
    }

    #[test]
    fn resource_alignment_must_be_a_nonzero_power_of_two() {
        for alignment in [0, 3, 6, 12, u64::MAX] {
            let item = resource_item(objects(1, 1)[4], 4, alignment);
            assert_rejected(
                item,
                RenderFabricFrameError::InvalidResourceAlignment {
                    object: item.object_id,
                    role: item.payload_role(),
                    alignment,
                },
            );
        }
        for shift in 0..u64::BITS {
            let item = resource_item(objects(1, 1)[4], 1, 1u64 << shift);
            let mut frame = RenderFabricFrame::default();
            frame.replace(1, [item]).unwrap();
            assert_eq!(frame.items(), &[item]);
        }
    }

    #[test]
    fn resource_layout_and_binding_checks_are_left_to_compile_and_runtime() {
        let mut item = resource_item(objects(1, 1)[1], u64::MAX, 1);
        if let DomainPayload::Resource(ref mut payload) = item.payload {
            // Intentionally inconsistent metadata: frame storage must not
            // duplicate template/layout validation or require physical imports.
            payload.layout = ResourceLayout::Buffer {
                bytes: 0,
                usage: BufferUsages::NONE,
            };
        }
        let mut frame = RenderFabricFrame::default();
        frame.replace(1, [item]).unwrap();
        assert_eq!(frame.items(), &[item]);
    }

    #[test]
    fn contains_object_handles_many_roles_domains_and_generations() {
        let mut frame = RenderFabricFrame::default();
        let mut items = Vec::new();
        for index in [2, 4, 6] {
            for object in objects(index, 3) {
                for history in [0, 2] {
                    for role in 0..4 {
                        items.push(value_item(object, history, role));
                    }
                }
            }
        }
        frame.replace(3, items.into_iter().rev()).unwrap();
        for index in 0..=8 {
            for object in objects(index, 3) {
                assert_eq!(frame.contains_object(object), [2, 4, 6].contains(&index));
            }
            for object in objects(index, 4) {
                assert!(!frame.contains_object(object));
            }
        }
        frame.clear(4);
        assert!(!frame.contains_object(objects(2, 3)[0]));
    }

    #[test]
    fn validation_error_implements_standard_error_and_display() {
        fn assert_error<T: std::error::Error>() {}
        assert_error::<RenderFabricFrameError>();
        let error = RenderFabricFrameError::NonFiniteValue {
            object: objects(1, 1)[0],
            role: PayloadRole(4),
        };
        assert!(alloc::format!("{error}").contains("NonFiniteValue"));
    }
}
