//! Backend-neutral history publication and import validation.
//!
//! The external renderer owns allocation, submission and completion. This
//! module deliberately records only a binding that has *already completed*;
//! it does not manufacture a queue fence or treat command recording as
//! completion.

use super::*;
use crate::frame_graph::ResourceLifetime;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt;

/// Completion value supplied by the external queue/fence owner after the
/// submission that produced a history binding has completed.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct HistoryCompletion(pub u64);

/// A completed persistent binding eligible for an exact later-frame import.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HistoryPublication {
    binding: FrameResourceBinding,
    completion: HistoryCompletion,
}

impl HistoryPublication {
    pub const fn binding(self) -> FrameResourceBinding {
        self.binding
    }

    pub const fn completion(self) -> HistoryCompletion {
        self.completion
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HistoryError {
    NotPersistent(ResourceKey),
    InvalidBinding {
        resource: ResourceKey,
        frame: u64,
    },
    DuplicatePublication {
        resource: ResourceKey,
        frame: u64,
    },
    OutOfOrderPublication {
        resource: ResourceKey,
        previous: u64,
        next: u64,
    },
    MissingPublication {
        resource: ResourceKey,
        frame: u64,
    },
    BindingMismatch {
        resource: ResourceKey,
        frame: u64,
    },
}

impl fmt::Display for HistoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "render fabric history error: {self:?}")
    }
}

impl std::error::Error for HistoryError {}

/// Per-resource completed-history ledger.
///
/// Entries are exact `(ResourceKey, frame)` publications. A read never falls
/// back to an older entry, and callers can retire entries only after proving no
/// future compiled graph can import them.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ResourceHistory {
    publications: BTreeMap<(ResourceKey, u64), HistoryPublication>,
    latest: BTreeMap<ResourceKey, u64>,
}

impl ResourceHistory {
    /// Publish a binding after its producing submission has completed. The
    /// binding must be persistent and match its declared layout/alignment.
    pub fn publish(
        &mut self,
        template: ResourceTemplate,
        binding: FrameResourceBinding,
        completion: HistoryCompletion,
    ) -> Result<(), HistoryError> {
        if template.lifetime != ResourceLifetime::Persistent {
            return Err(HistoryError::NotPersistent(template.key));
        }
        if binding.key != template.key || !binding.matches(template) {
            return Err(HistoryError::InvalidBinding {
                resource: binding.key,
                frame: binding.frame,
            });
        }
        let address = (binding.key, binding.frame);
        if self.publications.contains_key(&address) {
            return Err(HistoryError::DuplicatePublication {
                resource: binding.key,
                frame: binding.frame,
            });
        }
        if let Some(&previous) = self.latest.get(&binding.key)
            && binding.frame <= previous
        {
            return Err(HistoryError::OutOfOrderPublication {
                resource: binding.key,
                previous,
                next: binding.frame,
            });
        }
        self.publications.insert(
            address,
            HistoryPublication {
                binding,
                completion,
            },
        );
        self.latest.insert(binding.key, binding.frame);
        Ok(())
    }

    /// Validates an exact resource/frame import against a completed
    /// publication. Equality covers physical storage, layout, alignment and
    /// content version, so stale aliases cannot cross a `FrameLag` edge.
    pub fn validate_import(
        &self,
        binding: FrameResourceBinding,
    ) -> Result<HistoryPublication, HistoryError> {
        let publication = self
            .publications
            .get(&(binding.key, binding.frame))
            .copied()
            .ok_or(HistoryError::MissingPublication {
                resource: binding.key,
                frame: binding.frame,
            })?;
        if publication.binding != binding {
            return Err(HistoryError::BindingMismatch {
                resource: binding.key,
                frame: binding.frame,
            });
        }
        Ok(publication)
    }

    /// Drop entries strictly older than `first_live_frame` for one resource.
    /// This is an explicit external-lifetime handoff, never an automatic guess.
    pub fn retire_before(&mut self, resource: ResourceKey, first_live_frame: u64) {
        // Bound work to this resource's retired prefix, not every resource in
        // every frame still retained by the renderer.
        let retired: Vec<_> = self
            .publications
            .range((resource, 0)..(resource, first_live_frame))
            .map(|(&address, _)| address)
            .collect();
        for address in retired {
            self.publications.remove(&address);
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LagOneHistory {
    published: Option<u64>,
}

impl LagOneHistory {
    /// Missing frames never fall back to an older, semantically wrong version.
    pub fn read_slot(&self, frame: u64) -> Option<usize> {
        let previous = frame.checked_sub(1)?;
        (self.published == Some(previous)).then_some(Self::write_slot(previous))
    }

    pub const fn write_slot(frame: u64) -> usize {
        (frame & 1) as usize
    }

    pub fn can_publish(&self, frame: u64) -> bool {
        self.published.is_none_or(|last| frame > last)
    }

    /// This is only a frame-slot helper. Use [`ResourceHistory`] when a resource
    /// binding crosses a real `FrameLag` edge.
    pub fn publish(&mut self, frame: u64) -> bool {
        if !self.can_publish(frame) {
            return false;
        }
        self.published = Some(frame);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_graph::ResourceLifetime;

    const FEATURE: FeatureId = FeatureId(1);
    const RESOURCE: ResourceKey = ResourceKey::of(FEATURE, 0);

    fn template(lifetime: ResourceLifetime) -> ResourceTemplate {
        ResourceTemplate {
            key: RESOURCE,
            lifetime,
            layout: ResourceLayout::Buffer {
                bytes: 64,
                usage: BufferUsages::STORAGE,
            },
            alignment: 16,
        }
    }

    fn binding(frame: u64, version: u32, id: u32) -> FrameResourceBinding {
        let template = template(ResourceLifetime::Persistent);
        FrameResourceBinding {
            key: RESOURCE,
            frame,
            version,
            physical: PhysicalResource::Buffer {
                id: BufferId::from_parts(id, 1),
                offset: 0,
            },
            layout: template.layout,
            alignment: template.alignment,
        }
    }

    #[test]
    fn lag_one_requires_exact_publication_and_distinct_current_slot() {
        let mut history = LagOneHistory::default();
        assert_eq!(history.read_slot(0), None);
        assert_eq!(history.read_slot(1), None);
        assert!(history.publish(1));
        assert_eq!(history.read_slot(2), Some(1));
        assert_ne!(history.read_slot(2).unwrap(), LagOneHistory::write_slot(2));
        assert!(!history.publish(1));
        assert!(!history.publish(0));
        assert_eq!(history.read_slot(3), None);
        assert!(history.publish(4));
        assert_eq!(history.read_slot(5), Some(0));
    }

    #[test]
    fn resource_history_requires_completed_exact_binding_and_explicit_retirement() {
        let persistent = template(ResourceLifetime::Persistent);
        let mut history = ResourceHistory::default();
        let frame_4 = binding(4, 7, 10);
        history
            .publish(persistent, frame_4, HistoryCompletion(41))
            .unwrap();
        assert_eq!(
            history.validate_import(frame_4).unwrap().completion(),
            HistoryCompletion(41)
        );

        let mut stale_version = frame_4;
        stale_version.version += 1;
        assert_eq!(
            history.validate_import(stale_version),
            Err(HistoryError::BindingMismatch {
                resource: RESOURCE,
                frame: 4,
            })
        );
        assert_eq!(
            history.publish(persistent, frame_4, HistoryCompletion(42)),
            Err(HistoryError::DuplicatePublication {
                resource: RESOURCE,
                frame: 4,
            })
        );
        assert_eq!(
            history.publish(persistent, binding(3, 8, 11), HistoryCompletion(43)),
            Err(HistoryError::OutOfOrderPublication {
                resource: RESOURCE,
                previous: 4,
                next: 3,
            })
        );

        history.retire_before(RESOURCE, 5);
        assert_eq!(
            history.validate_import(frame_4),
            Err(HistoryError::MissingPublication {
                resource: RESOURCE,
                frame: 4,
            })
        );
        assert_eq!(
            history.publish(
                template(ResourceLifetime::Transient),
                binding(5, 9, 12),
                HistoryCompletion(44)
            ),
            Err(HistoryError::NotPersistent(RESOURCE))
        );
    }

    #[test]
    fn retirement_preserves_other_resources_and_publication_watermark() {
        let template = template(ResourceLifetime::Persistent);
        let other_template = ResourceTemplate {
            key: ResourceKey::of(FEATURE, 1),
            ..template
        };
        let mut history = ResourceHistory::default();
        let current = binding(4, 7, 10);
        let other = FrameResourceBinding {
            key: other_template.key,
            ..binding(3, 2, 11)
        };
        history
            .publish(template, current, HistoryCompletion(41))
            .unwrap();
        history
            .publish(other_template, other, HistoryCompletion(40))
            .unwrap();
        history.retire_before(RESOURCE, 4);
        assert!(history.validate_import(current).is_ok());
        history.retire_before(RESOURCE, 5);
        assert!(history.validate_import(other).is_ok());
        assert!(matches!(
            history.publish(template, current, HistoryCompletion(42)),
            Err(HistoryError::OutOfOrderPublication {
                previous: 4,
                next: 4,
                ..
            })
        ));
        history
            .publish(template, binding(5, 8, 12), HistoryCompletion(43))
            .unwrap();
    }

    #[test]
    fn import_checks_exact_physical_storage_and_metadata() {
        let template = template(ResourceLifetime::Persistent);
        let mut history = ResourceHistory::default();
        let current = binding(4, 7, 10);
        history
            .publish(template, current, HistoryCompletion(41))
            .unwrap();
        let other_allocation = FrameResourceBinding {
            physical: binding(4, 7, 11).physical,
            ..current
        };
        let other_alignment = FrameResourceBinding {
            alignment: 32,
            ..current
        };
        let other_layout = FrameResourceBinding {
            layout: ResourceLayout::Buffer {
                bytes: 128,
                usage: BufferUsages::STORAGE,
            },
            ..current
        };
        for changed in [other_allocation, other_alignment, other_layout] {
            assert_eq!(
                history.validate_import(changed),
                Err(HistoryError::BindingMismatch {
                    resource: RESOURCE,
                    frame: 4
                })
            );
        }
        assert_eq!(
            history.publish(template, other_layout, HistoryCompletion(42)),
            Err(HistoryError::InvalidBinding {
                resource: RESOURCE,
                frame: 4
            })
        );
    }
}
