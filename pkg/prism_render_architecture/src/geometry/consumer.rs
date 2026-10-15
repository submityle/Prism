//! One consumer contract for every runtime mesh representation.
//!
//! Visibility, depth, shadow, motion-vector, ray-tracing, collision, and
//! derived-geometry code must select a representation through this module.
//! The contract is deliberately GPU-independent: it carries stable geometry
//! identity and deformation/page identity, while a backend resolves those
//! identities to buffers, page slots, or acceleration-structure inputs.

use super::GeometryHandle;
use crate::deformation::DeformationHandle;
use crate::virtual_geometry::GeometryPageKey;

/// Version of the cross-consumer mesh contract.
pub const MESH_CONSUMER_CONTRACT_VERSION: u32 = 1;

/// A consumer that reads mesh geometry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MeshConsumer {
    /// The primary surface/visibility path.
    MainView,
    /// Depth-only prepass and HZB input.
    Depth,
    /// Shadow-map or virtual-shadow caster path.
    Shadow,
    /// Current/previous geometry used to generate motion vectors.
    MotionVector,
    /// BLAS build/refit or software ray-scene input.
    RayTracing,
    /// Collision shape or collision-triangle source.
    Collision,
    /// Any other cooked geometry product, such as SDF or navigation input.
    DerivedGeometry,
}

/// Temporal side of a mesh representation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MeshTemporal {
    /// The geometry produced for the current frame.
    Current,
    /// The geometry retained from the previous frame.
    Previous,
}

/// The representation selected by a mesh consumer.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MeshRepresentation {
    /// A resident standard/LOD draw representation.
    StaticLod { level: u32 },
    /// A resident deformed output owned by a deformation cache.
    Deformed {
        handle: DeformationHandle,
        frame: u64,
    },
    /// A cluster payload selected from a virtual-geometry page.
    VirtualCluster { page: GeometryPageKey, cluster: u32 },
}

/// A versioned build product consumed by collision, RT, SDF, or other derived
/// geometry code. It is intentionally not interchangeable with a render LOD.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MeshDerivedArtifact {
    /// Stable artifact identity from the build DAG.
    pub key: [u8; 32],
    /// Explicit artifact kind protocol ID.
    pub kind: u128,
    /// Explicit artifact schema.
    pub schema: u32,
}

/// All representations that belong to one geometry revision.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MeshConsumerBinding {
    /// Stable geometry handle shared by every consumer.
    pub geometry: GeometryHandle,
    /// Revision that all fields below were built from.
    pub revision: u32,
    /// Current-frame render representation.
    pub current: MeshRepresentation,
    /// Previous-frame render representation, when temporal history exists.
    pub previous: MeshRepresentation,
    /// Whether `previous` is a valid temporal predecessor of `current`.
    pub history_valid: bool,
    /// Explicit fallback used when the preferred representation is unavailable.
    pub fallback: MeshRepresentation,
    /// Cooked products used by non-render consumers.
    pub derived: Option<MeshDerivedArtifact>,
}

/// The resolved input for one consumer.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MeshConsumerSelection {
    /// One render representation.
    Render {
        geometry: GeometryHandle,
        representation: MeshRepresentation,
        used_fallback: bool,
    },
    /// A current/previous pair. `history_valid == false` means the consumer
    /// must treat the pair as a reset and emit zero velocity for this object.
    Motion {
        geometry: GeometryHandle,
        current: MeshRepresentation,
        previous: MeshRepresentation,
        history_valid: bool,
        used_fallback: bool,
    },
    /// A build artifact for collision or another derived consumer.
    Derived {
        geometry: GeometryHandle,
        artifact: MeshDerivedArtifact,
    },
}

/// Validation failures that would otherwise create mismatched pass inputs.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MeshConsumerBindingError {
    /// The geometry handle is the reserved invalid value.
    InvalidGeometry,
    /// Revisions are zero-based only for uninitialized records, never bindings.
    InvalidRevision,
    /// A virtual cluster cannot be used by a cooked-geometry consumer.
    VirtualClusterForDerivedConsumer,
    /// A derived consumer has no published artifact.
    MissingDerivedArtifact,
    /// An artifact schema of zero cannot be interpreted.
    InvalidDerivedSchema,
}

impl MeshConsumerBinding {
    /// Validates the identity and representation invariants.
    pub fn validate(self) -> Result<(), MeshConsumerBindingError> {
        if !self.geometry.is_valid() {
            return Err(MeshConsumerBindingError::InvalidGeometry);
        }
        if self.revision == 0 {
            return Err(MeshConsumerBindingError::InvalidRevision);
        }
        if let Some(artifact) = self.derived {
            if artifact.schema == 0 {
                return Err(MeshConsumerBindingError::InvalidDerivedSchema);
            }
        }
        Ok(())
    }

    /// Resolves a consumer without letting it silently choose another source.
    pub fn select(
        self,
        consumer: MeshConsumer,
    ) -> Result<MeshConsumerSelection, MeshConsumerBindingError> {
        self.validate()?;
        match consumer {
            MeshConsumer::MotionVector => Ok(MeshConsumerSelection::Motion {
                geometry: self.geometry,
                current: self.current,
                previous: if self.history_valid {
                    self.previous
                } else {
                    self.current
                },
                history_valid: self.history_valid,
                used_fallback: false,
            }),
            MeshConsumer::Collision | MeshConsumer::DerivedGeometry => {
                let artifact = self
                    .derived
                    .ok_or(MeshConsumerBindingError::MissingDerivedArtifact)?;
                if matches!(self.current, MeshRepresentation::VirtualCluster { .. }) {
                    return Err(MeshConsumerBindingError::VirtualClusterForDerivedConsumer);
                }
                Ok(MeshConsumerSelection::Derived {
                    geometry: self.geometry,
                    artifact,
                })
            }
            MeshConsumer::MainView
            | MeshConsumer::Depth
            | MeshConsumer::Shadow
            | MeshConsumer::RayTracing => Ok(MeshConsumerSelection::Render {
                geometry: self.geometry,
                representation: self.current,
                used_fallback: false,
            }),
        }
    }

    /// Resolves a render consumer with an explicit fallback representation.
    /// This is used by page residency code after it has determined that the
    /// preferred virtual page is unavailable.
    pub fn select_with_fallback(
        self,
        consumer: MeshConsumer,
        preferred_resident: bool,
    ) -> Result<MeshConsumerSelection, MeshConsumerBindingError> {
        if preferred_resident {
            return self.select(consumer);
        }
        match consumer {
            MeshConsumer::MainView
            | MeshConsumer::Depth
            | MeshConsumer::Shadow
            | MeshConsumer::RayTracing => {
                self.validate()?;
                Ok(MeshConsumerSelection::Render {
                    geometry: self.geometry,
                    representation: self.fallback,
                    used_fallback: true,
                })
            }
            MeshConsumer::MotionVector
            | MeshConsumer::Collision
            | MeshConsumer::DerivedGeometry => self.select(consumer),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::GenerationalHandle;

    fn binding(history_valid: bool, derived: bool) -> MeshConsumerBinding {
        MeshConsumerBinding {
            geometry: GenerationalHandle::new(4, 2),
            revision: 7,
            current: MeshRepresentation::Deformed {
                handle: DeformationHandle(9),
                frame: 20,
            },
            previous: MeshRepresentation::Deformed {
                handle: DeformationHandle(9),
                frame: 19,
            },
            history_valid,
            fallback: MeshRepresentation::StaticLod { level: 3 },
            derived: derived.then_some(MeshDerivedArtifact {
                key: [5; 32],
                kind: 11,
                schema: 1,
            }),
        }
    }

    #[test]
    fn every_render_consumer_uses_the_same_current_representation() {
        let binding = binding(true, true);
        for consumer in [
            MeshConsumer::MainView,
            MeshConsumer::Depth,
            MeshConsumer::Shadow,
            MeshConsumer::RayTracing,
        ] {
            let MeshConsumerSelection::Render { representation, .. } =
                binding.select(consumer).unwrap()
            else {
                panic!("render consumer must resolve to one representation");
            };
            assert_eq!(representation, binding.current);
        }
    }

    #[test]
    fn invalid_history_resets_motion_without_faking_validity() {
        let MeshConsumerSelection::Motion {
            current,
            previous,
            history_valid,
            ..
        } = binding(false, true)
            .select(MeshConsumer::MotionVector)
            .unwrap()
        else {
            panic!("motion consumer must resolve to a pair");
        };
        assert_eq!(current, previous);
        assert!(!history_valid);
    }

    #[test]
    fn missing_or_invalid_derived_artifacts_are_rejected() {
        assert_eq!(
            binding(true, false).select(MeshConsumer::Collision),
            Err(MeshConsumerBindingError::MissingDerivedArtifact)
        );
        let mut invalid = binding(true, true);
        invalid.derived = Some(MeshDerivedArtifact {
            schema: 0,
            ..invalid.derived.unwrap()
        });
        assert_eq!(
            invalid.select(MeshConsumer::DerivedGeometry),
            Err(MeshConsumerBindingError::InvalidDerivedSchema)
        );
    }

    #[test]
    fn page_miss_uses_only_the_declared_fallback() {
        let binding = MeshConsumerBinding {
            current: MeshRepresentation::VirtualCluster {
                page: GeometryPageKey::new(4, 9),
                cluster: 2,
            },
            ..binding(true, true)
        };
        let MeshConsumerSelection::Render {
            representation,
            used_fallback,
            ..
        } = binding
            .select_with_fallback(MeshConsumer::Shadow, false)
            .unwrap()
        else {
            panic!("shadow fallback must remain a render selection");
        };
        assert_eq!(representation, binding.fallback);
        assert!(used_fallback);
    }
}
