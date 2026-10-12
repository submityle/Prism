//! Pure StateTree host contracts for schemas and read-only contexts.
//!
//! This module deliberately stops at the core boundary.  It does not know
//! about an ECS world, entities, reflection paths, or a scheduler.  An
//! integration crate owns context gathering and can implement [`Schema`] with
//! its own borrowed context type.

/// Stable identity of a StateTree host schema.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SchemaId(pub u128);

impl SchemaId {
    /// The reserved empty identity.  A schema descriptor may not use it.
    pub const NIL: Self = Self(0);

    /// Creates a schema identity from a stable 128-bit value.
    #[must_use]
    pub const fn new(value: u128) -> Self {
        Self(value)
    }

    /// Returns the raw stable identity.
    #[must_use]
    pub const fn get(self) -> u128 {
        self.0
    }
}

/// Backward-compatible schema contract version.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SchemaVersion {
    /// Breaking contract version.
    pub major: u16,
    /// Additive contract version within a major version.
    pub minor: u16,
}

impl SchemaVersion {
    /// Creates a schema contract version.
    #[must_use]
    pub const fn new(major: u16, minor: u16) -> Self {
        Self { major, minor }
    }
}

/// Host capabilities declared by a schema.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct SchemaCapabilities(u32);

impl SchemaCapabilities {
    /// No optional host capabilities.
    pub const NONE: Self = Self(0);
    /// The host can apply schema-defined tag gates.
    pub const TAG_GATING: Self = Self(1 << 0);
    /// The host supplies replication-policy metadata.
    pub const REPLICATION: Self = Self(1 << 1);
    /// The host exposes fields suitable for debugger inspection.
    pub const DEBUG_FIELDS: Self = Self(1 << 2);
    /// The host can provide compiled typed bindings.
    pub const TYPED_BINDING: Self = Self(1 << 3);

    const KNOWN_BITS: u32 =
        Self::TAG_GATING.0 | Self::REPLICATION.0 | Self::DEBUG_FIELDS.0 | Self::TYPED_BINDING.0;

    /// Combines capability flags.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns whether all flags in `required` are present.
    #[must_use]
    pub const fn contains(self, required: Self) -> bool {
        self.0 & required.0 == required.0
    }

    /// Returns the raw capability bitset.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    const fn unknown_bits(self) -> u32 {
        self.0 & !Self::KNOWN_BITS
    }
}

/// Stable schema metadata shared by compiler, runtime, and tooling layers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SchemaDescriptor {
    /// Stable schema identity.
    pub id: SchemaId,
    /// Compatibility version of the schema contract.
    pub version: SchemaVersion,
    /// Optional host features exposed by this schema.
    pub capabilities: SchemaCapabilities,
}

impl SchemaDescriptor {
    /// Creates a schema descriptor.
    #[must_use]
    pub const fn new(
        id: SchemaId,
        version: SchemaVersion,
        capabilities: SchemaCapabilities,
    ) -> Self {
        Self {
            id,
            version,
            capabilities,
        }
    }

    /// Checks invariants that can be enforced without a host world.
    pub const fn validate(self) -> Result<(), SchemaError> {
        if self.id.0 == SchemaId::NIL.0 {
            return Err(SchemaError::EmptySchemaId);
        }
        let unknown_bits = self.capabilities.unknown_bits();
        if unknown_bits != 0 {
            return Err(SchemaError::UnknownCapabilities { unknown_bits });
        }
        Ok(())
    }
}

/// A structured error at the schema/context boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SchemaError {
    /// The descriptor used the reserved [`SchemaId::NIL`] identity.
    EmptySchemaId,
    /// The descriptor contains capability bits unknown to this core version.
    UnknownCapabilities {
        /// The unsupported bits.
        unknown_bits: u32,
    },
    /// A context was produced for another schema identity.
    ContextSchemaMismatch {
        /// The schema requested by the runtime.
        expected: SchemaId,
        /// The schema identity reported by the context.
        found: SchemaId,
    },
}

/// Read-only view borrowed from a host integration.
///
/// The core only receives a shared reference to this trait and exposes no
/// mutation or ownership operation.  Concrete integrations may add typed
/// accessors on their own context type; the core does not reinterpret or
/// discover those fields.
pub trait ContextRef<'w>: 'w {
    /// Returns the schema identity this view belongs to.
    fn schema_id(&self) -> SchemaId;
}

/// Host schema contract for a StateTree plan.
///
/// `Context` is a borrowed, integration-owned view.  Gathering it from an ECS
/// world is intentionally outside this crate; an ECS adapter should perform
/// that operation and then call [`Schema::validate_context`] before execution.
pub trait Schema: Send + Sync + 'static {
    /// Borrowed read-only context supplied by the host integration.
    type Context<'w>: ContextRef<'w>
    where
        Self: 'w;

    /// Returns stable schema metadata.
    fn descriptor() -> SchemaDescriptor;

    /// Validates static descriptor and context identity invariants.
    fn validate_context<'w>(context: &Self::Context<'w>) -> Result<(), SchemaError>
    where
        Self: 'w,
    {
        let descriptor = Self::descriptor();
        descriptor.validate()?;
        let found = context.schema_id();
        if found != descriptor.id {
            return Err(SchemaError::ContextSchemaMismatch {
                expected: descriptor.id,
                found,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ContextRef, Schema, SchemaCapabilities, SchemaDescriptor, SchemaError, SchemaId,
        SchemaVersion,
    };

    const TEST_ID: SchemaId = SchemaId::new(0xfeed);

    struct TestContext<'w> {
        schema: SchemaId,
        value: &'w u32,
    }

    impl<'w> ContextRef<'w> for TestContext<'w> {
        fn schema_id(&self) -> SchemaId {
            self.schema
        }
    }

    struct TestSchema;

    impl Schema for TestSchema {
        type Context<'w> = TestContext<'w>;

        fn descriptor() -> SchemaDescriptor {
            SchemaDescriptor::new(
                TEST_ID,
                SchemaVersion::new(1, 2),
                SchemaCapabilities::TAG_GATING.union(SchemaCapabilities::DEBUG_FIELDS),
            )
        }
    }

    #[test]
    fn descriptor_is_stable_and_capabilities_are_queryable() {
        let descriptor = TestSchema::descriptor();
        assert_eq!(descriptor.id, TEST_ID);
        assert_eq!(descriptor.version, SchemaVersion::new(1, 2));
        assert!(descriptor
            .capabilities
            .contains(SchemaCapabilities::TAG_GATING));
        assert!(!descriptor
            .capabilities
            .contains(SchemaCapabilities::REPLICATION));
        assert_eq!(descriptor.validate(), Ok(()));
    }

    #[test]
    fn context_identity_is_checked_without_gathering_a_world() {
        let value = 7;
        let context = TestContext {
            schema: TEST_ID,
            value: &value,
        };
        assert_eq!(TestSchema::validate_context(&context), Ok(()));

        let other = TestContext {
            schema: SchemaId::new(8),
            value: &value,
        };
        assert_eq!(
            TestSchema::validate_context(&other),
            Err(SchemaError::ContextSchemaMismatch {
                expected: TEST_ID,
                found: SchemaId::new(8),
            })
        );
        assert_eq!(*context.value, 7);
    }

    #[test]
    fn invalid_descriptor_is_rejected_structurally() {
        assert_eq!(
            SchemaDescriptor::new(
                SchemaId::NIL,
                SchemaVersion::default(),
                SchemaCapabilities::NONE,
            )
            .validate(),
            Err(SchemaError::EmptySchemaId)
        );
        assert_eq!(
            SchemaDescriptor::new(
                TEST_ID,
                SchemaVersion::default(),
                SchemaCapabilities::new_for_test(1 << 31),
            )
            .validate(),
            Err(SchemaError::UnknownCapabilities {
                unknown_bits: 1 << 31,
            })
        );
    }

    impl SchemaCapabilities {
        const fn new_for_test(bits: u32) -> Self {
            Self(bits)
        }
    }
}
