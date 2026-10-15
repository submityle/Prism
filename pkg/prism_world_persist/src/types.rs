use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt;

/// Stable identity of an object across sessions, materialization, and replay.
#[derive(Clone, Copy, Default, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct PersistentId(pub u128);

impl fmt::Debug for PersistentId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("PersistentId")
            .field(&self.0)
            .finish()
    }
}

/// Stable identity of an author or replica participating in a log.
#[derive(Clone, Copy, Default, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct ActorId(pub u128);

impl fmt::Debug for ActorId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("ActorId").field(&self.0).finish()
    }
}

/// Stable identifier of a field in an object's schema.
#[derive(Clone, Copy, Default, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct FieldId(pub u64);

impl fmt::Debug for FieldId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("FieldId").field(&self.0).finish()
    }
}

/// Globally unique identifier of one immutable transaction.
#[derive(Clone, Copy, Default, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct TransactionId(pub u128);

impl fmt::Debug for TransactionId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("TransactionId")
            .field(&self.0)
            .finish()
    }
}

/// Monotonic logical time used by the LWW register ordering.
#[derive(Clone, Copy, Default, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct LogicalClock(pub u64);

impl fmt::Debug for LogicalClock {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("LogicalClock")
            .field(&self.0)
            .finish()
    }
}

/// A scalar or deletion value stored by an override delta.
#[derive(Clone)]
pub enum OverrideValue {
    /// A signed integer value.
    Integer(i64),
    /// An unsigned integer value.
    Unsigned(u64),
    /// A finite IEEE-754 double-precision value.
    Float(f64),
    /// A boolean value.
    Boolean(bool),
    /// An opaque byte value.
    Bytes(Vec<u8>),
    /// Removes the field from the materialized state.
    Delete,
}

impl OverrideValue {
    /// Constructs a finite floating-point override.
    pub fn float(value: f64) -> Result<Self, ValueError> {
        if value.is_finite() {
            Ok(Self::Float(value))
        } else {
            Err(ValueError::NonFiniteFloat)
        }
    }

    /// Validates values that cannot be represented deterministically.
    pub fn validate(&self) -> Result<(), ValueError> {
        if matches!(self, Self::Float(value) if !value.is_finite()) {
            Err(ValueError::NonFiniteFloat)
        } else {
            Ok(())
        }
    }

    /// Returns the stable wire tag used by the binary codec.
    pub(crate) fn tag(&self) -> u8 {
        match self {
            Self::Integer(_) => 1,
            Self::Unsigned(_) => 2,
            Self::Float(_) => 3,
            Self::Boolean(_) => 4,
            Self::Bytes(_) => 5,
            Self::Delete => 6,
        }
    }
}

impl PartialEq for OverrideValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Integer(a), Self::Integer(b)) => a == b,
            (Self::Unsigned(a), Self::Unsigned(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => a.to_bits() == b.to_bits(),
            (Self::Boolean(a), Self::Boolean(b)) => a == b,
            (Self::Bytes(a), Self::Bytes(b)) => a == b,
            (Self::Delete, Self::Delete) => true,
            _ => false,
        }
    }
}

impl Eq for OverrideValue {}

impl fmt::Debug for OverrideValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Integer(value) => formatter.debug_tuple("Integer").field(value).finish(),
            Self::Unsigned(value) => formatter.debug_tuple("Unsigned").field(value).finish(),
            Self::Float(value) => formatter.debug_tuple("Float").field(value).finish(),
            Self::Boolean(value) => formatter.debug_tuple("Boolean").field(value).finish(),
            Self::Bytes(value) => formatter.debug_tuple("Bytes").field(value).finish(),
            Self::Delete => formatter.write_str("Delete"),
        }
    }
}

/// A single field update inside an [`OverrideDelta`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FieldOverride {
    /// Stable field identity.
    pub field: FieldId,
    /// New value, or [`OverrideValue::Delete`].
    pub value: OverrideValue,
}

/// Validation failures for one override value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueError {
    /// NaN and infinities are rejected because they do not form a portable
    /// deterministic numeric value.
    NonFiniteFloat,
}

/// Validation failures for a field delta.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeltaError {
    /// A field occurred more than once in one canonical delta.
    DuplicateField(FieldId),
    /// A value failed deterministic validation.
    InvalidValue(ValueError),
    /// A transaction inverse did not cover exactly the same fields.
    InverseFieldMismatch,
}

/// Canonically ordered field changes for one object.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OverrideDelta {
    changes: Vec<FieldOverride>,
}

impl OverrideDelta {
    /// Creates a delta, sorting fields by [`FieldId`] and rejecting duplicates.
    pub fn new(mut changes: Vec<FieldOverride>) -> Result<Self, DeltaError> {
        changes.sort_unstable_by_key(|change| change.field);
        for pair in changes.windows(2) {
            if pair[0].field == pair[1].field {
                return Err(DeltaError::DuplicateField(pair[0].field));
            }
        }
        for change in &changes {
            change.value.validate().map_err(DeltaError::InvalidValue)?;
        }
        Ok(Self { changes })
    }

    /// Creates an empty delta.
    pub const fn empty() -> Self {
        Self {
            changes: Vec::new(),
        }
    }

    /// Creates a one-field delta.
    pub fn single(field: FieldId, value: OverrideValue) -> Result<Self, DeltaError> {
        Self::new(alloc::vec![FieldOverride { field, value }])
    }

    /// Returns the number of fields in this delta.
    pub const fn len(&self) -> usize {
        self.changes.len()
    }

    /// Returns true when this delta has no fields.
    pub const fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    /// Returns canonical field changes in ascending field order.
    pub fn iter(&self) -> core::slice::Iter<'_, FieldOverride> {
        self.changes.iter()
    }

    /// Returns the canonical field changes as a slice.
    pub fn as_slice(&self) -> &[FieldOverride] {
        &self.changes
    }

    /// Finds one field change.
    pub fn get(&self, field: FieldId) -> Option<&OverrideValue> {
        self.changes
            .binary_search_by_key(&field, |change| change.field)
            .ok()
            .and_then(|index| self.changes.get(index))
            .map(|change| &change.value)
    }

    pub(crate) fn same_fields(&self, other: &Self) -> bool {
        self.changes
            .iter()
            .map(|change| change.field)
            .eq(other.changes.iter().map(|change| change.field))
    }
}

/// Immutable transaction containing forward and reverse field changes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Transaction {
    id: TransactionId,
    actor: ActorId,
    clock: LogicalClock,
    sequence: u64,
    object: PersistentId,
    delta: OverrideDelta,
    inverse: OverrideDelta,
}

impl Transaction {
    /// Constructs a transaction. `sequence` must be the next log sequence.
    pub fn new(
        id: TransactionId,
        actor: ActorId,
        clock: LogicalClock,
        sequence: u64,
        object: PersistentId,
        delta: OverrideDelta,
        inverse: OverrideDelta,
    ) -> Result<Self, DeltaError> {
        if !delta.same_fields(&inverse) {
            return Err(DeltaError::InverseFieldMismatch);
        }
        Ok(Self {
            id,
            actor,
            clock,
            sequence,
            object,
            delta,
            inverse,
        })
    }

    /// Returns the transaction identity.
    pub const fn id(&self) -> TransactionId {
        self.id
    }

    /// Returns the author/replica identity.
    pub const fn actor(&self) -> ActorId {
        self.actor
    }

    /// Returns the logical clock.
    pub const fn clock(&self) -> LogicalClock {
        self.clock
    }

    /// Returns the append-only sequence number.
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Returns the changed object identity.
    pub const fn object(&self) -> PersistentId {
        self.object
    }

    /// Returns the forward delta.
    pub const fn delta(&self) -> &OverrideDelta {
        &self.delta
    }

    /// Returns the recorded reverse delta.
    pub const fn inverse(&self) -> &OverrideDelta {
        &self.inverse
    }

    pub(crate) fn validate(&self) -> Result<(), DeltaError> {
        if !self.delta.same_fields(&self.inverse) {
            return Err(DeltaError::InverseFieldMismatch);
        }
        for change in self.delta.iter().chain(self.inverse.iter()) {
            change.value.validate().map_err(DeltaError::InvalidValue)?;
        }
        Ok(())
    }
}

/// Current field map for one object.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ObjectState {
    fields: BTreeMap<FieldId, OverrideValue>,
}

impl ObjectState {
    /// Returns a field's materialized value, or `None` if absent/deleted.
    pub fn get(&self, field: FieldId) -> Option<&OverrideValue> {
        self.fields.get(&field)
    }

    /// Iterates fields in ascending field order.
    pub fn iter(&self) -> alloc::collections::btree_map::Iter<'_, FieldId, OverrideValue> {
        self.fields.iter()
    }

    /// Returns the number of materialized fields.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Returns true when no fields are materialized.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

/// Materialized world override state, deterministically ordered by object and field.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct WorldState {
    objects: BTreeMap<PersistentId, ObjectState>,
}

impl WorldState {
    /// Creates an empty world state.
    pub const fn new() -> Self {
        Self {
            objects: BTreeMap::new(),
        }
    }

    /// Returns one object's materialized state.
    pub fn object(&self, object: PersistentId) -> Option<&ObjectState> {
        self.objects.get(&object)
    }

    /// Returns one field's materialized value.
    pub fn get(&self, object: PersistentId, field: FieldId) -> Option<&OverrideValue> {
        self.object(object).and_then(|state| state.get(field))
    }

    /// Iterates objects in ascending [`PersistentId`] order.
    pub fn iter(&self) -> alloc::collections::btree_map::Iter<'_, PersistentId, ObjectState> {
        self.objects.iter()
    }

    /// Returns the number of objects with at least one materialized field.
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// Returns true when the state has no materialized fields.
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    pub(crate) fn inverse_for(&self, object: PersistentId, delta: &OverrideDelta) -> OverrideDelta {
        let changes = delta
            .iter()
            .map(|change| FieldOverride {
                field: change.field,
                value: self
                    .get(object, change.field)
                    .cloned()
                    .unwrap_or(OverrideValue::Delete),
            })
            .collect();
        OverrideDelta { changes }
    }

    pub(crate) fn apply_delta(&mut self, object: PersistentId, delta: &OverrideDelta) {
        let object_state = self.objects.entry(object).or_default();
        for change in delta.iter() {
            match &change.value {
                OverrideValue::Delete => {
                    object_state.fields.remove(&change.field);
                }
                value => {
                    object_state.fields.insert(change.field, value.clone());
                }
            }
        }
        if object_state.fields.is_empty() {
            self.objects.remove(&object);
        }
    }
}
