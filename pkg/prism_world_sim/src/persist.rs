//! Optional bridge from simulation deltas to the persistence transaction model.

use prism_world_persist::{
    ActorId, DeltaError, FieldId, LogicalClock, OverrideDelta, OverrideValue, PersistentId,
    Transaction, TransactionId,
};

use crate::{decode_delta, encode_delta, CodecError, SimulationDelta};

/// Stable persistence field containing an encoded [`SimulationDelta`].
///
/// The value is intentionally opaque to the persistence core. The simulation
/// codec owns the bytes and can evolve its record representation independently
/// from the generic field-value model.
pub const SIMULATION_DELTA_FIELD: FieldId = FieldId(0x5052_4953_4d53_494d);

/// Errors returned while converting between simulation deltas and transactions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SimulationPersistError {
    /// The simulation delta failed canonical encoding or decoding.
    Codec(CodecError),
    /// The generic persistence transaction rejected the supplied field deltas.
    Persistence(DeltaError),
    /// The transaction did not contain exactly the simulation field shape.
    InvalidFieldShape {
        /// Number of fields in the forward or inverse delta.
        forward_fields: usize,
        /// Number of fields in the inverse delta.
        inverse_fields: usize,
    },
    /// The transaction contained a different field identifier.
    UnexpectedField(FieldId),
    /// The forward value was not opaque bytes.
    ForwardValueNotBytes,
    /// The inverse value was not a deletion tombstone.
    InverseValueNotDelete,
}

/// Converts one simulation delta into a generic persistence transaction.
///
/// `transaction_id`, `actor`, `clock`, `sequence`, and `object` are copied
/// exactly into the resulting transaction. The forward value is the stable
/// little-endian output of [`crate::encode_delta`], while the inverse uses a
/// deletion tombstone for the same field. The generic persistence constructor
/// therefore enforces that forward and inverse address identical fields.
pub fn simulation_delta_to_transaction(
    transaction_id: TransactionId,
    actor: ActorId,
    clock: LogicalClock,
    sequence: u64,
    object: PersistentId,
    delta: &SimulationDelta,
) -> Result<Transaction, SimulationPersistError> {
    let bytes = encode_delta(delta).map_err(SimulationPersistError::Codec)?;
    let forward = OverrideDelta::single(SIMULATION_DELTA_FIELD, OverrideValue::Bytes(bytes))
        .map_err(SimulationPersistError::Persistence)?;
    let inverse = OverrideDelta::single(SIMULATION_DELTA_FIELD, OverrideValue::Delete)
        .map_err(SimulationPersistError::Persistence)?;
    Transaction::new(
        transaction_id,
        actor,
        clock,
        sequence,
        object,
        forward,
        inverse,
    )
    .map_err(SimulationPersistError::Persistence)
}

/// Decodes a transaction produced by [`simulation_delta_to_transaction`].
///
/// This checks the field count, stable field identifier, forward value type,
/// and inverse tombstone before handing the opaque bytes to the simulation
/// codec. Arbitrary generic transactions are rejected instead of being
/// silently interpreted as simulation state.
pub fn transaction_to_simulation_delta(
    transaction: &Transaction,
) -> Result<SimulationDelta, SimulationPersistError> {
    let forward = transaction.delta();
    let inverse = transaction.inverse();
    if forward.len() != 1 || inverse.len() != 1 {
        return Err(SimulationPersistError::InvalidFieldShape {
            forward_fields: forward.len(),
            inverse_fields: inverse.len(),
        });
    }
    let forward_change = forward.as_slice().first().expect("length checked above");
    let inverse_change = inverse.as_slice().first().expect("length checked above");
    if forward_change.field != SIMULATION_DELTA_FIELD {
        return Err(SimulationPersistError::UnexpectedField(
            forward_change.field,
        ));
    }
    if inverse_change.field != SIMULATION_DELTA_FIELD {
        return Err(SimulationPersistError::UnexpectedField(
            inverse_change.field,
        ));
    }
    if !matches!(inverse_change.value, OverrideValue::Delete) {
        return Err(SimulationPersistError::InverseValueNotDelete);
    }
    let OverrideValue::Bytes(bytes) = &forward_change.value else {
        return Err(SimulationPersistError::ForwardValueNotBytes);
    };
    decode_delta(bytes).map_err(SimulationPersistError::Codec)
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::{CellCoord, EcologyDelta, FlowDelta, FragmentDelta, VoxelDelta};

    fn sample_delta() -> SimulationDelta {
        SimulationDelta {
            tick: 19,
            voxel_deltas: vec![VoxelDelta {
                coord: CellCoord::new(-2, 1, 4),
                material: 3,
                before_mass: 9,
                after_mass: 6,
                removed_mass: 3,
            }],
            fragment_deltas: vec![FragmentDelta {
                stable_key: 11,
                origin: CellCoord::new(-2, 1, 4),
                mass: 3,
                material: 3,
                velocity: [0, -1, 2],
            }],
            flow_deltas: vec![FlowDelta {
                from: CellCoord::new(0, 0, 0),
                to: CellCoord::new(1, 0, 0),
                amount: 4,
            }],
            ecology_deltas: vec![EcologyDelta {
                coord: CellCoord::new(2, 0, 0),
                before_biomass: 4,
                after_biomass: 5,
                moisture: 100,
                stress: 20,
            }],
        }
    }

    fn transaction(delta: &SimulationDelta) -> Transaction {
        simulation_delta_to_transaction(
            TransactionId(7),
            ActorId(8),
            LogicalClock(9),
            10,
            PersistentId(11),
            delta,
        )
        .expect("valid simulation transaction")
    }

    #[test]
    fn transaction_round_trips_delta_and_preserves_stable_inputs() {
        let delta = sample_delta();
        let transaction = transaction(&delta);
        assert_eq!(transaction.id(), TransactionId(7));
        assert_eq!(transaction.actor(), ActorId(8));
        assert_eq!(transaction.clock(), LogicalClock(9));
        assert_eq!(transaction.sequence(), 10);
        assert_eq!(transaction.object(), PersistentId(11));
        assert_eq!(transaction.delta().len(), 1);
        assert_eq!(transaction.inverse().len(), 1);
        assert!(matches!(
            transaction.delta().get(SIMULATION_DELTA_FIELD),
            Some(OverrideValue::Bytes(_))
        ));
        assert_eq!(
            transaction.inverse().get(SIMULATION_DELTA_FIELD),
            Some(&OverrideValue::Delete)
        );
        assert_eq!(transaction_to_simulation_delta(&transaction), Ok(delta));
    }

    #[test]
    fn equal_inputs_produce_byte_identical_transactions() {
        let delta = sample_delta();
        assert_eq!(transaction(&delta), transaction(&delta));
    }

    #[test]
    fn invalid_simulation_delta_reports_codec_error() {
        let invalid = SimulationDelta {
            voxel_deltas: vec![VoxelDelta {
                coord: CellCoord::new(0, 0, 0),
                material: 1,
                before_mass: 1,
                after_mass: 1,
                removed_mass: 1,
            }],
            ..SimulationDelta::default()
        };
        assert_eq!(
            simulation_delta_to_transaction(
                TransactionId(1),
                ActorId(2),
                LogicalClock(3),
                4,
                PersistentId(5),
                &invalid,
            ),
            Err(SimulationPersistError::Codec(CodecError::InvalidRecord))
        );
    }

    #[test]
    fn wrong_forward_and_inverse_values_are_rejected() {
        let forward_integer =
            OverrideDelta::single(SIMULATION_DELTA_FIELD, OverrideValue::Unsigned(1)).unwrap();
        let inverse_delete =
            OverrideDelta::single(SIMULATION_DELTA_FIELD, OverrideValue::Delete).unwrap();
        let transaction = Transaction::new(
            TransactionId(1),
            ActorId(2),
            LogicalClock(3),
            4,
            PersistentId(5),
            forward_integer,
            inverse_delete,
        )
        .unwrap();
        assert_eq!(
            transaction_to_simulation_delta(&transaction),
            Err(SimulationPersistError::ForwardValueNotBytes)
        );

        let forward_bytes =
            OverrideDelta::single(SIMULATION_DELTA_FIELD, OverrideValue::Bytes(vec![1, 2, 3]))
                .unwrap();
        let inverse_bytes =
            OverrideDelta::single(SIMULATION_DELTA_FIELD, OverrideValue::Bytes(vec![4])).unwrap();
        let transaction = Transaction::new(
            TransactionId(1),
            ActorId(2),
            LogicalClock(3),
            4,
            PersistentId(5),
            forward_bytes,
            inverse_bytes,
        )
        .unwrap();
        assert_eq!(
            transaction_to_simulation_delta(&transaction),
            Err(SimulationPersistError::InverseValueNotDelete)
        );
    }

    #[test]
    fn mismatched_fields_are_rejected_by_persistence_constructor() {
        let forward =
            OverrideDelta::single(SIMULATION_DELTA_FIELD, OverrideValue::Bytes(vec![1])).unwrap();
        let inverse = OverrideDelta::single(FieldId(99), OverrideValue::Delete).unwrap();
        assert_eq!(
            Transaction::new(
                TransactionId(1),
                ActorId(2),
                LogicalClock(3),
                4,
                PersistentId(5),
                forward,
                inverse,
            ),
            Err(DeltaError::InverseFieldMismatch)
        );
    }
}
