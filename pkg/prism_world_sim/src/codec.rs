use alloc::vec::Vec;

use crate::ecology::EcologyDelta;
use crate::hydrology::FlowDelta;
use crate::sim::SimulationDelta;
use crate::types::CellCoord;
use crate::voxel::{FragmentDelta, VoxelDelta};

const MAGIC: [u8; 4] = *b"PWSM";
const VERSION: u16 = 1;

/// Stable binary codec failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodecError {
    /// Input ended before a complete value was available.
    Truncated,
    /// The four-byte stream marker was not recognized.
    InvalidMagic,
    /// The stream version is not supported.
    UnsupportedVersion,
    /// A record count cannot fit in the remaining input.
    CountOverflow,
    /// A record contains an invalid value.
    InvalidRecord,
    /// Bytes remain after the final declared record.
    TrailingBytes,
}

/// Encodes a simulation delta using stable little-endian integer records.
pub fn encode_delta(delta: &SimulationDelta) -> Vec<u8> {
    let mut output = Vec::new();
    output.extend_from_slice(&MAGIC);
    put_u16(&mut output, VERSION);
    put_u64(&mut output, delta.tick);
    put_u32(&mut output, delta.voxel_deltas.len() as u32);
    put_u32(&mut output, delta.fragment_deltas.len() as u32);
    put_u32(&mut output, delta.flow_deltas.len() as u32);
    put_u32(&mut output, delta.ecology_deltas.len() as u32);
    for item in &delta.voxel_deltas {
        put_coord(&mut output, item.coord);
        put_u16(&mut output, item.material);
        put_u32(&mut output, item.before_mass);
        put_u32(&mut output, item.after_mass);
        put_u32(&mut output, item.removed_mass);
    }
    for item in &delta.fragment_deltas {
        put_u64(&mut output, item.stable_key);
        put_coord(&mut output, item.origin);
        put_u32(&mut output, item.mass);
        put_u16(&mut output, item.material);
        for velocity in item.velocity {
            put_i16(&mut output, velocity);
        }
    }
    for item in &delta.flow_deltas {
        put_coord(&mut output, item.from);
        put_coord(&mut output, item.to);
        put_u32(&mut output, item.amount);
    }
    for item in &delta.ecology_deltas {
        put_coord(&mut output, item.coord);
        put_u32(&mut output, item.before_biomass);
        put_u32(&mut output, item.after_biomass);
        put_u16(&mut output, item.moisture);
        put_u16(&mut output, item.stress);
    }
    output
}

/// Decodes a stable little-endian simulation delta.
pub fn decode_delta(input: &[u8]) -> Result<SimulationDelta, CodecError> {
    let mut reader = Reader { input, offset: 0 };
    if reader.take(4)? != MAGIC {
        return Err(CodecError::InvalidMagic);
    }
    if reader.u16()? != VERSION {
        return Err(CodecError::UnsupportedVersion);
    }
    let tick = reader.u64()?;
    let voxel_count = reader.count(24)?;
    let fragment_count = reader.count(28)?;
    let flow_count = reader.count(28)?;
    let ecology_count = reader.count(20)?;
    let mut delta = SimulationDelta {
        tick,
        voxel_deltas: Vec::with_capacity(voxel_count),
        fragment_deltas: Vec::with_capacity(fragment_count),
        flow_deltas: Vec::with_capacity(flow_count),
        ecology_deltas: Vec::with_capacity(ecology_count),
    };
    for _ in 0..voxel_count {
        let coord = reader.coord()?;
        let material = reader.u16()?;
        let before_mass = reader.u32()?;
        let after_mass = reader.u32()?;
        let removed_mass = reader.u32()?;
        if before_mass < after_mass || before_mass - after_mass != removed_mass {
            return Err(CodecError::InvalidRecord);
        }
        delta.voxel_deltas.push(VoxelDelta {
            coord,
            material,
            before_mass,
            after_mass,
            removed_mass,
        });
    }
    for _ in 0..fragment_count {
        let stable_key = reader.u64()?;
        let origin = reader.coord()?;
        let mass = reader.u32()?;
        let material = reader.u16()?;
        let velocity = [reader.i16()?, reader.i16()?, reader.i16()?];
        if mass == 0 {
            return Err(CodecError::InvalidRecord);
        }
        delta.fragment_deltas.push(FragmentDelta {
            stable_key,
            origin,
            mass,
            material,
            velocity,
        });
    }
    for _ in 0..flow_count {
        let from = reader.coord()?;
        let to = reader.coord()?;
        let amount = reader.u32()?;
        if amount == 0 || from == to {
            return Err(CodecError::InvalidRecord);
        }
        delta.flow_deltas.push(FlowDelta { from, to, amount });
    }
    for _ in 0..ecology_count {
        let coord = reader.coord()?;
        let before_biomass = reader.u32()?;
        let after_biomass = reader.u32()?;
        let moisture = reader.u16()?;
        let stress = reader.u16()?;
        delta.ecology_deltas.push(EcologyDelta {
            coord,
            before_biomass,
            after_biomass,
            moisture,
            stress,
        });
    }
    if reader.offset != input.len() {
        return Err(CodecError::TrailingBytes);
    }
    Ok(delta)
}

struct Reader<'a> {
    input: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, length: usize) -> Result<&'a [u8], CodecError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(CodecError::Truncated)?;
        let value = self
            .input
            .get(self.offset..end)
            .ok_or(CodecError::Truncated)?;
        self.offset = end;
        Ok(value)
    }

    fn count(&mut self, record_size: usize) -> Result<usize, CodecError> {
        let count = self.u32()? as usize;
        let bytes = count
            .checked_mul(record_size)
            .ok_or(CodecError::CountOverflow)?;
        if bytes > self.input.len().saturating_sub(self.offset) {
            return Err(CodecError::CountOverflow);
        }
        Ok(count)
    }

    fn u16(&mut self) -> Result<u16, CodecError> {
        Ok(u16::from_le_bytes(
            self.take(2)?
                .try_into()
                .map_err(|_| CodecError::Truncated)?,
        ))
    }

    fn i16(&mut self) -> Result<i16, CodecError> {
        Ok(i16::from_le_bytes(
            self.take(2)?
                .try_into()
                .map_err(|_| CodecError::Truncated)?,
        ))
    }

    fn u32(&mut self) -> Result<u32, CodecError> {
        Ok(u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| CodecError::Truncated)?,
        ))
    }

    fn u64(&mut self) -> Result<u64, CodecError> {
        Ok(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| CodecError::Truncated)?,
        ))
    }

    fn coord(&mut self) -> Result<CellCoord, CodecError> {
        Ok(CellCoord::new(
            i32::from_le_bytes(
                self.take(4)?
                    .try_into()
                    .map_err(|_| CodecError::Truncated)?,
            ),
            i32::from_le_bytes(
                self.take(4)?
                    .try_into()
                    .map_err(|_| CodecError::Truncated)?,
            ),
            i32::from_le_bytes(
                self.take(4)?
                    .try_into()
                    .map_err(|_| CodecError::Truncated)?,
            ),
        ))
    }
}

fn put_u16(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn put_i16(output: &mut Vec<u8>, value: i16) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn put_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn put_coord(output: &mut Vec<u8>, value: CellCoord) {
    output.extend_from_slice(&value.x.to_le_bytes());
    output.extend_from_slice(&value.y.to_le_bytes());
    output.extend_from_slice(&value.z.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::SimulationDelta;
    use alloc::vec;

    #[test]
    fn codec_round_trip_is_stable() {
        let delta = SimulationDelta {
            tick: 7,
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
        };
        let encoded = encode_delta(&delta);
        assert_eq!(decode_delta(&encoded), Ok(delta.clone()));
        assert_eq!(encode_delta(&decode_delta(&encoded).unwrap()), encoded);
    }
}
