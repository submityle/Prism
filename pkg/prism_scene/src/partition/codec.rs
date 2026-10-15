use std::collections::BTreeSet;

use prism_ecs::{entity::Entity, partition::cell::CellCoord};

use super::{PartitionError, MAX_CELL_BYTES};
use crate::PersistentId;

#[derive(Clone, Debug)]
pub(super) struct Member {
    pub source: Entity,
    pub id: PersistentId,
    pub archived: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Edge {
    pub name: String,
    pub policy: u8,
    pub source: PersistentId,
    pub target: PersistentId,
}

pub(super) struct CellArchive {
    pub cell: CellCoord,
    pub members: Vec<Member>,
    pub edges: BTreeSet<Edge>,
    pub payload: Vec<u8>,
}

fn invalid() -> PartitionError {
    PartitionError::InvalidArchive("malformed cell envelope")
}

// Integrity check for accidental truncation/bit flips, not authentication.
fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

impl CellArchive {
    pub fn encode(&self) -> Result<Vec<u8>, PartitionError> {
        let mut bytes = b"PSCL\x01".to_vec();
        for axis in [self.cell.x, self.cell.y, self.cell.z] {
            bytes.extend_from_slice(&axis.to_le_bytes());
        }
        push_count(&mut bytes, self.members.len())?;
        for member in &self.members {
            bytes.extend_from_slice(&member.source.to_bits().to_le_bytes());
            bytes.extend_from_slice(&member.id.to_u128().to_le_bytes());
            bytes.push(u8::from(member.archived));
        }
        push_count(&mut bytes, self.edges.len())?;
        for edge in &self.edges {
            bytes.extend_from_slice(&edge.source.to_u128().to_le_bytes());
            bytes.extend_from_slice(&edge.target.to_u128().to_le_bytes());
            bytes.push(edge.policy);
            push_count(&mut bytes, edge.name.len())?;
            bytes.extend_from_slice(edge.name.as_bytes());
        }
        push_count(&mut bytes, self.payload.len())?;
        bytes.extend_from_slice(&self.payload);
        let hash = checksum(&bytes);
        bytes.extend_from_slice(&hash.to_le_bytes());
        if bytes.len() > MAX_CELL_BYTES {
            return Err(PartitionError::InvalidArchive("cell exceeds byte limit"));
        }
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8], cell: CellCoord) -> Result<Self, PartitionError> {
        if bytes.len() > MAX_CELL_BYTES || bytes.len() < 8 {
            return Err(invalid());
        }
        let (body, hash) = bytes.split_at(bytes.len() - 8);
        if checksum(body) != u64::from_le_bytes(hash.try_into().map_err(|_| invalid())?) {
            return Err(PartitionError::InvalidArchive("cell checksum mismatch"));
        }
        let mut reader = Reader { bytes: body, at: 0 };
        if reader.take(5)? != b"PSCL\x01" {
            return Err(invalid());
        }
        for axis in [cell.x, cell.y, cell.z] {
            if reader.take(4)? != axis.to_le_bytes() {
                return Err(PartitionError::InvalidArchive("cell coordinate mismatch"));
            }
        }
        let count = reader.count(25)?;
        let mut members = Vec::with_capacity(count);
        let mut sources = BTreeSet::new();
        let mut ids = BTreeSet::new();
        for _ in 0..count {
            let source = Entity::from_bits(reader.u64()?).ok_or_else(invalid)?;
            let id = reader.id()?;
            let archived = match reader.take(1)?[0] {
                0 => false,
                1 => true,
                _ => return Err(invalid()),
            };
            if !sources.insert(source) || !ids.insert(id) {
                return Err(PartitionError::InvalidArchive("duplicate cell member"));
            }
            members.push(Member {
                source,
                id,
                archived,
            });
        }
        let count = reader.count(38)?;
        let mut edges = BTreeSet::new();
        for _ in 0..count {
            let source = reader.id()?;
            let target = reader.id()?;
            let policy = reader.take(1)?[0];
            // Three booleans and two 2-bit CleanupPolicy discriminants.
            if policy & 0x80 != 0 || ((policy >> 3) & 3) > 2 || ((policy >> 5) & 3) > 2 {
                return Err(invalid());
            }
            let length = reader.count(1)?;
            if length == 0 || length > 4096 {
                return Err(invalid());
            }
            let name = std::str::from_utf8(reader.take(length)?)
                .map_err(|_| invalid())?
                .to_owned();
            if !ids.contains(&source)
                || !edges.insert(Edge {
                    name,
                    policy,
                    source,
                    target,
                })
            {
                return Err(PartitionError::InvalidArchive(
                    "invalid source-owned relation",
                ));
            }
        }
        let length = reader.count(1)?;
        let payload = reader.take(length)?.to_vec();
        if reader.at != body.len() {
            return Err(invalid());
        }
        Ok(Self {
            cell,
            members,
            edges,
            payload,
        })
    }
}

fn push_count(bytes: &mut Vec<u8>, count: usize) -> Result<(), PartitionError> {
    let count = u32::try_from(count).map_err(|_| invalid())?;
    bytes.extend_from_slice(&count.to_le_bytes());
    Ok(())
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], PartitionError> {
        let end = self.at.checked_add(count).ok_or_else(invalid)?;
        let slice = self.bytes.get(self.at..end).ok_or_else(invalid)?;
        self.at = end;
        Ok(slice)
    }

    fn count(&mut self, minimum: usize) -> Result<usize, PartitionError> {
        let count = u32::from_le_bytes(self.take(4)?.try_into().map_err(|_| invalid())?) as usize;
        if count > (self.bytes.len() - self.at) / minimum {
            return Err(invalid());
        }
        Ok(count)
    }

    fn u64(&mut self) -> Result<u64, PartitionError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().map_err(|_| invalid())?,
        ))
    }

    fn id(&mut self) -> Result<PersistentId, PartitionError> {
        let id = PersistentId::from_u128(u128::from_le_bytes(
            self.take(16)?.try_into().map_err(|_| invalid())?,
        ));
        if id.is_nil() {
            return Err(invalid());
        }
        Ok(id)
    }
}
