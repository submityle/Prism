//! Canonical bounded metadata inside the shared version-one envelope codec.
use super::*;

pub(super) const META_LIMIT: usize = 1024 * 1024;
pub(super) fn digest(bytes: &[u8]) -> u64 {
    codec::fold(0xcbf29ce484222325, bytes)
}
pub(super) fn identity(magic: [u8; 8]) -> EnvelopeIdentity {
    EnvelopeIdentity {
        magic,
        coord: crate::CellCoord::new(0, 0, 0),
        guid: StableGuid::from_u128(1),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Pointer {
    pub generation: u64,
    pub id: StableGuid,
    pub hash: u64,
}
impl Default for Pointer {
    fn default() -> Self {
        Self {
            generation: 0,
            id: StableGuid::from_u128(0),
            hash: 0,
        }
    }
}
impl Pointer {
    pub fn encode(self) -> Vec<u8> {
        let mut out = self.generation.to_le_bytes().to_vec();
        out.extend(self.id.to_u128().to_le_bytes());
        out.extend(self.hash.to_le_bytes());
        out
    }
    pub fn decode(bytes: &[u8]) -> Result<Self, FileCellIoError> {
        let mut r = Reader(bytes);
        let p = Self::read(&mut r)?;
        r.end()?;
        Ok(p)
    }
    pub fn read(r: &mut Reader<'_>) -> Result<Self, FileCellIoError> {
        let p = Self {
            generation: r.u64()?,
            id: StableGuid::from_u128(r.u128()?),
            hash: r.u64()?,
        };
        if (p.generation == 0 && (p.hash != 0 || !p.id.is_nil()))
            || (p.generation != 0 && p.id.is_nil())
        {
            return Err(FileCellIoError::Corrupt("root identity"));
        }
        Ok(p)
    }
}

pub(super) fn target(out: &mut Vec<u8>, t: &WorldFileTarget) {
    let (kind, x, y, z, guid) = t.key();
    out.push(kind);
    for n in [x, y, z] {
        out.extend(n.to_le_bytes());
    }
    out.extend(guid.to_le_bytes());
}

pub(super) fn manifest(request: &WorldTransactionRequest, files: &[WorldFileSnapshot]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend((files.len() as u32).to_le_bytes());
    for f in files {
        target(&mut out, &f.target);
        out.extend(f.revision.to_le_bytes());
        out.extend((f.bytes.len() as u64).to_le_bytes());
        out.extend(digest(&f.bytes).to_le_bytes());
    }
    out.extend((request.changes.len() as u32).to_le_bytes());
    for c in &request.changes {
        target(&mut out, &c.target);
        match c.base {
            ActorRevisionBase::Missing => out.push(0),
            ActorRevisionBase::Revision(r) => {
                out.push(1);
                out.extend(r.to_le_bytes());
            }
        }
        match c.value {
            None => out.push(0),
            Some((r, _)) => {
                out.push(1);
                out.extend(r.to_le_bytes());
            }
        }
    }
    out
}

pub(super) struct Entry {
    pub target: WorldFileTarget,
    pub revision: u64,
    pub length: usize,
    pub hash: u64,
}
pub(super) struct Descriptor {
    pub target: WorldFileTarget,
    pub base: ActorRevisionBase,
    pub revision: Option<u64>,
}
pub(super) fn decode_manifest(
    bytes: &[u8],
    limits: WorldTransactionLimits,
) -> Result<(Vec<Entry>, Vec<Descriptor>), FileCellIoError> {
    let mut r = Reader(bytes);
    let count = r.u32()? as usize;
    if count > limits.entries {
        return Err(FileCellIoError::Capacity);
    }
    let mut entries = Vec::with_capacity(count);
    let mut total = 0usize;
    for _ in 0..count {
        let target = r.target()?;
        let revision = r.u64()?;
        let length = r.u64()?;
        let hash = r.u64()?;
        if length > limits.file_bytes as u64 {
            return Err(FileCellIoError::TooLarge {
                limit: limits.file_bytes,
                actual: length,
            });
        }
        total = total
            .checked_add(length as usize)
            .ok_or(FileCellIoError::Capacity)?;
        if total > limits.snapshot_bytes {
            return Err(FileCellIoError::Capacity);
        }
        if entries
            .last()
            .is_some_and(|e: &Entry| e.target.key() >= target.key())
        {
            return Err(FileCellIoError::Corrupt("manifest order"));
        }
        entries.push(Entry {
            target,
            revision,
            length: length as usize,
            hash,
        });
    }
    let count = r.u32()? as usize;
    if count == 0 || count > limits.entries * 2 {
        return Err(FileCellIoError::Corrupt("change count"));
    }
    let mut changes = Vec::with_capacity(count);
    for _ in 0..count {
        let target = r.target()?;
        let base = match r.byte()? {
            0 => ActorRevisionBase::Missing,
            1 => ActorRevisionBase::Revision(r.u64()?),
            _ => return Err(FileCellIoError::Corrupt("base tag")),
        };
        let revision = match r.byte()? {
            0 => None,
            1 => Some(r.u64()?),
            _ => return Err(FileCellIoError::Corrupt("mutation tag")),
        };
        if changes
            .last()
            .is_some_and(|d: &Descriptor| d.target.key() >= target.key())
        {
            return Err(FileCellIoError::Corrupt("change order"));
        }
        changes.push(Descriptor {
            target,
            base,
            revision,
        });
    }
    r.end()?;
    Ok((entries, changes))
}

pub(super) struct Reader<'a>(pub &'a [u8]);
impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], FileCellIoError> {
        let bytes = self
            .0
            .get(..N)
            .ok_or(FileCellIoError::Corrupt("truncated metadata"))?;
        let value = bytes.try_into().expect("fixed metadata range");
        self.0 = &self.0[N..];
        Ok(value)
    }
    pub fn byte(&mut self) -> Result<u8, FileCellIoError> {
        Ok(self.take::<1>()?[0])
    }
    pub fn u32(&mut self) -> Result<u32, FileCellIoError> {
        Ok(u32::from_le_bytes(self.take()?))
    }
    pub fn u64(&mut self) -> Result<u64, FileCellIoError> {
        Ok(u64::from_le_bytes(self.take()?))
    }
    pub fn u128(&mut self) -> Result<u128, FileCellIoError> {
        Ok(u128::from_le_bytes(self.take()?))
    }
    pub fn target(&mut self) -> Result<WorldFileTarget, FileCellIoError> {
        let kind = self.byte()?;
        let c = crate::CellCoord::new(
            i32::from_le_bytes(self.take()?),
            i32::from_le_bytes(self.take()?),
            i32::from_le_bytes(self.take()?),
        );
        let guid = StableGuid::from_u128(self.u128()?);
        let t = match kind {
            0 => WorldFileTarget::Cell(CellFileIdentity {
                coord: c,
                stable_id: guid,
                relative_path: format!("cells/{}_{}_{}.prismcell", c.x, c.y, c.z),
            }),
            1 => WorldFileTarget::Actor(
                ActorFileIdentity::new(guid, c)
                    .map_err(|_| FileCellIoError::Corrupt("actor identity"))?,
            ),
            _ => return Err(FileCellIoError::Corrupt("target tag")),
        };
        t.validate()
            .map_err(|_| FileCellIoError::Corrupt("target identity"))?;
        Ok(t)
    }
    pub fn end(&self) -> Result<(), FileCellIoError> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(FileCellIoError::Corrupt("metadata trailing bytes"))
        }
    }
}
