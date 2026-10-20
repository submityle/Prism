//! Version-one canonical little-endian membership wire format.
use super::*;

const MAGIC: &[u8; 8] = b"PEWPROJ\0";
const VERSION: u32 = 1;
const FIXED_BYTES: usize = 64;

pub(super) fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100_0000_01b3)
    })
}

impl WorldProject {
    pub(super) fn encoded_size(&self) -> Result<usize, ProjectError> {
        let mut size = FIXED_BYTES;
        let mut add = |n: usize| -> Result<(), ProjectError> {
            size = size
                .checked_add(n)
                .ok_or(ProjectError::Budget("encoded bytes"))?;
            Ok(())
        };
        for layer in self.layers.values() {
            add(10 + usize::from(layer.parent.is_some()) * 4)?;
            add(layer.name.len())?;
        }
        for layers in self.cells.values() {
            add(16)?;
            add(layers
                .len()
                .checked_mul(4)
                .ok_or(ProjectError::Budget("memberships"))?)?;
        }
        add(self
            .actors
            .len()
            .checked_mul(28)
            .ok_or(ProjectError::Budget("actors"))?)?;
        Ok(size)
    }

    /// Canonical, content-independent ordering and exact length with corruption
    /// checksum. This checksum detects corruption; it is not authentication.
    pub fn encode(&self, limits: ProjectLimits) -> Result<Vec<u8>, ProjectError> {
        self.validate(limits)?;
        let size = self.encoded_size()?;
        let mut out = Vec::with_capacity(size);
        out.extend(MAGIC);
        out.extend(VERSION.to_le_bytes());
        out.extend((size as u64).to_le_bytes());
        out.extend(self.identity.to_u128().to_le_bytes());
        out.extend(self.revision.to_le_bytes());
        for count in [self.layers.len(), self.cells.len(), self.actors.len()] {
            out.extend((count as u32).to_le_bytes());
        }
        for layer in self.layers.values() {
            out.extend(layer.id.get().to_le_bytes());
            out.extend((layer.name.len() as u32).to_le_bytes());
            out.extend(layer.name.as_bytes());
            out.push(u8::from(layer.parent.is_some()));
            if let Some(parent) = layer.parent {
                out.extend(parent.get().to_le_bytes());
            }
            out.push(u8::from(layer.visible) | (u8::from(layer.editable) << 1));
        }
        for (cell, layers) in self.cells() {
            put_coord(&mut out, cell);
            out.extend((layers.len() as u32).to_le_bytes());
            for id in layers {
                out.extend(id.get().to_le_bytes());
            }
        }
        for (actor, cell) in self.actors() {
            out.extend(actor.to_u128().to_le_bytes());
            put_coord(&mut out, cell);
        }
        out.extend(checksum(&out).to_le_bytes());
        debug_assert_eq!(out.len(), size);
        Ok(out)
    }

    /// Decodes only canonical version-one bytes. Counts and name/membership
    /// lengths are checked before allocations. Trailing bytes are rejected.
    pub fn decode(bytes: &[u8], limits: ProjectLimits) -> Result<Self, ProjectError> {
        limits.validate()?;
        if bytes.len() > limits.max_bytes {
            return Err(ProjectError::Budget("encoded bytes"));
        }
        if bytes.len() < FIXED_BYTES {
            return Err(ProjectError::Malformed("truncated header"));
        }
        let mut r = Reader(bytes);
        if r.take(8)? != MAGIC {
            return Err(ProjectError::Malformed("magic"));
        }
        if r.u32()? != VERSION {
            return Err(ProjectError::Malformed("version"));
        }
        if r.u64()? != bytes.len() as u64 {
            return Err(ProjectError::Malformed("length"));
        }
        let hash_offset = bytes.len() - 8;
        let mut tail = Reader(&bytes[hash_offset..]);
        if tail.u64()? != checksum(&bytes[..hash_offset]) {
            return Err(ProjectError::Malformed("checksum"));
        }
        let identity = StableGuid::from_u128(r.u128()?);
        let revision = r.u64()?;
        let layer_count = r.count(limits.max_layers, "layers")?;
        let cell_count = r.count(limits.max_cells, "cells")?;
        let actor_count = r.count(limits.max_actors, "actors")?;
        // Supported ceilings make all these products fit even a 32-bit usize.
        if FIXED_BYTES + layer_count * 10 + cell_count * 16 + actor_count * 28 > bytes.len() {
            return Err(ProjectError::Malformed("record counts exceed length"));
        }
        let mut project = Self::new(identity)?;
        project.revision = revision;
        for _ in 0..layer_count {
            let id = DataLayerId::new(r.u32()?);
            if project
                .layers
                .last_key_value()
                .is_some_and(|(previous, _)| *previous >= id)
            {
                return Err(ProjectError::Malformed("layer order"));
            }
            let len = r.count(limits.max_name_bytes, "name")?;
            let name = std::str::from_utf8(r.take(len)?)
                .map_err(|_| ProjectError::Malformed("UTF-8"))?
                .to_owned();
            let parent = match r.byte()? {
                0 => None,
                1 => Some(DataLayerId::new(r.u32()?)),
                _ => return Err(ProjectError::Malformed("parent tag")),
            };
            let flags = r.byte()?;
            if flags > 3 {
                return Err(ProjectError::Malformed("layer flags"));
            }
            project.layers.insert(
                id,
                ProjectLayer {
                    id,
                    name,
                    parent,
                    visible: flags & 1 != 0,
                    editable: flags & 2 != 0,
                },
            );
        }
        let mut links = 0usize;
        for _ in 0..cell_count {
            let key = CellKey::new(r.coord()?);
            if project
                .cells
                .last_key_value()
                .is_some_and(|(previous, _)| *previous >= key)
            {
                return Err(ProjectError::Malformed("cell order"));
            }
            let count = r.count(limits.max_memberships - links, "memberships")?;
            links += count;
            if count > r.0.len() / 4 {
                return Err(ProjectError::Malformed("membership length"));
            }
            let mut layers = BTreeSet::new();
            for _ in 0..count {
                let id = DataLayerId::new(r.u32()?);
                if layers.last().is_some_and(|previous| *previous >= id) {
                    return Err(ProjectError::Malformed("membership order"));
                }
                layers.insert(id);
            }
            project.cells.insert(key, layers);
        }
        for _ in 0..actor_count {
            let actor = StableGuid::from_u128(r.u128()?);
            if project
                .actors
                .last_key_value()
                .is_some_and(|(previous, _)| *previous >= actor)
            {
                return Err(ProjectError::Malformed("actor order"));
            }
            project.actors.insert(actor, CellKey::new(r.coord()?));
        }
        if r.0.len() != 8 {
            return Err(ProjectError::Malformed("trailing bytes"));
        }
        project.validate(limits)?;
        Ok(project)
    }
}

fn put_coord(out: &mut Vec<u8>, cell: CellCoord) {
    for n in [cell.x, cell.y, cell.z] {
        out.extend(n.to_le_bytes());
    }
}
struct Reader<'a>(&'a [u8]);
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], ProjectError> {
        if n > self.0.len() {
            return Err(ProjectError::Malformed("truncated record"));
        }
        let (head, tail) = self.0.split_at(n);
        self.0 = tail;
        Ok(head)
    }
    fn byte(&mut self) -> Result<u8, ProjectError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, ProjectError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("fixed slice"),
        ))
    }
    fn u64(&mut self) -> Result<u64, ProjectError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("fixed slice"),
        ))
    }
    fn u128(&mut self) -> Result<u128, ProjectError> {
        Ok(u128::from_le_bytes(
            self.take(16)?.try_into().expect("fixed slice"),
        ))
    }
    fn coord(&mut self) -> Result<CellCoord, ProjectError> {
        Ok(CellCoord::new(
            self.u32()? as i32,
            self.u32()? as i32,
            self.u32()? as i32,
        ))
    }
    fn count(&mut self, limit: usize, name: &'static str) -> Result<usize, ProjectError> {
        let n = self.u32()? as usize;
        if n > limit {
            return Err(ProjectError::Budget(name));
        }
        Ok(n)
    }
}
