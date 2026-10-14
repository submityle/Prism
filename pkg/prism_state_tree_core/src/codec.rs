//! Stable, typed instance-frame values and cold-path snapshot migration.

use alloc::{vec, vec::Vec};

use crate::{BindingSlot, FrameError, FrameLayout, InstanceFrame};

/// Asset-stable identity of a field, independent of its frame offset and slot index.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SlotGuid(pub u128);

/// Fixed-width, little-endian wire representation of a frame field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SlotType {
    /// One byte, restricted to zero or one.
    Bool,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 64-bit integer.
    U64,
    /// Signed 64-bit integer.
    I64,
    /// IEEE-754 32-bit float (bit-preserving).
    F32,
    /// IEEE-754 64-bit float (bit-preserving).
    F64,
}

impl SlotType {
    const fn size(self) -> usize {
        match self {
            Self::Bool => 1,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    const fn tag(self) -> u8 {
        match self {
            Self::Bool => 1,
            Self::U32 => 2,
            Self::I32 => 3,
            Self::U64 => 4,
            Self::I64 => 5,
            Self::F32 => 6,
            Self::F64 => 7,
        }
    }
}

/// Borrow-free scalar that can be written to or read from a typed frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SlotValue {
    /// Boolean value.
    Bool(bool),
    /// Unsigned 32-bit integer.
    U32(u32),
    /// Signed 32-bit integer.
    I32(i32),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// Signed 64-bit integer.
    I64(i64),
    /// IEEE-754 32-bit float.
    F32(f32),
    /// IEEE-754 64-bit float.
    F64(f64),
}

impl SlotValue {
    /// Returns the type tag used by the frame codec.
    pub const fn slot_type(self) -> SlotType {
        match self {
            Self::Bool(_) => SlotType::Bool,
            Self::U32(_) => SlotType::U32,
            Self::I32(_) => SlotType::I32,
            Self::U64(_) => SlotType::U64,
            Self::I64(_) => SlotType::I64,
            Self::F32(_) => SlotType::F32,
            Self::F64(_) => SlotType::F64,
        }
    }

    fn bytes(self) -> [u8; 8] {
        match self {
            Self::Bool(value) => [u8::from(value), 0, 0, 0, 0, 0, 0, 0],
            Self::U32(value) => pad32(value.to_le_bytes()),
            Self::I32(value) => pad32(value.to_le_bytes()),
            Self::U64(value) => value.to_le_bytes(),
            Self::I64(value) => value.to_le_bytes(),
            Self::F32(value) => pad32(value.to_bits().to_le_bytes()),
            Self::F64(value) => value.to_bits().to_le_bytes(),
        }
    }
}

const fn pad32(value: [u8; 4]) -> [u8; 8] {
    [value[0], value[1], value[2], value[3], 0, 0, 0, 0]
}

/// Compile-time declaration for one typed field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlotDecl {
    /// Runtime binding slot (can change across revisions).
    pub slot: BindingSlot,
    /// Stable identity (must not be recycled for another meaning).
    pub guid: SlotGuid,
    /// Fixed-width wire type.
    pub kind: SlotType,
    /// Schema version for this field.
    pub version: u32,
    /// Value used for fresh frames and newly introduced fields.
    pub default: SlotValue,
}

/// A compiled, shareable codec; encoding and decoding do not allocate.
#[derive(Clone, Debug)]
pub struct SlotCodec {
    declarations: Vec<SlotDecl>,
    layout: FrameLayout,
    fingerprint: u64,
}

/// Owned, checked wire image of a frame (padding is canonicalized).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrameSnapshot {
    /// Fingerprint of the exact slot schema and layout.
    pub fingerprint: u64,
    /// Canonical little-endian field bytes with zeroed padding.
    pub bytes: Vec<u8>,
    /// Stable integrity hash of fingerprint, length and bytes.
    pub checksum: u64,
}

/// Precompiled mapping from old field identities to a new layout.
#[derive(Clone, Debug)]
pub struct FrameMigration {
    old_fingerprint: u64,
    new_fingerprint: u64,
    copies: Vec<(BindingSlot, BindingSlot, usize)>,
}

fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x100000001b3);
    }
}

impl FrameSnapshot {
    fn checksum(fingerprint: u64, bytes: &[u8]) -> u64 {
        let mut hash = 0xcbf29ce484222325;
        hash_bytes(&mut hash, &fingerprint.to_le_bytes());
        hash_bytes(&mut hash, &(bytes.len() as u64).to_le_bytes());
        hash_bytes(&mut hash, bytes);
        hash
    }
}

impl SlotCodec {
    /// Validates identities, types and versions and computes a stable wire layout.
    pub fn compile(mut declarations: Vec<SlotDecl>) -> Result<Self, FrameError> {
        declarations.sort_by_key(|decl| decl.slot);
        for decl in &declarations {
            if decl.default.slot_type() != decl.kind || decl.version == 0 {
                return Err(FrameError::InvalidField { slot: decl.slot });
            }
        }
        let layout = FrameLayout::compile(
            declarations
                .iter()
                .map(|decl| (decl.slot, decl.kind.size(), decl.kind.size()))
                .collect(),
        )?;
        let mut guids: Vec<_> = declarations.iter().map(|decl| decl.guid).collect();
        guids.sort_unstable();
        for pair in guids.windows(2) {
            if pair[0] == pair[1] {
                return Err(FrameError::DuplicateGuid(pair[0]));
            }
        }
        let mut fingerprint = 0xcbf29ce484222325;
        hash_bytes(&mut fingerprint, &layout.fingerprint().to_le_bytes());
        for decl in &declarations {
            hash_bytes(&mut fingerprint, &decl.guid.0.to_le_bytes());
            hash_bytes(&mut fingerprint, &[decl.kind.tag()]);
            hash_bytes(&mut fingerprint, &decl.version.to_le_bytes());
            hash_bytes(&mut fingerprint, &decl.default.bytes()[..decl.kind.size()]);
        }
        Ok(Self {
            declarations,
            layout,
            fingerprint,
        })
    }

    /// Returns the exact wire identity, including schema and defaults.
    pub const fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// Returns the compiled byte layout.
    pub const fn layout(&self) -> &FrameLayout {
        &self.layout
    }

    /// Allocates a frame and installs all declared defaults.
    pub fn new_frame(&self) -> InstanceFrame {
        let mut frame = InstanceFrame::new(self.layout.clone());
        for decl in &self.declarations {
            self.encode(&mut frame, decl.slot, decl.default)
                .expect("validated default and matching layout");
        }
        frame
    }

    fn decl(&self, slot: BindingSlot) -> Result<&SlotDecl, FrameError> {
        self.declarations
            .binary_search_by_key(&slot, |decl| decl.slot)
            .map(|index| &self.declarations[index])
            .map_err(|_| FrameError::MissingSlot(slot))
    }

    fn check_frame(&self, frame: &InstanceFrame) -> Result<(), FrameError> {
        if frame.layout() != &self.layout {
            return Err(FrameError::LayoutMismatch);
        }
        Ok(())
    }

    /// Writes one scalar after checking both the codec and the target layout.
    pub fn encode(
        &self,
        frame: &mut InstanceFrame,
        slot: BindingSlot,
        value: SlotValue,
    ) -> Result<(), FrameError> {
        self.check_frame(frame)?;
        let decl = self.decl(slot)?;
        if decl.kind != value.slot_type() {
            return Err(FrameError::TypeMismatch(slot));
        }
        frame.write(
            slot,
            0..decl.kind.size(),
            &value.bytes()[..decl.kind.size()],
        )
    }

    /// Reads one scalar; malformed boolean bytes are rejected.
    pub fn decode(
        &self,
        frame: &InstanceFrame,
        slot: BindingSlot,
    ) -> Result<SlotValue, FrameError> {
        self.check_frame(frame)?;
        let kind = self.decl(slot)?.kind;
        let bytes = frame.read(slot, 0..kind.size())?;
        let mut raw = [0; 8];
        raw[..bytes.len()].copy_from_slice(bytes);
        Ok(match kind {
            SlotType::Bool => match raw[0] {
                0 => SlotValue::Bool(false),
                1 => SlotValue::Bool(true),
                _ => return Err(FrameError::InvalidValue(slot)),
            },
            SlotType::U32 => SlotValue::U32(u32::from_le_bytes(raw[..4].try_into().unwrap())),
            SlotType::I32 => SlotValue::I32(i32::from_le_bytes(raw[..4].try_into().unwrap())),
            SlotType::U64 => SlotValue::U64(u64::from_le_bytes(raw)),
            SlotType::I64 => SlotValue::I64(i64::from_le_bytes(raw)),
            SlotType::F32 => SlotValue::F32(f32::from_bits(u32::from_le_bytes(
                raw[..4].try_into().unwrap(),
            ))),
            SlotType::F64 => SlotValue::F64(f64::from_bits(u64::from_le_bytes(raw))),
        })
    }

    /// Captures a canonical image; never serializes uninitialized or nonzero padding.
    pub fn snapshot(&self, frame: &InstanceFrame) -> Result<FrameSnapshot, FrameError> {
        self.check_frame(frame)?;
        let mut bytes = vec![0; self.layout.size()];
        for field in self.layout.fields() {
            self.decode(frame, field.slot)?;
            bytes[field.offset..field.offset + field.size]
                .copy_from_slice(frame.read(field.slot, 0..field.size)?);
        }
        Ok(FrameSnapshot {
            fingerprint: self.fingerprint,
            checksum: FrameSnapshot::checksum(self.fingerprint, &bytes),
            bytes,
        })
    }

    /// Validates a snapshot before atomically replacing an existing frame's bytes.
    pub fn restore(
        &self,
        frame: &mut InstanceFrame,
        snapshot: &FrameSnapshot,
    ) -> Result<(), FrameError> {
        self.check_frame(frame)?;
        if snapshot.fingerprint != self.fingerprint {
            return Err(FrameError::LayoutMismatch);
        }
        if snapshot.bytes.len() != self.layout.size()
            || snapshot.checksum != FrameSnapshot::checksum(snapshot.fingerprint, &snapshot.bytes)
        {
            return Err(FrameError::CorruptSnapshot);
        }
        self.validate_bytes(&snapshot.bytes)?;
        frame.replace_bytes(&snapshot.bytes)
    }

    fn validate_bytes(&self, bytes: &[u8]) -> Result<(), FrameError> {
        let mut previous_end = 0;
        for field in self.layout.fields() {
            if bytes[previous_end..field.offset]
                .iter()
                .any(|&byte| byte != 0)
            {
                return Err(FrameError::CorruptSnapshot);
            }
            if self.decl(field.slot)?.kind == SlotType::Bool && bytes[field.offset] > 1 {
                return Err(FrameError::InvalidValue(field.slot));
            }
            previous_end = field.offset + field.size;
        }
        if bytes[previous_end..].iter().any(|&byte| byte != 0) {
            return Err(FrameError::CorruptSnapshot);
        }
        Ok(())
    }
}

impl FrameMigration {
    /// Compiles a GUID-based mapping. Changed types or field versions require an explicit converter.
    pub fn compile(old: &SlotCodec, new: &SlotCodec) -> Result<Self, FrameError> {
        let mut copies = Vec::new();
        for target in &new.declarations {
            if let Some(source) = old
                .declarations
                .iter()
                .find(|decl| decl.guid == target.guid)
            {
                if source.kind != target.kind || source.version != target.version {
                    return Err(FrameError::IncompatibleGuid(target.guid));
                }
                copies.push((source.slot, target.slot, target.kind.size()));
            }
        }
        Ok(Self {
            old_fingerprint: old.fingerprint,
            new_fingerprint: new.fingerprint,
            copies,
        })
    }

    /// Applies the mapping to a validated old snapshot, initializing new fields from defaults.
    pub fn migrate(
        &self,
        old: &SlotCodec,
        new: &SlotCodec,
        snapshot: &FrameSnapshot,
    ) -> Result<FrameSnapshot, FrameError> {
        if old.fingerprint != self.old_fingerprint || new.fingerprint != self.new_fingerprint {
            return Err(FrameError::LayoutMismatch);
        }
        let mut source = old.new_frame();
        old.restore(&mut source, snapshot)?;
        let mut target = new.new_frame();
        for &(from, to, size) in &self.copies {
            target.write(to, 0..size, source.read(from, 0..size)?)?;
        }
        new.snapshot(&target)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::{
        FrameMigration, FrameSnapshot, SlotCodec, SlotDecl, SlotGuid, SlotValue,
    };
    use crate::{BindingSlot, FrameError};

    fn decl(slot: u32, guid: u128, default: SlotValue) -> SlotDecl {
        SlotDecl {
            slot: BindingSlot(slot),
            guid: SlotGuid(guid),
            kind: default.slot_type(),
            version: 1,
            default,
        }
    }

    #[test]
    fn typed_roundtrip_defaults_and_errors_leave_bytes_untouched() {
        let codec = SlotCodec::compile(vec![
            decl(2, 12, SlotValue::F64(1.25)),
            decl(0, 10, SlotValue::Bool(true)),
            decl(1, 11, SlotValue::I32(-17)),
        ])
        .unwrap();
        let mut frame = codec.new_frame();
        assert_eq!(
            codec.decode(&frame, BindingSlot(0)),
            Ok(SlotValue::Bool(true))
        );
        assert_eq!(
            codec.decode(&frame, BindingSlot(1)),
            Ok(SlotValue::I32(-17))
        );
        assert_eq!(
            codec.decode(&frame, BindingSlot(2)),
            Ok(SlotValue::F64(1.25))
        );
        let before = frame.bytes().to_vec();
        assert_eq!(
            codec.encode(&mut frame, BindingSlot(1), SlotValue::U32(1)),
            Err(FrameError::TypeMismatch(BindingSlot(1)))
        );
        assert_eq!(before, frame.bytes());
        codec
            .encode(&mut frame, BindingSlot(1), SlotValue::I32(-42))
            .unwrap();
        assert_eq!(
            frame.read(BindingSlot(1), 0..4).unwrap(),
            &(-42_i32).to_le_bytes()
        );
        let snapshot = codec.snapshot(&frame).unwrap();
        let mut restored = codec.new_frame();
        codec.restore(&mut restored, &snapshot).unwrap();
        assert_eq!(frame, restored);
        assert_eq!(codec.snapshot(&restored).unwrap(), snapshot);
    }

    #[test]
    fn detects_corruption_padding_and_wrong_codec_without_mutation() {
        let codec = SlotCodec::compile(vec![
            decl(0, 1, SlotValue::Bool(false)),
            decl(1, 2, SlotValue::U64(0)),
        ])
        .unwrap();
        let mut frame = codec.new_frame();
        let original = frame.clone();
        let mut snapshot = codec.snapshot(&frame).unwrap();
        snapshot.bytes[0] = 2;
        snapshot.checksum = FrameSnapshot::checksum(snapshot.fingerprint, &snapshot.bytes);
        assert_eq!(
            codec.restore(&mut frame, &snapshot),
            Err(FrameError::InvalidValue(BindingSlot(0)))
        );
        assert_eq!(frame, original);
        snapshot.bytes[0] = 0;
        snapshot.bytes[1] = 9;
        snapshot.checksum = FrameSnapshot::checksum(snapshot.fingerprint, &snapshot.bytes);
        assert_eq!(
            codec.restore(&mut frame, &snapshot),
            Err(FrameError::CorruptSnapshot)
        );
        assert_eq!(frame, original);
        snapshot.bytes[1] = 0;
        assert_eq!(
            codec.restore(&mut frame, &snapshot),
            Err(FrameError::CorruptSnapshot)
        );
        let other = SlotCodec::compile(vec![decl(0, 3, SlotValue::Bool(false))]).unwrap();
        assert_eq!(
            other.restore(&mut other.new_frame(), &codec.snapshot(&frame).unwrap()),
            Err(FrameError::LayoutMismatch)
        );
    }

    #[test]
    fn migrates_by_guid_across_slot_reorder_and_add_remove() {
        let old = SlotCodec::compile(vec![
            decl(0, 100, SlotValue::I32(0)),
            decl(1, 200, SlotValue::U64(3)),
        ])
        .unwrap();
        let new = SlotCodec::compile(vec![
            decl(0, 200, SlotValue::U64(0)),
            decl(1, 300, SlotValue::Bool(true)),
        ])
        .unwrap();
        let mut source = old.new_frame();
        old.encode(&mut source, BindingSlot(1), SlotValue::U64(99))
            .unwrap();
        let original = old.snapshot(&source).unwrap();
        let migration = FrameMigration::compile(&old, &new).unwrap();
        let migrated = migration.migrate(&old, &new, &original).unwrap();
        let mut target = new.new_frame();
        new.restore(&mut target, &migrated).unwrap();
        assert_eq!(new.decode(&target, BindingSlot(0)), Ok(SlotValue::U64(99)));
        assert_eq!(
            new.decode(&target, BindingSlot(1)),
            Ok(SlotValue::Bool(true))
        );
        assert_eq!(old.snapshot(&source).unwrap(), original);
    }

    #[test]
    fn rejects_duplicate_guids_and_incompatible_versions_or_types() {
        assert!(matches!(
            SlotCodec::compile(vec![
                decl(0, 1, SlotValue::U32(0)),
                decl(1, 1, SlotValue::U32(1))
            ]),
            Err(FrameError::DuplicateGuid(SlotGuid(1)))
        ));
        let old = SlotCodec::compile(vec![decl(0, 1, SlotValue::U32(0))]).unwrap();
        let next = SlotCodec::compile(vec![decl(0, 1, SlotValue::I32(0))]).unwrap();
        assert!(matches!(
            FrameMigration::compile(&old, &next),
            Err(FrameError::IncompatibleGuid(SlotGuid(1)))
        ));
        let mut changed = decl(0, 1, SlotValue::U32(0));
        changed.version = 2;
        let next = SlotCodec::compile(vec![changed]).unwrap();
        assert!(matches!(
            FrameMigration::compile(&old, &next),
            Err(FrameError::IncompatibleGuid(SlotGuid(1)))
        ));
    }
}
