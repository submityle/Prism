//! Stable persistence for host-owned panel instance identities.

use alloc::{string::String, vec::Vec};
use core::fmt;

use crate::{
    PanelInstanceId, PanelInstanceRegistry, PanelInstanceRegistryError, PanelInstanceState,
};

const MAGIC: &[u8; 4] = b"PRIN";
const VERSION: u16 = 1;
const MAX_BYTES: usize = 16 * 1024 * 1024;
const MAX_INSTANCES: usize = 65_536;
const MAX_PANEL_ID_BYTES: usize = 1024 * 1024;

/// Failure while decoding or encoding instance registry bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstancePersistError {
    /// Input does not contain a complete header or value.
    Truncated,
    /// Input magic is not `PRIN`.
    BadMagic,
    /// Wire version is unsupported.
    UnsupportedVersion(u16),
    /// Input contains bytes after the canonical value.
    TrailingBytes,
    /// A checked count or string length exceeds its bound.
    ResourceLimit(&'static str),
    /// A panel ID is not valid UTF-8.
    InvalidUtf8,
    /// A state tag is unknown.
    InvalidState(u8),
    /// Registry invariants were violated.
    Registry(PanelInstanceRegistryError),
}

impl fmt::Display for InstancePersistError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => formatter.write_str("truncated instance persistence data"),
            Self::BadMagic => formatter.write_str("bad instance persistence magic"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported instance version: {version}")
            }
            Self::TrailingBytes => formatter.write_str("trailing instance persistence bytes"),
            Self::ResourceLimit(kind) => {
                write!(formatter, "instance resource limit exceeded: {kind}")
            }
            Self::InvalidUtf8 => formatter.write_str("invalid UTF-8 in instance persistence data"),
            Self::InvalidState(value) => write!(formatter, "invalid instance state tag: {value}"),
            Self::Registry(error) => write!(formatter, "invalid instance registry: {error}"),
        }
    }
}

/// Encodes all retained instance records, including closed identities.
pub fn encode_instance_registry(
    registry: &PanelInstanceRegistry,
) -> Result<Vec<u8>, InstancePersistError> {
    if registry.len() > MAX_INSTANCES {
        return Err(InstancePersistError::ResourceLimit("instances"));
    }
    let mut writer = Writer { bytes: Vec::new() };
    writer.bytes(MAGIC);
    writer.u16(VERSION);
    writer.u64(registry.next_id());
    writer.u32(
        u32::try_from(registry.len())
            .map_err(|_| InstancePersistError::ResourceLimit("instances"))?,
    );
    for instance in registry.iter() {
        writer.u64(instance.id().get());
        writer.string(instance.panel_id())?;
        writer.u8(match instance.state() {
            PanelInstanceState::Open => 0,
            PanelInstanceState::Closed => 1,
        })?;
    }
    if writer.bytes.len() > MAX_BYTES {
        return Err(InstancePersistError::ResourceLimit("bytes"));
    }
    Ok(writer.bytes)
}

/// Decodes and validates a complete instance registry snapshot.
pub fn decode_instance_registry(
    bytes: &[u8],
) -> Result<PanelInstanceRegistry, InstancePersistError> {
    if bytes.len() > MAX_BYTES {
        return Err(InstancePersistError::ResourceLimit("bytes"));
    }
    let mut reader = Reader { bytes, offset: 0 };
    if reader.take(4)? != MAGIC {
        return Err(InstancePersistError::BadMagic);
    }
    let version = reader.u16()?;
    if version != VERSION {
        return Err(InstancePersistError::UnsupportedVersion(version));
    }
    let next_id = reader.u64()?;
    let count = reader.u32()? as usize;
    if count > MAX_INSTANCES {
        return Err(InstancePersistError::ResourceLimit("instances"));
    }
    let mut instances = Vec::with_capacity(count);
    for _ in 0..count {
        let id = PanelInstanceId::from_raw(reader.u64()?);
        let panel_id = reader.string()?;
        let state = match reader.u8()? {
            0 => PanelInstanceState::Open,
            1 => PanelInstanceState::Closed,
            value => return Err(InstancePersistError::InvalidState(value)),
        };
        instances.push(PanelInstanceRegistry::persisted_instance(
            id, panel_id, state,
        ));
    }
    if reader.offset != bytes.len() {
        return Err(InstancePersistError::TrailingBytes);
    }
    PanelInstanceRegistry::from_persisted(next_id, instances)
        .map_err(InstancePersistError::Registry)
}

struct Writer {
    bytes: Vec<u8>,
}
impl Writer {
    fn bytes(&mut self, value: &[u8]) {
        self.bytes.extend_from_slice(value);
    }
    fn u8(&mut self, value: u8) -> Result<(), InstancePersistError> {
        self.bytes.push(value);
        Ok(())
    }
    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    fn string(&mut self, value: &str) -> Result<(), InstancePersistError> {
        if value.len() > MAX_PANEL_ID_BYTES {
            return Err(InstancePersistError::ResourceLimit("panel ID"));
        }
        self.u32(
            u32::try_from(value.len())
                .map_err(|_| InstancePersistError::ResourceLimit("panel ID"))?,
        );
        self.bytes(value.as_bytes());
        Ok(())
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], InstancePersistError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or(InstancePersistError::Truncated)?;
        if end > self.bytes.len() {
            return Err(InstancePersistError::Truncated);
        }
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }
    fn u8(&mut self) -> Result<u8, InstancePersistError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, InstancePersistError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, InstancePersistError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, InstancePersistError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, InstancePersistError> {
        let len = self.u32()? as usize;
        if len > MAX_PANEL_ID_BYTES {
            return Err(InstancePersistError::ResourceLimit("panel ID"));
        }
        core::str::from_utf8(self.take(len)?)
            .map(String::from)
            .map_err(|_| InstancePersistError::InvalidUtf8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Instancing, PanelDescriptor, PanelRegistry};

    fn panels() -> PanelRegistry {
        let mut panels = PanelRegistry::new();
        panels
            .register(PanelDescriptor::new("logs", "Logs").with_instancing(Instancing::Multi))
            .unwrap();
        panels
    }

    #[test]
    fn roundtrip_preserves_open_closed_identity_and_next_id() {
        let panels = panels();
        let mut registry = PanelInstanceRegistry::new();
        let first = registry.create(&panels, "logs").unwrap();
        registry.create(&panels, "logs").unwrap();
        registry.close(first.id()).unwrap();
        let bytes = encode_instance_registry(&registry).unwrap();
        assert_eq!(decode_instance_registry(&bytes).unwrap(), registry);
    }

    #[test]
    fn rejects_trailing_bytes_and_invalid_state() {
        let panels = panels();
        let mut registry = PanelInstanceRegistry::new();
        registry.create(&panels, "logs").unwrap();
        let mut bytes = encode_instance_registry(&registry).unwrap();
        bytes.push(1);
        assert_eq!(
            decode_instance_registry(&bytes),
            Err(InstancePersistError::TrailingBytes)
        );
        let mut bad = encode_instance_registry(&registry).unwrap();
        *bad.last_mut().unwrap() = 7;
        assert_eq!(
            decode_instance_registry(&bad),
            Err(InstancePersistError::InvalidState(7))
        );
    }
}
