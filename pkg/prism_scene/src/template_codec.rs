//! Deterministic binary codec for [`SceneTemplate`](crate::SceneTemplate).
//!
//! The codec is deliberately independent of any asset container. A container may
//! store these bytes under the template's stable GUID; decoding remains bounded,
//! deterministic, and validates the complete descriptor before publication.

use alloc::string::String;
use alloc::vec::Vec;

use prism_asset::StableGuid;

use crate::identity::{LocalEntityId, NameTable};
use crate::plan_codec::{decode_plan, encode_plan, PlanCodecError, PlanDecodeLimits};
use crate::residency::{ResidencyPlan, ResidencyTier};
use crate::template::{NestedSlot, SceneTemplate, TemplateError};

const MAGIC: &[u8; 4] = b"PRST";
const VERSION: u16 = 1;

/// Bounds applied while decoding an untrusted baked template.
#[allow(missing_docs)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TemplateDecodeLimits {
    pub max_bytes: usize,
    pub max_plan_bytes: usize,
    pub max_names: usize,
    pub max_name_bytes: usize,
    pub max_nested: usize,
    pub max_tiers: usize,
    pub max_dependencies: usize,
    pub plan: PlanDecodeLimits,
}

impl Default for TemplateDecodeLimits {
    fn default() -> Self {
        Self {
            max_bytes: 256 * 1024 * 1024,
            max_plan_bytes: 256 * 1024 * 1024,
            max_names: 1_048_576,
            max_name_bytes: 16 * 1024,
            max_nested: 1_048_576,
            max_tiers: 16_777_216,
            max_dependencies: 1_048_576,
            plan: PlanDecodeLimits::default(),
        }
    }
}

/// A malformed or unsafe template wire representation.
#[allow(missing_docs)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TemplateCodecError {
    BadMagic,
    UnsupportedVersion(u16),
    Truncated,
    TrailingBytes,
    ResourceLimit(&'static str),
    InvalidValue(&'static str),
    InvalidPlan(PlanCodecError),
    InvalidTemplate(TemplateError),
}

/// Encodes a validated template in the stable version-1 binary format.
pub fn encode_template(template: &SceneTemplate) -> Result<Vec<u8>, TemplateCodecError> {
    template
        .validate()
        .map_err(TemplateCodecError::InvalidTemplate)?;
    let plan = encode_plan(&template.plan).map_err(TemplateCodecError::InvalidPlan)?;
    let mut writer = Writer(Vec::new());
    writer.bytes(MAGIC);
    writer.u16(VERSION);
    writer.guid(template.guid);
    writer.u32(checked_len(plan.len(), "plan bytes")?);
    writer.bytes(&plan);
    writer.u32(checked_len(template.names.len(), "names")?);
    for (name, local) in template.names.iter() {
        writer.string(name)?;
        writer.u32(local.index());
    }
    writer.u32(checked_len(template.nested.len(), "nested")?);
    for slot in template.nested.iter() {
        writer.u32(slot.root.index());
        writer.guid(slot.template);
    }
    writer.u32(checked_len(template.tiers.len(), "tiers")?);
    for (local, tier) in template.tiers.iter() {
        writer.u32(local.index());
        writer.u8(match tier {
            ResidencyTier::Interactive => 0,
            ResidencyTier::Streamed => 1,
            ResidencyTier::Proxy => 2,
        });
    }
    writer.u32(checked_len(template.deps.len(), "dependencies")?);
    for dependency in template.deps.iter().copied() {
        writer.guid(dependency);
    }
    Ok(writer.0)
}

/// Decodes and validates a template from the stable version-1 binary format.
pub fn decode_template(
    bytes: &[u8],
    limits: TemplateDecodeLimits,
) -> Result<SceneTemplate, TemplateCodecError> {
    if bytes.len() > limits.max_bytes {
        return Err(TemplateCodecError::ResourceLimit("template bytes"));
    }
    let mut reader = Reader::new(bytes);
    if reader.take(4)? != MAGIC {
        return Err(TemplateCodecError::BadMagic);
    }
    let version = reader.u16()?;
    if version != VERSION {
        return Err(TemplateCodecError::UnsupportedVersion(version));
    }
    let guid = reader.guid()?;
    let plan_len = reader.count("plan bytes", limits.max_plan_bytes)?;
    let plan_bytes = reader.take(plan_len)?;
    let plan = decode_plan(plan_bytes, limits.plan).map_err(TemplateCodecError::InvalidPlan)?;

    let name_count = reader.count("names", limits.max_names)?;
    let mut names = NameTable::new();
    for _ in 0..name_count {
        let name = reader.string(limits.max_name_bytes)?;
        let local = LocalEntityId::new(reader.u32()?);
        if names.insert(name, local).is_some() {
            return Err(TemplateCodecError::InvalidValue("duplicate name"));
        }
    }
    let nested_count = reader.count("nested", limits.max_nested)?;
    let mut nested = Vec::with_capacity(nested_count);
    for _ in 0..nested_count {
        nested.push(NestedSlot::new(
            LocalEntityId::new(reader.u32()?),
            reader.guid()?,
        ));
    }
    let tier_count = reader.count("tiers", limits.max_tiers)?;
    let mut tiers = ResidencyPlan::new();
    for _ in 0..tier_count {
        let local = LocalEntityId::new(reader.u32()?);
        let tier = match reader.u8()? {
            0 => ResidencyTier::Interactive,
            1 => ResidencyTier::Streamed,
            2 => ResidencyTier::Proxy,
            _ => return Err(TemplateCodecError::InvalidValue("residency tier")),
        };
        if tiers.set(local, tier).is_some() {
            return Err(TemplateCodecError::InvalidValue("duplicate tier"));
        }
    }
    let dependency_count = reader.count("dependencies", limits.max_dependencies)?;
    let mut deps = Vec::with_capacity(dependency_count);
    for _ in 0..dependency_count {
        deps.push(reader.guid()?);
    }
    if !reader.is_empty() {
        return Err(TemplateCodecError::TrailingBytes);
    }
    let template = SceneTemplate {
        guid,
        plan,
        names,
        nested: nested.into_boxed_slice(),
        tiers,
        deps: deps.into_boxed_slice(),
    };
    template
        .validate()
        .map_err(TemplateCodecError::InvalidTemplate)?;
    Ok(template)
}

fn checked_len(len: usize, kind: &'static str) -> Result<u32, TemplateCodecError> {
    u32::try_from(len).map_err(|_| TemplateCodecError::ResourceLimit(kind))
}

struct Writer(Vec<u8>);
impl Writer {
    fn bytes(&mut self, value: &[u8]) {
        self.0.extend_from_slice(value);
    }
    fn u8(&mut self, value: u8) {
        self.0.push(value);
    }
    fn u16(&mut self, value: u16) {
        self.bytes(&value.to_le_bytes());
    }
    fn u32(&mut self, value: u32) {
        self.bytes(&value.to_le_bytes());
    }
    fn guid(&mut self, value: StableGuid) {
        self.bytes(&value.to_u128().to_le_bytes());
    }
    fn string(&mut self, value: &str) -> Result<(), TemplateCodecError> {
        let len = checked_len(value.len(), "name bytes")?;
        self.u32(len);
        self.bytes(value.as_bytes());
        Ok(())
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }
    fn take(&mut self, len: usize) -> Result<&'a [u8], TemplateCodecError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(TemplateCodecError::Truncated)?;
        if end > self.bytes.len() {
            return Err(TemplateCodecError::Truncated);
        }
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }
    fn u8(&mut self) -> Result<u8, TemplateCodecError> {
        Ok(*self.take(1)?.first().unwrap())
    }
    fn u16(&mut self) -> Result<u16, TemplateCodecError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, TemplateCodecError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn guid(&mut self) -> Result<StableGuid, TemplateCodecError> {
        Ok(StableGuid::from_u128(u128::from_le_bytes(
            self.take(16)?.try_into().unwrap(),
        )))
    }
    fn count(&mut self, kind: &'static str, max: usize) -> Result<usize, TemplateCodecError> {
        let value = self.u32()? as usize;
        if value > max {
            return Err(TemplateCodecError::ResourceLimit(kind));
        }
        Ok(value)
    }
    fn string(&mut self, max: usize) -> Result<String, TemplateCodecError> {
        let len = self.count("name bytes", max)?;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| TemplateCodecError::InvalidValue("name utf8"))
    }
    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overrides::ArchetypeSignature;
    use crate::plan::{ArchetypeGroup, InstantiationPlan};
    use alloc::boxed::Box;

    fn template() -> SceneTemplate {
        let plan = InstantiationPlan {
            entity_count: 1,
            groups: Box::new([ArchetypeGroup::new(
                ArchetypeSignature::default(),
                Box::new([LocalEntityId::new(0)]),
                Box::new([]),
            )]),
            entity_valued_fields: Box::new([]),
            hierarchy: Box::new([]),
            reserved_local_base: LocalEntityId::RESERVED_BASE,
        };
        let mut value = SceneTemplate::new(StableGuid::from_u128(1), plan);
        value.names.insert("root", LocalEntityId::new(0));
        value.nested = Box::new([NestedSlot::new(
            LocalEntityId::new(0),
            StableGuid::from_u128(2),
        )]);
        value.tiers.set(LocalEntityId::new(0), ResidencyTier::Proxy);
        value.deps = Box::new([StableGuid::from_u128(2)]);
        value
    }

    #[test]
    fn round_trip_is_deterministic() {
        let value = template();
        let encoded = encode_template(&value).unwrap();
        assert_eq!(encoded, encode_template(&value).unwrap());
        assert_eq!(
            decode_template(&encoded, TemplateDecodeLimits::default()).unwrap(),
            value
        );
    }

    #[test]
    fn rejects_truncation_and_trailing_bytes() {
        let encoded = encode_template(&template()).unwrap();
        assert_eq!(
            decode_template(
                &encoded[..encoded.len() - 1],
                TemplateDecodeLimits::default()
            ),
            Err(TemplateCodecError::Truncated)
        );
        let mut trailing = encoded;
        trailing.push(0);
        assert_eq!(
            decode_template(&trailing, TemplateDecodeLimits::default()),
            Err(TemplateCodecError::TrailingBytes)
        );
    }

    #[test]
    fn rejects_invalid_template_after_decode() {
        let mut value = template();
        value.nested[0] = NestedSlot::new(LocalEntityId::new(1), StableGuid::from_u128(3));
        let encoded = encode_template(&template()).unwrap();
        let _ = value;
        let mut invalid = encoded;
        // Replace the nested root while preserving the wire shape.
        let root_offset = 4 + 2 + 16 + 4 + encode_plan(&template().plan).unwrap().len() + 4 + 4 + 4;
        invalid[root_offset..root_offset + 4].copy_from_slice(&1u32.to_le_bytes());
        assert!(matches!(
            decode_template(&invalid, TemplateDecodeLimits::default()),
            Err(TemplateCodecError::InvalidTemplate(_))
        ));
    }
}
