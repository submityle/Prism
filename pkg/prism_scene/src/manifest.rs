//! Deterministic packaging wrapper for a validated [`SceneTemplate`].
//!
//! This is a manifest/container contract, not an authoring bake pipeline: it
//! does not perform source import, reflection migration, CAS storage, or I/O.

use crate::{
    decode_template, encode_template, SceneTemplate, TemplateCodecError, TemplateDecodeLimits,
};
use alloc::boxed::Box;
use alloc::vec::Vec;
use prism_asset::StableGuid;

const MAGIC: &[u8; 4] = b"PMNF";
const VERSION: u16 = 1;

/// A deterministic runtime manifest containing one encoded template payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SceneManifest {
    /// Identity duplicated outside the payload for catalog lookup.
    pub guid: StableGuid,
    /// Manifest schema version.
    pub schema: u32,
    /// Canonical dependency closure advertised by the template.
    pub dependencies: Box<[StableGuid]>,
    /// PRST template payload.
    pub payload: Vec<u8>,
}

/// Limits for decoding an untrusted manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ManifestDecodeLimits {
    /// Maximum complete manifest bytes.
    pub max_bytes: usize,
    /// Maximum dependency count.
    pub max_dependencies: usize,
    /// Maximum embedded payload bytes.
    pub max_payload_bytes: usize,
    /// Limits forwarded to the embedded template decoder.
    pub template: TemplateDecodeLimits,
}
impl Default for ManifestDecodeLimits {
    fn default() -> Self {
        Self {
            max_bytes: 256 * 1024 * 1024,
            max_dependencies: 1_048_576,
            max_payload_bytes: 256 * 1024 * 1024,
            template: TemplateDecodeLimits::default(),
        }
    }
}

/// Manifest construction or wire decoding failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ManifestError {
    /// Template validation or embedded payload failure.
    Template(TemplateCodecError),
    /// Header magic is wrong.
    BadMagic,
    /// Wire version is unsupported.
    UnsupportedVersion(u16),
    /// Input ended early.
    Truncated,
    /// Bytes remained after decoding.
    TrailingBytes,
    /// A configured resource bound was exceeded.
    ResourceLimit(&'static str),
    /// Header metadata disagrees with the embedded template.
    MetadataMismatch(&'static str),
}

impl SceneManifest {
    /// Packages a validated template without claiming source-graph baking.
    pub fn from_template(template: &SceneTemplate) -> Result<Self, ManifestError> {
        let payload = encode_template(template).map_err(ManifestError::Template)?;
        Ok(Self {
            guid: template.guid,
            schema: template.schema,
            dependencies: template.deps.clone(),
            payload,
        })
    }
    /// Decodes and verifies the embedded template and duplicated metadata.
    pub fn to_template(
        &self,
        limits: TemplateDecodeLimits,
    ) -> Result<SceneTemplate, ManifestError> {
        let template = decode_template(&self.payload, limits).map_err(ManifestError::Template)?;
        if self.schema != template.schema {
            return Err(ManifestError::MetadataMismatch("schema"));
        }
        if template.guid != self.guid {
            return Err(ManifestError::MetadataMismatch("guid"));
        }
        if template.deps.as_ref() != self.dependencies.as_ref() {
            return Err(ManifestError::MetadataMismatch("dependencies"));
        }
        Ok(template)
    }
}

/// Encodes a manifest with strict lengths and deterministic dependency order.
pub fn encode_manifest(manifest: &SceneManifest) -> Result<Vec<u8>, ManifestError> {
    if manifest
        .dependencies
        .windows(2)
        .any(|pair| pair[0] >= pair[1])
    {
        return Err(ManifestError::MetadataMismatch("dependency order"));
    }
    let _ = manifest.to_template(TemplateDecodeLimits::default())?;
    let dep_bytes = manifest
        .dependencies
        .len()
        .checked_mul(16)
        .ok_or(ManifestError::ResourceLimit("dependencies"))?;
    let total = 4usize
        .checked_add(2)
        .and_then(|n| n.checked_add(4 + 16 + 4 + dep_bytes + 4 + manifest.payload.len()))
        .ok_or(ManifestError::ResourceLimit("manifest bytes"))?;
    if manifest.dependencies.len() > u32::MAX as usize || manifest.payload.len() > u32::MAX as usize
    {
        return Err(ManifestError::ResourceLimit("manifest length"));
    }
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&manifest.schema.to_le_bytes());
    out.extend_from_slice(&manifest.guid.to_u128().to_le_bytes());
    out.extend_from_slice(&(manifest.dependencies.len() as u32).to_le_bytes());
    for guid in &manifest.dependencies {
        out.extend_from_slice(&guid.to_u128().to_le_bytes());
    }
    out.extend_from_slice(&(manifest.payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&manifest.payload);
    Ok(out)
}

/// Decodes a manifest and verifies its embedded template metadata.
pub fn decode_manifest(
    bytes: &[u8],
    limits: ManifestDecodeLimits,
) -> Result<SceneManifest, ManifestError> {
    if bytes.len() > limits.max_bytes {
        return Err(ManifestError::ResourceLimit("manifest bytes"));
    }
    let mut r = Reader { bytes, at: 0 };
    if r.take(4)? != MAGIC {
        return Err(ManifestError::BadMagic);
    }
    let version = r.u16()?;
    if version != VERSION {
        return Err(ManifestError::UnsupportedVersion(version));
    }
    let schema = r.u32()?;
    let guid = StableGuid::from_u128(r.u128()?);
    let count = r.u32()? as usize;
    if count > limits.max_dependencies {
        return Err(ManifestError::ResourceLimit("dependencies"));
    }
    let mut deps = Vec::with_capacity(count);
    for _ in 0..count {
        deps.push(StableGuid::from_u128(r.u128()?));
    }
    if deps.windows(2).any(|p| p[0] >= p[1]) {
        return Err(ManifestError::MetadataMismatch("dependency order"));
    }
    let payload_len = r.u32()? as usize;
    if payload_len > limits.max_payload_bytes {
        return Err(ManifestError::ResourceLimit("payload"));
    }
    let payload = r.take(payload_len)?.to_vec();
    if r.at != bytes.len() {
        return Err(ManifestError::TrailingBytes);
    }
    let manifest = SceneManifest {
        guid,
        schema,
        dependencies: deps.into_boxed_slice(),
        payload,
    };
    if manifest.schema == 0 {
        return Err(ManifestError::MetadataMismatch("schema"));
    }
    manifest.to_template(limits.template)?;
    Ok(manifest)
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], ManifestError> {
        let end = self.at.checked_add(n).ok_or(ManifestError::Truncated)?;
        let value = self
            .bytes
            .get(self.at..end)
            .ok_or(ManifestError::Truncated)?;
        self.at = end;
        Ok(value)
    }
    fn u16(&mut self) -> Result<u16, ManifestError> {
        let b = self.take(2)?;
        let v = u16::from_le_bytes([b[0], b[1]]);
        Ok(v)
    }
    fn u32(&mut self) -> Result<u32, ManifestError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u128(&mut self) -> Result<u128, ManifestError> {
        let b = self.take(16)?;
        Ok(u128::from_le_bytes(
            b.try_into().map_err(|_| ManifestError::Truncated)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SceneTemplate;

    #[test]
    fn manifest_round_trip_is_deterministic() {
        let template = SceneTemplate::empty(StableGuid::from_u128(7));
        let manifest = SceneManifest::from_template(&template).unwrap();
        let encoded = encode_manifest(&manifest).unwrap();
        assert_eq!(encoded, encode_manifest(&manifest).unwrap());
        let decoded = decode_manifest(&encoded, ManifestDecodeLimits::default()).unwrap();
        assert_eq!(
            decoded
                .to_template(TemplateDecodeLimits::default())
                .unwrap(),
            template
        );
    }

    #[test]
    fn manifest_rejects_trailing_bytes() {
        let template = SceneTemplate::empty(StableGuid::from_u128(8));
        let manifest = SceneManifest::from_template(&template).unwrap();
        let mut encoded = encode_manifest(&manifest).unwrap();
        encoded.push(1);
        assert_eq!(
            decode_manifest(&encoded, ManifestDecodeLimits::default()),
            Err(ManifestError::TrailingBytes)
        );
    }
}
