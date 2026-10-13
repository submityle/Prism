//! Versioned, renderer-independent persistence for workspaces.
//!
//! This module stores the ordered workspace/perspective collection and embeds
//! the existing validated [`DockLayout`](crate::DockLayout) codec for each
//! perspective. It intentionally does not perform file I/O or autosave.

use alloc::{string::String, vec::Vec};
use core::fmt;

use crate::{
    decode_layout, encode_layout, DecodeLimits, DecodeOptions, PanelRegistry, PersistError,
    WorkspaceError, WorkspaceManager,
};

const MAGIC: &[u8; 4] = b"PRWS";
const VERSION: u16 = 1;

/// Limits applied while decoding an untrusted workspace snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceDecodeLimits {
    /// Maximum encoded workspace snapshot size.
    pub max_bytes: usize,
    /// Maximum number of workspaces.
    pub max_workspaces: usize,
    /// Maximum number of perspectives in one workspace.
    pub max_perspectives: usize,
    /// Maximum UTF-8 byte length of a workspace or perspective name.
    pub max_name_bytes: usize,
    /// Maximum encoded layout size embedded in one perspective.
    pub max_layout_bytes: usize,
    /// Limits forwarded to the nested layout codec.
    pub layout: DecodeLimits,
}

impl Default for WorkspaceDecodeLimits {
    fn default() -> Self {
        Self {
            max_bytes: 32 * 1024 * 1024,
            max_workspaces: 256,
            max_perspectives: 256,
            max_name_bytes: 1024 * 1024,
            max_layout_bytes: 16 * 1024 * 1024,
            layout: DecodeLimits::default(),
        }
    }
}

/// Registry and resource policy used when decoding a workspace snapshot.
#[derive(Clone, Copy, Debug, Default)]
pub struct WorkspaceDecodeOptions<'a> {
    /// Registered panel identifiers used by nested layout decoding.
    pub known_panels: Option<&'a [&'a str]>,
    /// Registered document identifiers used by nested layout decoding.
    pub known_documents: Option<&'a [&'a str]>,
    /// Workspace and nested-layout resource limits.
    pub limits: WorkspaceDecodeLimits,
}

/// A structured workspace persistence or validation failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkspacePersistError {
    /// The input does not start with the workspace persistence magic.
    BadMagic,
    /// The encoded version is not supported.
    UnsupportedVersion(u16),
    /// Input ended before a complete value was available.
    Truncated,
    /// Extra bytes followed the decoded snapshot.
    TrailingBytes,
    /// A declared size or count exceeds its configured bound.
    ResourceLimit(&'static str),
    /// A length cannot be represented safely on this platform.
    LengthOverflow,
    /// A name or nested value contains invalid UTF-8.
    InvalidUtf8,
    /// An optional-value tag is not recognized.
    InvalidEnum {
        /// Name of the discriminant family.
        kind: &'static str,
        /// Raw discriminant value.
        value: u8,
    },
    /// The snapshot refers to an object that is not present in the snapshot.
    InvalidReference(&'static str),
    /// A nested layout failed its own persistence validation.
    Layout(PersistError),
    /// The snapshot contains a duplicate or otherwise invalid named object.
    Workspace(WorkspaceError),
}

impl fmt::Display for WorkspacePersistError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => formatter.write_str("bad workspace persistence magic"),
            Self::UnsupportedVersion(version) => {
                write!(
                    formatter,
                    "unsupported workspace persistence version: {version}"
                )
            }
            Self::Truncated => formatter.write_str("truncated workspace persistence data"),
            Self::TrailingBytes => formatter.write_str("trailing workspace persistence bytes"),
            Self::ResourceLimit(kind) => {
                write!(formatter, "workspace resource limit exceeded: {kind}")
            }
            Self::LengthOverflow => formatter.write_str("workspace length overflow"),
            Self::InvalidUtf8 => formatter.write_str("invalid UTF-8 in workspace persistence data"),
            Self::InvalidEnum { kind, value } => {
                write!(formatter, "invalid workspace {kind} tag: {value}")
            }
            Self::InvalidReference(kind) => {
                write!(
                    formatter,
                    "workspace snapshot has an invalid {kind} reference"
                )
            }
            Self::Layout(error) => write!(formatter, "nested layout persistence error: {error:?}"),
            Self::Workspace(error) => write!(formatter, "workspace model error: {error}"),
        }
    }
}

impl From<PersistError> for WorkspacePersistError {
    fn from(error: PersistError) -> Self {
        Self::Layout(error)
    }
}

impl From<WorkspaceError> for WorkspacePersistError {
    fn from(error: WorkspaceError) -> Self {
        Self::Workspace(error)
    }
}

/// Encodes the ordered workspace collection in a stable version-1 format.
pub fn encode_workspace(manager: &WorkspaceManager) -> Result<Vec<u8>, WorkspacePersistError> {
    let limits = WorkspaceDecodeLimits::default();
    if manager.workspace_count() > limits.max_workspaces {
        return Err(WorkspacePersistError::ResourceLimit("workspaces"));
    }
    for workspace in manager.workspaces() {
        if workspace.name().len() > limits.max_name_bytes
            || workspace.perspective_count() > limits.max_perspectives
        {
            return Err(WorkspacePersistError::ResourceLimit(
                if workspace.name().len() > limits.max_name_bytes {
                    "name bytes"
                } else {
                    "perspectives"
                },
            ));
        }
        if workspace
            .active_perspective_name()
            .is_some_and(|name| name.len() > limits.max_name_bytes)
        {
            return Err(WorkspacePersistError::ResourceLimit("name bytes"));
        }
        for perspective in workspace.perspectives() {
            if perspective.name().len() > limits.max_name_bytes {
                return Err(WorkspacePersistError::ResourceLimit("name bytes"));
            }
        }
    }
    if manager
        .active_workspace_name()
        .is_some_and(|name| name.len() > limits.max_name_bytes)
    {
        return Err(WorkspacePersistError::ResourceLimit("name bytes"));
    }
    let mut writer = Writer::new(limits.max_bytes);
    writer.bytes(MAGIC)?;
    writer.u16(VERSION)?;
    writer.count(manager.workspace_count())?;
    match manager.active_workspace_name() {
        None => writer.u8(0)?,
        Some(name) => {
            writer.u8(1)?;
            writer.string(name)?;
        }
    }

    for workspace in manager.workspaces() {
        writer.string(workspace.name())?;
        writer.count(workspace.perspective_count())?;
        match workspace.active_perspective_name() {
            None => writer.u8(0)?,
            Some(name) => {
                writer.u8(1)?;
                writer.string(name)?;
            }
        }
        for perspective in workspace.perspectives() {
            writer.string(perspective.name())?;
            let bytes = encode_layout(&perspective.layout())?;
            writer.count(bytes.len())?;
            writer.bytes(&bytes)?;
        }
    }
    Ok(writer.finish())
}

/// Decodes a workspace collection and validates every nested layout.
pub fn decode_workspace(
    bytes: &[u8],
    options: WorkspaceDecodeOptions<'_>,
) -> Result<WorkspaceManager, WorkspacePersistError> {
    if bytes.len() > options.limits.max_bytes {
        return Err(WorkspacePersistError::ResourceLimit("bytes"));
    }
    let mut reader = Reader {
        bytes,
        offset: 0,
        limits: options.limits,
    };
    if reader.take(4)? != MAGIC {
        return Err(WorkspacePersistError::BadMagic);
    }
    let version = reader.u16()?;
    if version != VERSION {
        return Err(WorkspacePersistError::UnsupportedVersion(version));
    }

    let workspace_count = reader.count(options.limits.max_workspaces, "workspaces")?;
    let active_workspace = reader.optional_name()?;
    if workspace_count == 0 && active_workspace.is_some() {
        return Err(WorkspacePersistError::InvalidReference("active workspace"));
    }

    let mut manager = WorkspaceManager::new();
    for _ in 0..workspace_count {
        let workspace_name = reader.name()?;
        let perspective_count = reader.count(options.limits.max_perspectives, "perspectives")?;
        let active_perspective = reader.optional_name()?;
        if perspective_count == 0 && active_perspective.is_some() {
            return Err(WorkspacePersistError::InvalidReference(
                "active perspective",
            ));
        }

        manager.add_workspace(workspace_name.clone())?;
        manager.switch_workspace(&workspace_name)?;
        for _ in 0..perspective_count {
            let perspective_name = reader.name()?;
            let layout_len = reader.count(options.limits.max_layout_bytes, "layout bytes")?;
            let layout_bytes = reader.take(layout_len)?;
            let layout = decode_layout(
                layout_bytes,
                DecodeOptions {
                    known_panels: options.known_panels,
                    known_documents: options.known_documents,
                    limits: options.limits.layout,
                },
            )?;
            manager.add_perspective(&perspective_name, &layout)?;
        }
        if let Some(name) = active_perspective {
            manager.switch_perspective(&name)?;
        }
    }

    if let Some(name) = active_workspace {
        manager.switch_workspace(&name)?;
    }
    if reader.offset != bytes.len() {
        return Err(WorkspacePersistError::TrailingBytes);
    }
    Ok(manager)
}

/// Decodes a workspace collection using a typed panel registry allow-list.
pub fn decode_workspace_with_registry(
    bytes: &[u8],
    registry: &PanelRegistry,
    known_documents: Option<&[&str]>,
    limits: WorkspaceDecodeLimits,
) -> Result<WorkspaceManager, WorkspacePersistError> {
    let panel_ids: Vec<&str> = registry
        .iter()
        .map(|descriptor| descriptor.id.as_str())
        .collect();
    decode_workspace(
        bytes,
        WorkspaceDecodeOptions {
            known_panels: Some(&panel_ids),
            known_documents,
            limits,
        },
    )
}

struct Writer {
    bytes: Vec<u8>,
    max_bytes: usize,
}

impl Writer {
    fn new(max_bytes: usize) -> Self {
        Self {
            bytes: Vec::new(),
            max_bytes,
        }
    }

    fn append(&mut self, value: &[u8]) -> Result<(), WorkspacePersistError> {
        let next = self
            .bytes
            .len()
            .checked_add(value.len())
            .ok_or(WorkspacePersistError::LengthOverflow)?;
        if next > self.max_bytes {
            return Err(WorkspacePersistError::ResourceLimit("bytes"));
        }
        self.bytes.extend_from_slice(value);
        Ok(())
    }

    fn bytes(&mut self, value: &[u8]) -> Result<(), WorkspacePersistError> {
        self.append(value)
    }

    fn u8(&mut self, value: u8) -> Result<(), WorkspacePersistError> {
        self.append(&[value])
    }

    fn u16(&mut self, value: u16) -> Result<(), WorkspacePersistError> {
        self.bytes(&value.to_le_bytes())
    }

    fn u32(&mut self, value: u32) -> Result<(), WorkspacePersistError> {
        self.bytes(&value.to_le_bytes())
    }

    fn count(&mut self, value: usize) -> Result<(), WorkspacePersistError> {
        self.u32(u32::try_from(value).map_err(|_| WorkspacePersistError::LengthOverflow)?)
    }

    fn string(&mut self, value: &str) -> Result<(), WorkspacePersistError> {
        self.count(value.len())?;
        self.bytes(value.as_bytes())
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
    limits: WorkspaceDecodeLimits,
}

impl Reader<'_> {
    fn take(&mut self, len: usize) -> Result<&[u8], WorkspacePersistError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(WorkspacePersistError::LengthOverflow)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(WorkspacePersistError::Truncated)?;
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, WorkspacePersistError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, WorkspacePersistError> {
        Ok(u16::from_le_bytes(
            self.take(2)?
                .try_into()
                .map_err(|_| WorkspacePersistError::Truncated)?,
        ))
    }

    fn u32(&mut self) -> Result<u32, WorkspacePersistError> {
        Ok(u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| WorkspacePersistError::Truncated)?,
        ))
    }

    fn count(&mut self, max: usize, kind: &'static str) -> Result<usize, WorkspacePersistError> {
        let count =
            usize::try_from(self.u32()?).map_err(|_| WorkspacePersistError::LengthOverflow)?;
        if count > max {
            return Err(WorkspacePersistError::ResourceLimit(kind));
        }
        Ok(count)
    }

    fn name(&mut self) -> Result<String, WorkspacePersistError> {
        let length = self.count(self.limits.max_name_bytes, "name bytes")?;
        let bytes = self.take(length)?;
        let value = core::str::from_utf8(bytes).map_err(|_| WorkspacePersistError::InvalidUtf8)?;
        Ok(String::from(value))
    }

    fn optional_name(&mut self) -> Result<Option<String>, WorkspacePersistError> {
        match self.u8()? {
            0 => Ok(None),
            1 => self.name().map(Some),
            value => Err(WorkspacePersistError::InvalidEnum {
                kind: "option",
                value,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DockLayout, DockNode};
    use alloc::format;

    fn layout(panel: &str) -> DockLayout {
        DockLayout {
            root: DockNode::panel(panel),
            ..DockLayout::default()
        }
    }

    fn options() -> WorkspaceDecodeOptions<'static> {
        WorkspaceDecodeOptions::default()
    }

    #[test]
    fn roundtrip_preserves_order_and_active_selection() {
        let mut manager = WorkspaceManager::new();
        manager.add_workspace("coding").unwrap();
        manager
            .add_perspective("base", &layout("outliner"))
            .unwrap();
        manager
            .add_perspective("focus", &layout("inspector"))
            .unwrap();
        manager.switch_perspective("focus").unwrap();
        manager.add_workspace("debug").unwrap();
        manager
            .add_perspective("default", &layout("console"))
            .unwrap();
        manager.switch_workspace("coding").unwrap();

        let decoded = decode_workspace(&encode_workspace(&manager).unwrap(), options()).unwrap();
        assert_eq!(decoded.active_workspace_name(), Some("coding"));
        assert_eq!(
            decoded.active_layout().unwrap().root,
            DockNode::panel("inspector")
        );
        assert_eq!(
            decoded
                .workspaces()
                .map(crate::Workspace::name)
                .collect::<Vec<_>>(),
            ["coding", "debug"]
        );
        assert_eq!(
            decoded
                .workspace("coding")
                .unwrap()
                .active_perspective_name(),
            Some("focus")
        );
    }

    #[test]
    fn registry_decode_filters_nested_unknown_panels() {
        let mut manager = WorkspaceManager::new();
        manager.add_workspace("coding").unwrap();
        manager.add_perspective("base", &layout("known")).unwrap();
        let bytes = encode_workspace(&manager).unwrap();
        let mut registry = PanelRegistry::new();
        registry
            .register(crate::PanelDescriptor::new("other", "Other"))
            .unwrap();
        let decoded = decode_workspace_with_registry(
            &bytes,
            &registry,
            None,
            WorkspaceDecodeLimits::default(),
        )
        .unwrap();
        assert_eq!(decoded.active_layout().unwrap().root, DockNode::Empty);
    }

    #[test]
    fn malformed_headers_truncation_and_trailing_bytes_are_rejected() {
        let mut manager = WorkspaceManager::new();
        manager.add_workspace("coding").unwrap();
        manager.add_perspective("base", &layout("panel")).unwrap();
        let bytes = encode_workspace(&manager).unwrap();

        let mut bad_magic = bytes.clone();
        bad_magic[0] = b'X';
        assert_eq!(
            decode_workspace(&bad_magic, options()),
            Err(WorkspacePersistError::BadMagic)
        );

        let mut bad_version = bytes.clone();
        bad_version[4..6].copy_from_slice(&99u16.to_le_bytes());
        assert_eq!(
            decode_workspace(&bad_version, options()),
            Err(WorkspacePersistError::UnsupportedVersion(99))
        );
        assert_eq!(
            decode_workspace(&bytes[..bytes.len() - 1], options()),
            Err(WorkspacePersistError::Truncated)
        );
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert_eq!(
            decode_workspace(&trailing, options()),
            Err(WorkspacePersistError::TrailingBytes)
        );
    }

    #[test]
    fn encoder_rejects_default_decoder_resource_overflow() {
        let mut too_many_workspaces = WorkspaceManager::new();
        for index in 0..=WorkspaceDecodeLimits::default().max_workspaces {
            too_many_workspaces
                .add_workspace(format!("workspace-{index}"))
                .unwrap();
        }
        assert_eq!(
            encode_workspace(&too_many_workspaces),
            Err(WorkspacePersistError::ResourceLimit("workspaces"))
        );

        let mut too_many_perspectives = WorkspaceManager::new();
        too_many_perspectives.add_workspace("workspace").unwrap();
        for index in 0..=WorkspaceDecodeLimits::default().max_perspectives {
            too_many_perspectives
                .add_perspective(format!("perspective-{index}"), &DockLayout::default())
                .unwrap();
        }
        assert_eq!(
            encode_workspace(&too_many_perspectives),
            Err(WorkspacePersistError::ResourceLimit("perspectives"))
        );

        let long_name = "x".repeat(WorkspaceDecodeLimits::default().max_name_bytes + 1);
        let mut too_long = WorkspaceManager::new();
        too_long.add_workspace(long_name).unwrap();
        assert_eq!(
            encode_workspace(&too_long),
            Err(WorkspacePersistError::ResourceLimit("name bytes"))
        );
    }

    #[test]
    fn resource_limits_are_checked_before_allocation() {
        let mut manager = WorkspaceManager::new();
        manager.add_workspace("coding").unwrap();
        manager.add_perspective("base", &layout("panel")).unwrap();
        let bytes = encode_workspace(&manager).unwrap();
        let limits = WorkspaceDecodeLimits {
            max_workspaces: 0,
            ..WorkspaceDecodeLimits::default()
        };
        assert_eq!(
            decode_workspace(
                &bytes,
                WorkspaceDecodeOptions {
                    limits,
                    ..options()
                }
            ),
            Err(WorkspacePersistError::ResourceLimit("workspaces"))
        );
    }
}
