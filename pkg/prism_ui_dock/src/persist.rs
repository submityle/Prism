//! Stable, versioned binary persistence core for [`DockLayout`](crate::DockLayout).
//!
//! This format is intended for saved layout data, not as a workspace database
//! or autosave coordinator. The wire format uses little-endian integers and
//! explicit enum tags; changing it requires a new version or a compatible
//! extension.

use crate::{
    AutoHideBars, AutoHideEntry, Axis, DockLayout, DockNode, FloatingWindow, Limits,
    MaximizeRestore, PanelRegistry, Ratio, Rect, Side, SplitChild,
};
use alloc::{boxed::Box, string::String, vec::Vec};

const MAGIC: &[u8; 4] = b"PRDK";
const VERSION: u16 = 1;

/// Additional limits for parsing untrusted encoded layouts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeLimits {
    /// Existing semantic layout limits, applied before and after filtering.
    pub layout: Limits,
    /// Maximum encoded input size.
    pub max_bytes: usize,
    /// Maximum UTF-8 byte length of one identifier.
    pub max_string_bytes: usize,
    /// Maximum entries in any encoded list (in addition to semantic limits).
    pub max_collection_len: usize,
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            layout: Limits::default(),
            max_bytes: 16 * 1024 * 1024,
            max_string_bytes: 1024 * 1024,
            max_collection_len: 16_384,
        }
    }
}

/// Registry and resource policy used when decoding a layout.
#[derive(Clone, Copy, Debug, Default)]
pub struct DecodeOptions<'a> {
    /// Registered panel identifiers; `None` disables panel filtering.
    pub known_panels: Option<&'a [&'a str]>,
    /// Registered document identifiers; `None` disables document filtering.
    pub known_documents: Option<&'a [&'a str]>,
    /// Parser and semantic validation limits.
    pub limits: DecodeLimits,
}

/// A structured persistence encoding, decoding, or validation failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PersistError {
    /// The input does not start with the persistence magic.
    BadMagic,
    /// The encoded version is not supported.
    UnsupportedVersion(u16),
    /// Input ended before a complete value was available.
    Truncated,
    /// Extra bytes followed the decoded layout.
    TrailingBytes,
    /// A declared length or count exceeds its configured bound.
    ResourceLimit(&'static str),
    /// A length cannot be represented safely on this platform.
    LengthOverflow,
    /// An identifier contains invalid UTF-8.
    InvalidUtf8,
    /// A tag or boolean field contains an unknown discriminant.
    InvalidEnum {
        /// Name of the discriminant family.
        kind: &'static str,
        /// Raw discriminant value.
        value: u8,
    },
    /// A value cannot be represented by this format.
    InvalidValue(&'static str),
    /// The decoded/encoded layout violates a model invariant.
    InvalidLayout(crate::LayoutError),
    /// Unknown-ID filtering removed content that could not be safely repaired.
    CannotRepair(&'static str),
}

/// Encodes a validated layout using the stable version-1 binary format.
pub fn encode_layout(layout: &DockLayout) -> Result<Vec<u8>, PersistError> {
    layout
        .validate_with(Limits::default())
        .map_err(PersistError::InvalidLayout)?;
    let mut out = Writer(Vec::new());
    out.bytes(MAGIC);
    out.u16(VERSION);
    out.node(&layout.root, 0)?;
    out.count(layout.floating.len())?;
    for window in &layout.floating {
        out.rect(window.rect);
        out.node(&window.root, 0)?;
        out.u32(window.z);
    }
    out.autohide(&layout.autohide)?;
    match &layout.maximized {
        None => out.u8(0),
        Some(snapshot) => {
            out.u8(1);
            out.string(&snapshot.panel)?;
            out.node(&snapshot.root, 0)?;
            out.count(snapshot.floating.len())?;
            for window in &snapshot.floating {
                out.rect(window.rect);
                out.node(&window.root, 0)?;
                out.u32(window.z);
            }
            out.autohide(&snapshot.autohide)?;
        }
    }
    if out.0.len() > DecodeLimits::default().max_bytes {
        return Err(PersistError::ResourceLimit("bytes"));
    }
    // Keep encoding symmetric with the default decoder's per-field limits.
    enforce_encode_limits(layout, DecodeLimits::default())?;
    Ok(out.0)
}

fn enforce_encode_limits(layout: &DockLayout, limits: DecodeLimits) -> Result<(), PersistError> {
    fn check_node(
        value: &DockNode,
        depth: usize,
        limits: DecodeLimits,
    ) -> Result<usize, PersistError> {
        if depth > limits.layout.max_depth {
            return Err(PersistError::ResourceLimit("depth"));
        }
        match value {
            DockNode::Empty => Ok(1),
            DockNode::TabStack { tabs, .. } | DockNode::Document { tabs, .. } => {
                if tabs.len() > limits.max_collection_len {
                    return Err(PersistError::ResourceLimit("collection"));
                }
                for id in tabs {
                    if id.len() > limits.max_string_bytes {
                        return Err(PersistError::ResourceLimit("string bytes"));
                    }
                }
                Ok(1)
            }
            DockNode::Split { children, .. } => {
                if children.len() > limits.max_collection_len {
                    return Err(PersistError::ResourceLimit("collection"));
                }
                let mut count = 1usize;
                for child in children {
                    count = count
                        .checked_add(check_node(&child.node, depth + 1, limits)?)
                        .ok_or(PersistError::LengthOverflow)?;
                }
                Ok(count)
            }
        }
    }
    fn windows(windows: &[FloatingWindow], limits: DecodeLimits) -> Result<(), PersistError> {
        if windows.len() > limits.max_collection_len {
            return Err(PersistError::ResourceLimit("collection"));
        }
        for window in windows {
            check_node(&window.root, 0, limits)?;
        }
        Ok(())
    }
    check_node(&layout.root, 0, limits)?;
    windows(&layout.floating, limits)?;
    if layout.autohide.entries.len() > limits.max_collection_len {
        return Err(PersistError::ResourceLimit("collection"));
    }
    for entry in &layout.autohide.entries {
        if entry.panel.len() > limits.max_string_bytes {
            return Err(PersistError::ResourceLimit("string bytes"));
        }
    }
    if let Some(active) = &layout.autohide.active {
        if active.len() > limits.max_string_bytes {
            return Err(PersistError::ResourceLimit("string bytes"));
        }
    }
    if let Some(snapshot) = &layout.maximized {
        if snapshot.panel.len() > limits.max_string_bytes {
            return Err(PersistError::ResourceLimit("string bytes"));
        }
        check_node(&snapshot.root, 0, limits)?;
        windows(&snapshot.floating, limits)?;
        if snapshot.autohide.entries.len() > limits.max_collection_len {
            return Err(PersistError::ResourceLimit("collection"));
        }
        for entry in &snapshot.autohide.entries {
            if entry.panel.len() > limits.max_string_bytes {
                return Err(PersistError::ResourceLimit("string bytes"));
            }
        }
    }
    Ok(())
}

/// Decodes a layout, filters unregistered IDs when registries are supplied,
/// repairs removable empty tree branches and active indices, then validates.
pub fn decode_layout(bytes: &[u8], options: DecodeOptions<'_>) -> Result<DockLayout, PersistError> {
    if bytes.len() > options.limits.max_bytes {
        return Err(PersistError::ResourceLimit("bytes"));
    }
    let mut reader = Reader {
        bytes,
        offset: 0,
        limits: options.limits,
        nodes: 0,
    };
    if reader.take(4)? != MAGIC {
        return Err(PersistError::BadMagic);
    }
    let version = reader.u16()?;
    if version != VERSION {
        return Err(PersistError::UnsupportedVersion(version));
    }
    let mut layout = DockLayout {
        root: reader.node(0)?,
        floating: reader.windows()?,
        autohide: reader.autohide()?,
        maximized: None,
    };
    match reader.u8()? {
        0 => {}
        1 => {
            let panel = reader.string()?;
            let root = reader.node(0)?;
            let floating = reader.windows()?;
            let autohide = reader.autohide()?;
            layout.maximized = Some(Box::new(MaximizeRestore {
                panel,
                root,
                floating,
                autohide,
            }));
        }
        value => {
            return Err(PersistError::InvalidEnum {
                kind: "option",
                value,
            })
        }
    }
    if reader.offset != bytes.len() {
        return Err(PersistError::TrailingBytes);
    }

    // Reject malformed invariants before filtering. Filtering may repair
    // ownership and active state, but it must never make a bad ratio valid.
    layout
        .validate_with(options.limits.layout)
        .map_err(PersistError::InvalidLayout)?;

    filter_node(
        &mut layout.root,
        options.known_panels,
        options.known_documents,
        true,
    )?;
    for window in &mut layout.floating {
        filter_node(
            &mut window.root,
            options.known_panels,
            options.known_documents,
            true,
        )?;
    }
    layout
        .floating
        .retain(|window| window.root != DockNode::Empty);
    filter_autohide(&mut layout.autohide, options.known_panels);
    if let Some(snapshot) = &mut layout.maximized {
        if options
            .known_panels
            .is_some_and(|known| !known.contains(&snapshot.panel.as_str()))
        {
            layout.maximized = None;
        } else {
            filter_node(
                &mut snapshot.root,
                options.known_panels,
                options.known_documents,
                true,
            )?;
            for window in &mut snapshot.floating {
                filter_node(
                    &mut window.root,
                    options.known_panels,
                    options.known_documents,
                    true,
                )?;
            }
            snapshot
                .floating
                .retain(|window| window.root != DockNode::Empty);
            filter_autohide(&mut snapshot.autohide, options.known_panels);
        }
    }
    layout
        .validate_with(options.limits.layout)
        .map_err(PersistError::InvalidLayout)?;
    Ok(layout)
}

/// Decodes a layout using a typed [`PanelRegistry`] as the panel allow-list.
///
/// This convenience boundary preserves the existing slice-based API while
/// ensuring registry users do not duplicate IDs into a second allow-list.
/// Document IDs remain an independent optional allow-list.
pub fn decode_layout_with_registry(
    bytes: &[u8],
    registry: &PanelRegistry,
    known_documents: Option<&[&str]>,
    limits: DecodeLimits,
) -> Result<DockLayout, PersistError> {
    let panel_ids: Vec<&str> = registry
        .iter()
        .map(|descriptor| descriptor.id.as_str())
        .collect();
    decode_layout(
        bytes,
        DecodeOptions {
            known_panels: Some(&panel_ids),
            known_documents,
            limits,
        },
    )
}

fn filter_autohide(bars: &mut AutoHideBars, known: Option<&[&str]>) {
    if let Some(known) = known {
        bars.entries
            .retain(|entry| known.contains(&entry.panel.as_str()));
        if bars
            .active
            .as_ref()
            .is_some_and(|active| !known.contains(&active.as_str()))
        {
            bars.active = None;
        }
    }
}

fn filter_node(
    node: &mut DockNode,
    panels: Option<&[&str]>,
    documents: Option<&[&str]>,
    root: bool,
) -> Result<(), PersistError> {
    match node {
        DockNode::TabStack { tabs, active } => {
            filter_tabs(tabs, active, panels);
            if tabs.is_empty() {
                *node = DockNode::Empty;
            }
        }
        DockNode::Document { tabs, active } => {
            filter_tabs(tabs, active, documents);
            if tabs.is_empty() {
                *node = DockNode::Empty;
            }
        }
        DockNode::Split { children, .. } => {
            for child in children.iter_mut() {
                filter_node(&mut child.node, panels, documents, false)?;
            }
            children.retain(|child| child.node != DockNode::Empty);
            match children.len() {
                0 => *node = DockNode::Empty,
                1 => *node = children.remove(0).node,
                _ => renormalize(children)?,
            }
        }
        DockNode::Empty => {}
    }
    let _ = root;
    Ok(())
}

fn filter_tabs(tabs: &mut Vec<String>, active: &mut usize, known: Option<&[&str]>) {
    if let Some(known) = known {
        let old_active = *active;
        let mut new_active = None;
        let mut kept = Vec::new();
        for (index, tab) in tabs.drain(..).enumerate() {
            if known.contains(&tab.as_str()) {
                if index == old_active {
                    new_active = Some(kept.len());
                }
                kept.push(tab);
            }
        }
        *active = new_active.unwrap_or_else(|| old_active.min(kept.len().saturating_sub(1)));
        *tabs = kept;
    }
}

fn renormalize(children: &mut [SplitChild]) -> Result<(), PersistError> {
    let total: u64 = children
        .iter()
        .map(|child| u64::from(child.ratio.millionths()))
        .sum();
    if total == 0 {
        return Err(PersistError::CannotRepair("split ratios"));
    }
    let scale = u64::from(Ratio::SCALE);
    let mut assigned = 0u64;
    let mut remainders = Vec::with_capacity(children.len());
    for (index, child) in children.iter_mut().enumerate() {
        let product = u64::from(child.ratio.millionths()) * scale;
        let base = product / total;
        child.ratio = Ratio::from_millionths(base as u32);
        assigned += base;
        remainders.push((index, product % total));
    }
    remainders.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    for (index, _) in remainders.into_iter().take((scale - assigned) as usize) {
        children[index].ratio = Ratio::from_millionths(children[index].ratio.millionths() + 1);
    }
    if children.iter().any(|child| child.ratio.millionths() == 0) {
        return Err(PersistError::CannotRepair("split ratios"));
    }
    Ok(())
}

struct Writer(Vec<u8>);
impl Writer {
    fn bytes(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
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
    fn i32(&mut self, value: i32) {
        self.bytes(&value.to_le_bytes());
    }
    fn count(&mut self, count: usize) -> Result<(), PersistError> {
        let count = u32::try_from(count).map_err(|_| PersistError::LengthOverflow)?;
        self.u32(count);
        Ok(())
    }
    fn string(&mut self, value: &str) -> Result<(), PersistError> {
        self.count(value.len())?;
        self.bytes(value.as_bytes());
        Ok(())
    }
    fn rect(&mut self, rect: Rect) {
        self.i32(rect.x);
        self.i32(rect.y);
        self.u32(rect.width);
        self.u32(rect.height);
    }
    fn node(&mut self, node: &DockNode, depth: usize) -> Result<(), PersistError> {
        if depth > Limits::default().max_depth {
            return Err(PersistError::ResourceLimit("depth"));
        }
        match node {
            DockNode::Empty => self.u8(0),
            DockNode::TabStack { tabs, active } => {
                self.u8(1);
                self.strings(tabs)?;
                self.count(*active)?;
            }
            DockNode::Document { tabs, active } => {
                self.u8(2);
                self.strings(tabs)?;
                self.count(*active)?;
            }
            DockNode::Split { axis, children } => {
                self.u8(3);
                self.u8(match axis {
                    Axis::Horizontal => 0,
                    Axis::Vertical => 1,
                });
                self.count(children.len())?;
                for child in children {
                    self.u32(child.ratio.millionths());
                    self.node(&child.node, depth + 1)?;
                }
            }
        }
        Ok(())
    }
    fn strings(&mut self, strings: &[String]) -> Result<(), PersistError> {
        self.count(strings.len())?;
        for value in strings {
            self.string(value)?;
        }
        Ok(())
    }
    fn autohide(&mut self, bars: &AutoHideBars) -> Result<(), PersistError> {
        self.count(bars.entries.len())?;
        for entry in &bars.entries {
            self.u8(match entry.side {
                Side::Left => 0,
                Side::Right => 1,
                Side::Top => 2,
                Side::Bottom => 3,
            });
            self.string(&entry.panel)?;
        }
        match &bars.active {
            None => self.u8(0),
            Some(panel) => {
                self.u8(1);
                self.string(panel)?;
            }
        }
        Ok(())
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
    limits: DecodeLimits,
    nodes: usize,
}
impl Reader<'_> {
    fn take(&mut self, len: usize) -> Result<&[u8], PersistError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(PersistError::LengthOverflow)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(PersistError::Truncated)?;
        self.offset = end;
        Ok(value)
    }
    fn u8(&mut self) -> Result<u8, PersistError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, PersistError> {
        Ok(u16::from_le_bytes(
            self.take(2)?
                .try_into()
                .map_err(|_| PersistError::Truncated)?,
        ))
    }
    fn u32(&mut self) -> Result<u32, PersistError> {
        Ok(u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| PersistError::Truncated)?,
        ))
    }
    fn i32(&mut self) -> Result<i32, PersistError> {
        Ok(i32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| PersistError::Truncated)?,
        ))
    }
    fn count(&mut self) -> Result<usize, PersistError> {
        let count = usize::try_from(self.u32()?).map_err(|_| PersistError::LengthOverflow)?;
        if count > self.limits.max_collection_len {
            return Err(PersistError::ResourceLimit("collection length"));
        }
        Ok(count)
    }
    fn string(&mut self) -> Result<String, PersistError> {
        let len = usize::try_from(self.u32()?).map_err(|_| PersistError::LengthOverflow)?;
        if len > self.limits.max_string_bytes {
            return Err(PersistError::ResourceLimit("string bytes"));
        }
        let bytes = self.take(len)?;
        let value = core::str::from_utf8(bytes).map_err(|_| PersistError::InvalidUtf8)?;
        Ok(String::from(value))
    }
    fn strings(&mut self) -> Result<Vec<String>, PersistError> {
        let len = self.count()?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(self.string()?);
        }
        Ok(values)
    }
    fn rect(&mut self) -> Result<Rect, PersistError> {
        Ok(Rect {
            x: self.i32()?,
            y: self.i32()?,
            width: self.u32()?,
            height: self.u32()?,
        })
    }
    fn node(&mut self, depth: usize) -> Result<DockNode, PersistError> {
        if depth > self.limits.layout.max_depth {
            return Err(PersistError::ResourceLimit("depth"));
        }
        self.nodes = self
            .nodes
            .checked_add(1)
            .ok_or(PersistError::LengthOverflow)?;
        if self.nodes > self.limits.layout.max_nodes {
            return Err(PersistError::ResourceLimit("nodes"));
        }
        match self.u8()? {
            0 => Ok(DockNode::Empty),
            1 => {
                let tabs = self.strings()?;
                let active = self.count()?;
                Ok(DockNode::TabStack { tabs, active })
            }
            2 => {
                let tabs = self.strings()?;
                let active = self.count()?;
                Ok(DockNode::Document { tabs, active })
            }
            3 => {
                let axis = match self.u8()? {
                    0 => Axis::Horizontal,
                    1 => Axis::Vertical,
                    value => {
                        return Err(PersistError::InvalidEnum {
                            kind: "axis",
                            value,
                        })
                    }
                };
                let len = self.count()?;
                let mut children = Vec::with_capacity(len);
                for _ in 0..len {
                    children.push(SplitChild {
                        ratio: Ratio::from_millionths(self.u32()?),
                        node: self.node(depth + 1)?,
                    });
                }
                Ok(DockNode::Split { axis, children })
            }
            value => Err(PersistError::InvalidEnum {
                kind: "node",
                value,
            }),
        }
    }
    fn windows(&mut self) -> Result<Vec<FloatingWindow>, PersistError> {
        let len = self.count()?;
        if len > self.limits.layout.max_floating {
            return Err(PersistError::ResourceLimit("floating windows"));
        }
        let mut windows = Vec::with_capacity(len);
        for _ in 0..len {
            windows.push(FloatingWindow {
                rect: self.rect()?,
                root: self.node(0)?,
                z: self.u32()?,
            });
        }
        Ok(windows)
    }
    fn autohide(&mut self) -> Result<AutoHideBars, PersistError> {
        let len = self.count()?;
        if len > self.limits.layout.max_panels {
            return Err(PersistError::ResourceLimit("panels"));
        }
        let mut entries = Vec::with_capacity(len);
        for _ in 0..len {
            let value = self.u8()?;
            let side = match value {
                0 => Side::Left,
                1 => Side::Right,
                2 => Side::Top,
                3 => Side::Bottom,
                _ => {
                    return Err(PersistError::InvalidEnum {
                        kind: "side",
                        value,
                    })
                }
            };
            entries.push(AutoHideEntry {
                side,
                panel: self.string()?,
            });
        }
        let active = match self.u8()? {
            0 => None,
            1 => Some(self.string()?),
            value => {
                return Err(PersistError::InvalidEnum {
                    kind: "option",
                    value,
                })
            }
        };
        Ok(AutoHideBars { entries, active })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn rich_layout() -> DockLayout {
        DockLayout {
            root: DockNode::Split {
                axis: Axis::Horizontal,
                children: vec![
                    SplitChild {
                        ratio: Ratio::HALF,
                        node: DockNode::panel("tool"),
                    },
                    SplitChild {
                        ratio: Ratio::HALF,
                        node: DockNode::document("doc"),
                    },
                ],
            },
            floating: vec![FloatingWindow {
                rect: Rect {
                    x: -2,
                    y: 3,
                    width: 400,
                    height: 300,
                },
                root: DockNode::panel("float"),
                z: 7,
            }],
            autohide: AutoHideBars {
                entries: vec![AutoHideEntry {
                    side: Side::Right,
                    panel: "hidden".into(),
                }],
                active: Some("hidden".into()),
            },
            maximized: None,
        }
    }

    fn opts<'a>(panels: Option<&'a [&'a str]>, docs: Option<&'a [&'a str]>) -> DecodeOptions<'a> {
        DecodeOptions {
            known_panels: panels,
            known_documents: docs,
            limits: DecodeLimits::default(),
        }
    }

    #[test]
    fn roundtrip_preserves_all_surfaces() {
        let layout = rich_layout();
        assert_eq!(
            decode_layout(&encode_layout(&layout).unwrap(), opts(None, None)).unwrap(),
            layout
        );
    }

    #[test]
    fn roundtrip_preserves_maximize_restore() {
        let mut layout = rich_layout();
        let restore = MaximizeRestore {
            panel: "tool".into(),
            root: DockNode::panel("old"),
            floating: vec![FloatingWindow {
                rect: Rect {
                    x: 1,
                    y: 2,
                    width: 10,
                    height: 11,
                },
                root: DockNode::panel("old-float"),
                z: 2,
            }],
            autohide: AutoHideBars::default(),
        };
        layout.root = DockNode::panel("tool");
        layout.floating.clear();
        layout.autohide = AutoHideBars::default();
        layout.maximized = Some(Box::new(MaximizeRestore { ..restore }));
        let bytes = encode_layout(&layout).unwrap();
        assert_eq!(decode_layout(&bytes, opts(None, None)).unwrap(), layout);
    }

    #[test]
    fn rejects_malformed_maximize_state_before_renderer_consumption() {
        let mut layout = rich_layout();
        layout.maximized = Some(Box::new(MaximizeRestore {
            panel: "max".into(),
            root: DockNode::panel("old"),
            floating: vec![],
            autohide: AutoHideBars::default(),
        }));
        assert!(matches!(
            encode_layout(&layout),
            Err(PersistError::InvalidLayout(crate::LayoutError {
                issue: crate::ValidationIssue::MaximizeSnapshotInvalid,
                ..
            }))
        ));
    }

    #[test]
    fn rejects_truncation() {
        let bytes = encode_layout(&rich_layout()).unwrap();
        assert_eq!(
            decode_layout(&bytes[..bytes.len() - 1], opts(None, None)),
            Err(PersistError::Truncated)
        );
    }

    #[test]
    fn rejects_bad_magic_and_version() {
        let mut bytes = encode_layout(&rich_layout()).unwrap();
        bytes[0] = b'X';
        assert_eq!(
            decode_layout(&bytes, opts(None, None)),
            Err(PersistError::BadMagic)
        );
        bytes[0..4].copy_from_slice(MAGIC);
        bytes[4..6].copy_from_slice(&99u16.to_le_bytes());
        assert_eq!(
            decode_layout(&bytes, opts(None, None)),
            Err(PersistError::UnsupportedVersion(99))
        );
    }

    #[test]
    fn filters_unknown_ids_and_repairs_active_and_empty_nodes() {
        let bytes = encode_layout(&rich_layout()).unwrap();
        let panels = ["tool"];
        let docs = ["missing"];
        let decoded = decode_layout(&bytes, opts(Some(&panels), Some(&docs))).unwrap();
        assert_eq!(decoded.root, DockNode::panel("tool"));
        assert!(decoded.floating.is_empty());
        assert!(decoded.autohide.active.is_none());
    }

    #[test]
    fn typed_registry_entry_filters_unknown_panels_without_manual_id_slice() {
        let bytes = encode_layout(&rich_layout()).unwrap();
        let mut registry = PanelRegistry::new();
        registry
            .register(crate::PanelDescriptor::new("tool", "Tool"))
            .unwrap();
        let documents: [&str; 0] = [];
        let decoded = decode_layout_with_registry(
            &bytes,
            &registry,
            Some(&documents),
            DecodeLimits::default(),
        )
        .unwrap();
        assert_eq!(decoded.root, DockNode::panel("tool"));
        assert!(decoded.floating.is_empty());
        assert!(decoded.autohide.active.is_none());
    }

    #[test]
    fn filtering_a_split_renormalizes_remaining_ratios_deterministically() {
        let layout = DockLayout {
            root: DockNode::Split {
                axis: Axis::Horizontal,
                children: vec![
                    SplitChild {
                        ratio: Ratio::from_millionths(250_000),
                        node: DockNode::panel("a"),
                    },
                    SplitChild {
                        ratio: Ratio::from_millionths(250_000),
                        node: DockNode::panel("b"),
                    },
                    SplitChild {
                        ratio: Ratio::from_millionths(500_000),
                        node: DockNode::panel("removed"),
                    },
                ],
            },
            ..DockLayout::default()
        };
        let bytes = encode_layout(&layout).unwrap();
        let decoded = decode_layout(&bytes, opts(Some(&["a", "b"]), None)).unwrap();
        let DockNode::Split { children, .. } = decoded.root else {
            panic!("two surviving split children must remain a split");
        };
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].ratio.millionths(), 500_000);
        assert_eq!(children[1].ratio.millionths(), 500_000);
    }

    #[test]
    fn enforces_byte_and_node_limits() {
        let bytes = encode_layout(&rich_layout()).unwrap();
        let mut limited = opts(None, None);
        limited.limits.max_bytes = bytes.len() - 1;
        assert_eq!(
            decode_layout(&bytes, limited),
            Err(PersistError::ResourceLimit("bytes"))
        );
        let mut limited = opts(None, None);
        limited.limits.layout.max_nodes = 1;
        assert_eq!(
            decode_layout(&bytes, limited),
            Err(PersistError::ResourceLimit("nodes"))
        );
    }

    #[test]
    fn rejects_duplicate_ownership() {
        let mut writer = Writer(Vec::new());
        writer.bytes(MAGIC);
        writer.u16(VERSION);
        writer.u8(1);
        writer.count(1).unwrap();
        writer.string("same").unwrap();
        writer.count(0).unwrap();
        writer.count(0).unwrap();
        writer.count(1).unwrap();
        writer.u8(0);
        writer.string("same").unwrap();
        writer.u8(0);
        writer.u8(0);
        assert!(matches!(
            decode_layout(&writer.0, opts(None, None)),
            Err(PersistError::InvalidLayout(_))
        ));
    }

    #[test]
    fn rejects_invalid_ratio_in_wire_data() {
        let mut writer = Writer(Vec::new());
        writer.bytes(MAGIC);
        writer.u16(VERSION);
        writer.u8(3);
        writer.u8(0);
        writer.count(2).unwrap();
        writer.u32(0);
        writer.u8(1);
        writer.count(1).unwrap();
        writer.string("a").unwrap();
        writer.count(0).unwrap();
        writer.u32(Ratio::SCALE);
        writer.u8(1);
        writer.count(1).unwrap();
        writer.string("b").unwrap();
        writer.count(0).unwrap();
        writer.count(0).unwrap();
        writer.count(0).unwrap();
        writer.u8(0);
        writer.u8(0);
        assert!(matches!(
            decode_layout(&writer.0, opts(None, None)),
            Err(PersistError::InvalidLayout(_))
        ));
    }

    #[test]
    fn rejects_malformed_utf8() {
        let mut bytes = encode_layout(&DockLayout {
            root: DockNode::panel("x"),
            ..DockLayout::default()
        })
        .unwrap();
        bytes[15] = 0xff;
        assert_eq!(
            decode_layout(&bytes, opts(None, None)),
            Err(PersistError::InvalidUtf8)
        );
    }

    #[test]
    fn rejects_invalid_enum() {
        let mut bytes = encode_layout(&DockLayout::default()).unwrap();
        bytes[6] = 9;
        assert_eq!(
            decode_layout(&bytes, opts(None, None)),
            Err(PersistError::InvalidEnum {
                kind: "node",
                value: 9
            })
        );
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut bytes = encode_layout(&DockLayout::default()).unwrap();
        bytes.push(0);
        assert_eq!(
            decode_layout(&bytes, opts(None, None)),
            Err(PersistError::TrailingBytes)
        );
    }

    #[test]
    fn encoder_rejects_snapshots_that_default_decoder_would_reject() {
        let oversized = "x".repeat(DecodeLimits::default().max_bytes);
        let layout = DockLayout {
            root: DockNode::panel(oversized),
            ..DockLayout::default()
        };
        assert_eq!(
            encode_layout(&layout),
            Err(PersistError::ResourceLimit("bytes"))
        );
    }
}
