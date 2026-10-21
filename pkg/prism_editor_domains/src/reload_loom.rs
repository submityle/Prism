//! Executable data-only Loom loader, reusing Prism's state-preserving reloader.
//! Dynamic Rust expressions, signals and for_each require Rust recompilation.

use crate::reload::{
    PreparedReload, ReloadCancellation, ReloadError, ReloadLimits, ReloadPipeline, ReloadProvider,
    ReloadSlot, ReloadSources,
};
use prism_ui::Element;
use prism_ui_hotreload::{HotReloader, NodeIdent, ReloadPlan, ReloadReport};
use prism_ui_style::{Keyword, StyleProp, StyleValue};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

/// Exact supported schema revision, not a content hash or timestamp.
pub const LOOM_LITERAL_SCHEMA: &[u8] = b"prism.loom.literal/1";

/// Real file provider for literal Loom views and base style classes.
pub struct LoomReloadProvider {
    view: PathBuf,
    styles: Vec<PathBuf>,
}
impl LoomReloadProvider {
    /// Creates a provider. All view/style paths must also be registered in the lane.
    pub fn new(view: PathBuf, styles: Vec<PathBuf>) -> Self {
        Self { view, styles }
    }
}

/// Fully parsed and validated candidate; contains a real Prism Element tree.
pub struct LoomCandidate {
    tree: Element,
    limits: ReloadLimits,
}
impl LoomCandidate {
    /// The exact tree that will be mounted.
    pub fn tree(&self) -> &Element {
        &self.tree
    }
    /// Computes the actual NodePath/state preservation plan without applying it.
    pub fn plan<T>(&self, live: &HotReloader<T>) -> Result<ReloadPlan, ReloadError> {
        validate_live(live.current(), &self.limits)?;
        Ok(live.plan_for(&self.tree))
    }
}

impl ReloadProvider for LoomReloadProvider {
    type Artifact = LoomCandidate;
    fn prepare(
        &self,
        sources: &ReloadSources,
        schema: &[u8],
        cancellation: &ReloadCancellation,
        limits: &ReloadLimits,
    ) -> Result<LoomCandidate, ReloadError> {
        if schema != LOOM_LITERAL_SCHEMA {
            return Err(ReloadError::provider("unsupported Loom schema"));
        }
        if self.styles.len() > limits.sources {
            return Err(ReloadError::Budget("style sources"));
        }
        let mut classes = BTreeMap::new();
        for path in &self.styles {
            cancellation.check()?;
            let mut parser = Parser::new(sources.text(path)?, limits, cancellation);
            while !parser.done() {
                parser.expect_word("class")?;
                let name = parser.string()?;
                let properties = parser.properties()?;
                if classes.len() >= limits.nodes {
                    return Err(ReloadError::Budget("style classes"));
                }
                if classes.insert(name, properties).is_some() {
                    return Err(parser.error("duplicate class"));
                }
            }
        }
        let mut parser = Parser::new(sources.text(&self.view)?, limits, cancellation);
        let tree = parser.node(1, &classes)?;
        parser.eat(';');
        if !parser.done() {
            return Err(parser.error("trailing Loom tokens"));
        }
        Ok(LoomCandidate {
            tree,
            limits: limits.clone(),
        })
    }
}

/// Applies the real plan only after exact source/schema fences pass. State T is
/// moved/preserved in place; it need not implement Clone. Undo is never borrowed.
pub fn publish_loom<T>(
    pipeline: &ReloadPipeline<LoomReloadProvider>,
    prepared: PreparedReload<LoomCandidate>,
    slot: &mut ReloadSlot<HotReloader<T>>,
) -> Result<ReloadReport, ReloadError> {
    validate_live(slot.get().current(), &prepared.artifact().limits)?;
    pipeline.publish(
        prepared,
        slot,
        |_| Ok(()),
        |live, candidate, ()| live.reload(candidate.tree),
    )
}

fn validate_live(tree: &Element, limits: &ReloadLimits) -> Result<(), ReloadError> {
    let mut stack = vec![(tree, 1)];
    let mut count = 0;
    while let Some((node, depth)) = stack.pop() {
        count += 1;
        if count > limits.nodes || depth > limits.depth {
            return Err(ReloadError::Budget("live Loom tree"));
        }
        if node.child_elements().len() > limits.nodes.saturating_sub(count) {
            return Err(ReloadError::Budget("live Loom children"));
        }
        let mut identities = BTreeSet::new();
        for (index, child) in node.child_elements().iter().enumerate() {
            let identity = identity(child, index);
            if !identities.insert(identity) {
                return Err(ReloadError::provider("duplicate live sibling identity"));
            }
            if stack.len() >= limits.nodes {
                return Err(ReloadError::Budget("live Loom traversal"));
            }
            stack.push((child, depth + 1));
        }
    }
    Ok(())
}
fn identity(node: &Element, index: usize) -> NodeIdent {
    if let Some(id) = node.stable_id() {
        NodeIdent::Stable(id.clone())
    } else if let Some(key) = node.explicit_key() {
        NodeIdent::Keyed(key.clone())
    } else {
        NodeIdent::positional(index, node.kind().clone())
    }
}

type Properties = BTreeMap<StyleProp, StyleValue>;
struct Parser<'a> {
    source: &'a str,
    offset: usize,
    nodes: usize,
    limits: &'a ReloadLimits,
    cancellation: &'a ReloadCancellation,
}
impl<'a> Parser<'a> {
    fn new(
        source: &'a str,
        limits: &'a ReloadLimits,
        cancellation: &'a ReloadCancellation,
    ) -> Self {
        Self {
            source,
            offset: 0,
            nodes: 0,
            limits,
            cancellation,
        }
    }
    fn error(&self, message: &str) -> ReloadError {
        ReloadError::provider(format!("Loom byte {}: {message}", self.offset))
    }
    fn skip(&mut self) {
        loop {
            while self
                .source
                .as_bytes()
                .get(self.offset)
                .is_some_and(u8::is_ascii_whitespace)
            {
                self.offset += 1;
            }
            if self.source[self.offset..].starts_with("//") {
                self.offset += self.source[self.offset..]
                    .find('\n')
                    .unwrap_or(self.source.len() - self.offset);
            } else {
                break;
            }
        }
    }
    fn done(&mut self) -> bool {
        self.skip();
        self.offset == self.source.len()
    }
    fn eat(&mut self, ch: char) -> bool {
        self.skip();
        if self.source[self.offset..].starts_with(ch) {
            self.offset += ch.len_utf8();
            true
        } else {
            false
        }
    }
    fn expect(&mut self, ch: char) -> Result<(), ReloadError> {
        if self.eat(ch) {
            Ok(())
        } else {
            Err(self.error(&format!("expected {ch}")))
        }
    }
    fn word(&mut self) -> Result<&'a str, ReloadError> {
        self.skip();
        let start = self.offset;
        while self
            .source
            .as_bytes()
            .get(self.offset)
            .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_')
        {
            self.offset += 1;
        }
        if self.offset == start {
            Err(self.error("expected identifier"))
        } else {
            Ok(&self.source[start..self.offset])
        }
    }
    fn expect_word(&mut self, word: &str) -> Result<(), ReloadError> {
        if self.word()? == word {
            Ok(())
        } else {
            Err(self.error(&format!("expected {word}")))
        }
    }
    fn string(&mut self) -> Result<String, ReloadError> {
        self.skip();
        let start = self.offset;
        self.expect('"')?;
        let mut escaped = false;
        while let Some(&byte) = self.source.as_bytes().get(self.offset) {
            self.offset += 1;
            if byte == b'"' && !escaped {
                return serde_json::from_str(&self.source[start..self.offset])
                    .map_err(ReloadError::provider);
            }
            escaped = byte == b'\\' && !escaped;
        }
        Err(self.error("unterminated string"))
    }
    fn number_token(&mut self) -> Result<&'a str, ReloadError> {
        self.skip();
        let start = self.offset;
        while self
            .source
            .as_bytes()
            .get(self.offset)
            .is_some_and(|b| b.is_ascii_digit() || matches!(b, b'-' | b'+' | b'.' | b'e' | b'E'))
        {
            self.offset += 1;
        }
        if start == self.offset {
            Err(self.error("expected literal number"))
        } else {
            Ok(&self.source[start..self.offset])
        }
    }
    fn number(&mut self) -> Result<f32, ReloadError> {
        let value = self
            .number_token()?
            .parse::<f32>()
            .map_err(ReloadError::provider)?;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(self.error("nonfinite number"))
        }
    }
    fn node(
        &mut self,
        depth: usize,
        classes: &BTreeMap<String, Properties>,
    ) -> Result<Element, ReloadError> {
        self.cancellation.check()?;
        self.nodes += 1;
        if self.nodes > self.limits.nodes || depth > self.limits.depth {
            return Err(ReloadError::Budget("Loom tree"));
        }
        let mut element = match self.word()? {
            "box" => Element::box_(),
            "text" => {
                self.expect('(')?;
                let text = self.string()?;
                self.expect(')')?;
                Element::text(text)
            }
            "custom" => {
                self.expect('(')?;
                let kind = self.string()?;
                self.expect(')')?;
                Element::custom(kind)
            }
            _ => {
                return Err(
                    self.error("expected box, text or custom; dynamic expressions are unsupported")
                )
            }
        };
        self.expect('{')?;
        let mut attributes = BTreeSet::new();
        let mut identities = BTreeSet::new();
        let mut inline = Properties::new();
        let mut resolved = Properties::new();
        while !self.eat('}') {
            self.cancellation.check()?;
            let start = self.offset;
            let name = self.word()?;
            if !matches!(name, "key" | "stable_id" | "class" | "style") {
                self.offset = start;
                let child = self.node(depth + 1, classes)?;
                if !identities.insert(identity(&child, element.child_elements().len())) {
                    return Err(self.error("duplicate sibling identity"));
                }
                element = element.child(child);
                continue;
            }
            if !attributes.insert(name) {
                return Err(self.error("duplicate node attribute"));
            }
            self.expect(':')?;
            match name {
                "stable_id" => element = element.with_stable_id(self.string()?),
                "key" => {
                    self.skip();
                    element = if self.source[self.offset..].starts_with('"') {
                        element.key_str(self.string()?)
                    } else {
                        element.key_int(
                            self.number_token()?
                                .parse()
                                .map_err(ReloadError::provider)?,
                        )
                    };
                }
                "class" => loop {
                    let class = self.string()?;
                    let properties = classes
                        .get(&class)
                        .ok_or_else(|| self.error("unknown class"))?;
                    resolved.extend(properties.iter().map(|(k, v)| (*k, v.clone())));
                    element = element.class(class);
                    if !self.eat(',') {
                        break;
                    }
                },
                "style" => inline = self.properties()?,
                _ => unreachable!(),
            }
            self.expect(';')?;
        }
        resolved.extend(inline);
        for (prop, value) in resolved {
            element = element.style(prop, value);
        }
        Ok(element)
    }
    fn properties(&mut self) -> Result<Properties, ReloadError> {
        self.expect('{')?;
        let mut properties = Properties::new();
        while !self.eat('}') {
            self.cancellation.check()?;
            let (property, kind) =
                property(self.word()?).ok_or_else(|| self.error("unsupported style property"))?;
            self.expect(':')?;
            let value = match kind {
                ValueKind::Length { negative } => {
                    let constructor = self.word()?;
                    match constructor {
                        "auto" => StyleValue::auto(),
                        "px" | "percent" => {
                            self.expect('(')?;
                            let number = self.number()?;
                            self.expect(')')?;
                            if !negative && number < 0.0 {
                                return Err(self.error("negative length"));
                            }
                            if constructor == "px" {
                                StyleValue::px(number)
                            } else {
                                StyleValue::percent(number)
                            }
                        }
                        _ => return Err(self.error("expected px, percent or auto")),
                    }
                }
                ValueKind::Number { upper } => {
                    let number = self.number()?;
                    if number < 0.0 || number > upper {
                        return Err(self.error("number outside property range"));
                    }
                    StyleValue::number(number)
                }
                ValueKind::Color => {
                    self.expect_word("rgba8")?;
                    self.expect('(')?;
                    let mut channels = [0u8; 4];
                    for (index, channel) in channels.iter_mut().enumerate() {
                        if index != 0 {
                            self.expect(',')?;
                        }
                        *channel = self
                            .number_token()?
                            .parse::<u8>()
                            .map_err(ReloadError::provider)?;
                    }
                    self.expect(')')?;
                    StyleValue::rgba8(channels[0], channels[1], channels[2], channels[3])
                }
                ValueKind::Keyword(allowed) => {
                    let name = self.word()?;
                    let value = allowed
                        .iter()
                        .find(|(n, _)| *n == name)
                        .map(|(_, v)| *v)
                        .ok_or_else(|| self.error("invalid keyword for property"))?;
                    StyleValue::keyword(value)
                }
            };
            self.expect(';')?;
            if properties.insert(property, value).is_some() {
                return Err(self.error("duplicate style property"));
            }
        }
        Ok(properties)
    }
}

enum ValueKind {
    Length { negative: bool },
    Number { upper: f32 },
    Color,
    Keyword(&'static [(&'static str, Keyword)]),
}
fn property(name: &str) -> Option<(StyleProp, ValueKind)> {
    use StyleProp as P;
    let length = ValueKind::Length { negative: false };
    let signed = ValueKind::Length { negative: true };
    Some(match name {
        "width" => (P::Width, length),
        "height" => (P::Height, length),
        "min_width" => (P::MinWidth, length),
        "min_height" => (P::MinHeight, length),
        "max_width" => (P::MaxWidth, length),
        "max_height" => (P::MaxHeight, length),
        "padding_top" => (P::PaddingTop, length),
        "padding_right" => (P::PaddingRight, length),
        "padding_bottom" => (P::PaddingBottom, length),
        "padding_left" => (P::PaddingLeft, length),
        "margin_top" => (P::MarginTop, signed),
        "margin_right" => (P::MarginRight, signed),
        "margin_bottom" => (P::MarginBottom, signed),
        "margin_left" => (P::MarginLeft, signed),
        "gap" => (P::Gap, length),
        "row_gap" => (P::RowGap, length),
        "column_gap" => (P::ColumnGap, length),
        "font_size" => (P::FontSize, length),
        "border_radius" => (P::BorderRadius, length),
        "border_width" => (P::BorderWidth, length),
        "flex_basis" => (P::FlexBasis, length),
        "top" => (P::Top, signed),
        "right" => (P::Right, signed),
        "bottom" => (P::Bottom, signed),
        "left" => (P::Left, signed),
        "opacity" => (P::Opacity, ValueKind::Number { upper: 1.0 }),
        "flex_grow" => (P::FlexGrow, ValueKind::Number { upper: f32::MAX }),
        "flex_shrink" => (P::FlexShrink, ValueKind::Number { upper: f32::MAX }),
        "font_weight" => (P::FontWeight, ValueKind::Number { upper: 1000.0 }),
        "color" => (P::Color, ValueKind::Color),
        "background_color" => (P::BackgroundColor, ValueKind::Color),
        "border_color" => (P::BorderColor, ValueKind::Color),
        "display" => (
            P::Display,
            ValueKind::Keyword(&[
                ("flex", Keyword::Flex),
                ("grid", Keyword::Grid),
                ("block", Keyword::Block),
                ("none", Keyword::None),
            ]),
        ),
        "flex_direction" => (
            P::FlexDirection,
            ValueKind::Keyword(&[
                ("row", Keyword::Row),
                ("column", Keyword::Column),
                ("row_reverse", Keyword::RowReverse),
                ("column_reverse", Keyword::ColumnReverse),
            ]),
        ),
        "position" => (
            P::Position,
            ValueKind::Keyword(&[
                ("relative", Keyword::Relative),
                ("absolute", Keyword::Absolute),
            ]),
        ),
        "justify_content" => (
            P::JustifyContent,
            ValueKind::Keyword(&[
                ("start", Keyword::Start),
                ("end", Keyword::End),
                ("center", Keyword::Center),
                ("space_between", Keyword::SpaceBetween),
                ("space_around", Keyword::SpaceAround),
                ("space_evenly", Keyword::SpaceEvenly),
            ]),
        ),
        "align_items" => (
            P::AlignItems,
            ValueKind::Keyword(&[
                ("start", Keyword::Start),
                ("end", Keyword::End),
                ("center", Keyword::Center),
                ("stretch", Keyword::Stretch),
                ("baseline", Keyword::Baseline),
            ]),
        ),
        _ => return None,
    })
}
