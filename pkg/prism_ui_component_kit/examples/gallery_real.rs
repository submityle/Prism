//! Real-instance gallery: renders a curated set of *actual* kit components —
//! not bare class swatches — through Loom's own layout pass and reference
//! rasterizer, one PNG per theme mode (light / dark) plus a side-by-side sheet.
//!
//! Each tile mounts a real [`Component`] via
//! [`prism_ui_component::mount_component`], so composite controls paint their
//! full sub-tree (a slider's track + fill + thumb, a card's header/body/footer,
//! a progress bar's fill, an alert's tinted surface, …). This is the honest
//! answer to "are the components empty?": they are not — the earlier single-box
//! swatch sheet simply could not show visuals that live in child elements.
//!
//! The reference glyph rasterizer now paints real font outlines (via
//! `prism_ui_font`'s embedded `FiraMono` vector tier), so labels render as
//! crisp, legible text at any size; the tiles exercise both typography and
//! surface/shape/color fidelity.

#![allow(
    clippy::print_stdout,
    clippy::std_instead_of_alloc,
    clippy::uninlined_format_args,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "dev-only CLI example: renders a PNG and prints its path; may use std and pixel casts"
)]

use std::fs;
use std::path::Path;

use prism_ui::layout::{AvailableSpace, Size};
use prism_ui::{Element, Ui};
use prism_ui_component::mount_component;
use prism_ui_component_kit::basics::{Avatar, AvatarProps, Button, ButtonProps, Tag, TagProps};
use prism_ui_component_kit::containers::{Card, CardProps};
use prism_ui_component_kit::feedback::{Alert, AlertProps, ProgressBar, ProgressBarProps};
use prism_ui_component_kit::inputs::{
    Checkbox, CheckboxProps, Slider, SliderProps, Toggle, ToggleProps,
};
use prism_ui_component_kit::{stylesheet, ButtonVariant, ControlSize, Tone};
use prism_ui_render_backend::raster::{rasterize, Framebuffer};
use prism_ui_render_backend::scene::RetainedScene;
use prism_ui_style::{Breakpoint, Color, Keyword, Length, StyleProp, StyleValue, Theme, TokenStore};
use prism_ui_theme::{compile_theme, ThemeDefinition, ThemeMode};

// --- gallery geometry (logical px) -----------------------------------------
const COLS: usize = 5;
const CELL_W: f32 = 220.0;
const CELL_H: f32 = 120.0;
const GAP: f32 = 16.0;
const PAD: f32 = 28.0;

fn len(px: f32) -> StyleValue {
    StyleValue::Length(Length::Px(px))
}

/// Builds a `prism_ui_style::Theme` from the kit's glass definition for `mode`,
/// returning the resolved page background color alongside it.
fn kit_theme(mode: &ThemeMode) -> (Theme, Color) {
    let def = ThemeDefinition::glass();
    let compiled = compile_theme(&def, mode).expect("glass theme compiles");
    let mut tokens = TokenStore::new();
    let mut background = Color::rgba8(255, 255, 255, 255);
    for (name, value) in compiled.iter() {
        if name == "color.background"
            && let StyleValue::Color(c) = value
        {
            background = *c;
        }
        tokens.insert(name, value.clone());
    }
    (
        Theme {
            tokens,
            breakpoints: Breakpoint::ALL,
        },
        background,
    )
}

/// The curated set of real component instances to showcase. Each entry is a
/// fully built [`Element`] subtree produced by a real component's `render`.
fn instances() -> Vec<Element> {
    let btn = |variant: ButtonVariant| {
        mount_component(
            &Button,
            ButtonProps::new("Button")
                .variant(variant)
                .size(ControlSize::Medium),
        )
    };
    vec![
        btn(ButtonVariant::Filled),
        btn(ButtonVariant::Tinted),
        btn(ButtonVariant::Gray),
        btn(ButtonVariant::Glass),
        btn(ButtonVariant::Plain),
        mount_component(&Tag, TagProps::new("Accent").tone(Tone::Accent)),
        mount_component(&Tag, TagProps::new("Success").tone(Tone::Success)),
        mount_component(&Tag, TagProps::new("Danger").tone(Tone::Danger)),
        mount_component(&Avatar, AvatarProps::new().initials("AB")),
        mount_component(&Toggle, ToggleProps::new().on(true)),
        mount_component(&Toggle, ToggleProps::new().on(false)),
        mount_component(&Checkbox, CheckboxProps::new().checked(true).label("Checked")),
        mount_component(&Slider, SliderProps::new().value(0.35)),
        mount_component(&Slider, SliderProps::new().value(0.7)),
        mount_component(&ProgressBar, ProgressBarProps::new(0.6)),
        mount_component(
            &ProgressBar,
            ProgressBarProps::new(0.9).tone(Tone::Success),
        ),
        mount_component(
            &Alert,
            AlertProps::new("Saved successfully")
                .title("Success")
                .tone(Tone::Success),
        ),
        mount_component(
            &Alert,
            AlertProps::new("Check your input").title("Warning").tone(Tone::Warning),
        ),
        mount_component(
            &Card,
            CardProps::new()
                .header(Element::box_().child(Element::text("Card")))
                .child(Element::text("Body content")),
        ),
        mount_component(
            &Card,
            CardProps::new()
                .glass(true)
                .child(Element::text("Glass card")),
        ),
    ]
}

/// Wraps a real instance in a fixed-size, centered tile so the sheet reads as a
/// grid regardless of each component's intrinsic size.
fn tile(instance: Element) -> Element {
    Element::box_()
        .style(StyleProp::Width, len(CELL_W))
        .style(StyleProp::Height, len(CELL_H))
        .style(StyleProp::MinWidth, len(CELL_W))
        .style(StyleProp::MinHeight, len(CELL_H))
        .style(StyleProp::Display, StyleValue::Keyword(Keyword::Flex))
        .style(StyleProp::FlexDirection, StyleValue::Keyword(Keyword::Column))
        .style(StyleProp::AlignItems, StyleValue::Keyword(Keyword::Center))
        .style(StyleProp::JustifyContent, StyleValue::Keyword(Keyword::Center))
        .style(StyleProp::PaddingTop, len(12.0))
        .style(StyleProp::PaddingBottom, len(12.0))
        .style(StyleProp::PaddingLeft, len(12.0))
        .style(StyleProp::PaddingRight, len(12.0))
        .child(instance)
}

/// Builds the gallery element tree: a padded column of fixed rows of tiles.
fn gallery(instances: Vec<Element>, background: Color, width: f32, height: f32) -> Element {
    let mut root = Element::box_()
        .style(StyleProp::Width, len(width))
        .style(StyleProp::Height, len(height))
        .style(StyleProp::Display, StyleValue::Keyword(Keyword::Flex))
        .style(StyleProp::FlexDirection, StyleValue::Keyword(Keyword::Column))
        .style(StyleProp::PaddingTop, len(PAD))
        .style(StyleProp::PaddingLeft, len(PAD))
        .style(StyleProp::PaddingRight, len(PAD))
        .style(StyleProp::PaddingBottom, len(PAD))
        .style(StyleProp::RowGap, len(GAP))
        .style(StyleProp::BackgroundColor, StyleValue::Color(background));

    let mut iter = instances.into_iter().peekable();
    while iter.peek().is_some() {
        let mut row = Element::box_()
            .style(StyleProp::Display, StyleValue::Keyword(Keyword::Flex))
            .style(StyleProp::FlexDirection, StyleValue::Keyword(Keyword::Row))
            .style(StyleProp::ColumnGap, len(GAP))
            .style(StyleProp::Height, len(CELL_H));
        for _ in 0..COLS {
            match iter.next() {
                Some(instance) => row = row.child(tile(instance)),
                None => break,
            }
        }
        root = root.child(row);
    }
    root
}

/// Renders `element` to a `Framebuffer` via the engine's layout + reference
/// rasterizer.
fn render(element: &Element, theme: Theme, width: f32, height: f32) -> Framebuffer {
    let mut ui = Ui::new(RetainedScene::new())
        .with_theme(theme)
        .with_stylesheet(stylesheet());
    ui.set_viewport_width(width);
    ui.mount(element);
    ui.compute_layout(Size::new(
        AvailableSpace::Definite(width),
        AvailableSpace::Definite(height),
    ));
    let list = ui.backend().to_draw_list();
    rasterize(&list, width as u32, height as u32)
}

// --- minimal PNG encoder (no external crates) -------------------------------

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b): (u32, u32) = (1, 0);
    for &byte in bytes {
        a = (a + u32::from(byte)) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

fn encode_png(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    let stride = (width * 4) as usize;
    let mut raw = Vec::with_capacity((stride + 1) * height as usize);
    for y in 0..height as usize {
        raw.push(0);
        raw.extend_from_slice(&rgba[y * stride..(y + 1) * stride]);
    }

    let mut z = Vec::new();
    z.push(0x78);
    z.push(0x01);
    let mut offset = 0usize;
    while offset < raw.len() {
        let block = (raw.len() - offset).min(0xFFFF);
        let final_block = u8::from(offset + block >= raw.len());
        z.push(final_block);
        z.extend_from_slice(&(block as u16).to_le_bytes());
        z.extend_from_slice(&(!(block as u16)).to_le_bytes());
        z.extend_from_slice(&raw[offset..offset + block]);
        offset += block;
    }
    z.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut out = Vec::new();
    out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &z);
    chunk(&mut out, b"IEND", &[]);
    out
}

fn framebuffer_to_rgba8(fb: &Framebuffer) -> Vec<u8> {
    let (w, h) = (fb.width(), fb.height());
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let p = fb.pixel(x, y);
            for channel in p {
                out.push((channel.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            }
        }
    }
    out
}

fn compose_side_by_side(left: &Framebuffer, right: &Framebuffer) -> (u32, u32, Vec<u8>) {
    let h = left.height().max(right.height());
    let gutter = 2u32;
    let w = left.width() + gutter + right.width();
    let mut out = vec![0u8; (w * h * 4) as usize];
    let mut blit = |fb: &Framebuffer, x0: u32| {
        for y in 0..fb.height() {
            for x in 0..fb.width() {
                let p = fb.pixel(x, y);
                let idx = ((y * w + x0 + x) * 4) as usize;
                for (c, channel) in p.iter().enumerate() {
                    out[idx + c] = (channel.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                }
            }
        }
    };
    blit(left, 0);
    blit(right, left.width() + gutter);
    (w, h, out)
}

fn write_png(path: &Path, width: u32, height: u32, rgba: &[u8]) {
    let bytes = encode_png(width, height, rgba);
    fs::write(path, bytes).expect("write png");
    println!("wrote {} ({}x{})", path.display(), width, height);
}

fn main() {
    let count = instances().len();
    let rows = count.div_ceil(COLS);
    let width = PAD * 2.0 + COLS as f32 * CELL_W + (COLS as f32 - 1.0) * GAP;
    let height = PAD * 2.0 + rows as f32 * CELL_H + (rows as f32 - 1.0) * GAP;
    println!(
        "{} real instances -> {} cols x {} rows, {:.0}x{:.0}px per theme",
        count, COLS, rows, width, height
    );

    let out_dir = std::env::var("GALLERY_OUT")
        .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
    let out_dir = Path::new(&out_dir);
    fs::create_dir_all(out_dir).expect("create out dir");

    let (light_theme, light_bg) = kit_theme(&ThemeMode::Light);
    let (dark_theme, dark_bg) = kit_theme(&ThemeMode::Dark);

    let light_el = gallery(instances(), light_bg, width, height);
    let dark_el = gallery(instances(), dark_bg, width, height);

    let light_fb = render(&light_el, light_theme, width, height);
    let dark_fb = render(&dark_el, dark_theme, width, height);

    write_png(
        &out_dir.join("gallery_real_light.png"),
        light_fb.width(),
        light_fb.height(),
        &framebuffer_to_rgba8(&light_fb),
    );
    write_png(
        &out_dir.join("gallery_real_dark.png"),
        dark_fb.width(),
        dark_fb.height(),
        &framebuffer_to_rgba8(&dark_fb),
    );

    let (cw, ch, combined) = compose_side_by_side(&light_fb, &dark_fb);
    write_png(&out_dir.join("gallery_real_all.png"), cw, ch, &combined);
}
