//! Loom assembly for the editor's persistent shell chrome.

use prism_editor_shell::{MenuItem, StatusItem, StatusRegion, ToolbarItem};
use prism_ui::Element;
use prism_ui_component::Component;
use prism_ui_component_kit::basics::{Button, ButtonProps};
use prism_ui_component_kit::kit::{ButtonVariant, ControlSize};

use crate::action::{ActionBinding, ActionBindings, ActionTemplate, BoundElement, UiBuildError};
use crate::support::label;

/// Resolved contribution slices used to build the shell chrome.
#[derive(Clone, Copy, Debug)]
pub struct ShellChromeProps<'a> {
    /// Application title shown in the title bar.
    pub title: &'a str,
    /// Active workspace/perspective label.
    pub workspace: &'a str,
    /// Menu items for the active menu surface, already resolved for context.
    pub menus: &'a [MenuItem],
    /// Toolbar actions for the active toolbar surface, already resolved.
    pub toolbar: &'a [ToolbarItem],
    /// Status items already resolved and sorted by the shell model.
    pub status: &'a [StatusItem],
}

/// Builds the stable shell chrome tree.
#[must_use]
pub fn shell_chrome(props: &ShellChromeProps<'_>) -> Element {
    build_shell_chrome(props)
}

/// Builds shell chrome and binds command-bearing menu and toolbar items.
pub fn shell_chrome_bound(props: &ShellChromeProps<'_>) -> Result<BoundElement, UiBuildError> {
    let mut bindings = ActionBindings::new();
    for item in props.menus {
        if let Some(command_id) = &item.command_id {
            bindings.bind(ActionBinding::activate(
                format!("menu:{}", item.id),
                ActionTemplate::execute_command(command_id.clone())?,
            )?)?;
        }
    }
    for item in props.toolbar {
        bindings.bind(ActionBinding::activate(
            format!("toolbar:{}", item.id),
            ActionTemplate::execute_command(item.command_id.clone())?,
        )?)?;
    }
    Ok(BoundElement::new(build_shell_chrome(props), bindings))
}

fn build_shell_chrome(props: &ShellChromeProps<'_>) -> Element {
    let title_bar = Element::box_()
        .class("prism-editor-shell__title-bar")
        .child(label("prism-editor-shell__title", props.title))
        .child(label("prism-editor-shell__workspace", props.workspace));

    let menu_bar = Element::box_()
        .class("prism-editor-shell__menu-bar")
        .children(props.menus.iter().enumerate().map(|(index, item)| {
            let mut row = Element::box_()
                .key_str(format!("menu:{}", item.id))
                .class("prism-editor-shell__menu-item")
                .child(label("prism-editor-shell__menu-label", item.label.clone()));
            if item.parent_id.is_some() {
                row = row.class("is-submenu");
            }
            if index > 0 && item.group != props.menus[index - 1].group {
                row = row.class("is-group-start");
            }
            row
        }));

    let toolbar = Element::box_()
        .class("prism-editor-shell__toolbar")
        .children(props.toolbar.iter().map(|item| {
            Button
                .render(
                    &ButtonProps::new(item.label.clone())
                        .variant(ButtonVariant::Glass)
                        .size(ControlSize::Small),
                )
                .key_str(format!("toolbar:{}", item.id))
                .class("prism-editor-shell__toolbar-item")
        }));

    let status_bar = Element::box_()
        .class("prism-editor-shell__status-bar")
        .child(status_region("left", props.status, StatusRegion::Left))
        .child(status_region("center", props.status, StatusRegion::Center))
        .child(status_region("right", props.status, StatusRegion::Right));

    Element::custom("prism.editor.shell")
        .class("prism-editor-shell")
        .child(title_bar)
        .child(menu_bar)
        .child(toolbar)
        .child(status_bar)
}

fn status_region(name: &str, items: &[StatusItem], region: StatusRegion) -> Element {
    let mut slot = Element::box_()
        .key_str(format!("status-region:{name}"))
        .class("prism-editor-shell__status-region")
        .class(format!("is-{name}"));
    for item in items.iter().filter(|item| item.region == region) {
        slot = slot.child(
            label("prism-editor-shell__status-item", item.text.clone())
                .key_str(format!("status:{}", item.id)),
        );
    }
    slot
}
