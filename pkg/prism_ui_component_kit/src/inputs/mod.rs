//! `inputs/` controls — the form/data-entry family. See the kit design doc,
//! section 5.
//!
//! Each control emits a data-only [`Element`](prism_ui::Element) carrying only
//! kit class names; this module's [`register_styles`] owns the token-backed
//! [`Class`](prism_ui_style::Class) rules for those names. Controls stay plain
//! [`Component`](prism_ui_component::Component)s so they compose and unit-test
//! without a running runtime.
//!
//! This family is the stateless *structure* layer: it renders the shape of a
//! field, toggle, checkbox, radio (and group), slider, select, stepper and
//! search field from its props. Live editing/selection state is owned by
//! `prism_ui_form`, so these controls never depend on it — they only draw what
//! their props describe.
//!
//! This family ships: [`TextField`], [`TextArea`], [`Toggle`], [`Checkbox`],
//! [`Radio`], [`RadioGroup`], [`Slider`], [`Select`], [`Stepper`] and
//! [`SearchField`].

use crate::preset::StyleSheet;

pub mod checkbox;
pub mod radio;
pub mod radio_group;
pub mod search_field;
pub mod select;
pub mod slider;
pub mod stepper;
pub mod text_area;
pub mod text_field;
pub mod toggle;
pub mod cascader;
pub mod dropzone;
pub mod fieldset;
pub mod form_error;
pub mod form_field;
pub mod form_label;
pub mod input_group;
pub mod masked_input;
pub mod mentions;
pub mod multi_select;
pub mod password_field;
pub mod toggle_group;
pub mod transfer;
pub mod tree_select;

pub use checkbox::{Checkbox, CheckboxProps};
pub use radio::{Radio, RadioProps};
pub use radio_group::{RadioGroup, RadioGroupProps};
pub use search_field::{SearchField, SearchFieldProps};
pub use select::{Select, SelectProps};
pub use slider::{Slider, SliderProps};
pub use stepper::{Stepper, StepperProps};
pub use text_area::{TextArea, TextAreaProps};
pub use text_field::{TextField, TextFieldProps};
pub use toggle::{Toggle, ToggleProps};
pub use cascader::{Cascader, CascaderProps};
pub use dropzone::{Dropzone, DropzoneProps};
pub use fieldset::{Fieldset, FieldsetProps};
pub use form_error::{FormError, FormErrorProps};
pub use form_field::{FormField, FormFieldProps};
pub use form_label::{FormLabel, FormLabelProps};
pub use input_group::{InputGroup, InputGroupProps};
pub use masked_input::{apply_mask, MaskedInput, MaskedInputProps};
pub use mentions::{Mentions, MentionsProps};
pub use multi_select::{MultiSelect, MultiSelectProps};
pub use password_field::{PasswordField, PasswordFieldProps};
pub use toggle_group::{ToggleGroup, ToggleGroupProps};
pub use transfer::{Transfer, TransferProps};
pub use tree_select::{TreeSelect, TreeSelectNode, TreeSelectProps};

/// Registers every `inputs/` control's token-backed classes into `sheet`.
pub fn register_styles(sheet: &mut StyleSheet) {
    text_field::register_styles(sheet);
    text_area::register_styles(sheet);
    toggle::register_styles(sheet);
    checkbox::register_styles(sheet);
    radio::register_styles(sheet);
    radio_group::register_styles(sheet);
    slider::register_styles(sheet);
    select::register_styles(sheet);
    stepper::register_styles(sheet);
    search_field::register_styles(sheet);
    cascader::register_styles(sheet);
    dropzone::register_styles(sheet);
    fieldset::register_styles(sheet);
    form_error::register_styles(sheet);
    form_field::register_styles(sheet);
    form_label::register_styles(sheet);
    input_group::register_styles(sheet);
    masked_input::register_styles(sheet);
    mentions::register_styles(sheet);
    multi_select::register_styles(sheet);
    password_field::register_styles(sheet);
    toggle_group::register_styles(sheet);
    transfer::register_styles(sheet);
    tree_select::register_styles(sheet);
}
