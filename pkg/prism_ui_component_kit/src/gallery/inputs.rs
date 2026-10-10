//! Gallery instances for the `inputs` family. Dev-only; see `gallery/mod.rs`.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::mount_component;

use super::Showcase;
use crate::inputs::{
    Cascader, CascaderProps, Checkbox, CheckboxProps, Combobox, ComboboxOption, ComboboxProps,
    Dropzone, DropzoneProps, Fieldset, FieldsetProps, FileField, FileFieldProps, FormError,
    FormErrorProps, FormField, FormFieldProps, FormLabel, FormLabelProps, InputGroup,
    InputGroupProps, MaskedInput, MaskedInputProps, Mentions, MentionsProps, MultiSelect,
    MultiSelectProps, NumberField, NumberFieldProps, PasswordField, PasswordFieldProps, PinInput,
    PinInputProps, Radio, RadioGroup, RadioGroupProps, RadioProps, RangeSlider, RangeSliderProps,
    SearchField, SearchFieldProps, Select, SelectProps, Slider, SliderProps, Stepper, StepperProps,
    TagInput, TagInputProps, TextArea, TextAreaProps, TextField, TextFieldProps, Toggle,
    ToggleGroup, ToggleGroupProps, ToggleProps, Transfer, TransferProps, TreeSelect, TreeSelectNode,
    TreeSelectProps,
};
use crate::ControlSize;

/// Real, named instances of every `inputs` control.
#[must_use]
pub fn instances() -> Vec<Showcase> {
    alloc::vec![
        Showcase::new(
            "TextField / Value",
            mount_component(
                &TextField,
                TextFieldProps::new()
                    .value("Ada Lovelace")
                    .placeholder("Full name")
                    .size(ControlSize::Medium)
                    .leading(Element::text("@"))
                    .trailing(Element::text("✓")),
            ),
        ),
        Showcase::new(
            "TextArea / Multiline",
            mount_component(
                &TextArea,
                TextAreaProps::new()
                    .value("The quick brown fox\njumps over the lazy dog.")
                    .placeholder("Write a note…")
                    .rows(4),
            ),
        ),
        Showcase::new(
            "Toggle / On",
            mount_component(&Toggle, ToggleProps::new().on(true)),
        ),
        Showcase::new(
            "Checkbox / Checked",
            mount_component(
                &Checkbox,
                CheckboxProps::new().checked(true).label("Enable sync"),
            ),
        ),
        Showcase::new(
            "Radio / Selected",
            mount_component(
                &Radio,
                RadioProps::new().selected(true).label("Standard shipping"),
            ),
        ),
        Showcase::new(
            "RadioGroup / Three",
            mount_component(
                &RadioGroup,
                RadioGroupProps::new()
                    .options(["Low", "Medium", "High"])
                    .selected(1),
            ),
        ),
        Showcase::new(
            "Slider / 60%",
            mount_component(&Slider, SliderProps::new().value(0.6)),
        ),
        Showcase::new(
            "Select / Chosen",
            mount_component(
                &Select,
                SelectProps::new()
                    .options(["Red", "Green", "Blue"])
                    .selected(2)
                    .placeholder("Pick a color"),
            ),
        ),
        Showcase::new(
            "Stepper / 3 of 0..10",
            mount_component(&Stepper, StepperProps::new(3, 0, 10)),
        ),
        Showcase::new(
            "SearchField / Query",
            mount_component(
                &SearchField,
                SearchFieldProps::new()
                    .value("loom runtime")
                    .placeholder("Search…"),
            ),
        ),
        Showcase::new(
            "Cascader / Open",
            mount_component(
                &Cascader,
                CascaderProps::new()
                    .columns([
                        alloc::vec!["Asia", "Europe"],
                        alloc::vec!["China", "Japan"],
                        alloc::vec!["Beijing", "Shanghai"],
                    ])
                    .path([0usize, 0usize, 1usize])
                    .open(true),
            ),
        ),
        Showcase::new(
            "Dropzone / Active",
            mount_component(
                &Dropzone,
                DropzoneProps::new("Drop files here or click to browse").active(true),
            ),
        ),
        Showcase::new(
            "Fieldset / Legend",
            mount_component(
                &Fieldset,
                FieldsetProps::new("Account details")
                    .child(Element::text("Name: Ada"))
                    .child(Element::text("Role: Engineer")),
            ),
        ),
        Showcase::new(
            "FormError / Message",
            mount_component(
                &FormError,
                FormErrorProps::new("This field is required."),
            ),
        ),
        Showcase::new(
            "FormField / Full",
            mount_component(
                &FormField,
                FormFieldProps::new()
                    .label(mount_component(&FormLabel, FormLabelProps::new("Email").required(true)))
                    .control(mount_component(
                        &TextField,
                        TextFieldProps::new().placeholder("you@example.com"),
                    ))
                    .error(mount_component(
                        &FormError,
                        FormErrorProps::new("Enter a valid email."),
                    ))
                    .help("We'll never share your email.")
                    .required(true),
            ),
        ),
        Showcase::new(
            "FormLabel / Required",
            mount_component(
                &FormLabel,
                FormLabelProps::new("Password").required(true),
            ),
        ),
        Showcase::new(
            "InputGroup / Prefix+Suffix",
            mount_component(
                &InputGroup,
                InputGroupProps::new()
                    .prefix(Element::text("https://"))
                    .input(mount_component(
                        &TextField,
                        TextFieldProps::new().value("example"),
                    ))
                    .suffix(Element::text(".com")),
            ),
        ),
        Showcase::new(
            "MaskedInput / Phone",
            mount_component(
                &MaskedInput,
                MaskedInputProps::new()
                    .value("4155550199")
                    .mask("(###) ###-####")
                    .placeholder("(___) ___-____"),
            ),
        ),
        Showcase::new(
            "Mentions / Open",
            mount_component(
                &Mentions,
                MentionsProps::new()
                    .value("Hey @ad")
                    .suggestions(["ada", "adam", "adrian"])
                    .open(true),
            ),
        ),
        Showcase::new(
            "MultiSelect / Open",
            mount_component(
                &MultiSelect,
                MultiSelectProps::new()
                    .options(["Rust", "Go", "Zig", "C++"])
                    .selected([0usize, 2usize])
                    .placeholder("Pick languages")
                    .open(true),
            ),
        ),
        Showcase::new(
            "PasswordField / Revealed",
            mount_component(
                &PasswordField,
                PasswordFieldProps::new()
                    .value("hunter2")
                    .placeholder("Password")
                    .revealed(true),
            ),
        ),
        Showcase::new(
            "ToggleGroup / Exclusive",
            mount_component(
                &ToggleGroup,
                ToggleGroupProps::new()
                    .options(["Left", "Center", "Right"])
                    .selected([1usize])
                    .exclusive(true),
            ),
        ),
        Showcase::new(
            "Transfer / Two lists",
            mount_component(
                &Transfer,
                TransferProps::new()
                    .source(["Apple", "Banana", "Cherry"])
                    .target(["Date", "Elderberry"]),
            ),
        ),
        Showcase::new(
            "TreeSelect / Open",
            mount_component(
                &TreeSelect,
                TreeSelectProps::new()
                    .roots([
                        TreeSelectNode::new("Fruits")
                            .expanded(true)
                            .children([
                                TreeSelectNode::new("Apple").selected(true),
                                TreeSelectNode::new("Banana"),
                            ]),
                        TreeSelectNode::new("Vegetables")
                            .children([TreeSelectNode::new("Carrot")]),
                    ])
                    .open(true),
            ),
        ),
        Showcase::new(
            "Combobox / Open",
            mount_component(
                &Combobox,
                ComboboxProps::new()
                    .value("Ap")
                    .options([
                        ComboboxOption::new("Apple", "apple"),
                        ComboboxOption::new("Apricot", "apricot"),
                        ComboboxOption::new("Avocado", "avocado"),
                    ])
                    .open(true),
            ),
        ),
        Showcase::new(
            "FileField / Chosen",
            mount_component(
                &FileField,
                FileFieldProps::new()
                    .label("Upload avatar")
                    .filename("portrait.png"),
            ),
        ),
        Showcase::new(
            "NumberField / 42",
            mount_component(
                &NumberField,
                NumberFieldProps::new()
                    .value(42.0)
                    .min(0.0)
                    .max(100.0)
                    .step(1.0),
            ),
        ),
        Showcase::new(
            "PinInput / Masked",
            mount_component(
                &PinInput,
                PinInputProps::new()
                    .length(6)
                    .value("1234")
                    .masked(true),
            ),
        ),
        Showcase::new(
            "RangeSlider / 25..75",
            mount_component(
                &RangeSlider,
                RangeSliderProps::new()
                    .min(0.0)
                    .max(100.0)
                    .low(25.0)
                    .high(75.0),
            ),
        ),
        Showcase::new(
            "TagInput / Tags",
            mount_component(
                &TagInput,
                TagInputProps::new()
                    .tags(["design", "rust", "ui"])
                    .placeholder("Add a tag…"),
            ),
        ),
    ]
}
