use alloc::vec::Vec;
use prism_render_material::{
    ClosureKind, GpuMaterialHeader, Illumination, LobeLayout, MaterialRenderClass,
    ResponseModeLayout, SURFACE_LOBE_CLOSURES, SURFACE_RESPONSE_MODES,
};

pub const MAX_SHADING_CLASSES: usize = 9;

/// The runtime shading bucket a pixel is binned into for the deferred/compute
/// shading pass. This is a *derived* projection of the orthogonal material axes
/// (`illumination` + `closure_mask` + `render_class`), not a stored field: the
/// material ABI no longer carries a single `shading_model`. Keeping the 9-way
/// enum lets the shading pass stay wavefront-coherent while the material side
/// stays fully orthogonal.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub enum MaterialShadingClass {
    #[default]
    Principled,
    Unlit,
    Subsurface,
    ClearCoat,
    Cloth,
    Hair,
    Water,
    Npr,
    Custom,
}

impl MaterialShadingClass {
    pub const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClassificationError {
    Inactive,
    NonSurfaceClass,
    UnsupportedIllumination(u32),
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadingWorkItem {
    pub pixel_index: u32,
    pub material_index: u32,
    pub shading_class: u32,
    pub flags: u32,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ShadingWorkPlan {
    pub offsets: [u32; MAX_SHADING_CLASSES],
    pub counts: [u32; MAX_SHADING_CLASSES],
    pub work: Vec<ShadingWorkItem>,
}

impl ShadingWorkPlan {
    pub fn build(
        pixels: impl IntoIterator<Item = (u32, super::VisibilityPixel)>,
        materials: &[GpuMaterialHeader],
        capacity: u32,
    ) -> Self {
        Self::build_with_region(
            pixels
                .into_iter()
                .map(|(pixel_index, pixel)| (pixel_index, pixel, u32::MAX)),
            materials,
            capacity,
        )
    }

    /// Build a wavefront plan from visibility pixels and their optional
    /// per-region illumination selectors. The selector is written by the
    /// visibility raster in metadata.z (`0` = Lit, `1` = Stylized,
    /// `u32::MAX` = no region mask). Keeping this value in the CPU plan API
    /// makes diagnostics/replay use the exact same material classifier as the
    /// GPU classify pass instead of silently falling back to material-wide
    /// illumination.
    pub fn build_with_region(
        pixels: impl IntoIterator<Item = (u32, super::VisibilityPixel, u32)>,
        materials: &[GpuMaterialHeader],
        capacity: u32,
    ) -> Self {
        let mut classified: [Vec<ShadingWorkItem>; MAX_SHADING_CLASSES] = Default::default();
        let mut accepted = 0usize;
        for (pixel_index, pixel, region_illumination) in pixels {
            if !pixel.is_valid() || accepted >= capacity as usize {
                continue;
            }
            let Some(header) = materials.get(pixel.material_index as usize) else {
                continue;
            };
            if header.generation != pixel.material_generation {
                continue;
            }
            let Ok(class) = classify_material_header_with_region(header, region_illumination)
            else {
                continue;
            };
            classified[class.index()].push(ShadingWorkItem {
                pixel_index,
                material_index: pixel.material_index,
                shading_class: class as u32,
                flags: header.feature_flags,
            });
            accepted += 1;
        }
        let mut plan = Self::default();
        for (class, items) in classified.into_iter().enumerate() {
            plan.offsets[class] = plan.work.len() as u32;
            plan.counts[class] = items.len() as u32;
            plan.work.extend(items);
        }
        plan
    }
}

/// True when a closure bit is present in the packed mask.
fn has_closure(mask: u32, kind: ClosureKind) -> bool {
    mask & (1 << kind as u32) != 0
}

/// What makes a material belong to a shading class (roadmap P2-j).
///
/// The class set is **registration-driven**: this table is built once at
/// pipeline-build time from the PCM registry (the response-mode rows) plus the
/// lobe catalogue (the closures that own a dedicated shading bucket), and the
/// per-pixel classifier is its verified specialization. Adding a response mode
/// to the catalogue therefore adds a class row rather than editing a closed
/// enum tree, while the hot path stays exactly as it is: a priority-ordered
/// walk of a table that was fixed before the first dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadingClassRule {
    /// A dedicated subsystem render class binds its bucket directly, whatever
    /// the illumination axis says (a stylized water shader is still water work).
    RenderClass(MaterialRenderClass),
    /// The illumination axis selects this bucket.
    Illumination(Illumination),
    /// `Illumination::Custom` **with a bound program handle** (§3.5.4).
    ///
    /// A custom response needs something to run. With no handle the material
    /// falls through to the default row, i.e. it renders on the shared
    /// screen-space über path — the specification's "no `respond_custom` ⇒ safe
    /// degradation" — instead of occupying a `Custom` bucket that would run the
    /// very same integrator under a different name.
    CustomProgram,
    /// A `Lit` surface carrying this dominant closure selects this bucket.
    Closure(ClosureKind),
    /// The fallback: a `Lit` surface whose closures name no specialized bucket.
    Default,
}

/// One row of the shading-class table, in **evaluation order** (first match
/// wins, so priority is the row order).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShadingClassRow {
    /// The bucket's stable id — the `MaterialShadingClass` discriminant the work
    /// plan and every WESL `CLASS_*` constant use.
    pub id: u32,
    /// Display name, for diagnostics and the WESL parity gate.
    pub name: &'static str,
    /// The predicate this row tests.
    pub rule: ShadingClassRule,
}

/// A failure while deriving the class table from the registry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClassTableError {
    /// A registered response mode with no shading implementation in the engine.
    ///
    /// Registration is a *declaration*; without an implementation the class
    /// table cannot be built. Failing here — at pipeline-build time — is the
    /// whole point: a registered-but-unshaded mode must be a loud error rather
    /// than a material that silently classifies as something else.
    UnhandledResponseMode(String),
    /// A lobe declared a closure-bucket name the shading side does not
    /// implement.
    ///
    /// The schema owns the declaration; the engine owns the implementation.
    /// Silently dropping an unknown name would let a schema change remove a
    /// bucket without any build failure, so this is a loud pipeline-build
    /// error.
    UnknownClosureName(String),
    /// A schema response mode named an illumination variant the runtime does
    /// not implement.
    UnknownResponseModeIllumination(String),
    /// A schema response mode has no matching `Kind::ResponseMode` registry row.
    UnregisteredResponseMode(String),
    /// The schema and the PCM registry disagree on a response-mode slot.
    ResponseModeSlotMismatch {
        /// Response-mode uid.
        uid: String,
        /// Slot declared by the schema.
        schema: u32,
        /// Slot registered in the PCM catalogue.
        registry: u32,
    },
    /// A schema response mode claims the default `Lit` illumination, which is
    /// not an extension and must not occupy a class-table row.
    DefaultResponseModeDeclared(String),
    /// A non-default illumination variant has no schema response mode.
    MissingResponseMode(Illumination),
    /// The class set does not tile `0..=class_count` (a bucket id is missing or
    /// duplicated in a way the work plan cannot index).
    NonDenseClassIds {
        /// The ids the table declared, in row order.
        ids: Vec<u32>,
    },
    /// The class set is larger than the shader-side capacity.
    CapacityExceeded {
        /// Buckets the table declares.
        needed: usize,
        /// The capacity `MAX_SHADING_CLASSES` fixes for the work-plan arrays.
        capacity: usize,
    },
}

impl core::fmt::Display for ClassTableError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnhandledResponseMode(uid) => write!(
                f,
                "response mode {uid:?} is registered but has no shading implementation"
            ),
            Self::UnknownClosureName(name) => write!(
                f,
                "lobe closure {name:?} is declared by the schema but has no shading implementation"
            ),
            Self::UnknownResponseModeIllumination(illumination) => write!(
                f,
                "response mode declares illumination {illumination:?}, which the runtime axis does not implement"
            ),
            Self::UnregisteredResponseMode(uid) => write!(
                f,
                "response mode {uid:?} is declared by the schema but has no PCM registry row"
            ),
            Self::ResponseModeSlotMismatch {
                uid,
                schema,
                registry,
            } => write!(
                f,
                "response mode {uid:?} declares slot {schema} but the PCM registry pins slot {registry}"
            ),
            Self::DefaultResponseModeDeclared(uid) => write!(
                f,
                "response mode {uid:?} claims the default `lit` illumination; `lit` is not an extension"
            ),
            Self::MissingResponseMode(illumination) => write!(
                f,
                "illumination variant {illumination:?} has no registered response mode"
            ),
            Self::NonDenseClassIds { ids } => {
                write!(f, "shading-class ids {ids:?} do not tile 0..=max")
            }
            Self::CapacityExceeded { needed, capacity } => write!(
                f,
                "the class set needs {needed} buckets but the shader-side capacity is {capacity}"
            ),
        }
    }
}

impl std::error::Error for ClassTableError {}

/// A schema-declared closure name resolved to the engine's shading bucket.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ClosureBucket {
    /// The closure bit the row tests.
    kind: ClosureKind,
    /// The `MaterialShadingClass` bucket the closure owns.
    class: MaterialShadingClass,
    /// The class-table/WESL name for the bucket.
    row_name: &'static str,
    /// Evaluation priority among closure rows (lower runs first).
    priority: u8,
}

/// Resolve a schema closure name against the shading side's closed
/// implementation set.
///
/// Names are intentionally resolved here rather than baked into the schema:
/// the schema declares *what the lobe owns*, while the shading crate owns the
/// `ClosureKind` values and their evaluation order. A name that has no
/// implementation below is an error, not a silently invented bucket.
fn closure_from_name(name: &str) -> Option<ClosureBucket> {
    match name {
        // Hair is a subsystem closure with its own render class; it is also
        // accepted here so a future schema declaration cannot be mistaken for
        // an unknown string.
        "hair" => Some(ClosureBucket {
            kind: ClosureKind::Hair,
            class: MaterialShadingClass::Hair,
            row_name: "hair",
            priority: 0,
        }),
        "subsurface" => Some(ClosureBucket {
            kind: ClosureKind::Subsurface,
            class: MaterialShadingClass::Subsurface,
            row_name: "subsurface",
            priority: 1,
        }),
        "clearcoat" => Some(ClosureBucket {
            kind: ClosureKind::ClearCoat,
            class: MaterialShadingClass::ClearCoat,
            row_name: "clearcoat",
            priority: 2,
        }),
        "sheen" => Some(ClosureBucket {
            kind: ClosureKind::Sheen,
            class: MaterialShadingClass::Cloth,
            row_name: "cloth",
            priority: 3,
        }),
        _ => None,
    }
}

/// Build the `Lit` closure rows from the schema-generated lobe→closure table.
///
/// The table order is canonical (lobe order), so rows are explicitly sorted by
/// the engine's most-specific-first priority. A closure declared for a lobe
/// this build does not carry is skipped, preserving the previous
/// "no lobe → no bucket" contract; an unknown *name* is always an error.
fn closure_rows(
    closures: &[(&str, &str)],
    lobes: &[LobeLayout],
) -> Result<Vec<ShadingClassRow>, ClassTableError> {
    let mut rows = Vec::new();
    for (lobe_id, closure_name) in closures.iter().copied() {
        let bucket = closure_from_name(closure_name)
            .ok_or_else(|| ClassTableError::UnknownClosureName(closure_name.to_owned()))?;
        if !lobes.iter().any(|lobe| lobe.id == lobe_id) {
            continue;
        }
        rows.push((
            bucket.priority,
            ShadingClassRow {
                id: bucket.class as u32,
                name: bucket.row_name,
                rule: ShadingClassRule::Closure(bucket.kind),
            },
        ));
    }
    rows.sort_by_key(|(priority, _)| *priority);
    Ok(rows.into_iter().map(|(_, row)| row).collect())
}

/// Build the response-mode rows from the schema-generated table and verify the
/// PCM registry mirrors it in both directions.
///
/// The schema owns the declaration and the registry owns the load-time
/// manifest. A missing row, a wrong slot, an unknown illumination spelling, a
/// declared `lit` default, or an unclaimed non-default variant are all
/// pipeline-build errors: any one of them would otherwise silently change the
/// class table.
fn response_mode_rows(
    registry: &prism_material_pcm::ExtensionRegistry,
    modes: &[ResponseModeLayout],
) -> Result<Vec<ShadingClassRow>, ClassTableError> {
    let mut rows = Vec::new();
    let mut seen = Vec::new();
    for mode in modes {
        let manifest = registry
            .manifest_by_uid(mode.uid)
            .ok_or_else(|| ClassTableError::UnregisteredResponseMode(mode.uid.to_owned()))?;
        if manifest.kind() != prism_material_pcm::Kind::ResponseMode {
            return Err(ClassTableError::UnregisteredResponseMode(
                mode.uid.to_owned(),
            ));
        }
        let registry_slot = manifest
            .registry_slot()
            .ok_or_else(|| ClassTableError::UnregisteredResponseMode(mode.uid.to_owned()))?;
        if registry_slot != mode.registry_slot {
            return Err(ClassTableError::ResponseModeSlotMismatch {
                uid: mode.uid.to_owned(),
                schema: mode.registry_slot,
                registry: registry_slot,
            });
        }
        let illumination = Illumination::from_name(mode.illumination).ok_or_else(|| {
            ClassTableError::UnknownResponseModeIllumination(mode.illumination.to_owned())
        })?;
        if illumination == Illumination::Lit {
            return Err(ClassTableError::DefaultResponseModeDeclared(
                mode.uid.to_owned(),
            ));
        }
        seen.push(illumination);
        let (id, name, rule) = match illumination {
            Illumination::Stylized => (
                MaterialShadingClass::Npr as u32,
                "npr",
                ShadingClassRule::Illumination(Illumination::Stylized),
            ),
            Illumination::Unlit => (
                MaterialShadingClass::Unlit as u32,
                "unlit",
                ShadingClassRule::Illumination(Illumination::Unlit),
            ),
            Illumination::Custom => (
                MaterialShadingClass::Custom as u32,
                "custom",
                ShadingClassRule::CustomProgram,
            ),
            Illumination::Lit => unreachable!(),
        };
        rows.push(ShadingClassRow { id, name, rule });
    }
    for illumination in Illumination::ALL {
        if illumination != Illumination::Lit && !seen.contains(&illumination) {
            return Err(ClassTableError::MissingResponseMode(illumination));
        }
    }
    for manifest in registry
        .iter()
        .filter(|manifest| manifest.kind() == prism_material_pcm::Kind::ResponseMode)
    {
        if !modes.iter().any(|mode| mode.uid == manifest.uid()) {
            return Err(ClassTableError::UnhandledResponseMode(
                manifest.uid().to_owned(),
            ));
        }
    }
    Ok(rows)
}

/// The shading-class table (roadmap P2-j).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ShadingClassTable {
    rows: Vec<ShadingClassRow>,
}

impl ShadingClassTable {
    /// Derive the class set from the PCM registry and the lobe catalogue.
    ///
    /// * Render-class buckets (`Water`, `Hair`) come first, because a dedicated
    ///   subsystem owns them regardless of the illumination axis.
    /// * The illumination buckets come from the registry's `Kind::ResponseMode`
    ///   rows, in slot order, so the registration order is the class order.
    /// * The `Lit` closure buckets exist only if the corresponding lobe is
    ///   declared by the schema: a lobe the engine does not carry cannot claim a
    ///   bucket.
    ///
    /// # Errors
    ///
    /// [`ClassTableError::UnhandledResponseMode`] when a registered response mode
    /// has no engine implementation, or [`ClassTableError::UnknownClosureName`]
    /// when a schema-declared closure name has no shading bucket implementation.
    pub fn from_registry(
        registry: &prism_material_pcm::ExtensionRegistry,
        lobes: &[LobeLayout],
    ) -> Result<Self, ClassTableError> {
        let mut rows = Vec::new();
        // Dedicated subsystem buckets, in the engine's fixed priority order.
        rows.push(ShadingClassRow {
            id: MaterialShadingClass::Water as u32,
            name: "water",
            rule: ShadingClassRule::RenderClass(MaterialRenderClass::Water),
        });
        rows.push(ShadingClassRow {
            id: MaterialShadingClass::Hair as u32,
            name: "hair",
            rule: ShadingClassRule::RenderClass(MaterialRenderClass::Hair),
        });

        // Registered response modes. The schema owns the set/order/slot; the
        // shading side resolves the declared illumination name against its own
        // axis vocabulary and verifies that the PCM registry mirrors the row.
        rows.extend(response_mode_rows(registry, &SURFACE_RESPONSE_MODES)?);

        // `Lit` closure buckets, most specific first — the schema-generated
        // lobe→closure table is the authority for which of them exist.
        // Hair is a *subsystem* closure with its own render class, not an
        // über-BSDF lobe: the schema's lobe table never carries it, so this row
        // is unconditional (the render-class row above already binds the same
        // bucket for hair materials).
        rows.push(ShadingClassRow {
            id: MaterialShadingClass::Hair as u32,
            name: "hair",
            rule: ShadingClassRule::Closure(ClosureKind::Hair),
        });
        rows.extend(closure_rows(&SURFACE_LOBE_CLOSURES, lobes)?);
        rows.push(ShadingClassRow {
            id: MaterialShadingClass::Principled as u32,
            name: "principled",
            rule: ShadingClassRule::Default,
        });
        let table = Self { rows };
        table.validate(MAX_SHADING_CLASSES)?;
        Ok(table)
    }

    /// Number of shading buckets: `max(id) + 1`, i.e. the width the work plan's
    /// per-class arrays must cover.
    ///
    /// The ids must tile `0..=max` — the work plan indexes by id, so a gap or a
    /// duplicate would silently fold two buckets together. [`Self::validate`]
    /// checks that, and [`Self::from_registry`] runs it.
    ///
    /// # Panics
    ///
    /// Panics on an empty table (a class set with no fallback row is a
    /// programming error, and `from_registry` always emits one).
    #[must_use]
    pub fn class_count(&self) -> u32 {
        self.rows
            .iter()
            .map(|row| row.id)
            .max()
            .map(|max_id| max_id + 1)
            .expect("a shading-class table always has a default row")
    }

    /// Check the structural invariants the work plan depends on.
    ///
    /// # Errors
    ///
    /// [`ClassTableError::NonDenseClassIds`] when the ids do not tile
    /// `0..=class_count`, [`ClassTableError::CapacityExceeded`] when the class
    /// set does not fit the shader-side capacity (the prefix-sum/scatter arrays
    /// are fixed-size in WGSL, so the *live* count is table-driven but the
    /// capacity is a build-time fact).
    pub fn validate(&self, capacity: usize) -> Result<(), ClassTableError> {
        let count = self.class_count() as usize;
        let mut seen = vec![false; count];
        for row in &self.rows {
            let index = row.id as usize;
            if index >= count {
                return Err(ClassTableError::NonDenseClassIds {
                    ids: self.rows.iter().map(|row| row.id).collect(),
                });
            }
            seen[index] = true;
        }
        if seen.iter().any(|present| !present) {
            return Err(ClassTableError::NonDenseClassIds {
                ids: self.rows.iter().map(|row| row.id).collect(),
            });
        }
        if count > capacity {
            return Err(ClassTableError::CapacityExceeded {
                needed: count,
                capacity,
            });
        }
        Ok(())
    }

    /// The rows, in evaluation order.
    #[must_use]
    pub fn rows(&self) -> &[ShadingClassRow] {
        &self.rows
    }

    /// Number of rows (a lobe/render class may own more than one row).
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the table has no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Classify a header by walking the table, first match wins.
    ///
    /// Mirrors [`classify_material_header`]'s preconditions exactly; the two are
    /// pinned against each other over an exhaustive sweep by the tests.
    ///
    /// # Errors
    ///
    /// [`ClassificationError`] for an inactive material, a non-surface render
    /// class, or an illumination value the axis does not define.
    pub fn classify(&self, header: &GpuMaterialHeader) -> Result<u32, ClassificationError> {
        if header.is_active == 0 {
            return Err(ClassificationError::Inactive);
        }
        if !is_surface_render_class(header.render_class) {
            return Err(ClassificationError::NonSurfaceClass);
        }
        // The illumination axis is parsed lazily: the dedicated render-class rows
        // come first and bind their bucket whatever the illumination says (a
        // stylized water shader is still water work), so an unknown value there
        // must not pre-empt them. Only a row that actually consults the axis can
        // reject it.
        let illumination = Illumination::from_u32(header.illumination);
        for row in &self.rows {
            let matched = match row.rule {
                ShadingClassRule::RenderClass(class) => header.render_class == class as u32,
                ShadingClassRule::Illumination(axis) => {
                    let Some(actual) = illumination else {
                        return Err(ClassificationError::UnsupportedIllumination(
                            header.illumination,
                        ));
                    };
                    actual == axis
                }
                ShadingClassRule::CustomProgram => {
                    let Some(actual) = illumination else {
                        return Err(ClassificationError::UnsupportedIllumination(
                            header.illumination,
                        ));
                    };
                    // A custom response needs a program to run; without one the
                    // row below (the default) claims it.
                    actual == Illumination::Custom && header.custom_program != u32::MAX
                }
                ShadingClassRule::Closure(kind) => {
                    let Some(actual) = illumination else {
                        return Err(ClassificationError::UnsupportedIllumination(
                            header.illumination,
                        ));
                    };
                    actual == Illumination::Lit && has_closure(header.closure_mask, kind)
                }
                ShadingClassRule::Default => {
                    // The fallback claims whatever the rows above did not: a
                    // `Lit` surface with no specialized closure, and a `Custom`
                    // surface with no bound program (the safe degradation).
                    let Some(actual) = illumination else {
                        return Err(ClassificationError::UnsupportedIllumination(
                            header.illumination,
                        ));
                    };
                    matches!(actual, Illumination::Lit | Illumination::Custom)
                }
            };
            if matched {
                return Ok(row.id);
            }
        }
        Err(ClassificationError::NonSurfaceClass)
    }
}

/// Whether a render class is one of the surface-domain blend families.
fn is_surface_render_class(render_class: u32) -> bool {
    matches!(
        render_class,
        x if x == MaterialRenderClass::Opaque as u32
            || x == MaterialRenderClass::OpaqueTwoSided as u32
            || x == MaterialRenderClass::Masked as u32
            || x == MaterialRenderClass::MaskedTwoSided as u32
            || x == MaterialRenderClass::Hair as u32
            || x == MaterialRenderClass::Water as u32
    )
}

/// Project a `Lit` surface's closure graph onto a coherent shading bucket. The
/// dominant physically-based closure wins; a plain metal/dielectric surface is
/// `Principled`. This mirrors the old per-model buckets without forcing the
/// material author to pick one exclusive model.
fn lit_class_from_closures(closure_mask: u32) -> MaterialShadingClass {
    if has_closure(closure_mask, ClosureKind::Hair) {
        MaterialShadingClass::Hair
    } else if has_closure(closure_mask, ClosureKind::Subsurface) {
        MaterialShadingClass::Subsurface
    } else if has_closure(closure_mask, ClosureKind::ClearCoat) {
        MaterialShadingClass::ClearCoat
    } else if has_closure(closure_mask, ClosureKind::Sheen) {
        MaterialShadingClass::Cloth
    } else {
        MaterialShadingClass::Principled
    }
}

pub fn classify_material_header(
    header: &GpuMaterialHeader,
) -> Result<MaterialShadingClass, ClassificationError> {
    classify_material_header_with_region(header, u32::MAX)
}

/// Classify a surface with an optional per-region illumination selector.
///
/// `region_illumination` is the visibility raster's mask selector:
/// `0` = `Lit`, `1` = `Stylized`, any other value (including `u32::MAX`) means
/// "use the material header". The selector only overrides the two physically
/// meaningful style axes; `Unlit` and `Custom` remain material-level modes and
/// ignore the region mask, because they are not PBR/NPR blend endpoints.
pub fn classify_material_header_with_region(
    header: &GpuMaterialHeader,
    region_illumination: u32,
) -> Result<MaterialShadingClass, ClassificationError> {
    if header.is_active == 0 {
        return Err(ClassificationError::Inactive);
    }
    // Surface-domain blend families only. `Npr*`/`Custom*` no longer exist as
    // render classes; style lives on the illumination axis instead.
    if !matches!(
        header.render_class,
        x if x == MaterialRenderClass::Opaque as u32
            || x == MaterialRenderClass::OpaqueTwoSided as u32
            || x == MaterialRenderClass::Masked as u32
            || x == MaterialRenderClass::MaskedTwoSided as u32
            || x == MaterialRenderClass::Hair as u32
            || x == MaterialRenderClass::Water as u32
    ) {
        return Err(ClassificationError::NonSurfaceClass);
    }

    // Dedicated subsystem render classes bind their bucket directly, regardless
    // of the illumination axis (a stylized water shader is still water work).
    if header.render_class == MaterialRenderClass::Water as u32 {
        return Ok(MaterialShadingClass::Water);
    }
    if header.render_class == MaterialRenderClass::Hair as u32 {
        return Ok(MaterialShadingClass::Hair);
    }

    let illumination = match header.illumination {
        0 | 1 => match region_illumination {
            0 | 1 => region_illumination,
            _ => header.illumination,
        },
        _ => header.illumination,
    };
    match Illumination::from_u32(illumination) {
        Some(Illumination::Unlit) => Ok(MaterialShadingClass::Unlit),
        Some(Illumination::Stylized) => Ok(MaterialShadingClass::Npr),
        // §3.5.4: a custom response needs a program. With no handle the material
        // degrades to the shared über path (the same integrator the `Custom`
        // bucket would run), so it classifies as `Principled`.
        Some(Illumination::Custom) if header.custom_program != u32::MAX => {
            Ok(MaterialShadingClass::Custom)
        }
        Some(Illumination::Custom) => Ok(MaterialShadingClass::Principled),
        Some(Illumination::Lit) => Ok(lit_class_from_closures(header.closure_mask)),
        None => Err(ClassificationError::UnsupportedIllumination(illumination)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::abi::GenerationalHandle;
    use prism_render_material::fallback_material_header;

    /// Set the illumination axis on a header, mirroring how the material
    /// registry emits it.
    fn with_illumination(mut header: GpuMaterialHeader, illum: Illumination) -> GpuMaterialHeader {
        header.illumination = illum as u32;
        header
    }

    /// Set a single closure bit (the "dominant" physically-based closure).
    fn with_closure(mut header: GpuMaterialHeader, kind: ClosureKind) -> GpuMaterialHeader {
        header.closure_mask = 1 << kind as u32;
        header
    }

    #[test]
    fn illumination_axis_maps_to_style_buckets_without_per_asset_permutations() {
        let base = fallback_material_header(1);
        assert_eq!(
            classify_material_header(&with_illumination(base, Illumination::Unlit)),
            Ok(MaterialShadingClass::Unlit)
        );
        assert_eq!(
            classify_material_header(&with_illumination(base, Illumination::Stylized)),
            Ok(MaterialShadingClass::Npr)
        );
        // §3.5.4: the custom bucket needs a program to run. With a handle it is
        // its own bucket; without one the material degrades to the shared über
        // path (the default row) instead of occupying a bucket that would run
        // the same integrator under a different name.
        let unbound = with_illumination(base, Illumination::Custom);
        assert_eq!(
            classify_material_header(&unbound),
            Ok(MaterialShadingClass::Principled)
        );
        let bound = GpuMaterialHeader {
            custom_program: 7,
            ..unbound
        };
        assert_eq!(
            classify_material_header(&bound),
            Ok(MaterialShadingClass::Custom)
        );
    }

    #[test]
    fn region_illumination_selects_lit_or_stylized_per_pixel() {
        let base = fallback_material_header(1);
        let lit = with_illumination(base, Illumination::Lit);
        assert_eq!(
            classify_material_header_with_region(&lit, 0),
            Ok(MaterialShadingClass::Principled)
        );
        assert_eq!(
            classify_material_header_with_region(&lit, 1),
            Ok(MaterialShadingClass::Npr)
        );
        assert_eq!(
            classify_material_header_with_region(&lit, u32::MAX),
            Ok(MaterialShadingClass::Principled),
            "the sentinel keeps the material-level illumination"
        );
        for invalid_selector in [2, 7, u32::MAX - 1] {
            assert_eq!(
                classify_material_header_with_region(&lit, invalid_selector),
                Ok(MaterialShadingClass::Principled),
                "out-of-contract visibility selectors must fall back to material illumination"
            );
        }

        let stylized = with_illumination(base, Illumination::Stylized);
        assert_eq!(
            classify_material_header_with_region(&stylized, 0),
            Ok(MaterialShadingClass::Principled)
        );
        assert_eq!(
            classify_material_header_with_region(&stylized, 1),
            Ok(MaterialShadingClass::Npr)
        );

        // The mask blends only the PBR/NPR endpoints; Unlit and Custom remain
        // material-level response modes and ignore the region selector.
        let unlit = with_illumination(base, Illumination::Unlit);
        assert_eq!(
            classify_material_header_with_region(&unlit, 1),
            Ok(MaterialShadingClass::Unlit)
        );
        let custom = GpuMaterialHeader {
            custom_program: 7,
            ..with_illumination(base, Illumination::Custom)
        };
        assert_eq!(
            classify_material_header_with_region(&custom, 0),
            Ok(MaterialShadingClass::Custom)
        );
    }

    #[test]
    fn lit_surfaces_derive_their_bucket_from_the_dominant_closure() {
        let base = with_illumination(fallback_material_header(1), Illumination::Lit);
        for (kind, expected) in [
            (ClosureKind::Diffuse, MaterialShadingClass::Principled),
            (ClosureKind::Conductor, MaterialShadingClass::Principled),
            (ClosureKind::Subsurface, MaterialShadingClass::Subsurface),
            (ClosureKind::ClearCoat, MaterialShadingClass::ClearCoat),
            (ClosureKind::Sheen, MaterialShadingClass::Cloth),
            (ClosureKind::Hair, MaterialShadingClass::Hair),
        ] {
            assert_eq!(
                classify_material_header(&with_closure(base, kind)),
                Ok(expected),
                "closure {kind:?} should classify as {expected:?}"
            );
        }
    }

    #[test]
    fn hair_and_water_render_classes_bind_their_bucket_directly() {
        for (render_class, expected) in [
            (MaterialRenderClass::Hair, MaterialShadingClass::Hair),
            (MaterialRenderClass::Water, MaterialShadingClass::Water),
        ] {
            let mut header = fallback_material_header(1);
            header.render_class = render_class as u32;
            assert_eq!(classify_material_header(&header), Ok(expected));
        }
    }

    #[test]
    fn work_plan_is_class_contiguous_and_generation_safe() {
        let mut pbr = fallback_material_header(1);
        pbr.generation = 2;
        let mut npr = fallback_material_header(1);
        npr.generation = 4;
        npr.illumination = Illumination::Stylized as u32;
        let pixel = |material: GenerationalHandle| {
            super::super::VisibilityPixel::new(
                GenerationalHandle {
                    index: 1,
                    generation: 1,
                },
                0,
                0,
                material,
                [1.0, 0.0, 0.0],
                1,
            )
            .unwrap()
        };
        let plan = ShadingWorkPlan::build(
            [
                (
                    9,
                    pixel(GenerationalHandle {
                        index: 1,
                        generation: 4,
                    }),
                ),
                (
                    3,
                    pixel(GenerationalHandle {
                        index: 0,
                        generation: 2,
                    }),
                ),
                (
                    7,
                    pixel(GenerationalHandle {
                        index: 1,
                        generation: 3,
                    }),
                ),
            ],
            &[pbr, npr],
            8,
        );

        assert_eq!(plan.counts[MaterialShadingClass::Principled.index()], 1);
        assert_eq!(plan.counts[MaterialShadingClass::Npr.index()], 1);
        assert_eq!(plan.work.len(), 2);
        assert_eq!(plan.work[0].pixel_index, 3);
        assert_eq!(plan.work[1].pixel_index, 9);
    }

    #[test]
    fn cpu_work_plan_honors_per_region_illumination() {
        let mut lit = fallback_material_header(1);
        lit.generation = 3;
        lit.illumination = Illumination::Lit as u32;
        let pixel = || {
            super::super::VisibilityPixel::new(
                GenerationalHandle {
                    index: 0,
                    generation: 1,
                },
                0,
                0,
                GenerationalHandle {
                    index: 0,
                    generation: 3,
                },
                [1.0, 0.0, 0.0],
                1,
            )
            .unwrap()
        };
        let plan =
            ShadingWorkPlan::build_with_region([(4, pixel(), 0), (9, pixel(), 1)], &[lit], 8);
        assert_eq!(plan.counts[MaterialShadingClass::Principled.index()], 1);
        assert_eq!(plan.counts[MaterialShadingClass::Npr.index()], 1);
        assert_eq!(plan.work[0].pixel_index, 4);
        assert_eq!(plan.work[1].pixel_index, 9);
    }

    #[test]
    fn the_class_table_is_derived_from_the_registry_and_the_lobe_catalogue() {
        let registry = prism_material_pcm::builtin::builtin_registry();
        let table =
            ShadingClassTable::from_registry(&registry, &prism_render_material::SURFACE_LOBES)
                .expect("the built-in catalogue is fully implemented");

        // One row per bucket, plus the second `hair` row (the render class and
        // the closure both own the same bucket).
        assert_eq!(table.len(), 10);
        let mut ids: Vec<u32> = table.rows().iter().map(|row| row.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), MAX_SHADING_CLASSES);

        // Priority is row order: the dedicated render classes first, then the
        // registered response modes in registration order, then the `Lit`
        // closure buckets most-specific-first, and the default last.
        let names: Vec<&str> = table.rows().iter().map(|row| row.name).collect();
        assert_eq!(
            names,
            vec![
                "water",
                "hair",
                // The response-mode rows follow the registry's slot order.
                "npr",
                "unlit",
                "custom",
                "hair",
                "subsurface",
                "clearcoat",
                "cloth",
                "principled",
            ]
        );
        assert_eq!(table.rows().last().unwrap().rule, ShadingClassRule::Default);
    }

    #[test]
    fn the_wesl_class_constants_match_the_table_row_for_row() {
        // The table is the registration-driven source of truth and the shader's
        // `CLASS_*` constants are the same ids in the same spelling. This gate
        // reads the shipping WESL, so a class added on either side without the
        // other is a build failure rather than a silent work-plan mismatch.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../prism_render_scene/src/shaders/material_classification.wesl");
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        let mut shader: std::collections::BTreeMap<String, u32> = std::collections::BTreeMap::new();
        for line in source.lines() {
            let line = line.trim();
            let Some(rest) = line.strip_prefix("const CLASS_") else {
                continue;
            };
            let (name, value) = rest
                .split_once(": u32 = ")
                .unwrap_or_else(|| panic!("unparsable class constant: {line}"));
            // WESL spells unsigned literals with a trailing `u` (`0u`).
            let value: u32 = value
                .trim_end_matches(';')
                .trim_end_matches('u')
                .parse()
                .unwrap_or_else(|_| panic!("unparsable class id: {line}"));
            shader.insert(name.to_ascii_lowercase(), value);
        }
        assert!(!shader.is_empty(), "parsed no class constants");

        let registry = prism_material_pcm::builtin::builtin_registry();
        let table =
            ShadingClassTable::from_registry(&registry, &prism_render_material::SURFACE_LOBES)
                .unwrap();
        for row in table.rows() {
            let id = shader.get(row.name).unwrap_or_else(|| {
                panic!(
                    "material_classification.wesl has no CLASS_{}",
                    row.name.to_ascii_uppercase()
                )
            });
            assert_eq!(*id, row.id, "class `{}`", row.name);
        }
        // …and the shader must not carry a class the table has never heard of.
        let names: std::collections::BTreeSet<&str> =
            table.rows().iter().map(|row| row.name).collect();
        for name in shader.keys() {
            assert!(
                names.contains(name.as_str()),
                "the shader declares CLASS_{} with no table row",
                name.to_ascii_uppercase()
            );
        }
    }

    #[test]
    fn the_class_count_is_derived_and_the_ids_tile_it() {
        let registry = prism_material_pcm::builtin::builtin_registry();
        let table =
            ShadingClassTable::from_registry(&registry, &prism_render_material::SURFACE_LOBES)
                .unwrap();
        // P2-j: the *live* class count comes from the table, not from a magic
        // constant — and today it happens to fill the shader-side capacity.
        assert_eq!(table.class_count(), MAX_SHADING_CLASSES as u32);
        table.validate(MAX_SHADING_CLASSES).unwrap();
        // The ids are dense, so the work plan can index by id.
        let mut ids: Vec<u32> = table.rows().iter().map(|row| row.id).collect();
        ids.sort_unstable();
        ids.dedup();
        let dense: Vec<u32> = (0..table.class_count()).collect();
        assert_eq!(ids, dense);
        // One row short of the capacity is a build error, not a silent
        // overwrite: the prefix-sum/scatter arrays are WGSL-fixed.
        assert_eq!(
            table.validate(MAX_SHADING_CLASSES - 1),
            Err(ClassTableError::CapacityExceeded {
                needed: MAX_SHADING_CLASSES,
                capacity: MAX_SHADING_CLASSES - 1,
            })
        );
    }

    #[test]
    fn a_registered_response_mode_without_an_implementation_fails_loudly() {
        // Registration is a declaration; the engine still has to implement the
        // response. A catalogue entry with no shading behind it must be a
        // build-time error, not a material that silently classifies as something
        // else.
        let mut registry = prism_material_pcm::builtin::builtin_registry();
        registry
            .register(prism_material_pcm::ExtensionManifest::new(
                "org.prism.mode.watercolor",
                27,
                prism_material_pcm::Namespace::Prism,
                prism_material_pcm::Kind::ResponseMode,
            ))
            .expect("the Prism region has room");
        let error =
            ShadingClassTable::from_registry(&registry, &prism_render_material::SURFACE_LOBES)
                .expect_err("an unhandled response mode must be rejected");
        assert_eq!(
            error,
            ClassTableError::UnhandledResponseMode(String::from("org.prism.mode.watercolor"))
        );
    }

    #[test]
    fn a_schema_closure_without_a_shading_implementation_fails_loudly() {
        // The table is generated from the schema, but the shading side still
        // owns the closure implementations. A name the codegen can carry but
        // the engine cannot resolve must fail pipeline build, never disappear
        // into the default bucket.
        let error = closure_rows(
            &[("subsurface", "not_a_shading_closure")],
            &prism_render_material::SURFACE_LOBES,
        )
        .expect_err("an unknown closure name must be rejected");
        assert_eq!(
            error,
            ClassTableError::UnknownClosureName(String::from("not_a_shading_closure"))
        );
    }

    fn response_mode(
        uid: &'static str,
        registry_slot: u32,
        illumination: &'static str,
    ) -> ResponseModeLayout {
        ResponseModeLayout {
            id: "test",
            uid,
            registry_slot,
            namespace: "PRISM",
            vendor: "prism",
            stability: "candidate",
            illumination,
            requires_caps: &[],
            view_dependent: true,
        }
    }

    #[test]
    fn a_schema_response_mode_with_an_unknown_illumination_fails_loudly() {
        let registry = prism_material_pcm::builtin::builtin_registry();
        let error = response_mode_rows(
            &registry,
            &[response_mode(
                prism_material_pcm::builtin::UID_MODE_STYLIZED,
                23,
                "not_an_illumination",
            )],
        )
        .expect_err("an unknown illumination spelling must be rejected");
        assert_eq!(
            error,
            ClassTableError::UnknownResponseModeIllumination(String::from("not_an_illumination"))
        );
    }

    #[test]
    fn a_schema_response_mode_without_a_registry_row_fails_loudly() {
        let registry = prism_material_pcm::builtin::builtin_registry();
        let error = response_mode_rows(
            &registry,
            &[response_mode(
                "org.prism.response_mode.watercolor",
                23,
                "stylized",
            )],
        )
        .expect_err("a schema row without a registry row must be rejected");
        assert_eq!(
            error,
            ClassTableError::UnregisteredResponseMode(String::from(
                "org.prism.response_mode.watercolor"
            ))
        );
    }

    #[test]
    fn a_schema_response_mode_slot_mismatch_fails_loudly() {
        let registry = prism_material_pcm::builtin::builtin_registry();
        let error = response_mode_rows(
            &registry,
            &[response_mode(
                prism_material_pcm::builtin::UID_MODE_STYLIZED,
                99,
                "stylized",
            )],
        )
        .expect_err("a schema/registry slot mismatch must be rejected");
        assert_eq!(
            error,
            ClassTableError::ResponseModeSlotMismatch {
                uid: String::from(prism_material_pcm::builtin::UID_MODE_STYLIZED),
                schema: 99,
                registry: 23,
            }
        );
    }

    #[test]
    fn a_missing_response_mode_variant_fails_loudly() {
        let registry = prism_material_pcm::builtin::builtin_registry();
        let error = response_mode_rows(&registry, &[])
            .expect_err("every non-default illumination variant must have a response mode");
        assert_eq!(
            error,
            ClassTableError::MissingResponseMode(Illumination::Stylized)
        );
    }

    #[test]
    fn the_table_walk_and_the_hot_path_classifier_agree_exhaustively() {
        // Roadmap P2-j: the table is the registration-driven source of truth and
        // `classify_material_header` is its allocation-free specialization. The
        // only way to keep that claim honest is to sweep every combination the
        // hot path can see.
        let registry = prism_material_pcm::builtin::builtin_registry();
        let table =
            ShadingClassTable::from_registry(&registry, &prism_render_material::SURFACE_LOBES)
                .unwrap();
        let mut base = fallback_material_header(1);
        base.is_active = 1;
        for active in [0u32, 1] {
            for render_class in [
                MaterialRenderClass::Opaque as u32,
                MaterialRenderClass::OpaqueTwoSided as u32,
                MaterialRenderClass::Masked as u32,
                MaterialRenderClass::MaskedTwoSided as u32,
                MaterialRenderClass::Hair as u32,
                MaterialRenderClass::Water as u32,
                // A non-surface class must be rejected by both.
                MaterialRenderClass::Decal as u32,
            ] {
                for illumination in [0u32, 1, 2, 3, 4, u32::MAX] {
                    for custom_program in [0u32, 7, u32::MAX] {
                        let mut closure_masks = vec![0u32, u32::MAX];
                        for kind in [
                            ClosureKind::Hair,
                            ClosureKind::Subsurface,
                            ClosureKind::ClearCoat,
                            ClosureKind::Sheen,
                        ] {
                            closure_masks.push(1 << kind as u32);
                        }
                        // Every pair of closures, so priority is exercised.
                        for a in 0..closure_masks.len() {
                            for b in 0..closure_masks.len() {
                                closure_masks.push(closure_masks[a] | closure_masks[b]);
                            }
                        }
                        for closure_mask in closure_masks {
                            let mut header = base;
                            header.is_active = active;
                            header.render_class = render_class;
                            header.illumination = illumination;
                            header.custom_program = custom_program;
                            header.closure_mask = closure_mask;
                            let hot = classify_material_header(&header)
                                .map(|class| class as u32)
                                .map_err(|error| error);
                            let walked = table.classify(&header);
                            assert_eq!(hot, walked, "{header:?}");
                        }
                    }
                }
            }
        }
    }
}
