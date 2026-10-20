//! CPU authority for the renderer's light scene.
//!
//! This module deliberately does not depend on ECS, Bevy, or a GPU backend.
//! Adapters publish [`SceneDelta`] values here; GPU scene builders can then
//! consume the same ordered stream without owning a second light table.
//!
//! [`LightHandle`](super::LightHandle) is an older index-only contract used by
//! existing CPU algorithms and remains unchanged. [`LightSceneHandle`] is the
//! generation-safe handle for retained scene state.
//!
//! Handles carry a scene-lineage identity and a globally unique light
//! incarnation in addition to slot/generation. Checkpoints and deltas preserve
//! lineage for replicas; independently created scenes cannot alias. CPU slot
//! reuse is immediate after removal; GPU consumers must still fence resources.

use alloc::{collections::BTreeSet, vec::Vec};
use core::{
    fmt,
    ops::{BitOr, BitOrAssign},
    sync::atomic::{AtomicU64, Ordering},
};

const CHECKPOINT_VERSION: u32 = 2;
const INITIAL_FIELD_EPOCH: u64 = 1;
const KNOWN_FIELD_BITS: u32 = 0x3f;
const PATCH_FIELD_BITS: u32 = 0x1f;
static NEXT_SCENE_IDENTITY: AtomicU64 = AtomicU64::new(1);
static NEXT_LIGHT_IDENTITY: AtomicU64 = AtomicU64::new(1);

fn allocate_identity(counter: &AtomicU64) -> Result<u64, SceneError> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| SceneError::IdentityExhausted)
}

/// A generation-safe handle into [`LightScene`].
///
/// Slot indices may be recycled, but a recycled slot always receives a new
/// generation. A handle from an older occupant therefore cannot address the
/// replacement light.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LightSceneHandle {
    scene: u64,
    slot: u32,
    generation: u32,
    identity: u64,
}

impl LightSceneHandle {
    /// Creates an unscoped raw handle. It is useful for serialization adapters,
    /// but cannot address a [`LightScene`] until obtained from that scene.
    #[must_use]
    pub const fn new(slot: u32, generation: u32) -> Self {
        Self {
            scene: 0,
            slot,
            generation,
            identity: 0,
        }
    }

    const fn in_scene(scene: u64, slot: u32, generation: u32, identity: u64) -> Self {
        Self {
            scene,
            slot,
            generation,
            identity,
        }
    }

    /// Returns the owning scene lineage identity, or zero for an unscoped raw handle.
    #[must_use]
    pub const fn scene_identity(self) -> u64 {
        self.scene
    }

    /// Returns the unique light incarnation identity.
    #[must_use]
    pub const fn identity(self) -> u64 {
        self.identity
    }

    /// Returns the slot index.
    #[must_use]
    pub const fn slot(self) -> u32 {
        self.slot
    }

    /// Returns the slot generation.
    #[must_use]
    pub const fn generation(self) -> u32 {
        self.generation
    }

    /// Returns whether this handle has a usable, non-zero generation.
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.scene != 0 && self.generation != 0 && self.identity != 0
    }
}

/// Descriptive alias for callers that want to emphasize generation safety.
pub type GenerationSafeLightHandle = LightSceneHandle;

/// Classification of a light emitter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LightClass {
    Directional,
    Point,
    Spot,
    Rect,
    Disk,
    Capsule,
    Sky,
    Emissive,
}

/// Authoring/runtime mobility class.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LightMobility {
    Static,
    Stationary,
    Dynamic,
}

/// Measurement unit of the emission value.
///
/// The value is kept explicit rather than silently treating every light as a
/// unitless RGB multiplier. Importers decide how their source format maps to
/// these units; the scene contract preserves the result in linear space. No
/// photometric/radiometric conversion is inferred from RGB alone.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LightUnit {
    /// Illuminance in lm/m², for directional emitters.
    Lux,
    /// Luminous intensity in lm/sr, for punctual emitters.
    Candela,
    /// Total luminous flux; a consumer needs the emitter's angular distribution.
    Lumen,
    /// Radiance in W/(m² sr), for extended, sky, and mesh emitters.
    Radiance,
    /// Luminance in cd/m², for extended, sky, and mesh emitters.
    Nits,
    /// Explicit non-physical authoring input, never implicitly converted to SI.
    Unitless,
}

impl LightUnit {
    /// Whether this unit is meaningful for the emitter class. Conversion to a
    /// backend's canonical unit remains the consumer's responsibility.
    #[must_use]
    pub const fn supports(self, class: LightClass) -> bool {
        match (self, class) {
            (Self::Unitless, _) => true,
            (Self::Lux, LightClass::Directional) => true,
            (Self::Candela | Self::Lumen, LightClass::Point | LightClass::Spot) => true,
            (
                Self::Lumen,
                LightClass::Rect | LightClass::Disk | LightClass::Capsule | LightClass::Emissive,
            ) => true,
            (
                Self::Radiance | Self::Nits,
                LightClass::Rect
                | LightClass::Disk
                | LightClass::Capsule
                | LightClass::Sky
                | LightClass::Emissive,
            ) => true,
            _ => false,
        }
    }
}

/// Classification fields changed together by one patch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LightClassification {
    pub class: LightClass,
    pub mobility: LightMobility,
}

/// Linear emission input with an explicit measurement unit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightEmission {
    /// Nonnegative scene-linear RGB tint in the renderer's working color space.
    /// Importers resolve their source color space before publishing this value.
    pub color: [f32; 3],
    pub value: f32,
    pub unit: LightUnit,
    /// Provenance only: `color` already includes any temperature conversion.
    /// Consumers must not apply the temperature a second time.
    pub temperature_kelvin: Option<f32>,
}

impl LightEmission {
    /// Creates a constant-temperature-independent emission value.
    #[must_use]
    pub const fn new(color: [f32; 3], value: f32, unit: LightUnit) -> Self {
        Self {
            color,
            value,
            unit,
            temperature_kelvin: None,
        }
    }

    /// Adds an optional color-temperature authoring value.
    #[must_use]
    pub const fn with_temperature(mut self, temperature_kelvin: f32) -> Self {
        self.temperature_kelvin = Some(temperature_kelvin);
        self
    }
}

impl Default for LightEmission {
    fn default() -> Self {
        Self::new([1.0; 3], 1.0, LightUnit::Candela)
    }
}

/// Geometry and transform data for each supported emitter class.
///
/// Positions, lengths, and ranges are in meters in a common scene coordinate
/// system. Directions and rotations must be normalized; spot angles are cone
/// half-angles in radians. A point/spot range of zero means unbounded influence.
/// Directional, spot, rect, and disk vectors point along light propagation
/// from the emitter into the scene; a capsule may degenerate to a sphere
/// (`a == b`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LightGeometry {
    Directional {
        direction: [f32; 3],
    },
    Point {
        position: [f32; 3],
        radius: f32,
        range: f32,
    },
    Spot {
        position: [f32; 3],
        direction: [f32; 3],
        inner_angle: f32,
        outer_angle: f32,
        radius: f32,
        range: f32,
    },
    Rect {
        position: [f32; 3],
        direction: [f32; 3],
        right: [f32; 3],
        up: [f32; 3],
        width: f32,
        height: f32,
    },
    Disk {
        position: [f32; 3],
        direction: [f32; 3],
        radius: f32,
    },
    Capsule {
        a: [f32; 3],
        b: [f32; 3],
        radius: f32,
    },
    Sky {
        rotation: [f32; 4],
    },
    Emissive {
        emitter_id: u64,
    },
}

impl LightGeometry {
    /// Returns the emitter class represented by this geometry.
    #[must_use]
    pub const fn class(self) -> LightClass {
        match self {
            Self::Directional { .. } => LightClass::Directional,
            Self::Point { .. } => LightClass::Point,
            Self::Spot { .. } => LightClass::Spot,
            Self::Rect { .. } => LightClass::Rect,
            Self::Disk { .. } => LightClass::Disk,
            Self::Capsule { .. } => LightClass::Capsule,
            Self::Sky { .. } => LightClass::Sky,
            Self::Emissive { .. } => LightClass::Emissive,
        }
    }
}

impl Default for LightGeometry {
    fn default() -> Self {
        Self::Point {
            position: [0.0; 3],
            radius: 0.0,
            range: 0.0,
        }
    }
}

/// Visibility controls that are independent from emitter geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LightVisibility {
    pub enabled: bool,
    pub layer_mask: u32,
}

impl Default for LightVisibility {
    fn default() -> Self {
        Self {
            enabled: true,
            layer_mask: u32::MAX,
        }
    }
}

/// Optional renderer behavior flags.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LightFlags(u32);

impl LightFlags {
    pub const CAST_SHADOW: Self = Self(1 << 0);
    pub const CONTRIBUTE_GI: Self = Self(1 << 1);
    pub const VISIBLE_TO_REFLECTIONS: Self = Self(1 << 2);
    pub const DEBUG_ONLY: Self = Self(1 << 3);

    /// Creates flags from raw bits.
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Returns raw flag bits.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Tests whether all requested flags are present.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl BitOr for LightFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for LightFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Dirty field groups carried by a scene update.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct LightFieldMask(u32);

impl LightFieldMask {
    pub const CLASSIFICATION: Self = Self(1 << 0);
    pub const GEOMETRY: Self = Self(1 << 1);
    pub const EMISSION: Self = Self(1 << 2);
    pub const VISIBILITY: Self = Self(1 << 3);
    pub const FLAGS: Self = Self(1 << 4);
    pub const LIFETIME: Self = Self(1 << 5);
    pub const PAYLOAD: Self = Self(PATCH_FIELD_BITS);
    pub const CREATE: Self = Self(PATCH_FIELD_BITS | Self::LIFETIME.0);
    pub const REMOVE: Self = Self(Self::LIFETIME.0);
    pub const ALL: Self = Self(Self::CREATE.0);

    /// Creates a mask from raw bits, rejecting unknown bits.
    pub const fn from_bits(bits: u32) -> Option<Self> {
        if bits & !KNOWN_FIELD_BITS == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }

    /// Returns raw mask bits.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Tests whether all requested fields are set.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Tests whether no field is dirty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Const-compatible union for patch builders and static masks.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl BitOr for LightFieldMask {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for LightFieldMask {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Monotonic epochs for independently invalidatable light field groups.
///
/// A field increments once for each explicit patch that names it (including a
/// same-value write). Lifetime invalidation uses the handle's generation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LightFieldEpochs {
    pub classification: u64,
    pub geometry: u64,
    pub emission: u64,
    pub visibility: u64,
    pub flags: u64,
}

impl LightFieldEpochs {
    /// Epochs assigned to a newly created light.
    pub const INITIAL: Self = Self {
        classification: INITIAL_FIELD_EPOCH,
        geometry: INITIAL_FIELD_EPOCH,
        emission: INITIAL_FIELD_EPOCH,
        visibility: INITIAL_FIELD_EPOCH,
        flags: INITIAL_FIELD_EPOCH,
    };

    /// Returns the next epochs or an overflow error.
    pub fn advanced(self, mask: LightFieldMask) -> Result<Self, SceneError> {
        Ok(Self {
            classification: bump(
                self.classification,
                mask.contains(LightFieldMask::CLASSIFICATION),
            )?,
            geometry: bump(self.geometry, mask.contains(LightFieldMask::GEOMETRY))?,
            emission: bump(self.emission, mask.contains(LightFieldMask::EMISSION))?,
            visibility: bump(self.visibility, mask.contains(LightFieldMask::VISIBILITY))?,
            flags: bump(self.flags, mask.contains(LightFieldMask::FLAGS))?,
        })
    }

    /// Returns the epoch associated with one payload field group.
    #[must_use]
    pub const fn for_field(self, field: LightFieldMask) -> Option<u64> {
        match field.bits() {
            x if x == LightFieldMask::CLASSIFICATION.bits() => Some(self.classification),
            x if x == LightFieldMask::GEOMETRY.bits() => Some(self.geometry),
            x if x == LightFieldMask::EMISSION.bits() => Some(self.emission),
            x if x == LightFieldMask::VISIBILITY.bits() => Some(self.visibility),
            x if x == LightFieldMask::FLAGS.bits() => Some(self.flags),
            _ => None,
        }
    }
}

fn bump(value: u64, enabled: bool) -> Result<u64, SceneError> {
    if enabled {
        value.checked_add(1).ok_or(SceneError::EpochExhausted)
    } else {
        Ok(value)
    }
}

/// Complete CPU light record stored by [`LightScene`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightRecord {
    pub classification: LightClassification,
    pub emission: LightEmission,
    pub geometry: LightGeometry,
    pub visibility: LightVisibility,
    pub flags: LightFlags,
    pub epochs: LightFieldEpochs,
}

impl LightRecord {
    /// Constructs a record with initial field epochs.
    #[must_use]
    pub const fn new(
        classification: LightClassification,
        emission: LightEmission,
        geometry: LightGeometry,
    ) -> Self {
        Self {
            classification,
            emission,
            geometry,
            visibility: LightVisibility {
                enabled: true,
                layer_mask: u32::MAX,
            },
            flags: LightFlags(0),
            epochs: LightFieldEpochs::INITIAL,
        }
    }

    /// Returns the emission unit without requiring callers to unpack the
    /// complete record.
    #[must_use]
    pub const fn unit(self) -> LightUnit {
        self.emission.unit
    }

    /// Validates geometry, classification, units, emission and nonzero epochs.
    pub fn validate(self) -> Result<(), SceneError> {
        if self.geometry.class() != self.classification.class {
            return Err(SceneError::ClassificationGeometryMismatch {
                classification: self.classification.class,
                geometry: self.geometry.class(),
            });
        }
        validate_emission(self.emission)?;
        if !self.emission.unit.supports(self.classification.class) {
            return Err(SceneError::IncompatibleUnit {
                class: self.classification.class,
                unit: self.emission.unit,
            });
        }
        validate_geometry(self.geometry)?;
        if [
            self.epochs.classification,
            self.epochs.geometry,
            self.epochs.emission,
            self.epochs.visibility,
            self.epochs.flags,
        ]
        .contains(&0)
        {
            return Err(SceneError::InvalidRecordEpochs);
        }
        Ok(())
    }
}

impl Default for LightRecord {
    fn default() -> Self {
        Self::new(
            LightClassification {
                class: LightClass::Point,
                mobility: LightMobility::Dynamic,
            },
            LightEmission::default(),
            LightGeometry::default(),
        )
    }
}

fn validate_emission(emission: LightEmission) -> Result<(), SceneError> {
    if !emission.value.is_finite() || emission.value < 0.0 {
        return Err(SceneError::InvalidEmission);
    }
    if emission
        .color
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(SceneError::InvalidEmission);
    }
    if let Some(temperature) = emission.temperature_kelvin
        && (!temperature.is_finite() || temperature <= 0.0)
    {
        return Err(SceneError::InvalidEmission);
    }
    Ok(())
}

fn validate_geometry(geometry: LightGeometry) -> Result<(), SceneError> {
    match geometry {
        LightGeometry::Directional { direction } => {
            validate_direction(direction)?;
        }
        LightGeometry::Point {
            position,
            radius,
            range,
        } => {
            validate_vector(position)?;
            validate_nonnegative(radius)?;
            validate_nonnegative(range)?;
        }
        LightGeometry::Spot {
            position,
            direction,
            inner_angle,
            outer_angle,
            radius,
            range,
        } => {
            validate_vector(position)?;
            validate_direction(direction)?;
            if !inner_angle.is_finite()
                || !outer_angle.is_finite()
                || inner_angle < 0.0
                || outer_angle <= 0.0
                || outer_angle > core::f32::consts::FRAC_PI_2
                || inner_angle > outer_angle
            {
                return Err(SceneError::InvalidGeometry);
            }
            validate_nonnegative(radius)?;
            validate_nonnegative(range)?;
        }
        LightGeometry::Rect {
            position,
            direction,
            right,
            up,
            width,
            height,
        } => {
            validate_vector(position)?;
            validate_direction(direction)?;
            validate_direction(right)?;
            validate_direction(up)?;
            if dot(direction, right).abs() > 1e-4
                || dot(direction, up).abs() > 1e-4
                || dot(right, up).abs() > 1e-4
            {
                return Err(SceneError::InvalidGeometry);
            }
            if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
                return Err(SceneError::InvalidGeometry);
            }
        }
        LightGeometry::Disk {
            position,
            direction,
            radius,
        } => {
            validate_vector(position)?;
            validate_direction(direction)?;
            if !radius.is_finite() || radius <= 0.0 {
                return Err(SceneError::InvalidGeometry);
            }
        }
        LightGeometry::Capsule { a, b, radius } => {
            validate_vector(a)?;
            validate_vector(b)?;
            if !radius.is_finite() || radius <= 0.0 {
                return Err(SceneError::InvalidGeometry);
            }
        }
        LightGeometry::Sky { rotation } => {
            let norm: f64 = rotation
                .iter()
                .map(|value| f64::from(*value) * f64::from(*value))
                .sum();
            if rotation.iter().any(|value| !value.is_finite()) || (norm - 1.0).abs() > 1e-4 {
                return Err(SceneError::InvalidGeometry);
            }
        }
        LightGeometry::Emissive { .. } => {}
    }
    Ok(())
}

fn validate_vector(vector: [f32; 3]) -> Result<(), SceneError> {
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(SceneError::InvalidGeometry);
    }
    Ok(())
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f64 {
    a.into_iter()
        .zip(b)
        .map(|(a, b)| f64::from(a) * f64::from(b))
        .sum()
}

fn validate_direction(direction: [f32; 3]) -> Result<(), SceneError> {
    validate_vector(direction)?;
    if (dot(direction, direction) - 1.0).abs() > 1e-4 {
        return Err(SceneError::InvalidGeometry);
    }
    Ok(())
}

fn validate_nonnegative(value: f32) -> Result<(), SceneError> {
    if !value.is_finite() || value < 0.0 {
        return Err(SceneError::InvalidGeometry);
    }
    Ok(())
}

/// A typed partial update. The mask is authoritative and must match the
/// populated optional fields; this prevents silent full-record uploads.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LightPatch {
    pub mask: LightFieldMask,
    pub classification: Option<LightClassification>,
    pub geometry: Option<LightGeometry>,
    pub emission: Option<LightEmission>,
    pub visibility: Option<LightVisibility>,
    pub flags: Option<LightFlags>,
    pub expected_epochs: Option<LightFieldEpochs>,
}

impl LightPatch {
    /// Creates an empty patch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            mask: LightFieldMask(0),
            classification: None,
            geometry: None,
            emission: None,
            visibility: None,
            flags: None,
            expected_epochs: None,
        }
    }

    /// Sets classification fields and marks them dirty.
    #[must_use]
    pub const fn with_classification(mut self, value: LightClassification) -> Self {
        self.mask = self.mask.union(LightFieldMask::CLASSIFICATION);
        self.classification = Some(value);
        self
    }

    /// Sets geometry/transform fields and marks them dirty.
    #[must_use]
    pub const fn with_geometry(mut self, value: LightGeometry) -> Self {
        self.mask = self.mask.union(LightFieldMask::GEOMETRY);
        self.geometry = Some(value);
        self
    }

    /// Sets emission fields and marks them dirty.
    #[must_use]
    pub const fn with_emission(mut self, value: LightEmission) -> Self {
        self.mask = self.mask.union(LightFieldMask::EMISSION);
        self.emission = Some(value);
        self
    }

    /// Sets visibility fields and marks them dirty.
    #[must_use]
    pub const fn with_visibility(mut self, value: LightVisibility) -> Self {
        self.mask = self.mask.union(LightFieldMask::VISIBILITY);
        self.visibility = Some(value);
        self
    }

    /// Sets flag fields and marks them dirty.
    #[must_use]
    pub const fn with_flags(mut self, value: LightFlags) -> Self {
        self.mask = self.mask.union(LightFieldMask::FLAGS);
        self.flags = Some(value);
        self
    }

    /// Adds optimistic-concurrency expectations for the fields being patched.
    #[must_use]
    pub const fn expecting_epochs(mut self, value: LightFieldEpochs) -> Self {
        self.expected_epochs = Some(value);
        self
    }

    fn apply_to(self, current: &LightRecord) -> Result<LightRecord, SceneError> {
        if self.mask.is_empty() || self.mask.bits() & !PATCH_FIELD_BITS != 0 {
            return Err(SceneError::InvalidPatchMask(self.mask));
        }
        if let Some(expected) = self.expected_epochs {
            for field in [
                LightFieldMask::CLASSIFICATION,
                LightFieldMask::GEOMETRY,
                LightFieldMask::EMISSION,
                LightFieldMask::VISIBILITY,
                LightFieldMask::FLAGS,
            ] {
                if self.mask.contains(field)
                    && expected.for_field(field) != current.epochs.for_field(field)
                {
                    return Err(SceneError::FieldEpochMismatch {
                        field,
                        expected: expected.for_field(field).unwrap_or_default(),
                        actual: current.epochs.for_field(field).unwrap_or_default(),
                    });
                }
            }
        }
        if self.mask.contains(LightFieldMask::CLASSIFICATION) != self.classification.is_some()
            || self.mask.contains(LightFieldMask::GEOMETRY) != self.geometry.is_some()
            || self.mask.contains(LightFieldMask::EMISSION) != self.emission.is_some()
            || self.mask.contains(LightFieldMask::VISIBILITY) != self.visibility.is_some()
            || self.mask.contains(LightFieldMask::FLAGS) != self.flags.is_some()
        {
            return Err(SceneError::PatchMaskValueMismatch);
        }

        let mut updated = *current;
        if let Some(value) = self.classification {
            updated.classification = value;
        }
        if let Some(value) = self.geometry {
            updated.geometry = value;
        }
        if let Some(value) = self.emission {
            updated.emission = value;
        }
        if let Some(value) = self.visibility {
            updated.visibility = value;
        }
        if let Some(value) = self.flags {
            updated.flags = value;
        }
        updated.epochs = current.epochs.advanced(self.mask)?;
        updated.validate()?;
        Ok(updated)
    }
}

/// One ordered scene operation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SceneOperation {
    Create {
        handle: LightSceneHandle,
        record: LightRecord,
    },
    Update {
        handle: LightSceneHandle,
        patch: LightPatch,
    },
    Remove {
        handle: LightSceneHandle,
    },
}

impl SceneOperation {
    /// Returns the dirty fields represented by this operation.
    #[must_use]
    pub const fn dirty_fields(self) -> LightFieldMask {
        match self {
            Self::Create { .. } => LightFieldMask::CREATE,
            Self::Update { patch, .. } => patch.mask,
            Self::Remove { .. } => LightFieldMask::REMOVE,
        }
    }

    /// Returns the operation's handle.
    #[must_use]
    pub const fn handle(self) -> LightSceneHandle {
        match self {
            Self::Create { handle, .. } | Self::Update { handle, .. } | Self::Remove { handle } => {
                handle
            }
        }
    }
}

/// An ordered, replayable change to a light scene.
///
/// A nonempty delta advances sequence and scene epoch by exactly one; frames
/// may repeat but never regress. Each slot can appear at most once. New slots
/// must be appended contiguously in operation order; holes and same-delta
/// remove/recreate operations are rejected. This permits prevalidation using
/// only the touched records, rather than cloning the entire scene.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneDelta {
    pub sequence: u64,
    pub frame: u64,
    pub base_epoch: u64,
    pub epoch: u64,
    pub operations: Vec<SceneOperation>,
}

impl SceneDelta {
    /// Creates a delta. Validation occurs when it is applied.
    #[must_use]
    pub const fn new(
        sequence: u64,
        frame: u64,
        base_epoch: u64,
        epoch: u64,
        operations: Vec<SceneOperation>,
    ) -> Self {
        Self {
            sequence,
            frame,
            base_epoch,
            epoch,
            operations,
        }
    }

    /// Returns the scene lineage carried by this delta, or zero when empty.
    #[must_use]
    pub fn scene_identity(&self) -> u64 {
        self.operations
            .first()
            .map(|operation| operation.handle().scene)
            .unwrap_or(0)
    }

    /// Returns the aggregate dirty field mask.
    #[must_use]
    pub fn dirty_fields(&self) -> LightFieldMask {
        self.operations
            .iter()
            .copied()
            .fold(LightFieldMask(0), |mask, operation| {
                mask | operation.dirty_fields()
            })
    }

    /// Returns whether the delta contains no operations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
}

/// One committed change and the post-change field epochs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightChange {
    pub handle: LightSceneHandle,
    pub fields: LightFieldMask,
    pub epochs: LightFieldEpochs,
}

/// Result of applying one delta.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneApplyReport {
    pub sequence: u64,
    pub epoch: u64,
    pub changes: Vec<LightChange>,
}

/// Errors that preserve the reason a delta or checkpoint was rejected.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SceneError {
    InvalidHandle(LightSceneHandle),
    ForeignScene {
        expected: u64,
        actual: u64,
    },
    IdentityExhausted,
    StaleHandle {
        handle: LightSceneHandle,
        current_generation: u32,
    },
    SlotOccupied(LightSceneHandle),
    DuplicateHandle(LightSceneHandle),
    InvalidRecordEpochs,
    InvalidEmission,
    IncompatibleUnit {
        class: LightClass,
        unit: LightUnit,
    },
    InvalidGeometry,
    ClassificationGeometryMismatch {
        classification: LightClass,
        geometry: LightClass,
    },
    InvalidPatchMask(LightFieldMask),
    PatchMaskValueMismatch,
    FieldEpochMismatch {
        field: LightFieldMask,
        expected: u64,
        actual: u64,
    },
    EmptyDelta,
    SequenceMismatch {
        expected: u64,
        actual: u64,
    },
    BaseEpochMismatch {
        expected: u64,
        actual: u64,
    },
    TargetEpochMismatch {
        expected: u64,
        actual: u64,
    },
    FrameRegression {
        previous: u64,
        actual: u64,
    },
    SequenceExhausted,
    EpochExhausted,
    GenerationExhausted(LightSceneHandle),
    SlotCapacityExhausted,
    CheckpointVersion {
        expected: u32,
        actual: u32,
    },
    InvalidCheckpoint,
}

impl fmt::Display for SceneError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHandle(handle) => write!(formatter, "invalid light handle {handle:?}"),
            Self::ForeignScene { expected, actual } => write!(
                formatter,
                "light handle belongs to scene {actual}, expected scene {expected}"
            ),
            Self::IdentityExhausted => formatter.write_str("light scene identity space exhausted"),
            Self::StaleHandle {
                handle,
                current_generation,
            } => write!(
                formatter,
                "stale light handle {handle:?}; current generation is {current_generation}"
            ),
            Self::SlotOccupied(handle) => write!(formatter, "light slot is occupied: {handle:?}"),
            Self::DuplicateHandle(handle) => {
                write!(formatter, "duplicate handle in delta: {handle:?}")
            }
            Self::InvalidRecordEpochs => {
                formatter.write_str("new light record must have non-zero initial epochs")
            }
            Self::InvalidEmission => {
                formatter.write_str("light emission contains an invalid value")
            }
            Self::IncompatibleUnit { class, unit } => write!(
                formatter,
                "unit {unit:?} is incompatible with light class {class:?}"
            ),
            Self::InvalidGeometry => {
                formatter.write_str("light geometry contains an invalid value")
            }
            Self::ClassificationGeometryMismatch {
                classification,
                geometry,
            } => write!(
                formatter,
                "light classification {classification:?} does not match geometry {geometry:?}"
            ),
            Self::InvalidPatchMask(mask) => {
                write!(formatter, "invalid light patch mask 0x{:x}", mask.bits())
            }
            Self::PatchMaskValueMismatch => {
                formatter.write_str("light patch mask does not match its values")
            }
            Self::FieldEpochMismatch {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "light field epoch mismatch for 0x{:x}: expected {expected}, got {actual}",
                field.bits()
            ),
            Self::EmptyDelta => formatter.write_str("scene delta cannot be empty"),
            Self::SequenceMismatch { expected, actual } => {
                write!(
                    formatter,
                    "scene sequence mismatch: expected {expected}, got {actual}"
                )
            }
            Self::BaseEpochMismatch { expected, actual } => {
                write!(
                    formatter,
                    "scene base epoch mismatch: expected {expected}, got {actual}"
                )
            }
            Self::TargetEpochMismatch { expected, actual } => {
                write!(
                    formatter,
                    "scene target epoch mismatch: expected {expected}, got {actual}"
                )
            }
            Self::FrameRegression { previous, actual } => {
                write!(
                    formatter,
                    "scene frame regressed from {previous} to {actual}"
                )
            }
            Self::SequenceExhausted => formatter.write_str("scene sequence exhausted"),
            Self::EpochExhausted => formatter.write_str("scene epoch exhausted"),
            Self::GenerationExhausted(handle) => {
                write!(formatter, "generation exhausted for {handle:?}")
            }
            Self::SlotCapacityExhausted => {
                formatter.write_str("light scene slot capacity exhausted")
            }
            Self::CheckpointVersion { expected, actual } => {
                write!(
                    formatter,
                    "checkpoint version mismatch: expected {expected}, got {actual}"
                )
            }
            Self::InvalidCheckpoint => formatter.write_str("invalid light scene checkpoint"),
        }
    }
}

impl core::error::Error for SceneError {}

#[derive(Clone, Copy, Debug, PartialEq)]
struct LightSlot {
    generation: u32,
    identity: u64,
    retired: bool,
    record: Option<LightRecord>,
}

/// Fully checked mutation; commit cannot return a validation error.
struct PreparedChange {
    change: LightChange,
    record: Option<LightRecord>,
}

/// A retained CPU light scene with stable slot allocation and ordered deltas.
///
/// Apply costs O(k log k + k log f) with O(k) scratch storage for k operations
/// and f reusable slots; it does not scan or clone all n live records. Lookup
/// and live counts are O(1). Checkpoint and slot-order iteration are O(capacity).
/// Slots do not shrink, because dropping generations could resurrect old handles.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LightScene {
    identity: u64,
    slots: Vec<LightSlot>,
    free_slots: BTreeSet<u32>,
    live_count: usize,
    scene_epoch: u64,
    next_sequence: u64,
    last_frame: Option<u64>,
}

/// A complete, cloneable scene checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct LightSceneCheckpoint {
    version: u32,
    identity: u64,
    slots: Vec<LightSlot>,
    scene_epoch: u64,
    next_sequence: u64,
    last_frame: Option<u64>,
}

impl LightSceneCheckpoint {
    /// Returns the checkpoint schema version.
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Returns the scene lineage identity preserved by restore and replay.
    #[must_use]
    pub const fn identity(&self) -> u64 {
        self.identity
    }

    /// Returns the last committed scene epoch.
    #[must_use]
    pub const fn scene_epoch(&self) -> u64 {
        self.scene_epoch
    }

    /// Returns the last committed delta sequence.
    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.next_sequence
    }
}

impl LightScene {
    /// Creates an empty scene.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the scene lineage identity. Empty, never-mutated scenes have identity zero.
    #[must_use]
    pub const fn identity(&self) -> u64 {
        self.identity
    }

    /// Returns the number of live lights.
    #[must_use]
    pub fn len(&self) -> usize {
        self.live_count
    }

    /// Returns whether no light is live.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of reusable slots.
    #[must_use]
    pub fn free_slot_count(&self) -> usize {
        self.free_slots.len()
    }

    /// Returns the current global scene epoch.
    #[must_use]
    pub const fn scene_epoch(&self) -> u64 {
        self.scene_epoch
    }

    /// Returns the last committed delta sequence.
    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.next_sequence
    }

    /// Looks up a live record, rejecting stale generations as a miss.
    #[must_use]
    pub fn get(&self, handle: LightSceneHandle) -> Option<&LightRecord> {
        if handle.scene != self.identity || handle.identity == 0 {
            return None;
        }
        let slot = self.slots.get(handle.slot as usize)?;
        if slot.generation == handle.generation && slot.identity == handle.identity && !slot.retired
        {
            slot.record.as_ref()
        } else {
            None
        }
    }

    /// Iterates live handles and records in deterministic slot order.
    pub fn iter(&self) -> impl Iterator<Item = (LightSceneHandle, &LightRecord)> + '_ {
        self.slots.iter().enumerate().filter_map(|(index, slot)| {
            slot.record.as_ref().map(|record| {
                (
                    LightSceneHandle::in_scene(
                        self.identity,
                        index as u32,
                        slot.generation,
                        slot.identity,
                    ),
                    record,
                )
            })
        })
    }

    /// Inserts a new light and returns its generation-safe handle plus delta.
    pub fn insert(
        &mut self,
        record: LightRecord,
        frame: u64,
    ) -> Result<(LightSceneHandle, SceneDelta), SceneError> {
        let handle = self.next_handle()?;
        let delta = self.make_delta(frame, SceneOperation::Create { handle, record })?;
        self.apply_delta(&delta)?;
        Ok((handle, delta))
    }

    /// Updates selected fields and returns the committed delta.
    pub fn update(
        &mut self,
        handle: LightSceneHandle,
        patch: LightPatch,
        frame: u64,
    ) -> Result<SceneDelta, SceneError> {
        let delta = self.make_delta(frame, SceneOperation::Update { handle, patch })?;
        self.apply_delta(&delta)?;
        Ok(delta)
    }

    /// Removes a light and returns the committed delta.
    pub fn remove(
        &mut self,
        handle: LightSceneHandle,
        frame: u64,
    ) -> Result<SceneDelta, SceneError> {
        let delta = self.make_delta(frame, SceneOperation::Remove { handle })?;
        self.apply_delta(&delta)?;
        Ok(delta)
    }

    /// Applies a delta atomically after validating sequence, epochs, handles,
    /// records, patches, and duplicate operations.
    pub fn apply_delta(&mut self, delta: &SceneDelta) -> Result<SceneApplyReport, SceneError> {
        let delta_scene = delta
            .operations
            .first()
            .map(|operation| operation.handle().scene)
            .unwrap_or(0);
        if delta.operations.is_empty() {
            return Err(SceneError::EmptyDelta);
        }
        if self.identity != 0 && delta_scene != self.identity {
            return Err(SceneError::ForeignScene {
                expected: self.identity,
                actual: delta_scene,
            });
        }
        let prepared = self.prepare_delta(delta)?;

        let mut changes = Vec::with_capacity(delta.operations.len());
        for item in prepared {
            self.commit_change(&item);
            changes.push(item.change);
        }
        self.scene_epoch = delta.epoch;
        self.next_sequence = delta.sequence;
        self.last_frame = Some(delta.frame);
        if self.identity == 0 {
            self.identity = delta_scene;
        }
        Ok(SceneApplyReport {
            sequence: delta.sequence,
            epoch: delta.epoch,
            changes,
        })
    }

    /// Captures all slot generations, live records, and replay cursors.
    #[must_use]
    pub fn checkpoint(&self) -> LightSceneCheckpoint {
        LightSceneCheckpoint {
            version: CHECKPOINT_VERSION,
            identity: self.identity,
            slots: self.slots.clone(),
            scene_epoch: self.scene_epoch,
            next_sequence: self.next_sequence,
            last_frame: self.last_frame,
        }
    }

    /// Restores a validated checkpoint without changing the caller's scene on
    /// failure.
    pub fn restore(&mut self, checkpoint: &LightSceneCheckpoint) -> Result<(), SceneError> {
        let restored = Self::from_checkpoint(checkpoint)?;
        *self = restored;
        Ok(())
    }

    /// Restores a checkpoint into a new scene.
    pub fn from_checkpoint(checkpoint: &LightSceneCheckpoint) -> Result<Self, SceneError> {
        if checkpoint.version != CHECKPOINT_VERSION {
            return Err(SceneError::CheckpointVersion {
                expected: CHECKPOINT_VERSION,
                actual: checkpoint.version,
            });
        }
        let mut scene = Self {
            slots: checkpoint.slots.clone(),
            identity: checkpoint.identity,
            free_slots: BTreeSet::new(),
            live_count: 0,
            scene_epoch: checkpoint.scene_epoch,
            next_sequence: checkpoint.next_sequence,
            last_frame: checkpoint.last_frame,
        };
        scene.validate_checkpoint_slots()?;
        for (index, slot) in scene.slots.iter().enumerate() {
            scene.live_count += usize::from(slot.record.is_some());
            if slot.record.is_none() && !slot.retired {
                scene.free_slots.insert(index as u32);
            }
        }
        Ok(scene)
    }

    /// Replays deltas from an empty scene.
    pub fn replay(deltas: &[SceneDelta]) -> Result<Self, SceneError> {
        Self::replay_from(&Self::new().checkpoint(), deltas)
    }

    /// Replays deltas from a checkpoint.
    pub fn replay_from(
        checkpoint: &LightSceneCheckpoint,
        deltas: &[SceneDelta],
    ) -> Result<Self, SceneError> {
        let mut scene = Self::from_checkpoint(checkpoint)?;
        for delta in deltas {
            scene.apply_delta(delta)?;
        }
        Ok(scene)
    }

    fn next_handle(&self) -> Result<LightSceneHandle, SceneError> {
        if let Some(&slot_index) = self.free_slots.iter().next_back() {
            let slot = &self.slots[slot_index as usize];
            let scene = if self.identity == 0 {
                allocate_identity(&NEXT_SCENE_IDENTITY)?
            } else {
                self.identity
            };
            let identity = allocate_identity(&NEXT_LIGHT_IDENTITY)?;
            return Ok(LightSceneHandle::in_scene(
                scene,
                slot_index,
                slot.generation,
                identity,
            ));
        }
        let slot_index =
            u32::try_from(self.slots.len()).map_err(|_| SceneError::SlotCapacityExhausted)?;
        let scene = if self.identity == 0 {
            allocate_identity(&NEXT_SCENE_IDENTITY)?
        } else {
            self.identity
        };
        let identity = allocate_identity(&NEXT_LIGHT_IDENTITY)?;
        Ok(LightSceneHandle::in_scene(scene, slot_index, 1, identity))
    }

    fn make_delta(&self, frame: u64, operation: SceneOperation) -> Result<SceneDelta, SceneError> {
        let sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(SceneError::SequenceExhausted)?;
        let epoch = self
            .scene_epoch
            .checked_add(1)
            .ok_or(SceneError::EpochExhausted)?;
        Ok(SceneDelta::new(
            sequence,
            frame,
            self.scene_epoch,
            epoch,
            Vec::from([operation]),
        ))
    }

    fn prepare_delta(&self, delta: &SceneDelta) -> Result<Vec<PreparedChange>, SceneError> {
        if delta.operations.is_empty() {
            return Err(SceneError::EmptyDelta);
        }
        let expected_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(SceneError::SequenceExhausted)?;
        if delta.sequence != expected_sequence {
            return Err(SceneError::SequenceMismatch {
                expected: expected_sequence,
                actual: delta.sequence,
            });
        }
        if delta.base_epoch != self.scene_epoch {
            return Err(SceneError::BaseEpochMismatch {
                expected: self.scene_epoch,
                actual: delta.base_epoch,
            });
        }
        let expected_epoch = self
            .scene_epoch
            .checked_add(1)
            .ok_or(SceneError::EpochExhausted)?;
        if delta.epoch != expected_epoch {
            return Err(SceneError::TargetEpochMismatch {
                expected: expected_epoch,
                actual: delta.epoch,
            });
        }
        if let Some(previous) = self.last_frame
            && delta.frame < previous
        {
            return Err(SceneError::FrameRegression {
                previous,
                actual: delta.frame,
            });
        }

        let mut seen = BTreeSet::new();
        let mut appended_slots = 0u32;
        let mut prepared = Vec::with_capacity(delta.operations.len());
        let expected_scene = if self.identity == 0 {
            delta.operations[0].handle().scene
        } else {
            self.identity
        };
        for operation in delta.operations.iter().copied() {
            if operation.handle().scene != expected_scene {
                return Err(SceneError::ForeignScene {
                    expected: expected_scene,
                    actual: operation.handle().scene,
                });
            }
            if !seen.insert(operation.handle().slot) {
                return Err(SceneError::DuplicateHandle(operation.handle()));
            }
            match operation {
                SceneOperation::Create { handle, record } => {
                    record.validate()?;
                    if record.epochs != LightFieldEpochs::INITIAL {
                        return Err(SceneError::InvalidRecordEpochs);
                    }
                    self.validate_create_handle(handle, appended_slots)?;
                    if handle.slot as usize == self.slots.len() + appended_slots as usize {
                        appended_slots = appended_slots
                            .checked_add(1)
                            .ok_or(SceneError::SlotCapacityExhausted)?;
                    }
                    prepared.push(PreparedChange {
                        change: LightChange {
                            handle,
                            fields: LightFieldMask::CREATE,
                            epochs: record.epochs,
                        },
                        record: Some(record),
                    });
                }
                SceneOperation::Update { handle, patch } => {
                    let current = self.live_record(handle)?;
                    let updated = patch.apply_to(current)?;
                    prepared.push(PreparedChange {
                        change: LightChange {
                            handle,
                            fields: patch.mask,
                            epochs: updated.epochs,
                        },
                        record: Some(updated),
                    });
                }
                SceneOperation::Remove { handle } => {
                    let old = self.live_record(handle)?;
                    prepared.push(PreparedChange {
                        change: LightChange {
                            handle,
                            fields: LightFieldMask::REMOVE,
                            epochs: old.epochs,
                        },
                        record: None,
                    });
                }
            }
        }
        Ok(prepared)
    }

    fn validate_create_handle(
        &self,
        handle: LightSceneHandle,
        appended_slots: u32,
    ) -> Result<(), SceneError> {
        let target_scene = if self.identity == 0 {
            handle.scene
        } else {
            self.identity
        };
        if target_scene == 0 || handle.scene != target_scene {
            return Err(SceneError::ForeignScene {
                expected: target_scene,
                actual: handle.scene,
            });
        }
        if handle.identity == 0 {
            return Err(SceneError::InvalidHandle(handle));
        }
        if !handle.is_valid() {
            return Err(SceneError::InvalidHandle(handle));
        }
        let index = handle.slot as usize;
        let expected_new_index = self
            .slots
            .len()
            .checked_add(appended_slots as usize)
            .ok_or(SceneError::SlotCapacityExhausted)?;
        if index == expected_new_index {
            if expected_new_index >= u32::MAX as usize {
                return Err(SceneError::SlotCapacityExhausted);
            }
            if handle.generation == 1 {
                if self
                    .slots
                    .iter()
                    .any(|slot| slot.identity == handle.identity)
                {
                    return Err(SceneError::InvalidHandle(handle));
                }
                return Ok(());
            }
            return Err(SceneError::StaleHandle {
                handle,
                current_generation: 1,
            });
        }
        let slot = self
            .slots
            .get(index)
            .ok_or(SceneError::InvalidHandle(handle))?;
        if slot.generation != handle.generation {
            return Err(SceneError::StaleHandle {
                handle,
                current_generation: slot.generation,
            });
        }
        if slot.identity != handle.identity {
            return Err(SceneError::InvalidHandle(handle));
        }
        if slot.retired {
            return Err(SceneError::GenerationExhausted(handle));
        }
        if slot.record.is_some() {
            return Err(SceneError::SlotOccupied(handle));
        }
        if !self.free_slots.contains(&handle.slot) {
            return Err(SceneError::InvalidCheckpoint);
        }
        Ok(())
    }

    fn live_record(&self, handle: LightSceneHandle) -> Result<&LightRecord, SceneError> {
        if self.identity != 0 && handle.scene != self.identity {
            return Err(SceneError::ForeignScene {
                expected: self.identity,
                actual: handle.scene,
            });
        }
        if !handle.is_valid() {
            return Err(SceneError::InvalidHandle(handle));
        }
        let slot = self
            .slots
            .get(handle.slot as usize)
            .ok_or(SceneError::InvalidHandle(handle))?;
        if slot.generation != handle.generation {
            return Err(SceneError::StaleHandle {
                handle,
                current_generation: slot.generation,
            });
        }
        if slot.identity != handle.identity {
            return Err(SceneError::InvalidHandle(handle));
        }
        if slot.retired {
            return Err(SceneError::GenerationExhausted(handle));
        }
        slot.record
            .as_ref()
            .ok_or(SceneError::InvalidHandle(handle))
    }

    fn commit_change(&mut self, item: &PreparedChange) {
        let handle = item.change.handle;
        match item.record {
            Some(record) if item.change.fields.contains(LightFieldMask::LIFETIME) => {
                let index = handle.slot as usize;
                if index == self.slots.len() {
                    self.slots.push(LightSlot {
                        generation: handle.generation,
                        identity: handle.identity,
                        retired: false,
                        record: Some(record),
                    });
                } else {
                    self.free_slots.remove(&handle.slot);
                    let slot = &mut self.slots[index];
                    slot.retired = false;
                    slot.identity = handle.identity;
                    slot.record = Some(record);
                }
                self.live_count += 1;
            }
            Some(record) => {
                let index = handle.slot as usize;
                self.slots[index].record = Some(record);
            }
            None => {
                let index = handle.slot as usize;
                self.slots[index].record = None;
                let slot = &mut self.slots[index];
                if let Some(generation) = slot.generation.checked_add(1) {
                    slot.generation = generation;
                    self.free_slots.insert(handle.slot);
                } else {
                    slot.retired = true;
                }
                self.live_count -= 1;
            }
        }
    }

    fn validate_checkpoint_slots(&self) -> Result<(), SceneError> {
        if self.slots.len() > u32::MAX as usize
            || self.scene_epoch != self.next_sequence
            || self.last_frame.is_some() != (self.scene_epoch > 0)
            || (self.scene_epoch == 0 && !self.slots.is_empty())
            || ((self.scene_epoch > 0 || !self.slots.is_empty()) && self.identity == 0)
        {
            return Err(SceneError::InvalidCheckpoint);
        }
        for slot in &self.slots {
            if slot.generation == 0
                || slot.identity == 0
                || (slot.retired && (slot.record.is_some() || slot.generation != u32::MAX))
                || (slot.record.is_none() && slot.generation == 1)
            {
                return Err(SceneError::InvalidCheckpoint);
            }
            if let Some(record) = slot.record {
                if slot.identity == 0 || self.identity == 0 {
                    return Err(SceneError::InvalidCheckpoint);
                }
                record.validate()?;
                for epoch in [
                    record.epochs.classification,
                    record.epochs.geometry,
                    record.epochs.emission,
                    record.epochs.visibility,
                    record.epochs.flags,
                ] {
                    if epoch > self.scene_epoch {
                        return Err(SceneError::InvalidCheckpoint);
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(value: f32) -> LightRecord {
        LightRecord::new(
            LightClassification {
                class: LightClass::Point,
                mobility: LightMobility::Dynamic,
            },
            LightEmission::new([value, 1.0, 0.5], value, LightUnit::Candela),
            LightGeometry::Point {
                position: [value, 0.0, 1.0],
                radius: 0.1,
                range: 20.0,
            },
        )
    }

    #[test]
    fn recycled_slot_gets_new_generation_and_rejects_stale_handle() {
        let mut scene = LightScene::new();
        let (old, _) = scene.insert(point(1.0), 1).expect("insert");
        scene.remove(old, 2).expect("remove");
        assert!(scene.get(old).is_none());

        let (new, _) = scene.insert(point(2.0), 3).expect("reuse");
        assert_eq!(old.slot(), new.slot());
        assert_ne!(old.generation(), new.generation());
        assert!(scene.get(old).is_none());
        assert_eq!(
            scene.get(new).map(|record| record.emission.value),
            Some(2.0)
        );

        let patch = LightPatch::new().with_emission(LightEmission::default());
        let error = scene
            .update(old, patch, 4)
            .expect_err("stale update must fail");
        assert!(matches!(error, SceneError::StaleHandle { .. }));
    }

    #[test]
    fn patch_mask_and_field_epochs_are_incremental() {
        let mut scene = LightScene::new();
        let (handle, _) = scene.insert(point(1.0), 1).expect("insert");
        let before = scene.get(handle).expect("record").epochs;
        let patch = LightPatch::new().with_geometry(LightGeometry::Point {
            position: [3.0, 0.0, 1.0],
            radius: 0.1,
            range: 20.0,
        });
        let delta = scene.update(handle, patch, 2).expect("update");
        assert_eq!(delta.dirty_fields(), LightFieldMask::GEOMETRY);
        let after = scene.get(handle).expect("record").epochs;
        assert_eq!(after.geometry, before.geometry + 1);
        assert_eq!(after.emission, before.emission);
        assert_eq!(delta.operations[0].dirty_fields(), LightFieldMask::GEOMETRY);
    }

    #[test]
    fn expected_epoch_rejects_lost_update_without_mutating_scene() {
        let mut scene = LightScene::new();
        let (handle, _) = scene.insert(point(1.0), 1).expect("insert");
        let current = *scene.get(handle).expect("record");
        let stale = LightPatch::new()
            .with_emission(LightEmission::default())
            .expecting_epochs(LightFieldEpochs::INITIAL);
        scene.update(handle, stale, 2).expect("first update");
        let before = *scene.get(handle).expect("record");

        let stale_again = LightPatch::new()
            .with_emission(LightEmission::new([2.0; 3], 2.0, LightUnit::Candela))
            .expecting_epochs(current.epochs);
        let error = scene
            .update(handle, stale_again, 3)
            .expect_err("lost update");
        assert!(matches!(error, SceneError::FieldEpochMismatch { .. }));
        assert_eq!(*scene.get(handle).expect("record"), before);
    }

    #[test]
    fn delta_application_is_ordered_and_rejects_invalid_sequence() {
        let mut source = LightScene::new();
        let (_, delta) = source.insert(point(1.0), 7).expect("insert");
        let mut replica = LightScene::new();
        let report = replica.apply_delta(&delta).expect("replay delta");
        assert_eq!(report.sequence, 1);
        assert_eq!(replica, source);

        let error = replica.apply_delta(&delta).expect_err("duplicate sequence");
        assert!(matches!(error, SceneError::SequenceMismatch { .. }));
    }

    #[test]
    fn checkpoint_replay_preserves_free_slot_order_and_state() {
        let mut scene = LightScene::new();
        let (first, _) = scene.insert(point(1.0), 1).expect("first");
        let (second, _) = scene.insert(point(2.0), 1).expect("second");
        let checkpoint = scene.checkpoint();
        let remove_delta = scene.remove(first, 2).expect("remove");
        let (reused, insert_delta) = scene.insert(point(3.0), 3).expect("reuse");
        assert_eq!(reused.slot(), first.slot());
        assert_ne!(reused.generation(), first.generation());

        let replayed =
            LightScene::replay_from(&checkpoint, &[remove_delta.clone(), insert_delta.clone()])
                .expect("replay");
        assert_eq!(replayed, scene);
        assert_eq!(replayed.get(second), scene.get(second));
        assert_eq!(checkpoint.sequence(), 2);
        assert_eq!(checkpoint.scene_epoch(), 2);
    }

    #[test]
    fn invalid_record_and_patch_are_rejected_atomically() {
        let mut scene = LightScene::new();
        let invalid = LightRecord::new(
            LightClassification {
                class: LightClass::Spot,
                mobility: LightMobility::Dynamic,
            },
            LightEmission::default(),
            LightGeometry::Point {
                position: [0.0, 0.0, 1.0],
                radius: 0.1,
                range: 5.0,
            },
        );
        let error = scene.insert(invalid, 1).expect_err("class mismatch");
        assert!(matches!(
            error,
            SceneError::ClassificationGeometryMismatch { .. }
        ));
        assert!(scene.is_empty());

        let (handle, _) = scene.insert(point(1.0), 2).expect("insert");
        let malformed = LightPatch {
            mask: LightFieldMask::GEOMETRY,
            ..LightPatch::new()
        };
        let error = scene
            .update(handle, malformed, 3)
            .expect_err("missing value");
        assert!(matches!(error, SceneError::PatchMaskValueMismatch));
        assert_eq!(scene.sequence(), 1);
        assert_eq!(scene.scene_epoch(), 1);
    }

    #[test]
    fn independent_scenes_reject_same_slot_generation_handles_and_deltas() {
        let mut left = LightScene::new();
        let mut right = LightScene::new();
        let (left_handle, left_delta) = left.insert(point(1.0), 1).expect("left insert");
        let (right_handle, _) = right.insert(point(2.0), 1).expect("right insert");

        assert_eq!(
            (left_handle.slot(), left_handle.generation()),
            (right_handle.slot(), right_handle.generation())
        );
        assert_ne!(left_handle.scene_identity(), right_handle.scene_identity());
        assert_ne!(left_handle.identity(), right_handle.identity());
        assert!(left.get(right_handle).is_none());
        let before = right.clone();
        let error = right.remove(left_handle, 2).expect_err("foreign handle");
        assert!(matches!(error, SceneError::ForeignScene { .. }));
        assert_eq!(right, before);

        let error = right
            .apply_delta(&left_delta)
            .expect_err("foreign lineage delta");
        assert!(matches!(error, SceneError::ForeignScene { .. }));
        assert_eq!(right, before);
        assert!(left.get(left_handle).is_some());
    }

    #[test]
    fn checkpoint_restore_and_replay_preserve_lineage_without_resurrecting_future_handles() {
        let mut scene = LightScene::new();
        let (original, _) = scene.insert(point(1.0), 1).expect("insert");
        let checkpoint = scene.checkpoint();
        let remove_delta = scene.remove(original, 2).expect("remove");
        let (future, create_delta) = scene.insert(point(2.0), 3).expect("recreate");
        assert_eq!(future.slot(), original.slot());
        assert_eq!(future.generation(), original.generation() + 1);

        let replayed =
            LightScene::replay_from(&checkpoint, &[remove_delta.clone(), create_delta.clone()])
                .expect("replay deltas");
        assert_eq!(replayed.identity(), checkpoint.identity());
        assert_eq!(replayed.get(future), scene.get(future));

        scene.restore(&checkpoint).expect("restore checkpoint");
        assert_eq!(scene.identity(), checkpoint.identity());
        assert!(scene.get(future).is_none());
        scene.remove(original, 2).expect("remove after restore");
        let (restored_branch, _) = scene.insert(point(3.0), 3).expect("branch recreate");
        assert_eq!(restored_branch.slot(), future.slot());
        assert_eq!(restored_branch.generation(), future.generation());
        assert_ne!(restored_branch.identity(), future.identity());
        assert!(scene.get(future).is_none());
        assert!(scene.get(restored_branch).is_some());
    }

    #[test]
    fn raw_handles_and_mixed_lineage_delta_are_rejected_atomically() {
        let mut first = LightScene::new();
        let mut second = LightScene::new();
        let (first_handle, first_delta) = first.insert(point(1.0), 1).expect("first insert");
        let (second_handle, _) = second.insert(point(2.0), 1).expect("second insert");
        assert!(first
            .get(LightSceneHandle::new(
                first_handle.slot(),
                first_handle.generation()
            ))
            .is_none());

        let mixed = SceneDelta::new(
            1,
            1,
            0,
            1,
            alloc::vec![
                first_delta.operations[0],
                SceneOperation::Create {
                    handle: second_handle,
                    record: point(2.0)
                },
            ],
        );
        let mut replica = LightScene::new();
        let error = replica.apply_delta(&mixed).expect_err("mixed lineage");
        assert!(matches!(error, SceneError::ForeignScene { .. }));
        assert!(replica.is_empty());
        assert_eq!(replica.sequence(), 0);
    }

    #[test]
    fn invalid_late_operation_does_not_partially_commit_delta() {
        let mut source = LightScene::new();
        let (_, first_delta) = source.insert(point(1.0), 1).expect("first insert");
        let (second_handle, _) = source.insert(point(2.0), 1).expect("second insert");
        let invalid = LightRecord::new(
            LightClassification {
                class: LightClass::Spot,
                mobility: LightMobility::Dynamic,
            },
            LightEmission::default(),
            LightGeometry::Point {
                position: [0.0, 0.0, 1.0],
                radius: 0.1,
                range: 5.0,
            },
        );
        let delta = SceneDelta::new(
            1,
            1,
            0,
            1,
            alloc::vec![
                first_delta.operations[0],
                SceneOperation::Create {
                    handle: second_handle,
                    record: invalid
                },
            ],
        );
        let mut replica = LightScene::new();
        let error = replica
            .apply_delta(&delta)
            .expect_err("invalid second create");
        assert!(matches!(
            error,
            SceneError::ClassificationGeometryMismatch { .. }
        ));
        assert!(replica.is_empty());
        assert_eq!(replica.sequence(), 0);
        assert_eq!(replica.scene_epoch(), 0);
    }

    #[test]
    fn generation_exhaustion_retires_slot_without_wrapping() {
        let mut scene = LightScene::new();
        let (original, _) = scene.insert(point(1.0), 1).expect("insert");
        scene.slots[original.slot() as usize].generation = u32::MAX;
        let exhausted = LightSceneHandle::in_scene(
            scene.identity,
            original.slot(),
            u32::MAX,
            original.identity(),
        );

        scene
            .remove(exhausted, 2)
            .expect("remove exhausted generation");
        let retired = &scene.slots[exhausted.slot() as usize];
        assert!(retired.retired);
        assert_eq!(retired.generation, u32::MAX);
        assert_eq!(scene.free_slot_count(), 0);
        let (next, _) = scene.insert(point(2.0), 3).expect("append instead of wrap");
        assert_ne!(next.slot(), exhausted.slot());
    }

    #[test]
    fn identity_and_sequence_exhaustion_fail_closed() {
        let exhausted_counter = AtomicU64::new(u64::MAX);
        assert!(matches!(
            allocate_identity(&exhausted_counter),
            Err(SceneError::IdentityExhausted)
        ));
        assert_eq!(exhausted_counter.load(Ordering::Relaxed), u64::MAX);

        let mut scene = LightScene::new();
        scene.next_sequence = u64::MAX;
        scene.scene_epoch = u64::MAX;
        scene.last_frame = Some(0);
        let before = scene.clone();
        let error = scene.insert(point(1.0), 1).expect_err("sequence exhausted");
        assert!(matches!(error, SceneError::SequenceExhausted));
        assert_eq!(scene, before);

        let mut epoch_scene = LightScene::new();
        epoch_scene.scene_epoch = u64::MAX;
        epoch_scene.next_sequence = u64::MAX - 1;
        epoch_scene.last_frame = Some(0);
        let before = epoch_scene.clone();
        let error = epoch_scene
            .insert(point(1.0), 1)
            .expect_err("epoch exhausted");
        assert!(matches!(error, SceneError::EpochExhausted));
        assert_eq!(epoch_scene, before);
    }
}
