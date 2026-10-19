//! Self-contained OCIO-style display configuration and CPU reference transform.
//!
//! This module deliberately owns only colour configuration and reference math.
//! It has no frame-graph or pass dependency, and it does not require an OCIO
//! runtime. The wire format is canonical little-endian binary so a config can
//! be persisted, hashed, compared, and rejected before it reaches a renderer.

use alloc::vec::Vec;
use core::fmt;

/// Current canonical serialization schema for [`OcioConfig`].
pub const OCIO_CONFIG_SCHEMA_VERSION: u16 = 1;
/// Algorithm revision bound into the config identity.
pub const OCIO_CONFIG_ALGORITHM_VERSION: u32 = 1;
const CONFIG_MAGIC: [u8; 8] = *b"PRMOCIO1";
const CONFIG_BYTES: usize = 8 + 2 + 4 + 1 + 1 + 1 + 1 + 4 + 12;
const MIN_EXPOSURE_STOPS: f32 = -32.0;
const MAX_EXPOSURE_STOPS: f32 = 32.0;
const MIN_WHITE_BALANCE_GAIN: f32 = 0.0;
const MAX_WHITE_BALANCE_GAIN: f32 = 8.0;
const MAX_TONEMAP_INPUT: f32 = 1.0e10;
const DIGEST_DOMAIN: &[u8] = b"prism.lighting.ocio.v1\0";

/// A tone-mapping operator applied to exposed linear scene values.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum OcioToneMapping {
    /// Preserve linear values; display encoding still happens afterward.
    Linear = 0,
    /// Narkowicz's compact ACES filmic fit.
    Aces = 1,
    /// A compact AgX fit suitable for a CPU reference and a matching shader.
    Agx = 2,
}

impl OcioToneMapping {
    const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Linear),
            1 => Some(Self::Aces),
            2 => Some(Self::Agx),
            _ => None,
        }
    }
}

/// Output display gamut. Input values are always linear sRGB/D65 scene values.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum OcioDisplayGamut {
    /// Linear sRGB / Rec.709 primaries.
    Srgb = 0,
    /// Display-P3 primaries with a D65 white point.
    DisplayP3 = 1,
    /// Rec.2020 primaries with a D65 white point.
    Rec2020 = 2,
}

impl OcioDisplayGamut {
    const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Srgb),
            1 => Some(Self::DisplayP3),
            2 => Some(Self::Rec2020),
            _ => None,
        }
    }
}

/// Display transfer function applied after gamut conversion.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum OcioTransferFunction {
    /// Leave the display signal linear.
    Linear = 0,
    /// The piecewise IEC 61966-2-1 sRGB OETF.
    Srgb = 1,
    /// A power-law 2.2 display encoding.
    Gamma22 = 2,
    /// A power-law 2.4 display encoding.
    Gamma24 = 3,
}

impl OcioTransferFunction {
    const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Linear),
            1 => Some(Self::Srgb),
            2 => Some(Self::Gamma22),
            3 => Some(Self::Gamma24),
            _ => None,
        }
    }
}

/// Linear RGB gains used to neutralize the scene white point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OcioWhiteBalance {
    /// Red multiplier.
    pub red_gain: f32,
    /// Green multiplier.
    pub green_gain: f32,
    /// Blue multiplier.
    pub blue_gain: f32,
}

impl OcioWhiteBalance {
    /// Neutral daylight balance.
    pub const NEUTRAL: Self = Self {
        red_gain: 1.0,
        green_gain: 1.0,
        blue_gain: 1.0,
    };

    /// Creates a white-balance gain triplet.
    #[must_use]
    pub const fn new(red_gain: f32, green_gain: f32, blue_gain: f32) -> Self {
        Self {
            red_gain,
            green_gain,
            blue_gain,
        }
    }

    fn validate(self) -> Result<(), OcioConfigError> {
        for (field, value) in [
            ("white_balance.red_gain", self.red_gain),
            ("white_balance.green_gain", self.green_gain),
            ("white_balance.blue_gain", self.blue_gain),
        ] {
            if !value.is_finite() {
                return Err(OcioConfigError::NonFinite(field));
            }
            if !(MIN_WHITE_BALANCE_GAIN..=MAX_WHITE_BALANCE_GAIN).contains(&value) {
                return Err(OcioConfigError::OutOfRange(field));
            }
        }
        if self.red_gain == 0.0 && self.green_gain == 0.0 && self.blue_gain == 0.0 {
            return Err(OcioConfigError::DegenerateWhiteBalance);
        }
        Ok(())
    }

    const fn gains(self) -> [f32; 3] {
        [self.red_gain, self.green_gain, self.blue_gain]
    }
}

/// Version and algorithm identity of a validated OCIO configuration.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct OcioConfigVersion {
    /// Canonical wire schema version.
    pub schema: u16,
    /// Reference-transform algorithm revision.
    pub algorithm: u32,
}

/// BLAKE3 identity of canonical [`OcioConfig`] bytes.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct OcioConfigDigest(pub [u8; 32]);

impl OcioConfigDigest {
    /// Returns the digest bytes in canonical order.
    #[must_use]
    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

/// Complete display colour configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OcioConfig {
    /// Configuration schema version. Only [`OCIO_CONFIG_SCHEMA_VERSION`] is
    /// accepted by [`OcioConfig::validate`].
    pub schema_version: u16,
    /// Reference transform algorithm revision.
    pub algorithm_version: u32,
    /// Tone mapper used before display gamut conversion.
    pub tone_mapping: OcioToneMapping,
    /// Exposure adjustment in photographic stops.
    pub exposure_stops: f32,
    /// Linear RGB white-balance gains.
    pub white_balance: OcioWhiteBalance,
    /// Destination display gamut.
    pub display_gamut: OcioDisplayGamut,
    /// Destination transfer function.
    pub transfer_function: OcioTransferFunction,
}

impl Default for OcioConfig {
    fn default() -> Self {
        Self::srgb()
    }
}

impl OcioConfig {
    /// Creates the neutral linear+sRGB display configuration.
    #[must_use]
    pub const fn srgb() -> Self {
        Self {
            schema_version: OCIO_CONFIG_SCHEMA_VERSION,
            algorithm_version: OCIO_CONFIG_ALGORITHM_VERSION,
            tone_mapping: OcioToneMapping::Linear,
            exposure_stops: 0.0,
            white_balance: OcioWhiteBalance::NEUTRAL,
            display_gamut: OcioDisplayGamut::Srgb,
            transfer_function: OcioTransferFunction::Srgb,
        }
    }

    /// Creates a neutral ACES/sRGB configuration.
    #[must_use]
    pub const fn aces() -> Self {
        Self {
            tone_mapping: OcioToneMapping::Aces,
            ..Self::srgb()
        }
    }

    /// Creates a neutral AgX/sRGB configuration.
    #[must_use]
    pub const fn agx() -> Self {
        Self {
            tone_mapping: OcioToneMapping::Agx,
            ..Self::srgb()
        }
    }

    /// Returns this config's version identity.
    #[must_use]
    pub const fn version(self) -> OcioConfigVersion {
        OcioConfigVersion {
            schema: self.schema_version,
            algorithm: self.algorithm_version,
        }
    }

    /// Validates all scalar ranges and version gates.
    pub fn validate(&self) -> Result<(), OcioConfigError> {
        if self.schema_version != OCIO_CONFIG_SCHEMA_VERSION {
            return Err(OcioConfigError::UnsupportedSchema(self.schema_version));
        }
        if self.algorithm_version != OCIO_CONFIG_ALGORITHM_VERSION {
            return Err(OcioConfigError::UnsupportedAlgorithm(
                self.algorithm_version,
            ));
        }
        if !self.exposure_stops.is_finite() {
            return Err(OcioConfigError::NonFinite("exposure_stops"));
        }
        if !(MIN_EXPOSURE_STOPS..=MAX_EXPOSURE_STOPS).contains(&self.exposure_stops) {
            return Err(OcioConfigError::OutOfRange("exposure_stops"));
        }
        self.white_balance.validate()
    }

    /// Serializes this config into its fixed-size canonical wire format.
    pub fn serialize(&self) -> Result<Vec<u8>, OcioConfigError> {
        self.validate()?;
        let mut bytes = Vec::with_capacity(CONFIG_BYTES);
        bytes.extend_from_slice(&CONFIG_MAGIC);
        bytes.extend_from_slice(&self.schema_version.to_le_bytes());
        bytes.extend_from_slice(&self.algorithm_version.to_le_bytes());
        bytes.push(self.tone_mapping as u8);
        bytes.push(self.display_gamut as u8);
        bytes.push(self.transfer_function as u8);
        bytes.push(0);
        bytes.extend_from_slice(&self.exposure_stops.to_bits().to_le_bytes());
        for value in self.white_balance.gains() {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        debug_assert_eq!(bytes.len(), CONFIG_BYTES);
        Ok(bytes)
    }

    /// Decodes and validates a canonical config.
    pub fn deserialize(bytes: &[u8]) -> Result<Self, OcioConfigError> {
        if bytes.len() != CONFIG_BYTES {
            return Err(OcioConfigError::InvalidLength(bytes.len()));
        }
        if bytes[0..8] != CONFIG_MAGIC {
            return Err(OcioConfigError::InvalidMagic);
        }
        if bytes[17] != 0 {
            return Err(OcioConfigError::InvalidReservedByte(bytes[17]));
        }
        let schema_version = u16::from_le_bytes([bytes[8], bytes[9]]);
        let algorithm_version = u32::from_le_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]);
        let tone_mapping = OcioToneMapping::from_code(bytes[14])
            .ok_or(OcioConfigError::UnknownToneMapping(bytes[14]))?;
        let display_gamut = OcioDisplayGamut::from_code(bytes[15])
            .ok_or(OcioConfigError::UnknownDisplayGamut(bytes[15]))?;
        let transfer_function = OcioTransferFunction::from_code(bytes[16])
            .ok_or(OcioConfigError::UnknownTransferFunction(bytes[16]))?;
        let exposure_stops = f32::from_bits(u32::from_le_bytes([
            bytes[18], bytes[19], bytes[20], bytes[21],
        ]));
        let red_gain = read_f32(&bytes[22..26]);
        let green_gain = read_f32(&bytes[26..30]);
        let blue_gain = read_f32(&bytes[30..34]);
        let config = Self {
            schema_version,
            algorithm_version,
            tone_mapping,
            exposure_stops,
            white_balance: OcioWhiteBalance::new(red_gain, green_gain, blue_gain),
            display_gamut,
            transfer_function,
        };
        config.validate()?;
        Ok(config)
    }

    /// Computes a domain-separated digest of canonical serialized bytes.
    pub fn digest(&self) -> Result<OcioConfigDigest, OcioConfigError> {
        let bytes = self.serialize()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(DIGEST_DOMAIN);
        hasher.update(&bytes);
        Ok(OcioConfigDigest(*hasher.finalize().as_bytes()))
    }

    /// Applies the validated CPU reference transform to one linear HDR RGB.
    pub fn reference_transform(&self, input: [f32; 3]) -> Result<[f32; 3], OcioTransformError> {
        self.validate().map_err(OcioTransformError::InvalidConfig)?;
        if input.iter().any(|value| !value.is_finite()) {
            return Err(OcioTransformError::NonFiniteInput);
        }
        let exposure = 2.0_f32.powf(self.exposure_stops);
        let gains = self.white_balance.gains();
        let exposed = [
            finite_scale(input[0].max(0.0), gains[0], exposure),
            finite_scale(input[1].max(0.0), gains[1], exposure),
            finite_scale(input[2].max(0.0), gains[2], exposure),
        ];
        let mapped = exposed.map(|value| tone_map(value.min(MAX_TONEMAP_INPUT), self.tone_mapping));
        let gamut = gamut_matrix(self.display_gamut);
        let converted = [
            dot(gamut[0], mapped),
            dot(gamut[1], mapped),
            dot(gamut[2], mapped),
        ];
        Ok(converted.map(|value| transfer(value.max(0.0), self.transfer_function)))
    }
}

/// Errors raised while validating or decoding a config.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OcioConfigError {
    /// The byte slice does not have the fixed canonical size.
    InvalidLength(usize),
    /// The byte slice does not carry the Prism OCIO magic.
    InvalidMagic,
    /// The wire-format reserved byte was not zero.
    InvalidReservedByte(u8),
    /// The serialized schema is not supported.
    UnsupportedSchema(u16),
    /// The reference algorithm revision is not supported.
    UnsupportedAlgorithm(u32),
    /// A scalar is NaN or infinite.
    NonFinite(&'static str),
    /// A scalar lies outside its contract range.
    OutOfRange(&'static str),
    /// All white-balance gains are zero.
    DegenerateWhiteBalance,
    /// A tone-map code is not recognized.
    UnknownToneMapping(u8),
    /// A gamut code is not recognized.
    UnknownDisplayGamut(u8),
    /// A transfer-function code is not recognized.
    UnknownTransferFunction(u8),
}

impl fmt::Display for OcioConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength(length) => write!(formatter, "invalid OCIO config length {length}"),
            Self::InvalidMagic => formatter.write_str("invalid OCIO config magic"),
            Self::InvalidReservedByte(value) => {
                write!(formatter, "invalid OCIO reserved byte {value}")
            }
            Self::UnsupportedSchema(version) => {
                write!(formatter, "unsupported OCIO schema {version}")
            }
            Self::UnsupportedAlgorithm(version) => {
                write!(formatter, "unsupported OCIO algorithm {version}")
            }
            Self::NonFinite(field) => write!(formatter, "non-finite OCIO field {field}"),
            Self::OutOfRange(field) => write!(formatter, "out-of-range OCIO field {field}"),
            Self::DegenerateWhiteBalance => formatter.write_str("degenerate OCIO white balance"),
            Self::UnknownToneMapping(code) => write!(formatter, "unknown OCIO tone mapping {code}"),
            Self::UnknownDisplayGamut(code) => {
                write!(formatter, "unknown OCIO display gamut {code}")
            }
            Self::UnknownTransferFunction(code) => {
                write!(formatter, "unknown OCIO transfer function {code}")
            }
        }
    }
}

/// Errors raised by the reference transform.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OcioTransformError {
    /// The config failed validation.
    InvalidConfig(OcioConfigError),
    /// The input HDR pixel contains NaN or infinity.
    NonFiniteInput,
}

fn read_f32(bytes: &[u8]) -> f32 {
    f32::from_bits(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn finite_scale(value: f32, gain: f32, exposure: f32) -> f32 {
    let scaled = value * gain * exposure;
    if scaled.is_finite() {
        scaled
    } else {
        f32::MAX
    }
}

fn tone_map(value: f32, operator: OcioToneMapping) -> f32 {
    match operator {
        OcioToneMapping::Linear => value,
        OcioToneMapping::Aces => {
            let numerator = value * (2.51 * value + 0.03);
            let denominator = value * (2.43 * value + 0.59) + 0.14;
            (numerator / denominator).clamp(0.0, 1.0)
        }
        OcioToneMapping::Agx => {
            let numerator = value * (value + 0.024_578_6) - 0.000_090_537;
            let denominator = value * (0.983_729 * value + 0.432_951) + 0.238_081;
            (numerator / denominator).clamp(0.0, 1.0)
        }
    }
}

fn gamut_matrix(gamut: OcioDisplayGamut) -> [[f32; 3]; 3] {
    match gamut {
        OcioDisplayGamut::Srgb => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        OcioDisplayGamut::DisplayP3 => [
            [0.822_592_7, 0.177_533_0, 0.000_000_0],
            [0.033_199_5, 0.966_783_5, 0.000_000_0],
            [0.017_085_3, 0.072_395_7, 0.910_301_5],
        ],
        OcioDisplayGamut::Rec2020 => [
            [0.627_404_0, 0.329_282_0, 0.043_313_6],
            [0.069_097_0, 0.919_540_0, 0.011_361_2],
            [0.016_391_6, 0.088_013_2, 0.895_595_2],
        ],
    }
}

fn dot(row: [f32; 3], value: [f32; 3]) -> f32 {
    row[0] * value[0] + row[1] * value[1] + row[2] * value[2]
}

fn transfer(value: f32, function: OcioTransferFunction) -> f32 {
    let value = value.clamp(0.0, 1.0);
    match function {
        OcioTransferFunction::Linear => value,
        OcioTransferFunction::Srgb => {
            if value <= 0.003_130_8 {
                12.92 * value
            } else {
                1.055 * value.powf(1.0 / 2.4) - 0.055
            }
        }
        OcioTransferFunction::Gamma22 => value.powf(1.0 / 2.2),
        OcioTransferFunction::Gamma24 => value.powf(1.0 / 2.4),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(left: f32, right: f32) -> bool {
        (left - right).abs() <= 1e-5
    }

    #[test]
    fn canonical_round_trip_preserves_every_field() {
        let config = OcioConfig {
            schema_version: OCIO_CONFIG_SCHEMA_VERSION,
            algorithm_version: OCIO_CONFIG_ALGORITHM_VERSION,
            tone_mapping: OcioToneMapping::Agx,
            exposure_stops: 1.25,
            white_balance: OcioWhiteBalance::new(1.2, 0.9, 0.8),
            display_gamut: OcioDisplayGamut::DisplayP3,
            transfer_function: OcioTransferFunction::Gamma22,
        };
        let bytes = config.serialize().expect("valid config");
        assert_eq!(bytes.len(), CONFIG_BYTES);
        let decoded = OcioConfig::deserialize(&bytes).expect("canonical bytes decode");
        assert_eq!(decoded.schema_version, config.schema_version);
        assert_eq!(decoded.algorithm_version, config.algorithm_version);
        assert_eq!(decoded.tone_mapping, config.tone_mapping);
        assert_eq!(
            decoded.exposure_stops.to_bits(),
            config.exposure_stops.to_bits()
        );
        assert_eq!(decoded.white_balance, config.white_balance);
        assert_eq!(decoded.display_gamut, config.display_gamut);
        assert_eq!(decoded.transfer_function, config.transfer_function);
        assert_eq!(decoded.serialize().expect("re-serialize"), bytes);
    }

    #[test]
    fn digest_is_stable_and_changes_with_configuration() {
        let first = OcioConfig::aces();
        let second = OcioConfig::aces();
        assert_eq!(
            first.digest().expect("digest"),
            second.digest().expect("digest")
        );
        let mut changed = first;
        changed.exposure_stops = 1.0;
        assert_ne!(
            first.digest().expect("digest"),
            changed.digest().expect("digest")
        );
        assert_eq!(first.version().schema, OCIO_CONFIG_SCHEMA_VERSION);
        assert_eq!(first.version().algorithm, OCIO_CONFIG_ALGORITHM_VERSION);
    }

    #[test]
    fn invalid_values_are_rejected_before_serialization_or_transform() {
        let mut invalid = OcioConfig::srgb();
        invalid.exposure_stops = f32::NAN;
        assert_eq!(
            invalid.validate(),
            Err(OcioConfigError::NonFinite("exposure_stops"))
        );
        assert!(matches!(
            invalid.serialize(),
            Err(OcioConfigError::NonFinite(_))
        ));
        assert!(matches!(
            invalid.reference_transform([1.0, 1.0, 1.0]),
            Err(OcioTransformError::InvalidConfig(
                OcioConfigError::NonFinite(_)
            ))
        ));

        let mut invalid = OcioConfig::srgb();
        invalid.white_balance = OcioWhiteBalance::new(0.0, 0.0, 0.0);
        assert_eq!(
            invalid.validate(),
            Err(OcioConfigError::DegenerateWhiteBalance)
        );

        let mut bytes = OcioConfig::srgb().serialize().expect("valid config");
        bytes[14] = 255;
        assert_eq!(
            OcioConfig::deserialize(&bytes),
            Err(OcioConfigError::UnknownToneMapping(255))
        );
    }

    #[test]
    fn reference_transform_is_deterministic_and_responds_to_controls() {
        let input = [4.0, 1.0, 0.25];
        let config = OcioConfig {
            tone_mapping: OcioToneMapping::Aces,
            exposure_stops: 1.0,
            white_balance: OcioWhiteBalance::new(1.1, 1.0, 0.9),
            display_gamut: OcioDisplayGamut::Rec2020,
            transfer_function: OcioTransferFunction::Srgb,
            ..OcioConfig::srgb()
        };
        let first = config.reference_transform(input).expect("valid transform");
        let second = config.reference_transform(input).expect("valid transform");
        assert_eq!(first, second);
        assert!(first.iter().all(|value| (0.0..=1.0).contains(value)));
        let neutral = OcioConfig::srgb()
            .reference_transform([0.18, 0.18, 0.18])
            .expect("valid transform");
        assert!(approx(neutral[0], neutral[1]));
        assert!(approx(neutral[1], neutral[2]));
        assert_ne!(first, neutral);
    }

    #[test]
    fn transfer_functions_and_gamuts_are_finite() {
        for tone_mapping in [
            OcioToneMapping::Linear,
            OcioToneMapping::Aces,
            OcioToneMapping::Agx,
        ] {
            for display_gamut in [
                OcioDisplayGamut::Srgb,
                OcioDisplayGamut::DisplayP3,
                OcioDisplayGamut::Rec2020,
            ] {
                for transfer_function in [
                    OcioTransferFunction::Linear,
                    OcioTransferFunction::Srgb,
                    OcioTransferFunction::Gamma22,
                    OcioTransferFunction::Gamma24,
                ] {
                    let config = OcioConfig {
                        tone_mapping,
                        display_gamut,
                        transfer_function,
                        ..OcioConfig::srgb()
                    };
                    let output = config
                        .reference_transform([0.0, 0.18, 16.0])
                        .expect("valid transform");
                    assert!(output.iter().all(|value| value.is_finite()));
                    assert!(output.iter().all(|value| (0.0..=1.0).contains(value)));
                }
            }
        }
    }

    #[test]
    fn extreme_finite_hdr_input_stays_finite_and_display_bounded() {
        let config = OcioConfig {
            tone_mapping: OcioToneMapping::Linear,
            exposure_stops: MAX_EXPOSURE_STOPS,
            white_balance: OcioWhiteBalance::new(MAX_WHITE_BALANCE_GAIN, 1.0, 1.0),
            display_gamut: OcioDisplayGamut::Rec2020,
            transfer_function: OcioTransferFunction::Gamma24,
            ..OcioConfig::srgb()
        };
        let output = config
            .reference_transform([f32::MAX, f32::MAX, f32::MAX])
            .expect("finite HDR input is valid");
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output.iter().all(|value| (0.0..=1.0).contains(value)));
    }
}
