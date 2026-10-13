use prism_asset::StableGuid;
use prism_asset_build_core::{AssetId, AssetPackageMapping};
use std::fmt;

const HEADER: &str = "version=1";

#[derive(Debug)]
pub enum MappingError {
    Empty,
    Header,
    Line { number: usize, message: String },
}

impl fmt::Display for MappingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("mapping must contain version=1 and at least one record"),
            Self::Header => f.write_str("mapping must start with version=1"),
            Self::Line { number, message } => write!(f, "mapping line {number}: {message}"),
        }
    }
}

impl std::error::Error for MappingError {}

pub fn parse(input: &str) -> Result<Vec<AssetPackageMapping>, MappingError> {
    let mut lines = input.lines();
    if lines.next() != Some(HEADER) {
        return if input.is_empty() {
            Err(MappingError::Empty)
        } else {
            Err(MappingError::Header)
        };
    }
    let mut mappings = Vec::new();
    for (index, line) in lines.enumerate() {
        let number = index + 2;
        if line.is_empty() {
            continue;
        }
        let mapping = parse_record(line, number)?;
        if mappings
            .iter()
            .any(|existing: &AssetPackageMapping| existing.asset == mapping.asset)
        {
            return err(number, format!("duplicate asset {}", mapping.asset.0));
        }
        mappings.push(mapping);
    }
    if mappings.is_empty() {
        return Err(MappingError::Empty);
    }
    Ok(mappings)
}

fn parse_record(line: &str, number: usize) -> Result<AssetPackageMapping, MappingError> {
    let expected = ["asset", "guid", "container", "priority", "chunk_priority"];
    let fields = line.split(';').collect::<Vec<_>>();
    if fields.len() != expected.len() {
        return err(number, "expected five ordered fields");
    }
    let mut values = [""; 5];
    for (index, field) in fields.iter().enumerate() {
        let Some((name, value)) = field.split_once('=') else {
            return err(number, "field must use name=value");
        };
        if name != expected[index] {
            return err(number, format!("expected field {:?}", expected[index]));
        }
        if value.is_empty() {
            return err(number, format!("field {name:?} is empty"));
        }
        values[index] = value;
    }
    Ok(AssetPackageMapping {
        asset: AssetId(parse_u128(values[0], number, "asset")?),
        guid: parse_guid(values[1], number)?,
        container_name: safe_name(values[2], number, "container")?,
        priority: parse_i32(values[3], number, "priority")?,
        chunk_priority: parse_u32(values[4], number, "chunk_priority")?,
    })
}

fn parse_guid(value: &str, number: usize) -> Result<StableGuid, MappingError> {
    if let Some(path) = value.strip_prefix("path:") {
        return Ok(StableGuid::from_path(&safe_path(path, number)?));
    }
    if let Some(value) = value.strip_prefix("sub:") {
        let Some((path, label)) = value.rsplit_once('#') else {
            return err(number, "sub guid must be sub:path#label");
        };
        let path = safe_path(path, number)?;
        if label.is_empty()
            || label.contains(['#', '/', '\\'])
            || label.chars().any(char::is_control)
        {
            return err(number, "sub guid label is unsafe");
        }
        return Ok(StableGuid::derive_sub(StableGuid::from_path(&path), label));
    }
    if let Some(hex) = value.strip_prefix("raw:") {
        if hex.len() != 32 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return err(number, "raw guid must be 32 hexadecimal digits");
        }
        let guid = StableGuid::from_u128(u128::from_str_radix(hex, 16).unwrap());
        return if guid.is_nil() {
            err(number, "raw guid cannot be nil")
        } else {
            Ok(guid)
        };
    }
    err(number, "guid must use path:, sub:, or raw:")
}

fn safe_path(path: &str, number: usize) -> Result<String, MappingError> {
    if path.is_empty()
        || path.starts_with(['/', '\\'])
        || path.contains(':')
        || path.as_bytes().contains(&0)
    {
        return err(number, "path must be a non-empty project-relative path");
    }
    let mut parts = Vec::new();
    for part in path.split(['/', '\\']) {
        if part.is_empty() || part == "." {
            continue;
        }
        if part == ".." || part.chars().any(char::is_control) {
            return err(number, "path traversal or control character is not allowed");
        }
        parts.push(part);
    }
    if parts.is_empty() {
        return err(number, "path must contain a segment");
    }
    Ok(parts.join("/"))
}

fn safe_name(value: &str, number: usize, field: &str) -> Result<String, MappingError> {
    if value == "."
        || value == ".."
        || value.contains(['/', '\\', ':'])
        || value.as_bytes().contains(&0)
        || value.chars().any(char::is_control)
    {
        return err(number, format!("{field} must be one safe relative name"));
    }
    Ok(value.to_owned())
}

fn parse_u128(value: &str, number: usize, field: &str) -> Result<u128, MappingError> {
    if !decimal(value) {
        return err(number, format!("{field} must be canonical decimal"));
    }
    value.parse().map_err(|_| MappingError::Line {
        number,
        message: format!("{field} is out of range"),
    })
}

fn parse_u32(value: &str, number: usize, field: &str) -> Result<u32, MappingError> {
    Ok(parse_u128(value, number, field)?
        .try_into()
        .map_err(|_| MappingError::Line {
            number,
            message: format!("{field} is out of range"),
        })?)
}

fn parse_i32(value: &str, number: usize, field: &str) -> Result<i32, MappingError> {
    let digits = value.strip_prefix('-').unwrap_or(value);
    if !decimal(digits) || (value.starts_with('-') && digits == "0") {
        return err(number, format!("{field} must be canonical decimal"));
    }
    value.parse().map_err(|_| MappingError::Line {
        number,
        message: format!("{field} is out of range"),
    })
}

fn decimal(value: &str) -> bool {
    !value.is_empty()
        && value.bytes().all(|byte| byte.is_ascii_digit())
        && (value == "0" || !value.starts_with('0'))
}

fn err<T>(number: usize, message: impl Into<String>) -> Result<T, MappingError> {
    Err(MappingError::Line {
        number,
        message: message.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_multiple_records_and_normalizes_paths() {
        let mappings = parse("version=1\nasset=6;guid=sub:./textures\\hero.png#source;container=core;priority=0;chunk_priority=0\nasset=7;guid=path:textures/hero.png;container=core;priority=0;chunk_priority=1\n").unwrap();
        assert_eq!(mappings.len(), 2);
        assert_eq!(mappings[1].guid, StableGuid::from_path("textures/hero.png"));
    }

    #[test]
    fn rejects_unknown_duplicate_missing_and_unsafe_fields() {
        for record in [
            "asset=7;guid=path:a.png;container=core;priority=0",
            "asset=7;guid=path:a.png;guid=path:b.png;container=core;priority=0;chunk_priority=1",
            "asset=7;extra=x;container=core;priority=0;chunk_priority=1",
            "asset=7;guid=path:../a.png;container=core;priority=0;chunk_priority=1",
        ] {
            assert!(parse(&format!("version=1\n{record}\n")).is_err());
        }
    }
}
