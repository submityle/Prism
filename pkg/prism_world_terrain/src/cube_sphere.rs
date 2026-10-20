//! Cube-sphere face coordinates and deterministic direction mapping.

/// The six oriented faces of the cube used by the cube-sphere.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum CubeFace {
    /// Positive X face.
    PositiveX = 0,
    /// Negative X face.
    NegativeX = 1,
    /// Positive Y face.
    PositiveY = 2,
    /// Negative Y face.
    NegativeY = 3,
    /// Positive Z face.
    PositiveZ = 4,
    /// Negative Z face.
    NegativeZ = 5,
}

impl CubeFace {
    /// Faces in their stable serialization order.
    pub const ALL: [Self; 6] = [
        Self::PositiveX,
        Self::NegativeX,
        Self::PositiveY,
        Self::NegativeY,
        Self::PositiveZ,
        Self::NegativeZ,
    ];

    /// Returns the stable numeric face index.
    pub const fn index(self) -> u8 {
        self as u8
    }
}

/// A point in a face-local cube coordinate, where both coordinates are in
/// the closed interval `[-1, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceUv {
    face: CubeFace,
    u: f32,
    v: f32,
}

impl FaceUv {
    /// Creates a validated face-local coordinate.
    pub fn new(face: CubeFace, u: f32, v: f32) -> Result<Self, CubeSphereError> {
        if !u.is_finite() || !v.is_finite() {
            return Err(CubeSphereError::NonFinite);
        }
        if !(-1.0..=1.0).contains(&u) || !(-1.0..=1.0).contains(&v) {
            return Err(CubeSphereError::FaceCoordinateOutOfRange);
        }
        Ok(Self { face, u, v })
    }

    /// Returns the face.
    pub const fn face(self) -> CubeFace {
        self.face
    }

    /// Returns the local horizontal coordinate.
    pub const fn u(self) -> f32 {
        self.u
    }

    /// Returns the local vertical coordinate.
    pub const fn v(self) -> f32 {
        self.v
    }
}

/// A normalized direction on the unit sphere.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitDirection {
    components: [f32; 3],
}

impl UnitDirection {
    /// Returns the normalized Cartesian components.
    pub const fn components(self) -> [f32; 3] {
        self.components
    }
}

/// Errors returned by cube-sphere coordinate conversion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CubeSphereError {
    /// An input contained a NaN or infinity.
    NonFinite,
    /// A face-local coordinate was outside `[-1, 1]`.
    FaceCoordinateOutOfRange,
    /// A direction had zero length.
    ZeroDirection,
}

/// Maps a face-local cube coordinate to a normalized sphere direction.
pub fn face_uv_to_direction(face_uv: FaceUv) -> UnitDirection {
    let [x, y, z] = match face_uv.face {
        CubeFace::PositiveX => [1.0, face_uv.v, -face_uv.u],
        CubeFace::NegativeX => [-1.0, face_uv.v, face_uv.u],
        CubeFace::PositiveY => [face_uv.u, 1.0, -face_uv.v],
        CubeFace::NegativeY => [face_uv.u, -1.0, face_uv.v],
        CubeFace::PositiveZ => [face_uv.u, face_uv.v, 1.0],
        CubeFace::NegativeZ => [-face_uv.u, face_uv.v, -1.0],
    };
    normalize([x, y, z])
}

/// Maps a non-zero Cartesian direction to a deterministic face-local point.
pub fn direction_to_face_uv(direction: [f32; 3]) -> Result<FaceUv, CubeSphereError> {
    if direction.iter().any(|component| !component.is_finite()) {
        return Err(CubeSphereError::NonFinite);
    }
    let direction = normalize_result(direction)?;
    let [x, y, z] = direction.components;
    let ax = x.abs();
    let ay = y.abs();
    let az = z.abs();
    let (face, u, v) = if ax >= ay && ax >= az {
        if x >= 0.0 {
            (CubeFace::PositiveX, -z / x, y / x)
        } else {
            let denominator = -x;
            (CubeFace::NegativeX, z / denominator, y / denominator)
        }
    } else if ay >= az {
        if y >= 0.0 {
            (CubeFace::PositiveY, x / y, -z / y)
        } else {
            let denominator = -y;
            (CubeFace::NegativeY, x / denominator, z / denominator)
        }
    } else if z >= 0.0 {
        (CubeFace::PositiveZ, x / z, y / z)
    } else {
        let denominator = -z;
        (CubeFace::NegativeZ, x / z, y / denominator)
    };
    FaceUv::new(face, u, v)
}

fn normalize(direction: [f32; 3]) -> UnitDirection {
    normalize_result(direction).expect("face coordinates always produce a non-zero direction")
}

fn normalize_result(direction: [f32; 3]) -> Result<UnitDirection, CubeSphereError> {
    let scale = direction
        .iter()
        .map(|component| component.abs())
        .fold(0.0_f32, f32::max);
    if scale == 0.0 {
        return Err(CubeSphereError::ZeroDirection);
    }
    let scaled = [
        direction[0] / scale,
        direction[1] / scale,
        direction[2] / scale,
    ];
    let length = sqrt_f32(scaled[0] * scaled[0] + scaled[1] * scaled[1] + scaled[2] * scaled[2]);
    Ok(UnitDirection {
        components: [scaled[0] / length, scaled[1] / length, scaled[2] / length],
    })
}

fn sqrt_f32(value: f32) -> f32 {
    let mut estimate = if value > 1.0 { value } else { 1.0 };
    for _ in 0..8 {
        estimate = 0.5 * (estimate + value / estimate);
    }
    estimate
}
