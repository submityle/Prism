use glam::{Quat, Vec3};

pub(crate) fn finite(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

pub(crate) fn non_negative(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

pub(crate) fn unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

pub(crate) fn vec3(value: Vec3) -> Vec3 {
    Vec3::new(finite(value.x), finite(value.y), finite(value.z))
}

pub(crate) fn normalize(value: Vec3, fallback: Vec3) -> Vec3 {
    let value = vec3(value);
    let scale = value.abs().max_element();
    if scale > 1.0e-6 {
        let scaled = value / scale;
        scaled / scaled.length()
    } else {
        fallback
    }
}

/// Saturating addition keeps one overflowing contribution from erasing others.
pub(crate) fn add(a: Vec3, b: Vec3) -> Vec3 {
    let component = |a: f32, b: f32| {
        (f64::from(finite(a)) + f64::from(finite(b)))
            .clamp(-f64::from(f32::MAX), f64::from(f32::MAX)) as f32
    };
    Vec3::new(
        component(a.x, b.x),
        component(a.y, b.y),
        component(a.z, b.z),
    )
}

pub(crate) fn sub(a: Vec3, b: Vec3) -> Vec3 {
    let component = |a: f32, b: f32| {
        (f64::from(finite(a)) - f64::from(finite(b)))
            .clamp(-f64::from(f32::MAX), f64::from(f32::MAX)) as f32
    };
    Vec3::new(
        component(a.x, b.x),
        component(a.y, b.y),
        component(a.z, b.z),
    )
}

pub(crate) fn cross(a: Vec3, b: Vec3) -> Vec3 {
    let component = |a: f32, b: f32, c: f32, d: f32| {
        (f64::from(finite(a)) * f64::from(finite(b)) - f64::from(finite(c)) * f64::from(finite(d)))
            .clamp(-f64::from(f32::MAX), f64::from(f32::MAX)) as f32
    };
    Vec3::new(
        component(a.y, b.z, a.z, b.y),
        component(a.z, b.x, a.x, b.z),
        component(a.x, b.y, a.y, b.x),
    )
}

pub(crate) fn clamp_magnitude(value: Vec3, maximum: f32) -> Vec3 {
    let value = vec3(value);
    let maximum = non_negative(maximum);
    if maximum == 0.0 {
        return Vec3::ZERO;
    }
    let scale = value.abs().max_element();
    if scale <= 1.0e-12 {
        return value;
    }
    let unit_value = value / scale;
    let unit_length = unit_value.length();
    if !unit_length.is_finite() || scale <= maximum / unit_length {
        value
    } else {
        unit_value * (maximum / unit_length)
    }
}

pub(crate) fn rotation(value: Quat) -> Quat {
    let value = Quat::from_xyzw(
        finite(value.x),
        finite(value.y),
        finite(value.z),
        finite(value.w),
    );
    let length_squared = value.length_squared();
    if length_squared.is_finite() && length_squared > 1.0e-12 {
        value.normalize()
    } else {
        Quat::IDENTITY
    }
}

pub(crate) fn smoothstep01(value: f32) -> f32 {
    let value = unit(value);
    value * value * (3.0 - 2.0 * value)
}
