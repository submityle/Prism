//! 内置 pin 构造助手。

use crate::model::{Pin, PinDir};
use crate::value::{Value, ValueType};

pub fn exec_in() -> Pin {
    Pin {
        name: "exec_in".into(),
        dir: PinDir::In,
        ty: ValueType::Exec,
        default: None,
    }
}
pub fn exec_out(name: &str) -> Pin {
    Pin {
        name: name.into(),
        dir: PinDir::Out,
        ty: ValueType::Exec,
        default: None,
    }
}
pub fn data_in(name: &str, ty: ValueType, default: Option<Value>) -> Pin {
    Pin {
        name: name.into(),
        dir: PinDir::In,
        ty,
        default,
    }
}
pub fn data_out(name: &str, ty: ValueType) -> Pin {
    Pin {
        name: name.into(),
        dir: PinDir::Out,
        ty,
        default: None,
    }
}
