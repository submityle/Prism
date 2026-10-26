//! 蓝图值与类型系统。
//!
//! 与 `wasmtime::component::Val` 对齐，便于接入宿主 WASM ABI。

use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// 蓝图值类型（编译期）。
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValueType {
    Bool,
    I32,
    I64,
    F32,
    F64,
    String,
    Json,
    List(Box<ValueType>),
    /// 仅编辑期允许；编译期必须解析为具体类型。
    Wildcard,
    /// 控制信号（exec pin 携带）。
    Exec,
}

impl ValueType {
    /// 是否能由 `from` 隐式转换到 `self`。
    pub fn accepts(&self, from: &ValueType) -> bool {
        if self == from {
            return true;
        }
        matches!(
            (self, from),
            (_, ValueType::Wildcard)
                | (ValueType::Wildcard, _)
                | (ValueType::F64, ValueType::I32)
                | (ValueType::F64, ValueType::I64)
                | (ValueType::F64, ValueType::F32)
                | (ValueType::I64, ValueType::I32)
                | (ValueType::Json, _)
                | (ValueType::String, ValueType::I32)
                | (ValueType::String, ValueType::I64)
                | (ValueType::String, ValueType::F32)
                | (ValueType::String, ValueType::F64)
                | (ValueType::String, ValueType::Bool)
        )
    }

    pub fn default_value(&self) -> Value {
        match self {
            ValueType::Bool => Value::Bool(false),
            ValueType::I32 => Value::I32(0),
            ValueType::I64 => Value::I64(0),
            ValueType::F32 => Value::F32(0.0),
            ValueType::F64 => Value::F64(0.0),
            ValueType::String => Value::String(Arc::from("")),
            ValueType::Json => Value::Json(Arc::new(serde_json::Value::Null)),
            ValueType::List(_) => Value::List(Arc::new(Vec::new())),
            ValueType::Wildcard | ValueType::Exec => Value::Unit,
        }
    }
}

/// 运行期值。小标量内联，大对象用 `Arc` 共享。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Value {
    Unit,
    Bool(bool),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    String(Arc<str>),
    Json(Arc<serde_json::Value>),
    List(Arc<Vec<Value>>),
}

impl Value {
    pub fn ty(&self) -> ValueType {
        match self {
            Value::Unit => ValueType::Exec,
            Value::Bool(_) => ValueType::Bool,
            Value::I32(_) => ValueType::I32,
            Value::I64(_) => ValueType::I64,
            Value::F32(_) => ValueType::F32,
            Value::F64(_) => ValueType::F64,
            Value::String(_) => ValueType::String,
            Value::Json(_) => ValueType::Json,
            Value::List(xs) => {
                let inner = xs.first().map(|v| v.ty()).unwrap_or(ValueType::Wildcard);
                ValueType::List(Box::new(inner))
            }
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        if let Value::Bool(b) = self {
            Some(*b)
        } else {
            None
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::I32(x) => Some(*x as i64),
            Value::I64(x) => Some(*x),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::I32(x) => Some(*x as f64),
            Value::I64(x) => Some(*x as f64),
            Value::F32(x) => Some(*x as f64),
            Value::F64(x) => Some(*x),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Value::String(s) = self {
            Some(s)
        } else {
            None
        }
    }

    /// 隐式转换：把 `self` 转为目标类型 `to`，失败返回原值。
    pub fn coerce(self, to: &ValueType) -> Result<Value, Value> {
        if &self.ty() == to {
            return Ok(self);
        }
        match (to, &self) {
            (ValueType::F64, _) => self.as_f64().map(Value::F64).ok_or(self),
            (ValueType::I64, _) => self.as_i64().map(Value::I64).ok_or(self),
            (ValueType::String, _) => Ok(Value::String(Arc::from(format_value(&self)))),
            (ValueType::Json, _) => Ok(Value::Json(Arc::new(to_json_value(&self)))),
            _ => Err(self),
        }
    }
}

fn format_value(v: &Value) -> String {
    match v {
        Value::Unit => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::I32(x) => x.to_string(),
        Value::I64(x) => x.to_string(),
        Value::F32(x) => x.to_string(),
        Value::F64(x) => x.to_string(),
        Value::String(s) => s.to_string(),
        Value::Json(j) => j.to_string(),
        Value::List(_) => format!("{:?}", v),
    }
}

fn to_json_value(v: &Value) -> serde_json::Value {
    match v {
        Value::Unit => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::I32(x) => serde_json::Value::Number(serde_json::Number::from(*x)),
        Value::I64(x) => serde_json::Value::Number(serde_json::Number::from(*x)),
        Value::F32(x) => serde_json::Number::from_f64(*x as f64)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Value::F64(x) => serde_json::Number::from_f64(*x)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Value::String(s) => serde_json::Value::String(s.to_string()),
        Value::Json(j) => (**j).clone(),
        Value::List(xs) => serde_json::Value::Array(xs.iter().map(to_json_value).collect()),
    }
}
