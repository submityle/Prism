//! 内置节点：事件 / 流程控制 / 数学 / 比较 / 调试 / 变量。

mod compare;
mod debug;
mod events;
mod flow;
mod math;
mod pins;
mod vars;

use crate::registry::NodeLibrary;

pub fn register_all(lib: &mut NodeLibrary) {
    events::register(lib);
    flow::register(lib);
    math::register(lib);
    compare::register(lib);
    debug::register(lib);
    vars::register(lib);
}
