//! 多沙箱（多租户）示例。
//!
//! 三个租户共用一个 [`SandboxEngine`]：
//!   - alice  ：调用加法 component
//!   - bob    ：调用乘法 component
//!   - carol  ：调用 *同一份* 加法字节，验证「同摘要只编译一次」
//!
//! 通过 [`SandboxRegistry`] 隔离命名空间：实际安装的 component 名是
//! `tenant::module`，租户彼此看不到对方的逻辑名。

use std::sync::Arc;
use std::time::Duration;

use prism_wasm::{Attestor, Policy, Sandbox, SandboxEngine, SandboxRegistry};

fn component_with(op: &str) -> anyhow::Result<Vec<u8>> {
    let wat = format!(
        r#"(component
             (core module $m
               (func (export "op") (param i32 i32) (result i32)
                 local.get 0 local.get 1 {op}))
             (core instance $i (instantiate $m))
             (func (export "compute") (param "a" s32) (param "b" s32) (result s32)
               (canon lift (core func $i "op"))))"#
    );
    Ok(wat::parse_str(&wat)?)
}

fn make_sandbox(name: &str, engine: Arc<SandboxEngine>, fuel: u64) -> Sandbox {
    let policy = Policy::builder()
        .fuel(fuel)
        .deadline(Duration::from_millis(500))
        .memory_pages(64)
        .build();
    Sandbox::new(name, engine, policy, Arc::new(Attestor::ephemeral()))
}

fn main() -> anyhow::Result<()> {
    let engine = Arc::new(SandboxEngine::new()?);
    let registry = SandboxRegistry::new(engine.clone());

    // 注册三个租户，各自有不同燃料预算（演示策略隔离）。
    registry.register(make_sandbox("alice", engine.clone(), 1_000_000));
    registry.register(make_sandbox("bob", engine.clone(), 5_000_000));
    registry.register(make_sandbox("carol", engine.clone(), 2_000_000));

    // 安装：alice 与 carol 共享同一份字节（同摘要 → 仅编译一次）。
    let add_bytes = component_with("i32.add")?;
    let mul_bytes = component_with("i32.mul")?;
    let d_a = registry.install_for("alice", "calc", add_bytes.clone())?;
    let d_b = registry.install_for("bob", "calc", mul_bytes)?;
    let d_c = registry.install_for("carol", "calc", add_bytes)?;
    println!("alice 摘要 = {}", hex::encode(d_a));
    println!("bob   摘要 = {}", hex::encode(d_b));
    println!("carol 摘要 = {}", hex::encode(d_c));
    println!("alice == carol? {}", d_a == d_c);

    // 各自调用：彼此命名空间隔离，但底层 InstancePre 在同摘要间共享。
    let ra = registry.call::<(i32, i32), (i32,)>("alice", "calc", "compute", (10, 20))?;
    let rb = registry.call::<(i32, i32), (i32,)>("bob", "calc", "compute", (10, 20))?;
    let rc = registry.call::<(i32, i32), (i32,)>("carol", "calc", "compute", (10, 20))?;
    println!("alice(10,20) = {:?}  // 加法应得 30", ra.returns);
    println!("bob  (10,20) = {:?}  // 乘法应得 200", rb.returns);
    println!("carol(10,20) = {:?}  // 加法应得 30", rc.returns);

    println!("当前租户列表 = {:?}", registry.tenants());
    Ok(())
}
