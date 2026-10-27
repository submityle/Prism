//! 百万级调用优化 Demo：暴力方案 vs 批量方案性能对比。
//!
//! # 根本问题（160 秒的原因）
//!
//! 原始代码 N=1,000,000 unique 租户：
//!
//! ```text
//! 1M × register()         → 1M write lock 到 tenants HashMap
//! 1M × install_for_async()→ 1M write lock 到 current HashMap
//! 1M × call_async()       → 1M spawn_blocking，512 线程排队
//! 1M × JoinSet::spawn()   → 1M tokio 任务，~200B × 1M = 200MB 纯队列
//! ```
//!
//! # 正确姿势
//!
//! 实际业务中是「少数租户 × 大量请求」，而非「百万租户 × 一次请求」：
//!
//! ```text
//! 旧: 1,000,000 租户  × 1 次调用 = 1,000,000 spawn_blocking
//! 新:          20 租户 × 50,000 次调用
//!            = ceil(50000 / chunk) 个 spawn_blocking per tenant
//!            = 少量 spawn_blocking，每个内部顺序执行 chunk 次 WASM
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use prism_wasm::{no_wasi_ctx, Attestor, Policy, Sandbox, SandboxEngine, SandboxRegistry};
use tokio::task::JoinSet;

fn make_component(op: &str) -> anyhow::Result<Vec<u8>> {
    let wat = format!(
        r#"(component
             (core module $m
               (func (export "op") (param i32 i32) (result i32)
                 local.get 0 local.get 1 {op}))
             (core instance $i (instantiate $m))
             (func (export "compute") (param "x" s32) (param "y" s32) (result s32)
               (canon lift (core func $i "op"))))"#
    );
    Ok(wat::parse_str(&wat)?)
}

/// 普通沙箱（沿用旧方式，inherit_stdio）
fn make_sandbox_legacy(name: &str, engine: Arc<SandboxEngine>) -> Sandbox {
    let policy = Policy::builder()
        .fuel(1_000_000)
        .deadline(Duration::from_millis(500))
        .memory_pages(16)
        .build();
    Sandbox::new(name, engine, policy, Arc::new(Attestor::ephemeral()))
}

/// 优化沙箱：省去每次调用的 3 次 dup() 系统调用
fn make_sandbox_fast(name: &str, engine: Arc<SandboxEngine>) -> Sandbox {
    let policy = Policy::builder()
        .fuel(1_000_000)
        .deadline(Duration::from_millis(500))
        .memory_pages(16)
        .build();
    Sandbox::new(name, engine, policy, Arc::new(Attestor::ephemeral()))
        .with_wasi_ctx_fn(no_wasi_ctx) // 纯计算：省去 dup()
}

// ── 方案A：暴力（每次调用独占一个 spawn_blocking）────────────────────────

async fn bench_naive(
    registry: &Arc<SandboxRegistry>,
    tenants: &[String],
    calls_per_tenant: usize,
) -> anyhow::Result<(f64, usize)> {
    let total = tenants.len() * calls_per_tenant;
    let t = Instant::now();
    let mut set: JoinSet<anyhow::Result<()>> = JoinSet::new();
    let mut idx = 0usize;
    for tenant in tenants {
        for j in 0..calls_per_tenant {
            let reg = registry.clone();
            let t_name = tenant.clone();
            let args = (idx as i32, j as i32);
            idx += 1;
            set.spawn(async move {
                reg.call_async::<(i32, i32), (i32,)>(&t_name, "calc", "compute", args)
                    .await?;
                Ok(())
            });
        }
    }
    while let Some(r) = set.join_next().await {
        r??;
    }
    Ok((t.elapsed().as_secs_f64() * 1000.0, total))
}

// ── 方案B：批量（N/chunk 个 spawn_blocking，每个顺序执行 chunk 次 WASM）─

async fn bench_batched(
    registry: &Arc<SandboxRegistry>,
    tenants: &[String],
    calls_per_tenant: usize,
    chunk: usize,
) -> anyhow::Result<(f64, usize)> {
    let total = tenants.len() * calls_per_tenant;
    let t = Instant::now();

    // 每个租户独立并发，每个租户内部按 chunk 分批
    let mut set: JoinSet<anyhow::Result<()>> = JoinSet::new();
    for (ti, tenant) in tenants.iter().enumerate() {
        let reg = registry.clone();
        let t_name = tenant.clone();
        let all_args: Vec<(i32, i32)> = (0..calls_per_tenant)
            .map(|j| ((ti * calls_per_tenant + j) as i32, j as i32))
            .collect();

        set.spawn(async move {
            for chunk_args in all_args.chunks(chunk) {
                let batch = chunk_args.to_vec();
                let results = reg
                    .call_many_async::<(i32, i32), (i32,)>(&t_name, "calc", "compute", batch)
                    .await?;
                for r in results {
                    r?;
                }
            }
            Ok(())
        });
    }
    while let Some(r) = set.join_next().await {
        r??;
    }
    Ok((t.elapsed().as_secs_f64() * 1000.0, total))
}

// ── 主程序 ────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ── 你的原始参数 ──────────────────────────────────────────────────
    // const N: usize = 1_000_000; // ← 这是根本问题：1M 唯一租户 × 1 次调用
    //                                  = 1M spawn_blocking + 1M write lock
    //
    // 正确架构：少量租户 × 大量请求
    const TENANTS: usize = 20;
    const CALLS_PER_TENANT: usize = 1_000_000 / TENANTS; // 50,000 次/租户
    const TOTAL_CALLS: usize = TENANTS * CALLS_PER_TENANT; // 仍是 1,000,000 次

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "  总调用量: {} 次 ({} 租户 × {} 次/租户)",
        TOTAL_CALLS, TENANTS, CALLS_PER_TENANT
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // ── 1. 搭建引擎 + 注册表 ──────────────────────────────────────────
    let engine = Arc::new(SandboxEngine::new()?);
    let registry_slow = Arc::new(SandboxRegistry::new(engine.clone()));
    let registry_fast = Arc::new(SandboxRegistry::new(engine.clone()));

    let add_bytes = make_component("i32.add")?;
    let tenants: Vec<String> = (0..TENANTS).map(|i| format!("tenant_{i}")).collect();

    // 注册 + 安装（两套注册表分别用 legacy / fast 沙箱）
    let mut install_set: JoinSet<anyhow::Result<()>> = JoinSet::new();
    for name in &tenants {
        let reg_s = registry_slow.clone();
        let reg_f = registry_fast.clone();
        let eng = engine.clone();
        let n = name.clone();
        let bytes = add_bytes.clone();
        install_set.spawn(async move {
            reg_s.register(make_sandbox_legacy(&n, eng.clone()));
            reg_f.register(make_sandbox_fast(&n, eng.clone()));
            // 两个注册表共享同一个 engine，编译只发生一次
            reg_s.install_for_async(&n, "calc", bytes.clone()).await?;
            reg_f.install_for_async(&n, "calc", bytes).await?;
            Ok(())
        });
    }
    while let Some(r) = install_set.join_next().await {
        r??;
    }
    println!("✓ {} 个租户注册 + 安装完成\n", TENANTS);

    // ── 2. 预热（消除首次 InstancePre 构建的影响）────────────────────
    bench_batched(&registry_fast, &tenants, 200, 200).await?;

    // ── 3. 方案对比 ───────────────────────────────────────────────────
    println!("─── 方案 A: 暴力 (每次调用一个 spawn_blocking) ───");
    println!("    spawn_blocking 数量: {}", TOTAL_CALLS);
    // 只测 500 次避免等太久，然后外推
    let sample = 500usize;
    let (sample_ms, _) = bench_naive(&registry_slow, &tenants[..1], sample / TENANTS + 1).await?;
    let extrapolated_ms = sample_ms / sample as f64 * TOTAL_CALLS as f64;
    println!(
        "    样本 {} 次: {:.1} ms → 外推 {} 次: ~{:.0} ms  ({:.0} calls/sec)",
        sample,
        sample_ms,
        TOTAL_CALLS,
        extrapolated_ms,
        sample as f64 / (sample_ms / 1000.0)
    );

    println!("\n─── 方案 B: 批量 + no_wasi_ctx ───");
    for &chunk in &[100usize, 1000, 5000] {
        let (ms, total) = bench_batched(&registry_fast, &tenants, CALLS_PER_TENANT, chunk).await?;
        let spawns = TENANTS * ((CALLS_PER_TENANT + chunk - 1) / chunk);
        println!(
            "    chunk={:5}, spawn_blocking 数: {:5}, 耗时: {:8.1} ms  ({:>9.0} calls/sec)  加速 ~{:.0}x",
            chunk,
            spawns,
            ms,
            total as f64 / (ms / 1000.0),
            extrapolated_ms / ms
        );
    }

    // ── 4. 结果验证（抽样） ───────────────────────────────────────────
    println!("\n─── 正确性验证 ───");
    let verify_args: Vec<(i32, i32)> = (0..100).map(|i| (i, i)).collect();
    let results = registry_fast
        .call_many_async::<(i32, i32), (i32,)>("tenant_0", "calc", "compute", verify_args)
        .await?;
    for (i, r) in results.into_iter().enumerate() {
        let v = r?.returns.0;
        assert_eq!(v, (i as i32) * 2, "tenant_0 index {i} 结果错误: {v}");
    }
    println!("✓ 100 次抽样验证通过 (i+i=2i)");

    // ── 5. 根因与修复方案 ────────────────────────────────────────────
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  N=1,000,000 唯一租户 耗时 160,874 ms 的根因");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("① 1M × Sandbox::new()      → 1M 堆分配（~400B/个 = 400MB）");
    println!("② 1M × registry.register() → 1M write lock（序列化）");
    println!("③ 1M × install_for_async() → 1M write lock 到 current HashMap");
    println!("④ 1M × call_async()        → 1M spawn_blocking（512 线程 × 排队 ~2000 轮）");
    println!("⑤ 1M × JoinSet::spawn()    → 1M tokio 任务 + 调度开销");
    println!("");
    println!("修复：");
    println!("  租户数 20 + call_many_async(chunk=1000)");
    println!(
        "  → spawn_blocking: {} → {}",
        TOTAL_CALLS,
        TENANTS * ((CALLS_PER_TENANT + 999) / 1000)
    );
    println!(
        "  → no_wasi_ctx: 省去 {} 次 dup() 系统调用",
        TOTAL_CALLS * 3
    );

    Ok(())
}
