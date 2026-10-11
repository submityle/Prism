# 项目目录结构规则

## 1. 文件与目录命名规范

| 层级 | 规则 | 示例 |
|---|---|---|
| 顶层非 Rust 目录 | kebab-case（`-`） | `os-backends`、`docs`、`examples`、`benches`、`tools`；每 crate 另设 `text/`（与 `src/` 平级，见 §8） |
| Rust crate / 包目录 | snake_case（`_`）+ `prism_` 前缀 | `prism_render_driver`、`prism_ecs` |
| `src/` 内模块文件 | snake_case（`_`） | `archetype.rs` |
| 设计文档 | `<topic>_design_zh.md` | `prism_ecs_design_zh.md` |
| 平台后端 crate（`os-backends/`） | `<能力>_<后端>` | `prism_render_driver_wgpu` |
| 每版本变更文档 | `v<MAJOR>.<MINOR>.<PATCH>.md` | `v0.1.0.md` |

## 2. 顶级目录结构

```
crates/        # 上游 Bevy 原版保留层，仅承载暂留/永久保留项；随重构逐步退役
pkg/          # Prism 自研层，所有新能力以 prism_* crate 组织
os-backends/  # 平台/后端实现层，prism_*_<backend> 形态
```

- 新能力一律进 `pkg/`；`crates/` 只做保留与退役，禁止在 `crates/` 内新增 Prism 逻辑。
- `pkg/` 与 `crates/` 不得散落直接耦合。

| 目录 | 职责 |
|---|---|
| `pkg/` `crates/` `os-backends/` | 见上 |
| `docs/` | 引擎级设计/重构/移植文档 |
| `examples/` | 示例工程 |
| `benches/` | 基准测试 |
| `tests/` | 不再承载测试；跨 crate 集成测试统一收口到各 crate `text/integration/`（§8） |
| `assets/` | 示例资源，不参与构建 |
| `tools/` | 开发工具/CLI |
| `.github/` | CI workflow |

## 3. pkg 包规范

每个 `pkg/<crate>/` 必须：
1. 含 `README.md`，与 `src/` 平级。
2. 含 `docs/` 目录（设计文档 + 变更文档，§5）。
3. 为库 crate（`src/lib.rs`，非二进制）；二进制入口放 `prism_*_app` 或 `tools/`。
4. `Cargo.toml` 的 `name` = `prism_<domain>[_<sub>]`，与目录名一致。
5. 单一职责：一个 crate 一个能力。

```
pkg/prism_render_driver/
├── Cargo.toml
├── README.md
├── src/lib.rs            # 库入口，禁止含测试（§8）
├── docs/
│   ├── prism_render_driver_design_zh.md
│   └── version/{README.md, v0.1.0.md}
└── text/                 # 开发/测试隔离层（§8）
    ├── reference/        # 【条件选设】CPU 参考实现
    ├── unit/             # 【必设】单元测试（镜像 src/ 布局）
    └── integration/      # 【必设】集成测试
```

## 5. 文档约定

### 5.1 设计文档
- 命名 `<crate|topic>_design_zh.md`，中文撰写。
- 头部固定：版本号 + 适用引擎。
- 状态标记（其一）：`SHIPPED`（已实现通过测试）/ `PLANNED`（规划中）/ `前置待建`（依赖未具备）。
- 铁律：严格区分已实现与规划，不把未落地能力描述为已落地；提交按 `§章节` 引用设计规范。
- 须含「多维质量评估」段（§12），覆盖八维并写明取舍与未覆盖项。

### 5.2 每版本增量变更文档（crate 级 `version/`）
1. 位置：`pkg/<crate>/docs/version/`。
2. 命名：`v<MAJOR>.<MINOR>.<PATCH>.md`（三段式，对齐 crate 版本）；禁连字符、禁 `CHANGELOG-` 前缀。
3. 索引：`version/README.md` 按时间倒序列出版本链接 + 摘要。
4. 增量性：每版本新增一个文件，绝不改写历史；索引随发版追加，不删旧条目。
5. 单文件结构（固定）：
   ```markdown
   # <crate> v<MAJOR>.<MINOR>.<PATCH>
   ## 范围
   - 影响 crate：<列表>
   - 里程碑：M0–M4 之一
   - 对应设计：<topic>_design_zh.md §x.y
   ### Added / Changed / Deprecated / Removed / Fixed
   ### 破坏性变更（有则单列+迁移指引；无则写「无破坏性变更」）
   ```
6. 边界：仅记已落地变更；未发布/规划中写入设计文档 `PLANNED` 段；`## 范围` 须回链设计文档 §章节。
7. 触发与校验：每次发版或合并含 semver 变更的 PR，必须补 `v<MAJOR>.<MINOR>.<PATCH>.md`；CI 校验 `docs/version/` 存在且最新版本号 == crate 版本。

### 5.3 文档类型
| 文档 | 位置 | 记录 |
|---|---|---|
| 设计文档 `*_design_zh.md` | `pkg/<crate>/docs/` 或 `docs/` | 架构/契约/状态 |
| 重构/移植计划 `*_refactor_zh.md` / `*_plan_zh.md` | `docs/` | 切片队列与进度 |
| 每版本变更 `version/v<MAJOR>.<MINOR>.<PATCH>.md` | `pkg/<crate>/docs/version/` | 本次已落地差异 |
| 主题文档 | `pkg/<crate>/docs/` | 架构/路线图等 |

## 6. 代码规范

### 6.1 注释与 API 生成
- 公开项（`pub fn`/`struct`/`trait`/字段/宏）顶部用 `///` 文档注释。
- 模块文件顶部用 `//!` 说明职责。
- 注释讲「为什么」而非「是什么」。

### 6.2 no_std + alloc 地基
- 基础 crate 必须 `no_std + alloc`；仅 app/二进制层用 `std`。
- 后端无关类型系统不引入 `unsafe` 外的平台假设；平台相关代码收口 `os-backends/`。

### 6.3 测试与 Clippy
- 测试统一收口 `text/`（§8）：`src/` 禁 `#[cfg(test)]` 内联模块。
- CI 要求 Clippy 零告警：`cargo clippy --workspace --all-targets --all-features -- -D warnings`。

### 6.4 三件套闭环交付
每新 crate 须齐备，未达三件套不得合入 `main`：
1. 实现（`src/`，不含测试）；
2. 测试（`text/unit`、`text/integration`）；
3. 文档（设计文档标状态 + `docs/version/` 首份版本）。

## 7. 编码规范（结构性契约）

> 工具可强制项以 `docs/linters.md` 为准，不在本文重复。

### 7.1 错误处理
- 公开 API 错误统一返回 `Result<T, E>`；错误类型用 `thiserror`；禁止 `Box<dyn Error>` 作公开签名。
- 禁止静默吞错：`let _ = result` 须附注释说明可安全忽略。
- 跨 crate 错误转换经 `From`，不在调用点 `unwrap`。

### 7.2 unsafe 管控
- 后端无关/地基层零 `unsafe`。
- `unsafe` 仅允许 `os-backends/` 等平台绑定层，且：局部化到最小 `unsafe {}` 块、配 `// SAFETY:` 注释、附最小单元测试。
- 禁止以 `unsafe` 绕过借用检查赶进度。

### 7.3 禁止桩代码
- 公开 API 禁止 `todo!()` / `unimplemented!()`。
- 临时桩仅允许分支开发期、带 `// FIXME(<issue>):`，不得合入 `main`。

### 7.4 panic 纪律
- 库代码不主动 `panic`；仅启动期/文档化 precondition 可用 `expect`（带说明）。
- 热路径（帧循环/ECS 调度/渲染提交）无 panic、无分配、无锁竞争。

### 7.5 日志与追踪
- 统一 `tracing`，禁止 `println!`/`eprintln!`/`dbg!` 留入 `main`。
- 性能敏感路径用 `tracing::event!` 且受 feature 门控。

### 7.6 并发与 Send + Sync
- 跨线程句柄须 `Send + Sync`；共享状态经 `Mutex`/`RwLock`/无锁结构收口，热点不持锁。
- `no_std` 层不得跨线程依赖 `Rc`/`RefCell`。

### 7.7 命名与格式化
- 命名：类型 `UpperCamelCase`、函数/变量 `snake_case`、常量 `SCREAMING_SNAKE_CASE`、宏 `snake_case!`/`CamelCase!`。
- 以 `docs/linters.md` 为准：`cargo fmt --all`；Clippy `-D warnings` 零告警。

### 7.8 Feature flags
- 以 `docs/cargo_features.md` 为准；默认 feature 最小化，`std` 由后端 crate 按需开启；不暴露矛盾 feature 组合。

### 7.9 文档注释与 API 生成
- 公开项全覆盖 `///`；缺失即不合规。
- crate 根开启 `#![deny(missing_docs)]`；CI `cargo doc --no-deps --all-features -D warnings` 校验。
- `src/lib.rs` 顶部 `//!` 说明职责/入口/用法；每个 `src/*.rs` 顶部 `//!` 说明模块职责。
- 注释为合法 Markdown，非平凡 API 含摘要 + `##` 段或 ` ```rust ` 示例。
- doctest：` ```rust ` 示例须可编译/运行；CI `cargo test --doc`。

## 8. CPU 参考实现与测试隔离（text/ 隔离层）

### 8.0 适用范围
| 子规则 | 强制范围 |
|---|---|
| 测试隔离（§8.3–§8.5） | 全局强制，每 crate |
| CPU 参考实现 golden-twin（§8.1–§8.2） | 条件适用，仅当 crate 同时有 CPU 参考 + GPU 实现 |

### 8.1 语义分层（条件适用）
- 仅当同一能力同时有 CPU + GPU 实现时生效：CPU 实现定位开发/测试/对照基准（golden twin），GPU 为生产真实路径。
- CPU 实现仅用于：无 GPU 的 headless 渲染、parity 校验、CI 确定性测试；不得进入生产装配路径。
- GPU 实现（`feature = "gpu"` 门控）为真实路径；无适配器时 best-effort 跳过（`try_new` 返回 `None`）。

### 8.2 text/ 隔离目录（每 crate 与 src/ 平级）
- `text/unit/`、`text/integration/` 全局必设；`text/reference/` 条件选设（仅 §8.1 适用时）。
- CPU 参考实现统一放 `text/reference/`，不在 `src/` 暴露为生产 API；确需引用经 `#[path]`/`feature`/`cfg(test)` 门控，生产装配不缺省启用。
- 子目录 snake_case；`text/` 不进入 `no_std` 生产构建。

### 8.3 测试收口 text/，禁散落 src/
- `src/` 下禁止任何测试（含文件末尾 `#[cfg(test)] mod tests`、任何 `#[test]`）。
- 测试独立文件置于 `text/unit/`、`text/integration/`，目录镜像 `src/` 布局。
- 接入（显式其一）：
  1. `[[test]]` 目标：`Cargo.toml` 声明 `[[test]] name = "x" path = "text/unit/x.rs"`（推荐）；
  2. `#[path]` 引入：`#[cfg(test)] #[path = "text/unit/x.rs"] mod x_tests;`（测试体在 `text/`，不在 `src/*.rs`）。
- 仓库级跨 crate 集成测试置于根 `text/integration/`，不再用顶层 `tests/`。

### 8.4 测试须注释
- 测试文件/函数须含注释，讲清被测不变量、构造意图、预期边界。
- `text/` 下导出测试辅助 API 同样 `#![deny(missing_docs)]` + `///`。
- parity/golden-twin 测试须注释对照关系与公差。

### 8.5 校验（CI）
- 扫描 `src/**` 出现 `#[cfg(test)]`/`#[test]` 即失败（白名单除外）。
- 含 `text/reference/` 的 crate，扫描其是否被生产 `src/` 缺省依赖（非 `feature`/`cfg(test)` 不可见即告警）；不含则跳过。
- `cargo test --workspace` 覆盖 `text/` 全部测试目标；`cargo test --doc` 不受影响。

## 9. Cargo.toml / Workspace 契约

> 各 crate 元数据以根 `[workspace.package]` 为唯一事实来源，crate 一律继承。

### 9.1 版本号
- 禁止 crate 写独立 `version = "x.y.z"`；必须 `version.workspace = true` 继承根版本（当前 `0.20.0-dev`）。
- 发版只改根版本，全工作区统一推进，无 crate 单独升版本。
- CI：发现独立 `version =` 即失败。

### 9.2 其余继承项
- `edition`：统一 `2024`。
- `license`：统一 `MIT OR Apache-2.0`，`license.workspace = true`。
- `resolver`：根统一 `"3"`。
- 共享依赖（如 `wgpu`）优先 `workspace = true` 继承；

### 9.3 校验（CI）
- 无 crate 声明独立 `version`/`edition`/`license`；均 `.workspace = true`。
- 根 `version` 为全工作区唯一版本号，与各 crate `docs/version/` 最新文档版本一致。

## 10. 模块文件组织
- 优先 `src/foo.rs` + `src/foo/` 子目录，弃用 `src/foo/mod.rs`。
- `src/lib.rs` 为 crate 根；子模块 `mod foo;`（非 `mod foo/mod.rs;`）。
- 模块粒度对齐单一职责，避免巨型单文件。
- `text/` 下文件沿用同布局，与 `src/` 镜像。

## 11. 代码审查（CR）要求

### 11.1 自动门禁（CI 全绿，否则 CR 不通过）
- Clippy 零告警（§6.3/§7.7）。
- API 文档可生成、`missing_docs` 无告警（§7.9）。
- 测试收口 `text/`，`src/**` 无 `#[cfg(test)]`/`#[test]`（§8.3）；`cargo test --workspace` + `cargo test --doc` 通过。
- Workspace 契约：无独立 `version`/`edition`/`license`（§9）。
- 版本变更文档：含 semver 变更的 PR 须新增 `v<MAJOR>.<MINOR>.<PATCH>.md`，最新版本号 == 工作区版本（§5.2）。

### 11.2 三件套闭环（§6.4）
- 实现 + 测试（`text/`）+ 文档齐备；缺一则不得合入 `main`。

### 11.3 变更质量
- 禁桩代码合入 `main`（§7.3）。
- 状态诚实：SHIPPED/PLANNED 不混淆（§5.1）。
- 破坏性变更须在 version 文档单列迁移指引（§5.2）。

### 11.4 目录与命名合规
- 分层命名（§1）、`text/` 隔离（§8）、模块组织（§10）。

### 11.5 审查流程
- 每 PR 至少 1 reviewer；涉及 `os-backends`/跨边界改动须 owner 批准。
- 大迁移可分批多 PR，但每 PR self-contained、CI 不回归、不引入新违规。
- 提交流程/分支/CODEOWNERS 以 `CONTRIBUTING.md` 为准。

### 11.6 多维质量评估（§12）
- 核对设计与实现是否按 §12 八维论证、设计文档含评估段（§5.1）；高风险维度须具体证据：热路径性能（基准/无分配证明）、`unsafe` 安全（`// SAFETY` 完备）、破坏性变更迁移成本。

## 12. 设计与 CR 的多维质量评估

> 设计与 CR 均须从八维显式评估；维度取舍须在设计文档写明并获 reviewer 确认。

| 维度 | 须论证 | 锚点 |
|---|---|---|
| 性能 | 热路径零分配/无锁/无panic；批处理/instancing；帧预算；缓存友好 | §7.4；`prism_ui_render_backend` batch |
| 效果 | 正确性；保真 | §8 parity；`prism_render_driver` |
| 易用性 | API ergonomics；合理默认值；doctest；Plugin 装配简单 | §7.9 |
| 易维护性 | 单一职责；no_std；注释讲 why；无散落耦合 | §3；§6.2；§7.9 |
| 易拓展性 | trait/feature 扩展点；新后端=新 crate 不改 facade；ABI 稳定 | `RenderBackend`；§7.8；§9.2；`src/abi` |
| 易迭代性 | 增量文档；迁移指引；分批迁移；feature 门控 | §5.2；§11.5；§7.8 |
| 健壮性 | Result；无静默吞错；best-effort 获取；确定性测试 | §7.1；§7.4；`try_new` |
| 安全 | unsafe 局部化+//SAFETY；输入输出校验；依赖审计；无 UB | §7.2；§9.2 |

### 12.1 取舍记录（设计文档评估段须含）
1. 逐维现状：满足/部分/不适用 + 依据。
2. 关键取舍与理由：偏向哪维、牺牲哪维、为何可接受。
3. 已知未覆盖项 + 后续迭代计划（记 `PLANNED` 段或后续 `version/`）。

### 12.2 CR 重点质疑
- 热路径性能：基准/无分配证明，是否违反 §7.4。
- `unsafe` 安全：`// SAFETY` 完备、前置条件可证（§7.2）。
- 破坏性变更：迁移成本是否在 version 文档列明（§5.2）。

### 12.3 适用范围
- 适用于所有 `pkg/`、`os-backends/` 新能力设计及其 CR。
- 纯文档/工具改动可简化，但不得省略「安全」与「健壮性」。
