# Prism 次世代设置系统设计方案（Settings 2.0）

> 面向 Prism（后 Bevy 时代独立引擎）的 **设置内核 + 分层解析 + 事务化应用 + 持久化/迁移 + 编辑器/系统级作用域** 设计。它把设置从“一个内存 key-value + cvar 校验层”提升为引擎的一等契约：一份 schema 同时约束代码、文件、UI、控制台与远程；解析是纯函数；写入是事务；持久化有版本；失败有诊断；启动有 Bootstrap；治理有策略锁。
>
> 借形态不抄码。主要参考：
> - **显示确认/回滚、画质档、设备档**：Unreal Engine（GameUserSettings、RequestResolutionChange → ConfirmVideoMode / RevertVideoMode、Scalability、DeviceProfiles）
> - **作用域与来源检查**：VS Code（ConfigurationScope、`inspect()`、Restricted Mode、Settings Sync 排除 machine）
> - **策略优先级与受保护偏好**：Chromium（PrefValueStore：MANAGED > SUPERVISED > EXTENSION > COMMAND_LINE > USER > RECOMMENDED > DEFAULT；PrefHashStore；Local State 与 Profile 分离）
> - **系统级 schema/锁/可写性**：GSettings / dconf（schema、lock、`is_writable`）
> - **drop-in 片段与来源追踪**：systemd（`*.d/*.conf`、`cat-config`）、Git（`--show-origin`）
> - **项目覆盖与编辑器偏好分离**：Godot（`override.cfg`、EditorSettings）、Unity（SettingsProvider 的 User/Project scope）
> - **命名配置与确定性合并**：Bazel（`--config`）、Cargo（profile 继承）
> - **不可信项目配置**：Vim modeline / Emacs safe-local-variable（只借“防执行”，不借其机制）
>
> 本文为**设计稿**，不代表已实施；不含任何 AI/ML 内容。

- 版本: v1.5（v1.4 补操作级 journal 与恢复协议；v1.5 明确未发布 API 全量替换）
- 基线: `main@e248f12ec`（写作时）
- 状态: 实施中；M0 核心已落地并通过当前门禁，M0.5–M5 必须逐阶段完成后才算系统交付；完成度以代码、测试、运行行为与质量门禁共同证明
- 适用: 运行时 / 编辑器 / 机器级 / 构建管线
- 关键依赖: `prism_reflect`（元数据/版本/迁移/校验）、`prism_platform`（目录/文件/监视）、`prism_ui_reactive`、`prism_ui_i18n`、`prism_ui_form`、`prism_app`（应用集成；不承载旧设置 API 兼容层）
- 明确约束: 核心 `no_std + alloc`；文件 I/O 需 `std`；**不依赖任何 `bevy_*` crate**

**v1.4 变更摘要**

- 新增 §9.2.6 操作级 journal：`ApplyOp`、每 op 状态/补偿账本/`effect_handle`、恢复决策树与崩溃示例。
- §9.2.5 applier 契约改为 `apply_op/probe/compensate` 操作级接口。
- 全量替换未发布的 `prism_app::settings/cvar` API；不提供旧类型别名、行为兼容 facade 或旧文本格式承诺。
- 当前实现：`pkg/prism_settings` 已提供 no_std + alloc 的 Value/Def/Registry/Resolver/Snapshot/Inspect/Diff 核心；10k 定义、9 层解析基准在当前参考机 p50 约 4.05 ms、p95 约 4.32 ms、p99 约 4.64 ms。持久化、事务应用、编辑器和 `prism_app` 集成仍按后续里程碑推进，未完成部分不得宣称已交付。
- §9.2.3 明确 fsync 边界；§9.2.4 补 `CompensationUnknown`；不变量 19–20、§21 崩溃注入矩阵、§22 M3 验收同步。
- v1.3 的四点修复保留。

**v1.3 变更摘要**

- 修复里程碑信任边界：`TrustLevel` 数据模型进 M0、强制执行进 M2，M4 只做信任提示/管理。
- §9 事务改为三级承诺（状态原子 / 持久化可恢复 / 外部副作用分级），增加 journal、Degraded/NeedsRestart/RecoveryRequired 与崩溃恢复；不再承诺“全成功或全回滚”。
- 新增 §5.7 模型最小化验证与 M0.5 门禁：12–20 个真实键的完整优先级表 + 属性测试，未通过不进入 M1。
- §25.2.7 补参考环境、构建参数、统计方法、基准实现与跨机归一化。
- 同步改写不变量 8、§22 里程碑、§24 #5；v1.2 的 14 项裁决与 M3.5 保留。

---

## 目录

1. 设计哲学与目标
2. 现状与差距
3. 参考产品与取舍
4. 总体架构
5. 域与层
6. 核心模型
7. 类型、元数据、派生
8. 校验与约束
9. 应用语义
10. 持久化
11. 设置 schema 版本与迁移
12. 变更传播与可观测
13. UI / UX
14. 安全与信任
15. 并发、多世界、网络
16. no_std 与平台
17. 性能预算
18. API 草案
19. crate 布局
20. 不变量
21. 测试与验收
22. 里程碑
23. 风险与边界
24. 决策记录（原开放问题）
25. 质量属性工程
附录 A. 术语表
附录 B. 与现有 API 的映射
附录 C. 迁移算子
附录 D. 参考文档
附录 E. 质量门禁与评审清单
附录 F. 演进与升级 playbook

---

## 1. 设计哲学与目标

### 1.1 问题

现有设置内核能跑，但只覆盖“引擎/游戏运行时的少量标量”。把它放到次世代引擎的完整生命周期里，会暴露八个结构性问题：

1. **只活在内存**：`Settings` 不随 `App::new` 安装，也没有文件 I/O；重启即失。
2. **只有四种标量**：Bool / Int / Float / Str；枚举、数组、表（按键、分辨率、颜色）无处安放。
3. **两套键空间**：`Settings` 的字符串键与 `CvarRegistry` 的 cvar 各自生长，容易分裂。
4. **没有迁移**：加字段、改名、改单位即破坏旧配置；降级更无行为定义。
5. **没有应用语义**：哪些立即生效、哪些要重启、哪些要重建交换链，全靠子系统各自约定。
6. **没有启动前设置**：窗口、渲染器、日志在 `App` 之前创建，拿不到用户值。
7. **没有编辑器/项目/机器级模型**：编辑器偏好、项目设置、机器级策略混在一起，或干脆缺失。
8. **没有来源解释**：用户、客服、QA 都说不清“为什么最终是这个值”。

### 1.2 目标

1. **单一 schema**：`SettingDef` 是代码、文件、UI、控制台、文档的共同真相源。
2. **任意域与层**：运行时、编辑器、机器级、构建各自挂载源，共享同一 registry。
3. **类型安全可迁移**：富值模型、稳定枚举 ID、每域版本与迁移链。
4. **事务化应用**：校验 → 预检 → 应用 → 提交/回滚；可预览、可确认、可恢复。
5. **启动即用**：Bootstrap 子集先于 `App`/窗口/渲染器。
6. **默认安全**：项目配置是不可信数据；密钥不落明文；策略锁不可绕过。
7. **可观测可解释**：`inspect`/`explain`/`diff`/`journal`/`doctor` 是一等能力。
8. **编辑器与系统级原生支持**：不是外挂，而是四个域中的两个。
9. **面向未发布版本直接演进**：不承诺兼容现有 `prism_app::settings/cvar` API；调用方统一迁移到 schema 驱动的新接口。

### 1.3 非目标

- 存档系统（SaveGame）——设置不是存档，不能混。
- 通用密钥管理器——只做 `SecretRef` 与后端接口。
- 网络复制协议——只定义“哪些键可同步”的子集与校验。
- 资产管线依赖——设置不反向依赖资产系统。
- 旧 `crates/bevy_settings` 迁移/桥接——`pkg/` 不消费老栈，Bevy 正在被整体替换；不设迁移，必要时只做核心包之外的一次性导入器。
- 云同步服务端实现——本期只留 trait 与本地策略。

### 1.4 七条原则

1. **Schema 是代码，值是可覆盖的叶子。** 没有 schema 的键只是数据，不是设置。
2. **分层不销毁。** 高层覆盖低层；清除即回落，永远能回答“下一层是什么”。
3. **解析是纯函数。** 给定源与 registry，结果确定、可哈希、可复现。
4. **写入是事务。** 校验 → 预检 → 应用 → 提交/回滚；不允许半应用。
5. **失败是数据。** 结构化诊断，绝不 panic；损坏文件隔离而非拖垮启动。
6. **持久化键永远有定义。** 未知键保留但不参与解析；高版本文件只读已知键。
7. **UI 不拥有真相。** UI 只读 registry、写指定 scope；生效值永远由 resolver 决定。

---

## 2. 现状与差距

### 2.1 现有设置内核

`pkg/prism_app/src/settings.rs`（约 619 行）：

- `SettingsLayer` 五层：`EngineDefault → PlatformTier → User → CommandLine → Runtime`，`Ord` 即优先级。
- `SettingValue`：`Bool / Int / Float / Str`；`parse` 按 `true/false → i64 → f64 → str` 推断。
- `Settings`：`BTreeMap<String, BTreeMap<Layer, Value>>`；`get` 取最高层；`set`/`clear` 返回 `Option<SettingChange>`。
- `SettingChanged = SettingChange` 事件；`App` 有 `init_settings` / `insert_setting` / `clear_setting` / `apply_cli_overrides` / `setting*` 辅助。
- `Settings` **不随 `App::new` 安装**，是 opt-in。

### 2.2 现有 cvar 层

`pkg/prism_app/src/cvar.rs`（约 2117 行）：

- `CvarCategory` 九类：Render / Audio / Network / Input / Physics / Debug / Gameplay / System / General。
- `CvarFlags`：`CHEAT=1<<0`、`READ_ONLY=1<<1`、`ARCHIVE=1<<2`、`NOTIFY=1<<3`。
- `CvarBounds`：None / Int(i64,i64) / Float(f64,f64)。
- `CvarSpec`：name、category、default、bounds、flags、description、aliases。
- `register_cvar` 把默认值种入 `EngineDefault`；`set_cvar_at` 校验/钳制/权限后写层并可能发 `CvarChanged`。
- `CvarRegistry`：BTreeMap + `cheats_enabled` + 别名解析；`list_cvars`、`ConsoleOutcome`、`exec_console_script`。
- `load_user_config` 把 `name value` 行写入 User 层；`write_user_config` 仅输出 User 层 `ARCHIVE` cvar；`write_archive_config` 输出解析值；两者返回 `String`，**本 crate 不做文件 I/O**。
- 本质：cvar = Settings key + 前置 schema。

### 2.3 现有派生层

- `pkg/prism_app/src/reflect_config.rs`（约 1087 行）：从 `prism_reflect` 的 `#[reflect(default/range/clamp/rename/category)]` 字段元数据派生 `CvarSpec`；`register_reflected_cvars` 先克隆 registry 校验整批，再一次性提交。
- `pkg/prism_app/src/platform_tier.rs`（约 293 行）：五个键 `render.present`、`pacing.frame_limit_fps`、`pipeline.pipelined_rendering`、`loop.power_aware_variable_step`、`lifecycle.suspend_resume`，写入 `PlatformTier` 层。
- `pkg/prism_app/src/hot_reload.rs`（约 203 行）：`ConfigHotReload` 用内容哈希轮询，变化则走 `load_user_config` 重放。
- `prism_app` features：`default std`；`multi_thread` / `partition` / `pipelined` / `pipelined-tasks` / `determinism` / `winit` / `reflect` / `trace` / `hot-reload` / `module-reload` / `diagnostic`。

### 2.4 可复用底座

| 能力 | 位置 | 用法 |
|---|---|---|
| 类型元数据 | `pkg/prism_reflect/src/schema/metadata.rs` | 字段默认值/范围/别名/分类 |
| 版本与迁移 | `pkg/prism_reflect/src/schema/version.rs`、`migration.rs` | `SchemaVersion`、`Migration`、`MigrateError` |
| 校验/钳制 | `pkg/prism_reflect/src/schema/validate.rs`、`clamp.rs` | 数值/结构校验 |
| 版本化序列化 | `pkg/prism_reflect/src/schema/serde_integration.rs` | `to_versioned_binary` / `ron` |
| 目录 | `pkg/prism_platform/src/fs/dirs.rs` | home/config/data/cache/temp/exe/current |
| 文件 | `pkg/prism_platform/src/fs/file.rs` | read/write/rename/metadata/OpenOptions |
| 监视 | `pkg/prism_platform/src/fs/watch.rs` | 热重载 |
| 响应式 | `pkg/prism_ui_reactive` | signal/memo/effect/introspect |
| 国际化 | `pkg/prism_ui_i18n` | Catalog/PluralRules/lookup |
| 表单 | `pkg/prism_ui_form` | Form/FieldId/Validator |
| 工作区存储 | `pkg/prism_ui_dock/src/workspace_store.rs` | primary/last-good、原子写、文件锁、迁移 |

### 2.5 六个缺口

| 缺口 | 现状 | 后果 |
|---|---|---|
| 持久化与迁移 | 无文件 I/O、无版本 | 重启即失；改 schema 即破坏 |
| 富类型与约束 | 四种标量 | 枚举/数组/表只能塞字符串 |
| 应用语义 | 无 apply 概念 | 子系统各自处理“立即/重启” |
| 启动前设置 | `App` 之后才有 | 窗口/渲染器拿不到用户值 |
| 编辑器/系统级 | 无域模型 | 项目设置、机器级策略无处安放 |
| UI 与治理 | 无 schema→表单 | 手写 UI、文案分散、策略不可表达 |

### 2.6 未发布 API 的替换策略

当前 `prism_app::settings` 与 `cvar` 尚未发布，因此直接删除旧公开类型与入口，不维护兼容 facade、旧事件适配器或旧控制台语义。`prism_settings` 成为 schema、解析、写事务与诊断的唯一内核；应用层通过新 plugin/resource/command 接入。所有仓内调用方与测试在同一变更中迁移，禁止仅为让旧测试通过而保留旧 API。

持久化文件的 schema 迁移与 Rust API 兼容是两件事：已发行文件格式仍按 §11 迁移；旧的内存 API、未发行文本配置格式和旧调用路径不构成兼容承诺。若未来确有外部数据导入需求，只能增加一次性、默认关闭的导入工具，不将其重新变成运行时 API。

---

## 3. 参考产品与取舍

| 产品 | 借 | 不借 |
|---|---|---|
| **Unreal** | 显示确认/回滚、画质档、DeviceProfiles、项目/用户文件分离 | ini 巨型级联、配置即代码 |
| **VS Code** | scope 分类、`inspect()`、Restricted Mode、同步排除 machine | JSONC 唯一格式、pref 膨胀 |
| **Chromium** | Mandatory/Recommended 顺序、controlling source、Local State vs Profile、PrefHash | pref/flag 历史债 |
| **Firefox** | `user.js` 覆盖、locked 策略、StaticPrefs | 全部进 `about:config` |
| **GSettings/dconf** | schema + lock + `is_writable` | 强平台依赖、二进制后端 |
| **systemd / Git** | drop-in 片段、`cat-config` / `--show-origin` | 无 schema 的纯文本覆盖 |
| **Blender** | `--factory-startup`、prefs 与 startup 分离 | 二进制设置文件 |
| **Godot / Unity** | `override.cfg`、User/Project scope | 单文件合并冲突、PlayerPrefs |
| **Vim / Emacs** | 项目配置是不可信输入 | 配置执行代码 |
| **Bazel / Cargo** | 命名配置、profile 继承 | 无限层级/复杂表达式 |
| **NixOS** | 类型化 option + 文档生成 | 配置语言、求值模型 |

**结论**：借“作用域、来源、策略、确认、迁移、片段、诊断”的形态；不借“配置语言、单文件巨型级联、平台绑定、执行代码”。

---

## 4. 总体架构

### 4.1 分层图

```
消费者
  ├─ Runtime 子系统（render/audio/input/window/network）
  ├─ Editor（Preferences / Project Settings / System Settings / PIE）
  ├─ 工具（控制台、CI、doctor、远程运维）
  └─ UI（FormTree、搜索、预览对话框）
        │
Registry（启动期构建、冻结为 Arc）
  SettingDef · PresetDef · FlagDef · DomainSpec · Migration · UiMeta
        │
Resolver（纯函数）
  按域挂载源 → 选层 → Recommended 兜底 → 能力约束 → 计算 DAG → 校验 → 快照
        │
Sources
  Builtin · Project · PlatformProbe · Remote · User · Device · Machine
  · Session(CLI/env) · Runtime(console) · Policy · Observation(OS)
        │
Persistence
  runtime profiles · editor prefs · project files · machine.toml
  · policy.toml · remote cache · boot_state · workspace store（独立）
```

### 4.2 三条链路

- **写**：UI/控制台/API → 目标层 → 校验/钳制 → 标脏 → 候选快照 → 事务应用 → 持久化 + diff。
- **读**：`Arc<Snapshot>`（跨线程/子应用）或 `SettingsReader<T>`（主线程）→ O(1) 读 + generation 比较。
- **启动**：Bootstrap 子集（Machine + Policy + CLI + 少量 User）→ 安全模式判断 → 构建 registry → 加载各源 → 迁移/校验 → 解析 → 应用启动关键项 → App → 异步拉远程/云。

### 4.3 十五条核心决策

| # | 决策 | 选择 |
|---|---|---|
| 1 | 真相源 | 代码内 `SettingDef` 单一 schema，驱动运行时/持久化/UI/文档 |
| 2 | 值模型 | 动态 `Value` 树 + 稳定枚举 ID，不用 `TypeId`/ordinal 落盘 |
| 3 | 域模型 | Runtime / Editor / System / Build 四个域，层栈各自独立 |
| 4 | 分层 | 每域 2–9 层；Policy 分 **Mandatory（置顶锁）/ Recommended（低于 User）** |
| 5 | 每设置策略 | `PrecedenceProfile`（UserChoiceWins / ServerWins / DeviceScoped / Locked） |
| 6 | 持久化 | 分域版本化文件（TOML/JSON）+ 原子写 + 未知键保留 + `.bak` |
| 7 | 覆盖 | `ProjectLocal` + `settings.d/*.toml` 字典序片段 |
| 8 | 应用 | `ApplyGroup` + 分级事务（状态原子 / 持久化 journal / 副作用补偿），不承诺普遍回滚 |
| 9 | 预览 | 15s 确认 + `pending_confirm` 持久化 + `last_confirmed` 快照 |
| 10 | 启动 | Bootstrap 子集先于 App/窗口/渲染器 |
| 11 | 安全 | `TrustLevel`（项目设置是不可信数据）+ 密钥引用 + 受保护键哈希 |
| 12 | 可解释 | `inspect()` / `explain()` 是一等 API，不是 UI 便利 |
| 13 | 预设与开关 | `SettingsPreset` 一等公民；`FlagDef` 与 `SettingDef` 分型 |
| 14 | 编辑器与状态 | 设置编辑 = `Command`；工作区布局走 `workspace_store`，**不进设置系统** |
| 15 | 质量属性 | 性能/可维护/可扩展/易用作为门禁，而非事后优化（见 §25） |

---

## 5. 域与层

### 5.1 四域

**域决定“挂哪些源、写哪些层、可见哪些命名空间”；registry 全局共享。**

| 域 | 用途 | 典型设置 | 写入者 |
|---|---|---|---|
| **Runtime** | 游戏/应用运行时 | 画质、音频、输入、网络、窗口 | 玩家、运营、命令行 |
| **Editor** | 编辑器偏好与项目编辑 | 主题、快捷键、自动保存、项目默认 | 编辑器用户、团队、管理员 |
| **System** | 机器级/安装级 | GPU 选择、缓存路径、更新通道、遥测默认 | 安装器、管理员 |
| **Build** | 构建/CI | 目标平台、压缩、烘焙、符号 | CI、构建脚本 |

### 5.2 层栈

| 域 | 层栈（低 → 高） | 持久化根 |
|---|---|---|
| **Runtime** | EngineDefault < Project < Platform < Recommended < Remote < User < Device < Session < Runtime < **Mandatory** | `profiles/<id>/`、`device.toml`、`remote.cache.json` |
| **Editor** | EngineDefault < EditorDefault < Machine < Recommended < EditorUser < Project < ProjectLocal < Session < Runtime(PIE) < **Mandatory** | `editor/prefs.toml`、`ProjectSettings/*`、`machine.toml` |
| **System** | EngineDefault < Machine < Recommended < **Mandatory** | 机器级 `machine.toml` |
| **Build** | EngineDefault < Project < BuildProfile < Recommended < Session < **Mandatory** | `ProjectSettings/build.toml` |

规则：

- **Runtime 不挂 Machine**：机器级/安装级由 System 域在 Bootstrap 处理，避免游戏设置被机器路径污染。
- **Editor 的 Project > EditorUser**：项目对共享键（如渲染质量）有最终决定权；但 `editor.*` 默认禁止项目覆盖（见 5.4）。
- **PIE**：编辑器内运行时使用独立的 Runtime 层与快照，停止即丢弃。

### 5.3 Policy 两态

- **Mandatory**：置顶并锁。来源：托管策略、签名 `policy.toml`、管理员。任何非策略写入返回 `PolicyLocked`；UI 显示锁来源并禁用。
- **Recommended**：位于 User 之下、产品默认之上。仅当用户未显式设置该键时生效；用户一设即让位（Chromium `PrefValueStore` 的 `RECOMMENDED` 语义）。

### 5.4 Def 的域、作用域、覆盖

```rust
pub enum Domain { Runtime, Editor, System, Build }

pub enum Scope { Machine, User, Project }        // 默认写入目标

pub enum OverrideScope {                          // 值可以按什么粒度覆盖
    Global,
    Machine,
    Project,
    Resource,                                     // 每资源/每关卡
    Language,                                     // 每语言
}
```

- `domains`：该 def 在哪些域可见（位掩码）。
- `scope`：默认写到哪个作用域；编辑器 UI 的“为项目/为我/为这台机器”由它决定默认项。
- `project_overridable`（DefFlags）：项目是否可写该键；`editor.*`、`system.*` 默认 **false**。
- `override_scope`：优先做 **Project/Machine**；`Resource`/`Language` 为 M4 可选扩展，存储走 sidecar（如 `resources/<guid>.settings.toml`、`languages/<lang>.settings.toml`），不塞进主文件。

### 5.5 片段覆盖

每域支持 `settings.d/NN-*.toml` 按文件名字典序叠加，最后加载的优先。用途：

- 团队/平台覆盖（`20-console.toml`、`30-team.toml`）；
- 不修改主文件即可追加；
- `ProjectLocal` 是高优先级片段/本地目录，不进 VCS。

### 5.6 优先级示例

```
render.shadows.quality:
  EngineDefault=2 < Project=High < Platform=Medium < User=Ultra
  → Ultra
  控制台 set 0 → Runtime=0（最高，除 Mandatory）
  reset → 回落 Ultra
  Mandatory=Low → Low，写入被拒

telemetry.enabled:
  Mandatory=off → off；UI 显示“由组织管理”

render.resolution_scale:
  Recommended=1.0；用户未设 → 1.0
  用户设 1.25 → 1.25；Recommended 不再参与
```

### 5.7 模型最小化验证（M0.5 门禁）

四域 × 层栈 × OverrideScope × PrecedenceProfile × Mandatory/Recommended × Trust × 能力约束，组合数会快速膨胀。**模型冻结前必须先用少量真实设置做完整优先级表 + 属性测试**，否则不进入 M1。

验证集：12–20 个真实键，覆盖：

- 全局用户偏好（`audio.master_volume`）
- 设备相关（`render.resolution`、`render.refresh_rate`）
- 项目默认（`render.shadows.quality`）
- 服务器权威（`network.tick_rate`）
- 策略锁（`telemetry.enabled`）
- 信任边界（项目中的执行类键）
- 编辑器/项目本地（`editor.theme`、`ProjectLocal` 覆盖）
- 能力约束（`render.texture_quality` 受显存限制）
- 预设（`preset.quality=high`）
- 可计算值（`render.resolution_scale` 派生）

验证内容：

1. **完整优先级表**：每个键在四域、所有层、OverrideScope、PrecedenceProfile、Policy、Trust 下的期望值与来源。
2. **属性测试**：层覆盖单调性、reset 回落、Recommended 让位、Mandatory 不可绕过、Trust 拒绝、幂等、同层内顺序无关、`explain` 与解析一致。
3. **冲突矩阵**：同一键被多个来源设置时，结果唯一且可解释；不存在“看加载顺序”的歧义。
4. **组合剪枝**：若某维度组合无真实用例，删掉它（例如 `OverrideScope::Language` 延后）。

M0 可先实现用于跑通测试的**实验性 resolver**；在 M0.5 门禁通过前，层/策略组合不视为稳定 API，允许按验证结果删除或合并模型维度。M0.5 必须先于 M1 的 schema 与消费者集成，避免将未经验证的优先级语义扩散到 derive、持久化和 UI。

冻结标准：

- 所有验证键都有唯一期望值；属性测试全绿；
- `explain` 能复现优先级表；
- 无“必须靠文档解释”的歧义；
- 模型评审通过，之后才允许进入 M1。

不满足时优先简化：合并层、去掉 Resource/Language OverrideScope、限制 PrecedenceProfile 组合、把域可见性改成白名单；**不增加维度**。

---

## 6. 核心模型

### 6.1 Key / SettingId / Namespace

- **持久化标识**：点分小写 `render.shadows.quality`，稳定、永不复用；改名走别名 + 迁移。
- **运行时标识**：`SettingId(u32)`，registry 构建期 intern；读路径只比整数。
- **命名空间**：`render.*`、`audio.*`、`input.*`、`network.*`、`editor.*`、`system.*`、`build.*`、`plugin.<id>.*`。
- **命名空间所有权**：crate/插件注册时声明；mod 只能写 `mod.<id>.*`；卸载后保留但不参与解析。

### 6.2 Value

```rust
#[non_exhaustive]
pub enum Value {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Enum(EnumId),                    // 稳定字符串 id，不存 ordinal
    Array(Vec<Value>),
    Table(BTreeMap<String, Value>),  // 按键、分辨率、颜色
    Bytes(BytesRef),                 // 仅持久化/网络
    Secret(SecretRef),               // 句柄，不落明文
}
```

- 枚举：`EnumId` 是 intern 后的整数，但**落盘用稳定字符串**；改名/合并走 `EnumRemap`。
- 浮点：统一用 `f64` 解析，按 def 的 `ty` 收窄；`NaN/Inf` 拒绝。
- 表：键排序，保证序列化/hash 确定；表内字段也要有 schema 或允许自由键（需显式声明）。
- Bytes：仅用于持久化/网络；UI 不直接编辑。
- 实现层可把 `String`/`Array`/`Table` 内部用 `Arc`/`Box` 包装，使 `Value` clone O(1) 且 `size_of::<Value>() <= 32`；语义与序列化不变（见 §25.2）。

### 6.3 SettingDef

```rust
pub struct SettingDef {
    pub id: SettingId,
    pub key: &'static str,
    pub namespace: NamespaceId,
    pub ty: ValueType,
    pub default: Value,
    pub constraints: Constraints,
    pub apply: ApplyClass,
    pub apply_group: Option<ApplyGroupId>,
    pub flags: DefFlags,
    pub domains: DomainMask,
    pub scope: Scope,
    pub override_scope: OverrideScope,
    pub sync: SyncPolicy,
    pub precedence: PrecedenceProfile,
    pub ui: UiMeta,
    pub migration: Option<MigrationRef>,
    pub aliases: &'static [&'static str],
    pub doc: &'static str,
    pub since: Version,
    pub deprecated_since: Option<Version>,
}
```

`DefFlags`：

| Flag | 含义 |
|---|---|
| `persisted` | 是否落盘（默认 true） |
| `synced` | 是否参与云同步（默认按 `sync`） |
| `replicated` | 是否参与网络复制/hash 子集 |
| `bootstrap` | 是否在 App 之前可读 |
| `survives_safe_mode` | 安全模式下是否保留（a11y/语言/缩放） |
| `dev_only` | 仅开发版注册 |
| `cheat` | 需要 `cheats_enabled` |
| `read_only` | 运行期只读 |
| `hidden` | 不出现在普通设置 UI |
| `project_overridable` | 项目是否可写（`editor.*`/`system.*` 默认 false） |

`PrecedenceProfile`：

```rust
pub enum PrecedenceProfile {
    UserChoiceWins,   // 默认：User 高于 Remote
    ServerWins,       // Remote 高于 User（竞技/服务器权威）
    DeviceScoped,     // Device 高于 User（分辨率/手柄）
    Locked,           // 只允许 Mandatory/默认
}
```

### 6.4 Preset

```rust
pub struct SettingsPreset {
    pub id: PresetId,                 // low | medium | high | epic | custom
    pub extends: Option<PresetId>,
    pub values: Vec<(SettingKey, Value)>,
    pub target: LayerKind,            // Platform | User | Project
    pub platform: Option<PlatformProfileId>,
}
```

- 内置预设编译进二进制；项目/用户可定义 `presets.toml`。
- 预设是**数据**，走同一套 schema 校验；不是“设置分组”的别名。
- 应用预设 = 一个事务包多个 `set`，失败整体回滚。
- 不把预设做成“计算设置”：计算设置只做纯函数推导，预设做打包。

### 6.5 Flag

```rust
pub struct FlagDef {
    pub id: FlagId,
    pub default: bool,                // 默认关
    pub owner: &'static str,
    pub expires: Option<Version>,     // flag 有寿命
}
```

- Flag 不持久化、不迁移、不进普通设置 UI、不参与 `SettingsHash`。
- 值只来自 Remote/构建期；开发版允许 Runtime 覆盖。
- **Flag 不是 Setting**：避免“开关”污染设置 schema 与迁移链。

### 6.6 Registry

- 启动期构建，`finish` 后冻结为 `Arc<SettingsRegistry>`。
- 校验：重复 key、别名冲突、命名空间越界、默认值违约、计算环、apply group 合法、`bootstrap` 键的类型限制。
- 产出：UI 树、搜索索引、控制台命令表、文档表、hash 表。
- 提供：`def()`、`iter_namespace()`、`diff(a,b)`、`presets()`、`flags()`。

### 6.7 Resolver

- 每层 `Vec<Option<Value>>`（按 `SettingId` 索引）。
- 步骤：
  1. 按域挂载源；
  2. 过滤不可见/不可信；
  3. 按 `PrecedenceProfile` 选层；
  4. `Recommended` 兜底；
  5. 能力约束（GPU/显示/音频/输入）；
  6. 计算 DAG（拓扑序）；
  7. 类型/范围/跨设置校验；
  8. 产出 `Snapshot`。
- `generation` 单调递增；`Arc<Snapshot>` 跨线程共享；快照只含值 + 来源 + hash。

```rust
pub struct Snapshot {
    values: Vec<Option<Value>>,
    sources: Vec<LayerKind>,
    generation: u64,
    hash: SettingsHash,
}
```

### 6.8 Inspect / Explain

```rust
pub struct InspectResult {
    pub effective: Option<Value>,
    pub source: LayerKind,
    pub default: Value,
    pub by_layer: BTreeMap<LayerKind, Option<Value>>,
    pub fragment: Option<FragmentId>,   // 来自哪个 settings.d 片段
    pub locked_by: Option<PolicyLock>,
    pub modifiable: bool,
    pub preset: Option<PresetId>,
}
```

- `settings.explain <key>` 打印人类可读链路（来源、片段、被谁覆盖、为何不可写）。
- 编辑器控件显示“生效值 + 来源 badge”；客服报告直接带 explain。
- `inspect` 是只读、无副作用；不得触发 apply。

---

## 7. 类型、元数据、派生

### 7.1 derive 示例

```rust
#[derive(SettingsGroup, Reflect, Default)]
#[settings(namespace = "render", domain = "runtime,editor", scope = "user")]
pub struct RenderSettings {
    #[settings(default = Quality::High, title = "settings.render.quality",
               widget = Choice, preset = true)]
    pub quality: Quality,

    #[settings(default = 1.0, range = 0.25..=2.0, step = 0.05,
               unit = Ratio, apply = DisplayChange, group = Display,
               sync = never)]
    pub resolution_scale: f32,

    #[settings(default = false, apply = Restart, advanced, dev_only)]
    pub hdr: bool,
}
```

### 7.2 枚举稳定 ID

```rust
#[derive(SettingsEnum)]
pub enum Quality {
    #[settings(id = "low")]    Low,
    #[settings(id = "medium")] Medium,
    #[settings(id = "high")]   High,
    #[settings(id = "ultra")]  Ultra,
    #[settings(id = "epic", alias = "extreme")] Epic,
}
```

- 落盘 `"high"`，不写 ordinal；
- 只增不删；删除走 `EnumRemap` + 墓碑；
- 别名用于旧值读取，写回统一用新 ID。

### 7.3 单位

- 规范值存 **base unit**：比例 0..1、秒、Hz、dB、像素；
- `UiMeta.unit` 声明显示单位，UI 做本地化换算（`0.5` → `50%`，`1.5` → `150%`）；
- 单位变更必须 `UnitConvert`，不能只改注释。

### 7.4 UI 元数据

| 字段 | 用途 |
|---|---|
| `title_key` / `description_key` | i18n 文案键，不存自然语言 |
| `group` / `order` | 页面分组与排序 |
| `widget` | Toggle / Slider / Dropdown / Keybind / Resolution / Color / Path |
| `visible_if` / `enabled_if` | 条件显示/禁用（依赖其他设置或能力） |
| `advanced` | 高级折叠区 |
| `restart_badge` | 需要重启的视觉标记 |
| `tags` / `search_keywords` | 搜索 |
| `preset` | 是否参与画质预设 |

### 7.5 reflect 边界

- `prism_reflect` 负责**类型与结构**（字段、默认、范围、钳制）；
- `#[settings]` 负责**产品语义**（域、scope、apply、UI、同步、迁移）；
- 不让展示属性污染 reflect；也不让 reflect 元数据决定 apply 语义。
- 迁移只依赖持久化 key 与 schema 版本，不依赖 Rust 类型布局。

---

## 8. 校验与约束

### 8.1 五级

1. **类型**：能否解码（含枚举 ID 合法性）。
2. **范围/步长/正则/长度**：数值、字符串、数组。
3. **跨设置**：`requires` / `conflicts` / `implies` / `visible_if`。
4. **能力**：GPU 显存、最大纹理、显示模式、音频/输入设备。
5. **策略**：Mandatory、`read_only`、`cheat`、`Trust`。

### 8.2 违规处理

- 每 def 声明 `on_violation: Clamp | Reject`。
- 默认：用户/远程层 `Clamp`；项目/策略层 `Reject`（作者错误要大声失败）。
- 同一文件多个错误：收集全部，按 key 排序报告；不因一个错误丢弃整个文件（除非解析失败）。

### 8.3 计算设置

```rust
pub struct Computed {
    pub deps: &'static [SettingKey],
    pub eval: fn(&dyn SettingRead) -> Value,
    pub read_only: bool,
}
```

- 注册期建 DAG，检测环；拓扑序求值。
- 只允许纯函数；不得读文件/网络/环境。
- 必须有文档与依赖说明；UI 显示“自动计算”。
- 克制使用：画质档走 Preset，不走 Computed。

### 8.4 错误类型

```rust
pub enum SettingsError {
    UnknownKey(SettingKey),
    TypeMismatch { key: SettingKey, expected: ValueType },
    OutOfRange { key: SettingKey, value: Value, bounds: String },
    InvalidEnum { key: SettingKey, id: String },
    PolicyLocked { key: SettingKey, lock: PolicyLock },
    ReadOnly(SettingKey),
    CheatDisabled(SettingKey),
    UntrustedProject { path: String },
    MigrationFailed { from: u32, to: u32, key: Option<SettingKey> },
    Io { path: String, source: String },
    ApplyFailed { group: ApplyGroupId, reason: String },
    RollbackFailed { group: ApplyGroupId, reason: String },
    CycleDetected(Vec<SettingKey>),
}
```

所有错误实现 `Display`，带 key、来源、期望值；不 panic。

---

## 9. 应用语义

### 9.1 ApplyClass

| 类 | 含义 | 典型 |
|---|---|---|
| `Live` | 立即可用 | 音量、灵敏度 |
| `FrameBoundary` | 下一帧生效 | 阴影质量、后处理 |
| `DisplayChange` | 重建交换链/显示模式 | 分辨率、全屏、HDR、VRR |
| `Restart` | 需要重启进程 | 渲染后端、日志级别（部分） |
| `WorldReload` | 需要重载世界/关卡 | 物理精度、世界分区 |
| `DevOnly` | 仅开发版/控制台 | 调试绘制、统计 |

### 9.2 ApplyGroup + 事务边界

#### 9.2.1 事务保证分级

“原子”只对**我们拥有的状态**成立，不对所有外部副作用成立。分三级承诺：

| 级别 | 范围 | 保证 | 失败时 |
|---|---|---|---|
| **G1 状态原子** | 内存候选快照 → 生效快照 | 单指针交换；读者见旧或新，绝不见半更新 | 丢弃候选，旧快照不变 |
| **G2 持久化原子** | 我们写的文件 | 单文件 temp+fsync+rename；多文件用 manifest/journal 保证“旧完整集或新完整集” | 回退旧 generation；损坏隔离 |
| **G3 外部副作用** | applier 操作的显示/音频/设备/OS | **不承诺普遍回滚**；按 Reversible/Compensable/Irreversible 分级 | 补偿、标记 Degraded/NeedsRestart、重启/安全模式恢复 |

因此不再使用“全成功或全回滚”的表述；承诺改为：**状态提交原子、持久化可恢复、外部副作用分级可观测**。

#### 9.2.2 applier 可逆性分级

| 类 | 含义 | 例子 | 事务行为 |
|---|---|---|---|
| `Reversible` | 可用旧值精确恢复 | 音量、灵敏度、内存内开关 | 逆序补偿；保证恢复 |
| `Compensable` | 只能“再设一次旧值”，可能失败 | 显示模式、窗口大小、音频设备 | 补偿 + 确认窗口；失败进 Degraded |
| `Irreversible` | 无法恢复（进程/设备已重建） | 渲染后端切换、驱动重置、重启 | 必须最后执行；不做回滚承诺；置 NeedsRestart |
| `External` | 写 OS/外部系统，成功与否不由我们控制 | 系统音量、注册表、桌面策略 | 记录结果；不因失败回滚设置 |

每个 op 携带 `reversibility` 与 `idempotent`；`preflight` 必须报告计划中最高的不可逆等级与 `irreversible_index`。

#### 9.2.3 事务阶段与日志

```rust
pub enum TxPhase {
    Prepared, StateCommitted, Persisted, EffectsApplied,
    Confirmed, Degraded, RolledBack, RecoveryRequired,
}
```

阶段顺序按模式区分（`TxPhase` 是状态集合，不表示严格线性顺序）：

**直接提交（无预览）**

1. **Prepare**：校验、锁、能力检查、构建 apply plan；无副作用。
2. **Journal**：写 `apply.journal`（tx id、phase、old/new hash、op 计划），fsync。
3. **CommitState**：交换快照；发 `SettingsWillApply`。
4. **Persist**：按文件原子写 + manifest generation；失败则回滚快照、不执行 applier。
5. **Apply**：按 `order` 执行 op；每个 op 前后按 §9.2.6 回写 journal。
6. **Finalize**：清 journal；`Confirmed` 或 `Degraded`。

**预览（Display 等可补偿组）**

1. Prepare → Journal → CommitState。
2. **Apply preview**：只允许 Reversible/Compensable；记录旧值。
3. 等待确认：确认 → Persist → Finalize；超时/拒绝 → 补偿 + 回滚快照，**不落盘**。
4. 确认后持久化失败 → 补偿 + 回滚快照，报 `not_persisted`，不得假装已保存。

- 崩溃恢复：启动读 journal；`Prepared` 未提交 → 丢弃；`StateCommitted` 未持久化 → 按模式恢复；`Persisted` 未完成 apply → 只重跑**幂等** applier；非幂等/不可逆 → `RecoveryRequired` 或 `NeedsRestart`，交 doctor/用户决策。
- journal 记阶段 + 操作级事件（见 §9.2.6），不记全量值/密钥；有界、轮转；fsync 在 TxBegin/OpBegin/OpApplied/补偿边界。
- 多文件持久化用 `manifest.json`（generation + 文件 hash）；恢复选完整 generation；单文件仍是 temp+rename。

#### 9.2.4 失败状态与恢复

| 状态 | 含义 | 用户可见 | 恢复 |
|---|---|---|---|
| `Committed` | 状态与所有可逆副作用成功 | 正常 | — |
| `Degraded` | 状态已提交，部分副作用失败 | 该组显示“未完全生效” | 重试 / 重启 / 还原 |
| `NeedsRestart` | 状态已提交，等待重启 | 重启徽标 | 重启后清 |
| `RolledBack` | 状态恢复，可逆副作用已补偿 | 提示已还原 | — |
| `RecoveryRequired` | journal 有未决 op 或补偿不确定 | doctor 引导 | 选旧 generation / 安全模式 |

- 不可逆 applier 失败：不回滚状态，进 `Degraded` 并记录 `apply_error`；用户可显式“还原并重启”。
- 持久化失败：直接提交的 Persist 失败 → 回滚快照，状态不变；预览确认后的 Persist 失败 → 补偿 + 回滚，并报 `not_persisted`；绝不假装已保存。
- 预览超时/崩溃：走 `boot_state` 回滚到 `last_confirmed`（Display 组）。
- 补偿本身崩溃/失败 → `CompensationUnknown`；`probe` 不能确认时进 `RecoveryRequired`，不假设已还原。
- doctor 提供 `tx list` / `tx repair`；日志不记密钥。

#### 9.2.5 与 applier 的契约

```rust
pub trait SettingsApplier {
    fn group(&self) -> ApplyGroupId;
    fn order(&self) -> i32;
    fn preflight(&self, diff: &SettingsDiff, caps: &Capabilities)
        -> Result<ApplyPlan, ApplyError>;
    fn apply_op(&mut self, op: &ApplyOp) -> OpOutcome;
    fn probe(&self, op: &ApplyOp) -> EffectState;       // Applied/NotApplied/Unknown
    fn compensate(&mut self, op: &ApplyOp) -> CompensateOutcome; // 可失败
}
```

- `ApplyPlan` 由有序 `ApplyOp` 组成（§9.2.6）；`preflight` 报告 `max_reversibility` 与 `irreversible_index`。
- `apply_op` 必须遵守 `op.idempotent`：非幂等 op 重复调用返回 `AlreadyAttempted`，不得产生第二次副作用。
- `probe` 是恢复能力：能查询“效果是否已达目标/旧值”的必须实现；无法判断返回 `Unknown`。
- `compensate` 每次调用都写 `CompensateBegin/Compensated`；返回 `Unsupported` 的 op 必须在 preflight 阶段标成不可回滚。
- 未注册 applier 的键：只发 `SettingsDiff` 事件，无 op journal、不承诺恢复；这不是旧 API 兼容路径。
- 内置组：`Display`、`Audio`、`Input`、`Network`、`Content`、`Ui`、`System`；`Display` 默认 Compensable，`System` 默认 Irreversible/External。
- 回滚是“补偿”不是“时间倒流”：能恢复的写清楚，不能恢复的必须标记。

#### 9.2.6 操作级 journal 与恢复协议

阶段级 journal 不足以恢复“多个 applier 中途崩溃”。必须把最小副作用单元建模为 `ApplyOp`，每个 op 独立记录、独立状态、独立补偿账本。

**ApplyOp**

```rust
pub struct ApplyOp {
    pub index: u16,              // 执行序号 0..n
    pub id: OpId,                // stable hash(group, key, action)
    pub group: ApplyGroupId,
    pub key: SettingKey,
    pub action: OpAction,        // Set/Reset/Rebuild/Notify/...
    pub reversibility: Reversibility,
    pub idempotent: bool,
    pub input_digest: u64,       // 新值 + 目标句柄摘要
    pub effect_handle: Option<EffectHandle>, // apply 后回填；不含密钥
}
```

`OpOutcome = Applied { effect_handle } | Failed { error, effect_handle } | Deferred | AlreadyAttempted`；
`EffectState = Applied | NotApplied | Unknown`；
`CompensateOutcome = Done { verified } | Failed { error } | Unsupported`。

`Deferred` 只允许用于排队/预览前，不得已产生副作用；恢复按 Pending 处理。

- 一个 applier 可产生多个 op；op 是 journal/恢复的最小单位，applier 只是执行者。
- `preflight` 返回 `ApplyPlan { ops, max_reversibility, irreversible_index }`；顺序执行 `op.index`。
- 不可逆 op 前必须写 `IrreversibleBarrier`；之后进入 forward-only。

**journal 事件（append-only + checksum）**

每条记录含 `tx_id`、`seq`、`monotonic_ns`。事件：

| 事件 | 关键字段 | fsync |
|---|---|---|
| `TxBegin` | mode、hash_before/after、plan_digest、op_count、snapshot_gen、last_confirmed | 是 |
| `SnapshotCommitted` | snapshot_gen | 是 |
| `PersistBegin/End` | generation、manifest_hash | End 是 |
| `OpBegin` | op_index、op_id、group、key、action、reversibility、idempotent、input_digest | **调用前**，是 |
| `OpApplied` | op_index、effect_handle、result_digest | 是 |
| `OpFailed` | op_index、error、effect_handle | 是 |
| `IrreversibleBarrier` | op_index | **调用前**，是 |
| `CompensateBegin` | op_index、reason | 是 |
| `Compensated` | op_index、verified | 是 |
| `CompensateFailed` | op_index、error | 是 |
| `TxEnd` | outcome | 是 |

- `OpBegin` 必须先于副作用并 fsync；否则崩溃后无法判断是否已执行。
- `OpApplied/OpFailed` 后 fsync；失败也落盘，才能区分“未执行”与“执行失败”。
- 每 op 强同步（settings apply 是低频路径）；若基准显示不可接受，优化 op 合并/批量提交，不放宽“先写 `OpBegin` 再调用”的顺序。
- journal 写入/fsync 失败：立即停止，不调用下一个 op，进入 `RecoveryRequired`；不得继续产生副作用。
- 补偿本身也是操作：`CompensateBegin → Compensated/CompensateFailed`，带 `verified`；不得假设“调用 rollback 就成功”。
- journal 不记全量值/密钥；`effect_handle` 是回滚句柄（旧模式、备份路径、设备 id），可为空。
- 活动事务全量记录；`TxEnd` 后压缩为摘要 + 最近 N 条，轮转有界。
- 尾部截断/校验失败：丢弃到上一条有效记录；若最后有效记录是 `OpBegin/CompensateBegin`，按 `Unknown` 处理，不猜测。

**恢复重建表**

| 字段 | 取值 |
|---|---|
| `state` | Pending / InFlight / Applied / Failed / Compensating / Compensated / CompensationFailed / Unknown |
| `verification` | Unverified / VerifiedApplied / VerifiedNotApplied |
| `compensation` | NotNeeded / NotStarted / InFlight / Done / Failed / Unknown |
| `attempts` | 应用/补偿尝试次数 |
| `effect_handle` | 最后一次成功/失败返回的句柄 |
| `last_error` | 脱敏错误 |

`InFlight` 恢复时先转 `Unknown`：没有 `OpApplied/OpFailed` 就不能假设成功或失败。

**恢复决策**

```text
if TxEnd present: done
if no TxBegin / no SnapshotCommitted: discard (no side effects)
barrier = last IrreversibleBarrier
if barrier exists and OpBegin(barrier.op_index) exists:
    # 已进入不可逆区，forward-only
    match state(barrier.op_index):
      Applied -> TxEnd(Degraded) + NeedsRestart
      Unknown -> probe:
          VerifiedApplied -> mark Applied; TxEnd(Degraded) + NeedsRestart
          VerifiedNotApplied -> TxEnd(Degraded) + NeedsRestart (不重跑)
          Unknown -> RecoveryRequired (安全模式)
      other -> RecoveryRequired
    # 不启动任何新 op
else:
    # 未跨过不可逆执行点，可回滚
    for op in reverse order:
      Applied && compensation != Done -> compensate; 失败 -> RecoveryRequired
      Applied && compensation == Done -> no-op
      Failed -> no-op
      Unknown ->
        probe:
          VerifiedApplied -> compensate
          VerifiedNotApplied -> no-op
          Unknown && idempotent && !Irreversible -> compensate (设旧值可重入)
          Unknown -> RecoveryRequired
      Compensated -> no-op
      Compensating / CompensationFailed 或 compensation ∈ {Failed, Unknown} -> RecoveryRequired
    restore snapshot generation
    TxEnd(RolledBack) or RecoveryRequired
```

- `barrier` 存在但目标 op 无 `OpBegin`：不可逆副作用尚未开始，仍可回滚。
- `probe` 是恢复必需能力；不能判断就返回 `Unknown`，不得用“当前内存值”猜。
- **未知的非幂等 op 绝不自动重跑**；未知的不可逆 op 绝不自动回滚，进安全模式/人工选择。
- 补偿账本按 op 记录；只补偿 `Applied` 且 `compensation != Done` 的 op，逆序执行。
- 恢复后写 `RecoveryReport`（tx_id、每 op 最终状态、动作、未决项），doctor 可读；不写回 settings 文件。
- 恢复歧义 → `RecoveryRequired`：安全模式启动，提供“保留现状 / 回滚到 last_confirmed / 导出 journal”三个选项。

**示例（崩溃在不可逆 op 中途）**

```text
tx=7f3a mode=direct hash=9c1.. plan=4 ops
snapshot gen=42
op[0] begin Display/Set Compensable idem=true
op[0] applied handle="mode:2560x1440@165"
op[1] begin Audio/Set Reversible idem=true
op[1] applied handle="volume:0.8"
op[2] begin Render/Rebuild Irreversible idem=false
irreversible_barrier op=2
<crash>
```

恢复：op[2] 为 `Unknown` 且已跨屏障 → 不回滚、不重跑；`probe` 为 `VerifiedApplied` → 标 Applied，`TxEnd(Degraded)` + `NeedsRestart`；为 `Unknown` → `RecoveryRequired` → 安全模式。op[0]/op[1] 虽 Applied，forward-only 下不补偿。

**与 §9.2.3 的关系**

§9.2.3 描述阶段；本节描述阶段内每个 op 的记录与恢复。`TxPhase` 只是粗粒度状态，恢复以 op 级 journal 为准。

### 9.3 预览 / 确认 / 回滚

- `preview(group)`：应用 15 秒，UI 显示“保留 / 还原”。
- `pending_confirm` **写入 `boot_state.json`**：确认前崩溃/断电，下次启动自动回滚。
- `last_confirmed` 只存 Display 组的键（不是整份快照），回滚目标不是当前内存值。
- 适用：分辨率、全屏、HDR、VRR、色彩空间、多显示器。
- 预览期间再次修改：合并到同一 pending，重置计时；不允许嵌套预览。

### 9.4 安全模式 / 工厂重置

- `boot_state` 记录连续崩溃计数 + 待确认项。
- 连续失败 ≥ N：隔离可疑文件 → `EngineDefault + Mandatory + survives_safe_mode` 启动 → 恢复界面。
- CLI：`--safe-mode`、`--factory-settings`、`--reset-profile`、`--no-user-settings`。
- `survives_safe_mode`：UI 缩放、语言、字幕、色盲模式——绝不因安全模式丢掉无障碍设置。
- 工厂重置：删除用户/设备文件，保留机器级与策略；UI 二次确认；先备份到 `backups/`。

### 9.5 Bootstrap

固定小 schema，不依赖完整 registry；从同一批文件只抽取 bootstrap 键：

`system.language`、`ui.scale`、`ui.safe_mode`、`window.*`、`render.backend`、`render.adapter`、`log.level`、`profile.id`、`input.raw_input`。

- App 启动后完整 registry 再解析一次；二者对同一键必须一致（启动断言 + 测试）。
- Bootstrap 不加载项目/远程；只读 Machine + Policy + CLI + 少量 User。
- 专用服务器：无 User/Device，只读 Project + Session + Mandatory。

### 9.6 编辑器：设置编辑 = Command

```rust
pub struct SetSetting {
    key: SettingKey,
    scope: Scope,                 // EditorUser | Project | ProjectLocal | Machine
    old: Option<Value>,
    new: Option<Value>,
}
impl Command for SetSetting { /* apply / undo / merge / affected */ }
```

- 撤销/重做、宏录制、CI 回放、协同冲突检测（`affected`）复用编辑器通路。
- “应用画质预设” = 一个 `Transaction` 包多个 `SetSetting`。
- 预览 = apply + 定时 undo。
- 滑块拖动合并：同一 key 的连续 `SetSetting` 在时间窗内合并为一条撤销记录。

### 9.7 PIE

- 每实例独立 `Runtime` 层 + 独立快照 + 独立 applier 生命周期。
- 停止 PIE：丢弃实例层，恢复编辑器快照；不写持久化。
- 多实例：各自 generation；共享不可变 registry 与持久化源。

---

## 10. 持久化

### 10.1 目录布局

**Runtime**

```
<config_dir>/Prism/<app_id>/
  profiles/<id>/
    settings.toml
    settings.d/*.toml
    settings.unknown.toml        # 未知键保留
  device.toml
  policy.toml                    # 签名，只读
  remote.cache.json
  flags.cache.json
  boot_state.json
  backups/
  migrations.log
  secrets/                       # 只存引用
```

**Editor**

```
<config_dir>/Prism/Editor/
  prefs.toml
  prefs.d/*.toml
  machine.toml                   # 机器级；无权限时回退用户目录并记诊断
  workspaces/                    # 指向 workspace_store，不是设置
```

**Project**

```
<project>/ProjectSettings/
  project.toml                   # schema 版本、项目 id
  render.toml  audio.toml  input.toml  gameplay.toml  build.toml
  settings.d/NN-*.toml           # 团队覆盖，字典序
  local/                         # gitignored，本地覆盖
  .backup/                       # gitignored
```

**Machine**

- Windows：`%ProgramData%\Prism\machine.toml`
- macOS：`/Library/Application Support/Prism/machine.toml`
- Linux：`/etc/prism/machine.toml`
- 无写权限：回退 `<config_dir>/Prism/machine.toml`，doctor 记一条诊断。

### 10.2 格式

- 用户/项目/机器/策略：**TOML**（可手写、可 diff、合并冲突可读）。
- 远程/缓存/flag：**JSON**（机器生成、体积小）。
- 分享码：紧凑二进制 + base64url + 校验和。
- 编码器可插拔，语义只有一套（Value 树 + schema 版本）。
- 不引入配置语言/表达式；项目设置只当数据。

### 10.3 原子写

1. 写临时文件；
2. `fsync` 文件；
3. `rename` 覆盖；
4. `fsync` 目录；
5. 保留 `.bak`。

- 写失败不破坏旧文件；启动损坏则尝试 `.bak` → 安全模式。
- 项目文件：原子写 + `.backup/`（gitignored）；VCS 是最终备份。
- 写频率：用户编辑去抖（~250ms）；关闭/切项目/崩溃前 flush。

### 10.4 未知键与降级

- 未知键收进 `*.unknown.toml`，写回原样（保留注释/顺序尽量保真）。
- 文件版本高于当前：只读已知键、保留未知、警告、**绝不改写**。
- 解析失败：整文件隔离到 `backups/corrupt-<ts>/`，默认值启动，doctor 报告失败点。
- 单键类型错误：隔离该键，其余继续；报告里带原文片段。

### 10.5 Profile / Device

- `ProfileId`：本地多档案、主机多账号；切换 profile = 换持久化根 + 重新解析 + 事务应用。
- `DeviceId`：本机/手柄/显示器；`device.toml` 存分辨率、刷新率、手柄映射、音频设备。
- Device 层只放“与设备绑定”的键；`DeviceScoped` 的 def 才允许 Device > User。

### 10.6 云同步

- 按 `(key, updated_at, device_id)` per-setting LWW + 删除墓碑。
- `sync = never` 的键（GPU、分辨率、缓存路径、机器级）永不云同步。
- 不可合并冲突（按键绑定）进冲突 UI，不静默覆盖。
- 同步载荷只含 `synced` 键 + schema 版本 + device id；不含密钥。
- 服务端实现不在本期；先定 trait 与本地策略。

### 10.7 分享码 / 导入导出

- 只含 User 层（可选 Device）+ schema 版本 + 校验和。
- 导入：解码 → 迁移 → 校验 → 预览差异 → 应用（事务）→ 持久化。
- 导出：稳定排序；不含机器级/策略/密钥；不导出 `hidden` 键。

### 10.8 密钥

- `Secret(SecretRef)` 只存句柄；值在 `SecretStore`（OS keychain / 环境变量 / 0600 文件）。
- 日志、崩溃转储、遥测、控制台、UI 一律 `***`。
- 不参与 `SettingsHash` 与云同步。
- 缺失密钥：启动不失败；相关功能报“未配置”，UI 引导。

### 10.9 工作区状态

Dock 布局、窗口几何、打开标签、滚动位置、PIE 窗口位置走 `prism_ui_dock::workspace_store`：

- primary/last-good 双快照、临时文件 + fsync + 原子 rename、文件锁、迁移注册表；
- best-effort、可重置、**不承诺跨版本迁移**、不进设置 schema；
- 设置系统最多存“工作区预设名/引用”，不存布局本体。

---

## 11. 设置 schema 版本与迁移

### 11.1 每域独立版本

| 文件 | 版本键 |
|---|---|
| Runtime | `runtime_schema` |
| Editor | `editor_schema` |
| Project | `project_schema` |
| Machine | `machine_schema` |
| Build | `build_schema` |

避免“引擎升一次版本，所有文件被迫一起迁移”。每个文件头：

```toml
prism_schema = 3
```

本节说的“迁移”只指**设置数据的 schema 版本迁移**：键/值/枚举/单位/默认值/域层归属随 Prism 设置 schema 演进。它不涉及 Bevy，也不从 `crates/bevy_settings` 迁移——该老栈 `pkg/` 不消费，且 Bevy 正在被整体替换，所以不设桥接（见 §1.3）。若未来确有存量配置，只做核心包之外的一次性导入器，不进入迁移链。

### 11.2 迁移算子

| 算子 | 用途 | 例子 |
|---|---|---|
| `Rename` | 键改名 | `render.msaa` → `render.antialiasing` |
| `Split` | 一键拆多 | `audio.volume` → master/sfx/music |
| `Merge` | 多键合一 | `ui.scale_x` + `ui.scale_y` → `ui.scale` |
| `Convert` | 类型转换 | `"true"` → bool；整数 → 浮点 |
| `EnumRemap` | 枚举改名/合并 | `"epic"` → `"ultra"` |
| `UnitConvert` | 单位 | ms → s；百分比 → 比例 |
| `Drop` | 删除 | 必须写替代或移除说明 |
| `Default` | 补默认 | 新键按版本补值 |

### 11.3 迁移链与金样

- 迁移是纯函数：`fn migrate(from: u32, value: &mut Value) -> Result<(), MigrateError>`。
- 加载时版本低 → 逐级迁移；版本高 → 只读 + 保留未知 + 警告。
- 每个历史版本维护**金样文件**（golden corpus），CI 全量升级回归。
- 迁移失败：整文件隔离，默认值启动，doctor 报失败点；不部分写入。

### 11.4 兼容规则

- 每个 `Drop` 必须有替代或明确移除说明。
- 枚举只增不删；删除走 `EnumRemap` + 墓碑。
- 单位变更必须 `UnitConvert`，不能只改注释。
- 键永不复用；改名保留别名至少两个大版本。
- 文件降级：不删除未知键，不重写高版本文件。

---

## 12. 变更传播与可观测

### 12.1 事件与 diff

```rust
pub struct SettingChange {
    pub id: SettingId,
    pub key: &'static str,
    pub old: Option<Value>,
    pub new: Option<Value>,
    pub layer: LayerKind,
    pub fragment: Option<FragmentId>,
    pub cause: ChangeCause,
    pub tick: u64,
}

pub enum ChangeCause {
    UserEdit, Console, Cli, HotReload, Migration,
    RemoteFetch, PolicyUpdate, Rollback, Computed,
}

pub struct SettingsDiff {
    pub generation: u64,
    pub changes: Vec<SettingChange>,   // 按 key 排序
}
```

- 每帧聚合一次，按 key 排序，确定性。
- `SettingsDiff` 是唯一变更事件；不保留旧 `SettingChanged` 适配器。
- 订阅者可按 namespace/group 过滤；UI 只订阅可见键。

### 12.2 Journal

- 有界环形缓冲（4096 条）：时间、key、旧值/新值（脱敏）、来源、cause、调用栈（dev）。
- 崩溃转储、doctor、客服报告带 journal 尾部。
- 生产版默认不记录值，只记录 key + cause；开发版可记值。

### 12.3 SettingsHash

- 只覆盖 `replicated = true` 子集。
- 排序/编码稳定；排除设备相关项、密钥、UI 状态。
- 用途：网络加入校验、云同步冲突检测、bug 报告“设置指纹”。

### 12.4 遥测

- 默认关闭；只报计数/耗时/失败率，不报值。
- 隐私项（遥测开关）单独声明，且 `Mandatory` 可锁。
- 失败率按 key 聚合，帮助发现“某键在某平台总被钳制”。

### 12.5 Doctor

`settings.doctor` 检查：

- 未知键、弃用键、越界值、能力不支持；
- 迁移失败、策略冲突、密钥缺失；
- 文件权限/大小异常、备份存在；
- Bootstrap 与完整解析不一致；
- 项目设置被信任级别拦截。

输出：结构化列表 + 人类可读摘要 + 可导出 JSON。

---

## 13. UI / UX

### 13.1 schema → FormTree

- `UiMeta` 生成控件：Bool→Toggle、Enum→Dropdown、Float+range→Slider、Table→Keybind/Resolution/Color、Path→文件选择。
- 绑定走 `prism_ui_reactive`：值变化只更新订阅节点，不逐帧刷新。
- 校验走 `prism_ui_form`；文案走 `prism_ui_i18n`；不硬编码自然语言。

### 13.2 Scope-aware

每个控件显示：

- 生效值 + 来源 badge（EngineDefault/Project/User/Device/Mandatory…）；
- 各作用域值（可展开）；
- 操作：`[为项目设置] [为我设置] [为这台机器] [重置]`；
- Mandatory：锁图标 + 来源 + 禁用；
- 需要重启：badge + tooltip；
- 项目不可覆盖：隐藏写入口或显示“由项目决定”。

### 13.3 搜索

- 匹配：标题/描述/key/别名/关键词。
- 过滤：分类、只看已修改、只看需要重启、只看本机覆盖、只看项目覆盖。
- 排序：分组顺序 + 相关度；空结果给“清除过滤”入口。

### 13.4 交互

- 恢复：本节 / 全部 / 本设备；二次确认；先备份。
- 预览对话框：倒计时 + 保留/还原 + 失败原因。
- 冲突提示：按键重复、分辨率不支持、HDR 与显示器不匹配。
- 导入/导出分享码：预览差异，不盲写。
- 应用预设：显示“将修改 N 项”，可展开明细。

### 13.5 无障碍

- 角色/标签/焦点顺序/读屏公告。
- 不依赖颜色传达状态；高对比/色盲模式。
- 键盘/手柄全导航；焦点不丢。
- a11y 设置（缩放、语言、字幕、色盲）`survives_safe_mode`，且在 Bootstrap 生效。

### 13.6 编辑器页面

| 页面 | 作用域 | 存储 |
|---|---|---|
| Preferences | EditorUser | `editor/prefs.toml` |
| Project Settings | Project | `ProjectSettings/*` |
| System Settings | Machine | `machine.toml`（管理员） |
| Workspace | 工作区状态 | `workspace_store`（不进设置） |

### 13.7 控制台 / Inspector / 远程

- 控制台：`settings.list/get/set/diff/explain/doctor/export`；`cvar` 命令语法可作为产品别名保留，但只能读写新 registry 与事务路径。
- Inspector：显示 def 元数据、来源、约束、迁移历史。
- 远程运维：只读视图 + 受控写 + 鉴权 + 审计；不暴露密钥。

---

## 14. 安全与信任

### 14.1 不可信输入

- 文件大小、嵌套深度、键数量、字符串/数组长度设上限。
- 解析失败只产诊断，不 panic。
- 路径规范化，拒绝 `..` 与绝对路径（profile 名、导入码）。
- 远程配置限频、限大小、限深度。

### 14.2 TrustLevel

| 级别 | 来源 | 行为 |
|---|---|---|
| `Trusted` | 引擎/签名策略 | 全功能 |
| `User` | 用户目录 | 正常读写 |
| `Project` | 项目/VCS | 只当数据；禁用插件/脚本/远程/Machine 写；白名单设置 |
| `Restricted` | 未知/下载/工坊 | 只读已知键；不执行任何代码；首次打开弹信任提示 |

- 项目设置默认按**数据**解析，绝不执行代码/表达式/命令。
- Restricted 模式禁用项目插件、脚本、远程拉取、机器级写入。
- 用户可“信任此项目”，按项目路径记录；撤销信任立即生效。

### 14.3 Cheat

- 保留 `CHEAT` 语义；发行版默认关闭。
- 控制台/远程开启需显式开关 + 审计。
- 与成就/排行相关的键单独标记，不因 cheat 开关而静默启用。

### 14.4 Mandatory 策略

- pin + lock + hide；任何非策略写入 `PolicyLocked`。
- 签名验签在独立模块；验签失败 → 忽略策略 + 诊断（fail-open 还是 fail-closed 由产品定，默认 fail-closed 对 Mandatory 键）。
- UI 显示“由组织管理”，不显示可编辑控件。

### 14.5 受保护键哈希

- 对 cheat、遥测、策略、匹配/排行榜相关键存 HMAC/hash。
- 检测手动改文件；只作完整性信号，**不作安全边界**（客户端不可信）。
- 不哈希密钥值；不把 hash 写进日志。

### 14.6 密钥

见 10.8。补充：崩溃转储过滤、遥测过滤、控制台输出脱敏、UI 剪贴板保护。

### 14.7 mods

- 只能写 `mod.<id>.*`；不能写 core namespace。
- 卸载后设置保留但不参与解析；重装可恢复。
- mod 设置默认 `project_overridable = true`，但受 TrustLevel 限制。

---

## 15. 并发、多世界、网络

- **单写者**：主线程或显式事务线程；写路径不并发。
- **读者**：`Arc<Snapshot>`，渲染/子应用不加锁读；generation 比较决定是否刷新。
- **PIE/多 World**：共享不可变 registry 与持久化源；每实例独立 `Runtime` 层与快照。
- **网络**：
  - 只同步 `replicated = true` 子集；
  - 服务器权威；加入时下发 + hash 校验；
  - 不一致告警，不静默改客户端设置；
  - 服务器可强制 `Mandatory`/`ServerWins` 键。
- **热重载**：内容哈希轮询或平台 watch；变化 → 迁移/校验 → 候选快照 → 事务应用 → diff。
- **子应用**：`SubApp` 可持有独立 `SettingsReader`；默认继承主 App 快照。

---

## 16. no_std 与平台

- **core**：`no_std + alloc`，不依赖文件/时间/网络。
- **std feature**：I/O、env、时钟、文件监视。
- **wasm feature**：localStorage 后端；无文件系统路径。
- 平台差异（目录、换行、权限、注册表）集中在 `prism_settings_io`；core 不感知。
- 专用服务器：无 profile，只读 Project + Session + Mandatory。
- 移动端：`Device` 层 + 平台能力探测；设置文件走 app sandbox。

---

## 17. 性能预算

| 场景 | 目标 |
|---|---|
| 10k defs registry 构建 | < 10 ms |
| 冷启动全量解析（9 层） | < 5 ms |
| 单键写 + 重解析 + diff | < 50 µs |
| 类型化标量读取 | 无分配、常数时间 |
| 快照 clone | O(1) |
| 内存（10k defs + 全层） | < 4 MB |
| 每帧无变更开销 | 0 次全表扫描 |
| 迁移单文件（1k 键） | < 5 ms |
| 热路径读 p99 | < 100 ns |
| 设置页首屏（1k 控件） | < 50 ms |
| 启动设置总耗时 p95 | < 20 ms |

> 表中为**门禁预算**，不是愿望；测量方法、增量解析与回归策略见 §25.2。

实现要点：

- dense `Vec<Option<Value>>` + `SettingId` 索引；
- 脏位 bitset，只对脏键算 diff；
- generation 单调，读者按需 clone；
- 字符串 intern；枚举 intern；
- 序列化稳定排序，避免每次重排。

---

## 18. API 草案

### 18.1 注册

```rust
app.add_plugins(SettingsPlugin::new("com.example.game"));
app.register_settings::<RenderSettings>()?;
app.register_preset(Preset::high())?;
app.register_applier(DisplayApplier::new())?;
```

### 18.2 读

```rust
let s = app.settings();
let q = s.view::<RenderSettings>()?.quality();
let i = s.inspect(RenderSettings::QUALITY)?;      // InspectResult
let why = s.explain(RenderSettings::QUALITY);     // 人类可读链路
let snap = s.snapshot();                          // Arc<Snapshot>
```

### 18.3 写 / 事务

```rust
let mut tx = app.settings_mut().transaction(ApplyGroup::Display);
tx.set(RenderSettings::QUALITY, Quality::Ultra)?;
tx.apply_preset(PresetId::High)?;
match tx.commit()? {
    CommitOutcome::Applied => {}
    CommitOutcome::Reverted { failed } => { /* 报告 */ }
}
```

### 18.4 预览

```rust
let preview = app.settings_mut().preview(ApplyGroup::Display)?;
preview.confirm();
// 或 drop(preview) / 超时 → 自动回滚
```

### 18.5 开发控制台绑定

```rust
settings.register(SettingDef::new("render.shadows", Quality::High))?;
settings.write(scope, "render.shadows", Quality::Medium)?;
let value = settings.inspect("render.shadows")?;
```

控制台的 `cvar` 是用户界面与命令语法，不是独立注册表或旧 Rust API；它必须读写同一 `SettingDef`/事务路径。

### 18.6 编辑器

```rust
let cmd = SetSetting::new(scope, key, value);
editor.execute(Box::new(cmd))?;
```

### 18.7 控制台

```
settings.list render.*
settings.get render.shadows.quality
settings.set render.shadows.quality medium
settings.diff
settings.explain render.shadows.quality
settings.doctor
settings.export --scope user
```

### 18.8 错误处理

- 所有公开 API 返回 `Result<T, SettingsError>`。
- 写入失败不改变当前快照。
- apply 失败返回已回滚的 diff；调用方可选择重试/降级。

---

## 19. crate 布局

```
pkg/
  prism_settings/            # core：id/def/value/registry/resolver/constraints/
                             # computed/preset/diff/apply-trait/migration/inspect
                             # no_std+alloc；features: std, ecs, reflect
  prism_settings_derive/     # SettingsGroup / settings(...) / SettingsEnum
  prism_settings_io/         # TOML/JSON codec、原子 fs、profile/device/machine/
                             # project、片段、备份、导入导出、doctor
  prism_settings_ui/         # schema→FormTree、scope 选择器、搜索、预览、a11y
  prism_settings_editor/     # SettingsService、SetSettingCommand、settings_page
  prism_settings_test/       # 测试夹具、金样 harness、TestSettings、快照工具
  prism_gameplay_settings/   # 游戏层：画质预设、按键重绑、无障碍、难度（规划）
```

依赖方向：

```
gameplay → ui → editor → io → core
```

- **core 永不依赖 editor**。
- `prism_settings_ui` 只依赖 `prism_ui_*` 与 core 的只读接口。
- `prism_app` 只依赖 core（+ 可选 io），提供应用生命周期集成与事务调度；不 re-export 旧设置/cvar API。

**feature 切分（默认最小，按需开启）：**

| crate | 默认 | 可选 feature | 约束 |
|---|---|---|---|
| `prism_settings` | `std` | `std`、`reflect`、`trace`、`test-support` | `--no-default-features` 必须可构建 |
| `prism_settings_derive` | — | `ui` | 仅编译期，无运行时依赖 |
| `prism_settings_io` | `std` | `toml`、`json`、`watch`、`wasm` | 不依赖 UI/Editor |
| `prism_settings_ui` | `std` | `i18n`、`a11y` | 不依赖 IO/Editor |
| `prism_settings_editor` | `std` | `pie`、`remote` | 不依赖 gameplay |
| `prism_gameplay_settings` | `std` | `presets`、`rebind`、`a11y` | 不依赖 editor |
| `prism_settings_test` | — | — | 仅 dev-dependency |

---

## 20. 不变量

1. 不可信输入不 panic。
2. 解析结果永远满足类型与约束。
3. 高层覆盖不销毁低层；清除即回落。
4. 迭代/序列化/hash/diff 顺序确定。
5. 持久化键要么有 def，要么进 preserved，绝不整文件失败。
6. Mandatory 不可被非策略层绕过。
7. 迁移只前进；不改写高版本文件。
8. 状态提交原子；持久化可用 journal/manifest 恢复；外部副作用按 Reversible/Compensable/Irreversible 分级，不承诺普遍回滚。
9. 密钥不出现在日志/崩溃/遥测/UI。
10. 读者不阻塞写者，写者不阻塞读者。
11. 项目设置只当数据，绝不执行代码/表达式。
12. 工作区状态与设置分离。
13. OS 观测状态不当作我们的配置持久化。
14. Flag 不是 Setting：不迁移、不持久化、可过期。
15. 任何生效值都能 `explain` 出来。
16. 热路径读取无堆分配、无锁、无 I/O。
17. schema 是唯一真相源；任何生成物可重建，不手改。
18. 扩展点不得破坏核心不变量；第三方扩展不能写 core namespace。
19. 每个有副作用的 op 必须先写 `OpBegin` 并 fsync，再调用；`OpApplied/OpFailed/补偿` 必须落盘；`Unknown` 的非幂等 op 不得自动重跑。
20. journal 必须记录 `op_index/op_id/effect_handle/compensation`；恢复只依赖 journal + `probe`，不依赖内存推断。

---

## 21. 测试与验收

- **迁移金样**：每个历史版本升级到当前。
- **属性测试**：层覆盖单调性、序列化往返、钳制边界；§5.7 优先级模型属性（reset/Recommended/Mandatory/Trust/幂等/顺序无关/explain 一致）。
- **模型矩阵**：§5.7 的 12–20 键完整优先级表 + 属性测试，作为 M0.5 门禁。
- **事务恢复**：在 TxBegin/Snapshot/Persist/OpBegin/OpApplied/Barrier/CompensateBegin/Compensated/TxEnd 每个边界注入崩溃 × 四类 op；非幂等 Unknown 不自动重跑；补偿账本一致。
- **journal 重放**：同一 journal 重放结果确定；尾部截断回退到上一条有效记录并报 RecoveryRequired。
- **fuzz**：配置文件、导入码、远程 JSON。
- **故障注入**：写临时文件后崩溃、applier 中途失败、签名错误、能力缺失。
- **并发**：高频写 + 持续 clone 快照。
- **UI 快照**：FormTree 与文案索引。
- **确定性**：同输入两次 hash 相同。
- **基准**：10k defs 冷/热解析、单键写、diff、快照 clone。
- **替换完整性**：仓内调用方全部迁移；`prism_app::settings/cvar` 旧公开类型、方法和事件不存在；新内核/应用集成测试全绿。
- **启动一致性**：Bootstrap 与完整解析对同一键结果一致。
- **安全**：Restricted 项目配置不能触发代码/脚本/远程/机器写。
- **性能回归**：基准 p50/p95/p99 超预算或相对回退 > 5% 即失败。
- **文档构建**：`cargo doc` 无警告；设置参考由 schema 生成且与代码零 diff。
- **依赖方向**：CI 检查 core 不依赖 io/ui/editor。
- **易用性**：新设置从注册到 UI 出现 ≤ 3 步；错误信息必须含修复建议。
- **扩展性**：新增域/编解码/applier 的改动只落在约定 crate。

---

## 22. 里程碑

| 阶段 | 内容 | 功能验收 | 质量门禁 |
|---|---|---|---|
| **M0 内核** | id/def/value/registry/resolver/diff/inspect + 基准；`TrustLevel` 数据模型 + 执行类键拒绝；应用层接入新内核 | 新内核语义测试全绿；10k defs 基准达标；执行类键在非 Trusted 来源被拒；旧 API/测试调用方已迁移或删除 | `no_std` 构建通过；读路径零分配；API/信任边界评审 |
| **M0.5 模型验证** | §5.7 的 12–20 键完整优先级表 + 属性测试；冻结或简化层/策略模型 | 矩阵无歧义；属性测试全绿；`explain` 可复现 | 未通过不得进入 M1；简化杠杆已评审 |
| **M1 schema 统一** | derive + reflect 桥 + cvar 变开发者视图 + Preset | cvar 测试全绿；预设可事务应用 | 生成文档零 diff；`cargo doc` 零警告 |
| **M2 持久化** | io：TOML/JSON、原子写、profile/device/machine/project、片段、未知键、doctor、迁移金样；Trust 强制（项目/片段/机器写入边界、Restricted 沙箱） | 跨版本升级/降级/损坏恢复全绿；不可信输入/机器写入拒绝通过 | 迁移金样 100%；故障注入通过；fuzz 24h |
| **M3 应用语义** | ApplyGroup、操作级 journal、恢复状态机、预览+确认、安全模式、Bootstrap | 每个 journal 边界崩溃可恢复；Unknown 非幂等不重跑；Degraded/NeedsRestart/回滚正确 | 崩溃注入矩阵；不可逆 applier 演练；启动 p95 达标 |
| **M3.5 设置 UI** | schema→FormTree 通用表单、scope 选择器、inspect/解释、搜索、重置/撤销 | 运行时/编辑器共用一套表单；scope 切换正确 | 首屏 < 50 ms；a11y 初测；UI 快照 |
| **M4 编辑器域** | Editor/Project/ProjectLocal、信任提示/按路径管理（UI）、SetSettingCommand、PIE、插件自定义设置页 | 编辑器设置可撤销/预览/VCS diff 干净；信任撤销即时生效 | 撤销/预览演练；a11y 审计；VCS 合并测试 |
| **M5 治理** | System 域、Mandatory/Recommended、Remote 只读缓存 + `SyncTransport` trait、Flag、遥测、审计 | 策略不可绕过；同步排除正确 | 绕过测试；同步排除；审计日志；安全评审 |

---

## 23. 风险与边界

1. **范围蔓延**：先 M0–M3.5，再编辑器高级能力/系统域；不要第一版就做云同步。
2. **回滚不是免费的**：只承诺状态原子 + 分级补偿（见 §9.2）；不可逆 applier 必须标记并最后执行，文档必须写清楚“能否回滚、回滚到什么”。
3. **TOML 表达力有限**：复杂结构化值走 sidecar，语义仍一套。
4. **迁移是永久负债**：金样 + 墓碑 + 审计；不要为了“干净”删旧代码。
5. **云同步/托管策略涉及产品/法务**：本期只留接口。
6. **自动表单有 UX 上限**：游戏层必须能策展，不能把 registry 直接倒给玩家。
7. **性能要早测**：10k 设置不是极端值；dense 索引与脏位是硬要求。
8. **破坏性重构风险**：仓内调用方必须一次迁移到新契约；不得以旧 facade 延长过渡。控制台命令可保留为产品 UX，但必须改走新 registry 与事务。

---

## 24. 决策记录（原开放问题）

> 本章原为 14 项开放问题；v1.2 起全部裁决。裁决是**实施约束**，不是建议：除非走 ADR 并回写本章，否则不得偏离。带“延期”的条目是明确的**本期不做**，不是悬而未决。

### 24.1 裁决总表

| # | 问题 | 裁决 | 理由 / 约束 | 落地 |
|---|---|---|---|---|
| 1 | 四域是否够；Build 是否独立 | **四域够；Build 独立** | 构建是 headless/CI 生命周期，与 Editor 的 UI/撤销/权限模型不同；并入会让编辑器权限泄漏到构建机。自定义域作为 v2 扩展点，v1 不启用。 | §5.1；M0 |
| 2 | `ProjectLocal` + `settings.d` 是否 v1 | **v1 做解析/合并/来源追踪；片段编辑 UI 延后** | ProjectLocal 是本地实验刚需；片段是团队覆盖，读取/合并成本低；写入冲突 UI 复杂，延到 v1.1。 | §5.5、§10.1；M2 |
| 3 | 是否引入 `Recommended` | **v1 引入** | Chromium 语义：高于默认、低于 User；用户显式设置即让位；层栈已预留位置。 | §5.3；M0 |
| 4 | `SettingsPreset` 是否一等公民 | **是，v1** | 数据驱动、可 `extends`、事务应用；不新增层，由 `target` 决定写到 Platform/User/Project。 | §6.4；M1 |
| 5 | Trust/Restricted 是否 v1 | **M0 数据模型 + M2 强制；M4 仅信任 UI/管理** | 项目/片段在 M2 就会加载，不能等到 M4 才有边界；M0 定义 TrustLevel 并拒绝执行类键，M2 强制项目/机器写入边界与 Restricted 沙箱，M4 做提示/按路径管理。 | §14.2；M0/M2/M4 |
| 6 | 受保护键哈希范围 | **cheat、遥测、策略、匹配/排行榜相关键** | 客户端 hash 只作篡改信号，不是安全边界；排行榜以服务端签名/权威为准；不哈希密钥值。 | §14.5；M5 |
| 7 | 文件格式 | **用户/项目/机器/策略 TOML；远程/缓存/flag JSON；可选二进制快照** | TOML 可读可 diff；JSON 机器生成；语义只有一套（Value 树 + schema 版本）。 | §10.2；M2 |
| 8 | 云同步/托管策略是否落地 | **本期不落地**：只留 `SyncTransport` + 本地 policy + 只读远程缓存 | 涉及账号、服务端、法务；先定 trait 与本地策略。触发条件：出现账号体系、第二个消费应用或团队共享需求。 | §10.6、§15；M5 |
| 9 | 编辑器设置 UI 在 M4 还是 M3.5 | **通用表单/scope/inspect 提前到 M3.5；插件自定义页/PIE/remote 留 M4** | 用最短反馈环验证 schema→FormTree；避免 M4 一次性集成风险。 | §22；M3.5 |
| 10 | 对标基线 | **以 UE + VS Code + Chromium + GSettings 为主，其余补充** | 四者覆盖运行时/编辑器/策略/系统四类问题；不逐项对标，只借形态。 | §3 |
| 11 | `Value` 是否开放扩展 | **保持封闭枚举**；不开放 `Struct`/`Opaque` | 穷尽匹配、确定序列化、性能可预测、跨语言稳定；结构走 `Table` + `StructSchema`，二进制走 `Bytes` + codec；`Value` 标 `#[non_exhaustive]`，只允许内核新增变体。 | §6.2；M0 |
| 12 | registry 注册方式 | **默认显式调用**；`inventory`/`linkme` 仅可选 feature；构建脚本不改注册语义 | 顺序确定、可测试、wasm 友好；显式是唯一核心路径；可选 feature 只做便利。 | §25.4.2；M0 |
| 13 | 设置参考文档 | **双轨**：rustdoc 内嵌 API 参考 + 独立生成器产出用户参考/JSON；CI 零 diff | rustdoc 服务开发者；生成器服务用户、本地化、搜索；两者都从 schema 生成，不手写。 | §25.3.1；M1 |
| 14 | 性能门禁 | **双门禁**：CI 绝对预算 + PR 相对回退 >5% + 夜间长基准；CI 至少一个代表平台，夜间全平台矩阵 | 绝对值防“不可用”；相对值抓渐进劣化；夜间覆盖长尾。 | §25.2.7；M0 |

### 24.2 复审触发条件

以下裁决带**显式复审触发器**；触发前不重开讨论：

| 裁决 | 触发条件 | 复审方式 |
|---|---|---|
| #8 云同步延期 | 出现账号体系 / 第二个消费应用 / 团队共享设置需求 | 新 ADR + 服务端/法务评估 |
| #1 四域 | Build 需要独立写入者/权限模型，或出现第五种生命周期 | 新 ADR + 层栈兼容性分析 |
| #2 片段编辑 UI | 项目数/团队规模超过阈值，或片段冲突率上升 | v1.1 需求评审 |
| #5 完整信任管理 | 项目配置开始包含插件/脚本/远程源 | 安全评审 |
| #11 Value 扩展 | 出现 `Table`+`StructSchema` 无法表达的结构，且跨语言需要 | 大版本 ADR |

### 24.3 与 §4.3 的关系

§4.3 是 15 条**架构决策**；本章是 14 项**实施裁决**，用于消除歧义、锁定范围。两者冲突时，以本章较新的裁决为准，并回写 §4.3；不允许两份决策长期不一致。

---

## 25. 质量属性工程

> 前 24 章回答“系统长什么样”；本章回答“它凭什么在三年后仍然快、改得动、扩得开、用得顺”。四个质量属性都是**约束**，不是事后优化；每条都给出目标、机制、度量与门禁。

### 25.1 四维总览

| 属性 | 目标 | 关键机制 | 度量 / 门禁 |
|---|---|---|---|
| **性能** | 读 O(1) 无分配；写/应用有帧预算；启动 p95 < 20 ms | dense `SettingId` + 不可变快照 + 脏位增量 + 异步持久化 | 微基准 p50/p95/p99；相对回退 > 5% 失败；绝对预算硬失败 |
| **可维护性** | 单一真相源；改动局部化；迁移可审计 | derive 生成 + 生成物零 diff + crate 边界 + 每版本迁移模块 | 重复表 = 0；迁移金样覆盖 100%；`cargo doc` 零警告 |
| **可扩展性** | 新域/层/类型/编解码/applier/控件不改核心 | 注册 trait + 命名空间所有权 + `#[non_exhaustive]` + 稳定扩展点 | 扩展场景清单；第三方扩展不能写 core |
| **易用性** | 开发者 10 分钟接入；用户常见任务 > 95% 成功 | 默认可用 + 类型化视图 + `explain` + 预览/撤销 | 接入步骤 ≤ 3；错误含修复建议；a11y 审计通过 |

### 25.2 性能工程

#### 25.2.1 性能模型与预算

设置系统有四条不同性质的路径，不能用同一个“快”来衡量：

| 路径 | 频率 | 预算 | 关键约束 |
|---|---|---|---|
| 读（hot） | 每帧/每系统 | 标量 < 20 ns；p99 < 100 ns | 无分配、无锁、无 I/O、无虚调用 |
| 写 + 增量解析（warm） | 用户操作/控制台 | 单键 < 50 µs；100 键 < 1 ms | 只碰脏键；不整表重建 |
| 应用（apply） | 用户操作 | 单组 < 2 ms/帧 | 预检无副作用；重活异步/下一帧 |
| 启动/加载（cold） | 每次启动 | 全量 < 20 ms p95；Bootstrap < 5 ms | 主线程只做必要解析；其余异步 |
| UI | 打开/滚动/搜索 | 首屏 < 50 ms；滚动 60 fps | 虚拟化、增量更新、不轮询 |

**预算是一等契约**：基准失败与功能测试失败同等对待，阻塞合并。

#### 25.2.2 读路径

- `SettingId` 是 dense `u32`；每层 `Vec<Option<Value>>` 按 id 索引；快照是 `Box<[Option<Value>]> + Box<[SourceId]>`。
- `SettingsReader<T>` 借用一个 `Arc<Snapshot>`；标量按值返回；字符串/数组/表通过 `ValueRef<'_>` 借用，不 clone。
- `Value` 实现层把 `String`/`Array`/`Table` 内部用 `Arc`/`Box` 包装，使 clone O(1) 且 `size_of::<Value>() <= 32`；序列化与语义不变（见 §6.2）。
- 热路径不出现 `HashMap<String, Value>`：key 只在注册期查一次，运行期全用 id。
- 快照用 generation 比较；读者缓存最近一次快照，不每帧 clone。

#### 25.2.3 写与应用路径

- 写入先构造候选快照（copy-on-write cell），只校验/解析脏键。
- `ApplyGroup::preflight` 必须廉价、无副作用；`apply` 的耗时计入预算，超过 2 ms 的组拆帧或异步。
- 用户编辑去抖 250 ms；连续滑块合并；关闭/切项目/崩溃前 flush。
- 持久化异步：序列化在写线程；`fsync` 不占主线程；以 generation 解决“后写覆盖先写”。
- 显示模式等重操作走独立线程/下一帧，不阻塞输入。

#### 25.2.4 增量解析

- 全量解析只发生在加载、域切换、迁移后。
- 写入后只重算：脏键 + 反向依赖（Computed、`visible_if`、Preset 派生）+ 受影响 apply group。
- registry `finish` 时建反向依赖索引；新增 def 时增量维护。
- 最坏情况回退全量，但必须有上限与告警（doctor 记录“本次写入触发全量解析”）。
- 禁止每帧全量 resolve。

#### 25.2.5 启动

- Bootstrap 只解析 bootstrap 文件与 bootstrap 键；不构建完整 registry。
- App 启动后完整加载放到工作线程；加载完成前设置以只读默认值可用，UI 显示“加载中”。
- registry 构建用 derive 生成的静态表 + 排序数组；不做运行时反射扫描；10k defs < 10 ms。
- 机器/策略/用户/项目文件相互独立，可并行加载；合并顺序确定（按层优先级，不按完成时间）。

#### 25.2.6 UI

- 设置页虚拟化：只构建可见行；分组折叠时惰性构建。
- 值变化用 diff + memo 更新，不整页重建；订阅粒度到控件。
- 搜索索引注册期构建一次；过滤是索引查询，不是全表扫描。
- 预览/确认是覆盖层，不触发底层布局抖动。
- 不在每帧轮询设置；用 generation/信号。

#### 25.2.7 测量方法与门禁

**参考环境（绝对预算只在 reference runner 上判定）**

| 项 | 约定 |
|---|---|
| 机器 | CI 专用 runner；记录 CPU 型号/核数/内存/OS/存储；同代硬件 |
| 构建 | `--release`、`lto = "thin"`、`codegen-units = 1`、固定 rustc；bench 单独 profile |
| 运行时 | 固定 governor/电源模式；`taskset` 绑核；关闭后台噪声；记录温度/频率 |
| 统计 | 预热 ≥ 100 次；微基准样本 ≥ 10k，启动 ≥ 30；报 p50/p95/p99 + bootstrap 95% CI |
| 判定 | 用**上置信界**对比预算/主干；超预算或回退 > 5% 失败 |
| 交叉机器 | 非参考机只做相对比较；用校准基准（memcpy/hash/clock）归一化，不判定绝对预算 |

**基准矩阵（目标值；M0 实测后可按参考机校准，只允许收紧不允许放宽）**

| 基准 | 工作负载 | 规模 | 指标 | 绝对预算 |
|---|---|---|---|---|
| 读标量 | 10k defs，9 层 | 10k | p50/p99 ns | p50 < 20 ns，p99 < 100 ns |
| 读字符串/表 | 借用返回 | 10k | p99 ns | p99 < 150 ns |
| 写 + 增量解析 | 1 键 / 100 键 | 10k | µs/op | 1 键 < 50 µs；100 键 < 1 ms |
| 全量解析 | 9 层 | 1k/10k | ms p95 | 10k < 5 ms |
| registry 构建 | 静态表 | 10k | ms | < 10 ms |
| 迁移 | vN→vN+1 | 1k 键 | ms p95 | < 5 ms |
| UI 表单 | 1k 控件 | 1k | ms | 首屏 < 50 ms |
| 启动 | 完整 profile，冷进程 | — | ms p95 | settings ready < 20 ms；Bootstrap < 5 ms |

**基准实现**

- 微基准用 `criterion`（或等价 harness），固定输入生成器与 seed；计时区内禁止分配/打印。
- 启动基准：新进程 + 冷 page cache（另做 warm 对照）；分别测“进程启动→settings ready”“→App ready”。
- 内存：`size_of` 断言 + 分配计数（计数 allocator）+ 快照 RSS；10k defs + 全层 < 4 MB。
- 每次提交存 JSON 基线（机器指纹、rustc、commit）；PR 只对比同指纹基线；跨指纹只看趋势。
- 失败归因：相对回退 > 5% 附 flamegraph；噪声超 3% 先复跑，连续两次才判失败。

**门禁**

- CI 参考 runner 跑绝对预算；超过即失败。
- PR 相对回退 > 5% 失败；豁免需 owner、原因、到期日，并记录在 PR。
- 夜间长基准、内存剖析、macOS/Linux/Windows 全矩阵；`trace` 输出 span。
- 性能 PR 必须附 before/after 数字与基准名；不接受“感觉更快”。

#### 25.2.8 性能反模式

- 热路径 `HashMap<String, Value>` 查找。
- 每帧 clone 整份快照。
- 主线程序列化/fsync。
- 每帧扫描全部 defs。
- 每次读都分配 `String`。
- resolver 异步化或加锁。
- 用 `Arc<Mutex<...>>` 包裹整个 registry。

### 25.3 可维护性

#### 25.3.1 单一真相源与生成物

- `SettingDef` 是唯一真相源；derive 生成：`KEY` 常量、默认值、校验、UI 元数据、控制台补全、测试夹具。
- 设置参考文档由 schema 生成；CI 检查“重新生成后零 diff”。
- 禁止手写第二份 key/默认值/文案表；发现重复表即评审阻塞。
- 生成物不手改；改 schema 后重新生成。

#### 25.3.2 边界与依赖

| crate | 可以依赖 | 禁止依赖 | 所有者 |
|---|---|---|---|
| `prism_settings` | core/std/alloc、可选 reflect | io/ui/editor/app/gameplay | 设置内核组 |
| `prism_settings_derive` | 编译期库 | 运行时 crate | 设置内核组 |
| `prism_settings_io` | core、platform | ui/editor/app | 平台/持久化组 |
| `prism_settings_ui` | core、ui_* | io/editor/app | UI 组 |
| `prism_settings_editor` | core、io、ui | gameplay/app | 编辑器组 |
| `prism_gameplay_settings` | core、ui | editor/io | 游戏层组 |
| `prism_settings_test` | 全部（dev） | 生产依赖 | 设置内核组 |

- CI 检查依赖方向（`cargo tree` 白名单）；违反即失败。
- core 不依赖 `prism_app`；由 `prism_app` 依赖 core（依赖倒置）。
- 循环依赖零容忍。

#### 25.3.3 命名与版本

- key 语法：`<namespace>.<group>.<name>`，小写、点分、长度 ≤ 64；保留前缀 `engine.`、`system.`、`editor.`、`build.`。
- crate 走 semver；schema 每域独立版本；弃用窗口 ≥ 2 个 minor。
- 每个 def 记录 `since` / `deprecated_since` / `removed_in`；别名至少保留两个大版本。
- 维护“引擎版本 × schema 版本”兼容矩阵；升级/降级行为可查。

#### 25.3.4 迁移代码组织

- `migrations/<domain>/vN_to_vN+1.rs`，一个模块一个步骤；纯函数、不可变、只追加。
- 不修改已发布迁移步骤；新逻辑只能新增步骤。
- 每步旁边放金样与失败用例；CI 跑全版本升级。
- `migrations.log` 记录应用过的步骤与耗时；doctor 可查。
- 迁移债预算：每个 minor 新增迁移步骤数量有上限，超过需要架构评审。

#### 25.3.5 测试与诊断

| 层 | 覆盖 | 工具 |
|---|---|---|
| 单元 | def/resolver/constraints | `#[test]` |
| 属性 | 层单调、往返、钳制 | proptest/quickcheck |
| 金样 | 迁移、序列化 | `prism_settings_test` |
| fuzz | 文件/导入码/远程 | cargo-fuzz |
| 集成 | App/editor/PIE | 现成测试框架 |
| UI | FormTree/文案 | 快照 |
| 基准 | 热路径/启动 | bench harness |

- `prism_settings_test` 提供 `TestRegistry`、临时目录、`assert_inspect`、金样 harness。
- `doctor` 是维护工具：未知键、弃用、迁移失败、权限、Bootstrap 不一致。
- 错误类型统一；每条错误带 key、来源、期望、实际、修复建议、文档链接。

#### 25.3.6 文档与评审

- 四类文档：生成参考、手写指南、cookbook、迁移指南。
- 每个公开 API 有 rustdoc 示例；`cargo doc` 零警告。
- PR 必须过附录 E 清单；schema 变更需 namespace owner 评审。
- 跨切面决策写 ADR（域模型、Value 封闭、注册模型、性能预算）。

#### 25.3.7 可维护性反模式

- 手写 key 字符串与默认值。
- 巨型 registry 单文件。
- 迁移逻辑散落在 loader/UI/子系统。
- 隐式 apply 语义（“这个键应该会立即生效吧”）。
- 多套错误类型/多套诊断格式。
- 手改生成物。

### 25.4 可扩展性

#### 25.4.1 扩展点矩阵

| 扩展 | 接口/位置 | 稳定性 | 需要改核心？ |
|---|---|---|---|
| 新增设置 | `#[derive(SettingsGroup)]` + 插件注册 | 稳定 | 否 |
| 新增域 | `DomainSpec` 注册 | 实验 | 否 |
| 新增层 | `LayerSpec` + 优先级 | 实验 | 否 |
| 新增值类型 | `StructSchema` / `Bytes` codec | 大版本内稳定 | 否（封闭枚举） |
| 新增编解码 | `SettingsCodec` | 稳定 | 否 |
| 新增存储 | `SettingsStore` | 稳定 | 否 |
| 新增应用器 | `SettingsApplier` | 稳定 | 否 |
| 新增控件 | `WidgetFactory` | 实验 | 否 |
| 新增策略源 | `PolicyProvider` | 实验 | 否 |
| 新增平台 | `PlatformDirs` / `CapabilityProbe` | 稳定 | 否 |
| 新增编辑器页 | `SettingsPage` | 实验 | 否 |
| 新增预设 | 数据文件 | 稳定 | 否 |

#### 25.4.2 注册模型

- 默认**显式注册**：`SettingsPlugin::register(...)` 或 `SettingsExtension::register(&mut RegistryBuilder)`。
- 理由：顺序确定、无链接器魔法、wasm 友好、可测试；`inventory`/`linkme` 只作为可选 feature，默认关闭。
- 命名空间所有权：注册时拿 `NamespaceGuard`；重复注册 = 错误；mod 只能 `mod.<id>.*`。
- 第三方扩展不能：写 core namespace、注册 Mandatory、覆盖内置 key、修改 registry 已冻结内容。
- 注册期校验失败 → 启动失败并列出全部冲突，不静默跳过。

#### 25.4.3 域/层扩展

- `DomainSpec { id, name, layers, root, writable, visible_namespaces }`。
- 内置域保留 id 0–3；自定义从 4 开始。
- 层顺序显式声明，禁止“插到第 N 位”的魔法；`LayerSpec { id, kind, priority }`。
- 自定义域必须声明可见命名空间；默认只读，显式开启写。
- Resolver 泛化于 `LayerSet`，不为每个域写分支。

#### 25.4.4 值类型扩展

- `Value` 保持**封闭枚举**（`#[non_exhaustive]`，只允许内核新增变体）：穷尽匹配、序列化确定、无动态分发、性能可预测。
- 新结构类型走 `Value::Table` + `StructSchema`，新二进制类型走 `Value::Bytes` + codec。
- `ValueType` 标 `#[non_exhaustive]`；新增变体是 minor，消费方必须留 `_` 分支。
- 禁止 `Any`/trait object 进入 `Value`；禁止把 Rust 类型布局写进持久化。

#### 25.4.5 存储/编解码/应用/策略扩展

- `SettingsCodec`：`encode/decode`，必须报告未知字段/版本。
- `SettingsStore`：`load/save/list/backup`，fs/wasm/memory 三实现。
- `SettingsApplier`：注册到 group；group id 稳定；preflight/apply/rollback 契约。
- `PolicyProvider`：file/remote/platform；返回锁与来源。
- `CapabilityProbe`：GPU/显示/音频/输入；只读。
- `SecretStore`：keychain/env/file；只存引用。
- `SyncTransport`：本期只定义 trait。

#### 25.4.6 兼容性承诺

- 扩展 trait 用 `#[non_exhaustive]` + 默认方法；新增方法不破坏实现者。
- 枚举新增变体是 breaking（对穷尽匹配）；从第一天就 `#[non_exhaustive]`。
- 扩展点不得破坏 §20 不变量；注册期强制校验。
- 弃用先警告、再保留、最后移除；窗口 ≥ 2 个 minor。
- 第三方扩展的失败隔离：注册失败不拖垮内置设置；运行期错误只影响该扩展命名空间。

#### 25.4.7 可扩展性反模式

- 每个子系统自带一套 settings 文件与解析器。
- 把 `Value` 开放为 `Any`。
- 热路径用字符串 key 查 HashMap。
- 无命名空间的全局注册。
- UI 直接持有可变 registry。
- 扩展点放在热路径中间。

### 25.5 易用性

#### 25.5.1 开发者易用性

- 三步接入：`SettingsPlugin` → `#[derive(SettingsGroup)]` → `settings.view::<T>()`。
- 默认可用：derive 提供默认值/文案/UI；`SettingsPlugin` 自动注册核心 defs；不强制写样板。
- 类型化 API：`view::<T>()`、`get`、`set`、`reset`、`set_default`；`impl Into<Value>`。
- 错误可行动：`SettingsError` 带 key、来源、期望、实际、修复建议、文档链接。
- 测试友好：`TestSettings`、临时目录、金样断言、快照。
- 调试友好：`settings.explain`、`settings.doctor`、`tracing` span；dev 版显示来源链。
- 文档：每个公开 API 有示例；cookbook 覆盖“加一个设置/加一个域/加一个 applier”。

#### 25.5.2 用户易用性

- 零配置安全默认；不强制首次设置向导。
- 搜索、scope badge、重置、预览、撤销、预设。
- 重启最小化：能 Live 就不 Restart；必须重启时批量提示，不逐个弹窗。
- 失败可恢复：安全模式 UI、一键重置显示设置、doctor 报告。
- 无障碍：i18n、键盘/手柄、读屏、对比度、`survives_safe_mode`。
- 不堆术语：高级折叠；描述用用户语言；危险操作二次确认。

#### 25.5.3 编辑器易用性

- Scope 选择器 + “已修改”圆点 + 按 scope 重置 + 从其他 scope 复制值。
- 项目设置 diff 视图；VCS 友好排序。
- PIE 预览/应用/还原；不污染编辑器快照。
- 插件设置页自动生成；自定义控件有逃生舱。
- 命令面板集成：打开设置页、搜索键、跳到定义。

#### 25.5.4 运维/客服易用性

- 一条命令 `settings.doctor` 出报告；可导出脱敏 JSON。
- `settings.explain <key>` 直接回答“为什么是这个值”。
- 远程只读视图 + 审计；不暴露密钥。
- 支持包：journal 尾部 + doctor + schema 版本 + 平台能力。

#### 25.5.5 易用性度量

| 指标 | 目标 | 测量 |
|---|---|---|
| 开发者首次接入 | < 10 分钟 | 新同学任务测试 |
| 新增一个设置 | < 5 分钟 | 同上 |
| 常见用户任务成功率 | > 95% | 可用性测试（改分辨率/改键/重置） |
| 错误修复率 | > 80% | 给错误信息，观察能否自行修复 |
| 文档覆盖 | 100% def 有 title/description；公开 API 有示例 | CI 检查 |
| a11y | 审计通过 | 键盘/读屏/对比度 |

#### 25.5.6 易用性反模式

- 必须读源码才知道怎么注册。
- 错误只说“invalid value”。
- 需要重启却不提示。
- 设置项没有搜索/分组。
- 保存语义不清（自动保存 vs 手动保存）。
- 预览失败后无法恢复。

### 25.6 四维权衡与决策规则

| 冲突 | 取舍 | 规则 |
|---|---|---|
| 性能 vs 可扩展 | 热路径静态、扩展点放边界 | `Value` 封闭；注册只在启动期 |
| 性能 vs 可维护 | 生成代码/密集 id | 用 derive 换手写重复 |
| 可扩展 vs 可维护 | 稳定 trait + `#[non_exhaustive]` | 扩展点少而清晰，不开放内部 |
| 易用 vs 安全 | 安全默认 + 明确信任提示 | 项目配置默认不可信 |
| 易用 vs 性能 | 异步持久化 + 乐观 UI | 主线程不做 I/O/fsync |
| 可维护 vs 兼容 | 弃用窗口 + 别名 | 不静默破坏旧文件 |

决策规则：

1. 热路径只允许静态/密集结构；动态/字符串/反射留在冷路径。
2. 扩展点只放在边界（注册、codec、store、applier、widget、policy），不放进 resolver 核心。
3. 默认安全、默认可用；危险能力显式开启。
4. 先测量再优化；没有基准数字的“优化”不合并。
5. 任何新抽象必须回答：它减少了哪类重复，或打开了哪个扩展点。

---

## 附录 A. 术语表

| 术语 | 含义 |
|---|---|
| Domain | 域：Runtime / Editor / System / Build |
| Layer | 层：域内的来源优先级 |
| Scope | 作用域：Machine / User / Project |
| Def | `SettingDef`，一条设置的 schema |
| Flag | 特性开关，与 Setting 分型 |
| Preset | 一组设置值，可继承/平台化 |
| Snapshot | 解析后的不可变值集合 |
| Inspect | 查看生效值/来源/各层值 |
| Explain | 人类可读的来源链 |
| ApplyGroup | 可一起应用/回滚的组 |
| Applier | 子系统实现的应用器 |
| Bootstrap | App 之前的最小设置解析 |
| Fragment | `settings.d` 片段 |
| Mandatory | 策略锁，置顶 |
| Recommended | 管理员建议，低于 User |
| TrustLevel | 配置来源的可信级别 |

## 附录 B. 旧实现到新契约的替换清单

本附录是仓内迁移清单，不是 API 兼容承诺。旧类型、方法、事件、文本格式解析器和旧测试一并删除；每项能力迁入新模型后，只保留新契约的测试。

| 旧实现/能力 | 新模型 | 迁移动作 |
|---|---|---|
| `SettingsLayer`（5 层） | 域内 `LayerKind` + `SourceId` | 删除旧枚举；调用方声明 domain/source/layer |
| `SettingValue`（4 标量） | 封闭 `Value` 树 | 删除推断式 `parse`；输入按 `SettingDef` schema 解码与验证 |
| `Settings` BTreeMap | `Registry` + 不可变 `Snapshot` + `SettingsReader` | 删除旧资源；新 plugin 安装 registry/snapshot |
| `SettingChange` / `SettingChanged` | `SettingsDiff` | 删除旧事件；消费者订阅确定性 diff |
| `CvarSpec` / `CvarRegistry` | `SettingDef` 元数据 + console 查询视图 | 删除第二套 schema/注册表；控制台使用同一 registry 与写事务 |
| `set_cvar_at` / `reset_cvar` | scope 写入 + `SettingsTransaction` | 迁移校验、策略、apply、journal 全链路 |
| `load_user_config` / `write_user_config` | `prism_settings_io` 的版本化 codec/store | 删除旧 `name value` 文本格式，不承诺导入 |
| `platform_tier` | `PlatformProbe`/`Platform` source | 调用方迁移到能力/平台源模型 |
| `hot_reload` | io watcher → 候选快照 → apply transaction | 删除独立旧路径，复用统一事务 |
| `prism_ui_dock::workspace_store` | 独立 workspace state | 保持分离，不并入设置 |

## 附录 C. 迁移算子

| 算子 | 输入 | 输出 | 失败行为 |
|---|---|---|---|
| `Rename` | 旧键、新键 | 新键值 | 新键已存在 → 冲突诊断 |
| `Split` | 旧键、拆分函数 | 多个新键 | 任一目标非法 → 整条失败 |
| `Merge` | 多旧键、合并函数 | 新键 | 冲突 → 保留最高层 + 诊断 |
| `Convert` | 旧值、目标类型 | 新值 | 不可转换 → 默认值 + 诊断 |
| `EnumRemap` | 旧 ID、新 ID | 新 ID | 未知旧 ID → 丢弃 + 诊断 |
| `UnitConvert` | 旧单位、新单位 | 新值 | 溢出 → 钳制 + 诊断 |
| `Drop` | 旧键 | 无 | 必须声明替代/移除说明 |
| `Default` | 新键 | 默认值 | 无 |

## 附录 D. 参考文档

- `docs/prism_app_design_zh.md` §14（配置与设置分层）、§24.6（cvar 与配置级联）、§25.3（reflect 类型化读写校验）
- `docs/prism_gameplay_design_zh.md` §51（`prism_gameplay_settings` 规划）
- `docs/prism_editor_framework_design_zh.md` §4.1（Document/Layer）、§4.2（Command/Transaction/Undo）、§5（Shell）、§17（crate 矩阵）
- `docs/prism_editor_remote_design_zh.md`（编辑器远程协议）
- `pkg/prism_app/src/settings.rs`、`cvar.rs`、`reflect_config.rs`、`platform_tier.rs`、`hot_reload.rs`
- `pkg/prism_reflect/src/schema/`（metadata/version/migration/validate/clamp）
- `pkg/prism_platform/src/fs/`（dirs/file/dir/watch）
- `pkg/prism_ui_dock/src/workspace_store.rs`
- `crates/bevy_settings/src/lib.rs`（老栈，仅作对照，不桥接）

---

## 附录 E. 质量门禁与评审清单

### E.1 每个 PR

- [ ] schema：key 语法、默认值、约束、domains、scope、apply、flags 完整。
- [ ] 持久化：新增/改名/删除键有迁移，或明确声明“不持久化”。
- [ ] 迁移：金样 + 失败用例；不修改已发布步骤。
- [ ] 文档：title/description i18n key；公开 API 有 rustdoc 示例。
- [ ] 测试：单元 + 属性 + 集成；可见项有 UI 快照。
- [ ] 性能：热路径改动附 before/after；超预算需说明。
- [ ] 安全：不可信输入、密钥、策略、命名空间。
- [ ] 兼容：别名/弃用窗口；旧文件可读。
- [ ] 无障碍：标签、焦点、对比度、键盘。
- [ ] 生成物：重新生成后零 diff。

### E.2 每个 release

- [ ] 基准预算全绿；相对回退 ≤ 5%。
- [ ] 迁移金样覆盖全部历史版本。
- [ ] `cargo doc` 零警告；设置参考零 diff。
- [ ] feature 矩阵构建：`no_std`、`std`、`wasm`、editor。
- [ ] 依赖方向检查通过。
- [ ] fuzz 语料跑过；崩溃/挂起零。
- [ ] 升级/降级/损坏恢复演练。
- [ ] 安全模式演练。
- [ ] a11y 审计。
- [ ] 支持包/doctor 可用。

### E.3 新增一个设置的最小清单

1. 选 namespace 与 key（不重复、不复用）。
2. 写 def：类型、默认、约束、apply、scope、domains、flags。
3. 写 i18n title/description。
4. 决定是否持久化；需要则写迁移。
5. 写测试：默认、钳制、跨设置、迁移。
6. 确认 UI 出现位置与搜索关键词。
7. 更新 changelog 与兼容矩阵。

---

## 附录 F. 演进与升级 playbook

### F.1 新增设置

1. 确认 namespace 所有权与 key 语法。
2. 定义类型/默认/约束/apply/scope/domains/flags。
3. 补 i18n 与 UI 元数据。
4. 写单元 + 集成测试；可见项补 UI 快照。
5. 不需要迁移（新键按默认值）。
6. 更新 changelog；如需进入预设，更新 preset 数据。
7. 评审：附录 E.3。

### F.2 改名

1. 新 key 注册；旧 key 保留为 alias。
2. 写 `Rename` 迁移；金样覆盖旧文件。
3. UI/文档用新 key；旧 key 只读兼容。
4. 至少两个 minor 后删除 alias；删除前 doctor 提示。
5. 更新兼容矩阵。

### F.3 删除

1. 标记 `deprecated_since`；UI 显示弃用提示。
2. 提供替代键或说明；写 `Drop` 迁移（保留未知键策略）。
3. 两个 minor 后移除 def；旧文件迁移时丢弃并记诊断。
4. 更新 changelog 与迁移指南。

### F.4 改类型/单位

1. 评估是否 breaking；优先新增键而非改旧键。
2. 写 `Convert`/`UnitConvert`；覆盖边界/溢出/NaN。
3. 金样 + 属性测试；UI 单位换算。
4. 更新兼容矩阵；通知消费方。

### F.5 改默认值

1. 默认值不是持久化值；改默认不触发迁移。
2. 评估行为变化；更新预设与文档。
3. 加测试：未设置时取新默认；已设置不受影响。
4. changelog 标注“默认值变更”。

### F.6 新增域/层

1. 定义 `DomainSpec`/`LayerSpec`；分配 id。
2. 声明可见 namespace、可写层、持久化根。
3. Resolver/IO/UI 加集成测试；不写域分支。
4. feature flag 控制；默认关闭，评审后开启。
5. 文档：层栈图、优先级、迁移。

### F.7 新增 Value 类型/编解码

1. 优先用 `Table` + `StructSchema` 或 `Bytes` + codec。
2. 如必须新增 `Value` 变体：`#[non_exhaustive]`、序列化标识、旧版本降级策略。
3. codec 报告未知字段；round-trip 测试。
4. 更新兼容矩阵；大版本才允许破坏性变更。

### F.8 消费方升级

1. 读 changelog 与兼容矩阵。
2. 跑 `settings.doctor`；处理弃用警告。
3. 更新 key/类型/API；用别名过渡。
4. 回归：旧配置可读、UI 正常、apply 正常。
5. 回滚方案：保留旧版本二进制 + 备份。

### F.9 回滚

- schema 降级默认**只读**：高版本文件不被改写；未知键保留。
- 提供 `--reset-profile` / `--factory-settings` 作为最后手段。
- 回滚前备份 `backups/`；doctor 记录回滚原因。
- 破坏性回滚（删键/降类型）必须显式确认。

### F.10 发布节奏

- 每个 minor 允许新增设置/域/扩展点；破坏性变更只在大版本。
- 每个 release 跑附录 E.2。
- 迁移窗口：至少支持“上一个大版本 + 当前大版本”的文件。
- LTS 分支只接受安全/关键修复，不新增 schema。

---

## 结语

这套方案的核心不是“再加一个配置表”，而是把设置变成**引擎契约**：一份 schema 管代码/文件/UI/控制台/远程；域与层分离；写入是事务；确认可回滚；迁移有金样；来源可解释；项目配置当敌意输入；工作区状态不混进来。它能在不推翻现有 `prism_app::settings/cvar` 的前提下渐进落地，并同时支撑运行时、编辑器和机器级治理。

v1.1 进一步把性能、可维护性、可扩展性、易用性变成可度量、可门禁的工程约束——设计不止“能跑”，还要“跑得久、改得动、扩得开、用得顺”。
