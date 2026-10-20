# 12 个音频 crate 的无 Bevy 数学迁移

## 提交范围

2026-10-20 收尾。本切片只提交下表 12 个 crate 的已有迁移，以及本文档。
路径均相对于仓库根目录；文件数包含各包的 `Cargo.toml`。

| crate / 路径 | 修改文件数 |
| --- | ---: |
| `pkg/prism_audio_spatial` | 59 |
| `pkg/prism_audio_hrtf` | 11 |
| `pkg/prism_audio_assets` | 3 |
| `pkg/prism_audio_authoring` | 9 |
| `pkg/prism_audio_content` | 2 |
| `pkg/prism_audio_output` | 2 |
| `pkg/prism_audio_music` | 2 |
| `pkg/prism_audio_resample` | 9 |
| `pkg/prism_audio_governor` | 8 |
| `pkg/prism_audio_midi` | 2 |
| `pkg/prism_audio_procedural` | 12 |
| `pkg/prism_audio_wave` | 17 |

合计 136 个音频文件，加本文档共 137 个文件。通过逐个文件的
`git commit --only -- <明确文件列表>` 提交，不提交共享 index 中的其他内容。

前置提交为 `f2302de01`（`refactor(audio): migrate math contracts off Bevy`），
其中已提交 `prism_math::ops`、数学类型 serde 支持、旧兼容 feature 的删除，
以及 `prism_audio_core` 等另一批音频包的迁移。本提交依赖该前置提交，
不再次修改或提交 `prism_math`、`prism_audio_core` 或那批包。
`prism_audio_device`、根 `Cargo.toml` / `Cargo.lock` 及其他目录的并行改动均排除。

## 行为和 feature 边界

- 12 包均将直接数学依赖改为
  `prism_math = { path = "../prism_math", default-features = false }`。
- 源码、现有测试、doctest 和文档链接中的数学入口改为
  `prism_math::{Vec3, Quat, ...}` / `prism_math::ops`。仅迁移入口，
  不调整 DSP 算法、阈值、调度或数据格式；同步更新原数学后端注释。
- 保留 `default = ["std"]`，原 `std` / `serialize` 对数学依赖的转发
  分别改为 `prism_math/std` / `prism_math/serialize`，其他 feature 边不变。
  数学依赖自身不默认启用 `std`，标量 facade 在两种模式下均使用 libm。
- 不保留 `bevy_math` 依赖、源码引用、`nostd-libm` 或旧兼容 feature。
  公开 API 中的向量和四元数现在是 Prism 类型；旧 Bevy/glam 类型的调用方
  必须同步迁移，不能假定二者具有相同 Rust 类型身份或内存 ABI。

这里的 `no_std` 指保留已有 crate 配置和 feature 转发方式。原 manifest
对 `prism_audio_core`（以及 HRTF 对 spatial）的依赖仍启用依赖默认 feature；
本切片没有重写这些边。选中包的 `--no-default-features` 检查不能证明
整个依赖图都关闭了 `std`，也不能替代裸机目标验证。

## 已有编译和测试证据

本轮复核了此前会话的命令和输出，未重建依赖。历史验证发生于
2026-10-20 13:03–13:11 UTC，使用复制源码的隔离 workspace
`/private/tmp/prism-audio-migration-HWvhJQ`。该目录在收尾时已不存在；
以下命令记录用于说明当时的精确选择范围，重新执行前需重新准备隔离 workspace。

```sh
verify_dir=/private/tmp/prism-audio-migration-HWvhJQ
audio_packages=(
  -p prism_audio_authoring -p prism_audio_assets -p prism_audio_content
  -p prism_audio_governor -p prism_audio_hrtf -p prism_audio_midi
  -p prism_audio_music -p prism_audio_output -p prism_audio_procedural
  -p prism_audio_resample -p prism_audio_spatial -p prism_audio_wave
)

cargo check --manifest-path "$verify_dir/Cargo.toml" -p prism_math --no-default-features
cargo check --manifest-path "$verify_dir/Cargo.toml" -p prism_math --all-features
cargo check --manifest-path "$verify_dir/Cargo.toml" --all-features "${audio_packages[@]}"
cargo test --manifest-path "$verify_dir/Cargo.toml" --all-features -p prism_math "${audio_packages[@]}"
cargo check --manifest-path "$verify_dir/Cargo.toml" --no-default-features -p prism_math "${audio_packages[@]}"
```

以上命令最终均通过；测试覆盖所选 13 包的单元测试、集成测试和 doctest。
例如 spatial 为 766 项单元测试、12 项集成测试、59 项 doctest；
wave 为 77 项单元测试、8 项集成测试、8 项 doctest。日志部分输出被截断，
因此不据此给出全部测试总数。`--all-features` 覆盖启用 `std` 与
`serialize` 的组合，不代表所有 feature 组合均逐项验证。

隔离验证有以下限制：

1. 除 12 包外，复制了 `prism_math`、`prism_platform` 和 `prism_audio_core`。
   当时仓库 core 尚使用旧数学入口，因此**机械修改了临时副本的 core**
   依赖和源码引用才运行成功；该临时修改没有在当时写回仓库。
   当前 HEAD 已包含前述 core 迁移，但本轮没有重编译当前 HEAD，
   历史结果不能表述为当前完整提交树的构建验收。
2. 首轮 all-features 检查因 serde 的集合支持未启用而失败；补齐 math 的
   `serde/alloc` 与 `serde?/std` 转发并同步临时副本后，重跑通过。
   该 math 修复现已包含在前置提交中。
3. 隔离 workspace 使用精简的继承 lint 配置和独立依赖解析，
   不等同于根 workspace 的锁文件、配置或完整 lint 验收。
   隔离目录的 `cargo fmt --all -- --check` 当时未通过，存在大量格式差异；
   本文不声称全包格式检查通过。
4. 当时根 workspace 的两次 math 检查在依赖解析阶段即失败：
   `prism_render_scene` 请求 `prism_render_material` 已不存在的 `bevy` feature。
   这是历史阻断记录，本轮没有重跑，也不据此判断当前根 workspace 状态。

## 本轮轻量复核与未验证项

收尾逐文件比较当前差异与数学命名空间、依赖及 feature 替换规则；
另行检查 spatial / HRTF 的 Cargo 注释和 ambisonics 注释差异。
音频实现差异均属上述迁移范围；本轮仅整理 ambisonics 注释措辞并新增本文档。

从仓库根目录执行下列轻量检查（zsh/bash），不触发编译：

```sh
audio_paths=(
  pkg/prism_audio_spatial pkg/prism_audio_hrtf pkg/prism_audio_assets
  pkg/prism_audio_authoring pkg/prism_audio_content pkg/prism_audio_output
  pkg/prism_audio_music pkg/prism_audio_resample pkg/prism_audio_governor
  pkg/prism_audio_midi pkg/prism_audio_procedural pkg/prism_audio_wave
)
rg -n --hidden 'bevy_math|compat[-_]bevy|nostd-libm' "${audio_paths[@]}"
git diff --check -- "${audio_paths[@]}" docs/prism_audio_no_bevy_migration_zh.md
```

引用扫描无匹配（`rg` 退出码 1），差异空白检查通过。另以静态文件核对
确认 12 份 manifest 的数学依赖均关闭默认 feature，并保留 std / serialize
转发；提交后核对提交路径仅含上表 136 个文件和本文档。

受本轮磁盘余量约 4 GiB 及任务约束限制，未重新运行 Cargo 编译、测试
或全仓格式化。没有单独验证 `--no-default-features --features serialize`、
无操作系统目标、跨平台逐位一致性、设备输出/录音、硬件播放或完整 workspace；
不把本切片的直接依赖迁移结论扩展为整个 Prism 仓库已无 Bevy。
