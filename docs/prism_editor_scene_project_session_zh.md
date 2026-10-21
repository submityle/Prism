# Prism Editor Scene Project 会话实现说明

## 1. 结论与范围

本切片完成了 Prism Editor 的版本化 Scene Project 启动链路：项目可以在磁盘上安全创建或打开，初始空 Scene 在项目可见前已经形成 durable revision 0，打开后的 `DesktopDocumentSession` 会作为正式文档会话交给桌面事件循环。

这里的“完成”只指 Scene Project bootstrap 与 document session 绑定，不代表整个 Editor 已完成，也不代表已经存在完整的项目管理器、生产级 ECS 场景视图适配器或崩溃恢复产品流程。

实现入口：

- `SceneProject::create` / `SceneProject::create_with`
- `SceneProject::open` / `SceneProject::open_with`
- `SceneProjectOpen::into_desktop`
- `run_desktop_with_document`
- `DesktopDocumentSession::durable_revision`

## 2. 磁盘布局与 manifest

当前一个 Scene Project 的最小布局为：

```text
<project-root>/
  prism.project
  scene/
    cells/
      0_0_0.prismcell
```

`prism.project` 是固定长度 40 字节的小端二进制 manifest：

| 偏移 | 长度 | 内容 |
| ---: | ---: | --- |
| 0 | 8 | magic `PRISMPRJ` |
| 8 | 4 | manifest version，当前为 `1` |
| 12 | 16 | 持久化的非 nil Project `StableGuid` |
| 28 | 12 | Scene cell 坐标，当前固定为 `(0, 0, 0)` |

Project GUID 是项目身份，不由路径推导。项目目录整体移动后，重新打开仍读取并保留相同 GUID。解析要求精确长度；magic、版本、GUID、Scene 布局、截断或尾随字节任一不合法都会拒绝打开。

## 3. 创建流程

`SceneProject::create_with` 的可见提交顺序是：

1. 验证目标是不存在的路径或空的真实目录。
2. 创建 `scene` 存储根并生成非 nil Project GUID。
3. 为 `(0, 0, 0)` 建立 `EditorWorld` cell、文件 provider 和 `DocumentFileAdapter`。
4. 通过正常 document save 协议写入空文档，并等待保存 receipt。
5. receipt 成功 acknowledge 后，revision 0 才被视为 durable。
6. 最后发布 `prism.project`，再构造 UI 与 `DesktopDocumentSession`。

因此 manifest 不会先于初始 Scene 发布。调用成功时，返回对象满足：

- document durable revision 为 `Some(0)`；
- adapter 不处于 busy；
- 不要求 reconciliation；
- controller 不脏、没有未提交编辑、undo/redo 历史为空。

manifest 发布使用同目录 staging 文件：先完整写入并 `fsync` 文件，再用 hard link 以“不覆盖目标”的方式发布，移除 staging link 后 `fsync` 项目目录。若发布后的清理或目录同步失败，只回滚本次调用确实发布的目标；如果目标已由其他创建者占用，不会误删对方的 manifest。

创建失败时只清理本次创建的 Scene 数据和本次确认发布的 manifest。对于调用前已存在的非空目录，现有数据保持不变。

## 4. 打开流程

`SceneProject::open_with` 在返回桌面对象前完成全部验证和安装：

1. 验证项目根、manifest、Scene 根的路径和文件类型。
2. 解码并验证 manifest。
3. 使用 manifest 中的 Project GUID 建立 `EditorWorld` 与文件 adapter。
4. 加载 Scene cell，验证 PRISMDOC/document framing、schema 与资源上限。
5. prepare 完整 controller，再原子安装到 adapter/controller。
6. 构造 retained UI，挂载 content browser，并创建 `DesktopDocumentSession`。

失败的读取、格式校验、schema 校验或安装不会返回一个部分初始化的桌面会话。

## 5. Desktop document session 绑定

CLI 的 `--create-project` 与 `--open-project` 会得到 `SceneProjectOpen`，通过 `into_desktop` 将同一组已初始化的 `EditorDesktopUi` 和 `DesktopDocumentSession` 交给 `run_desktop_with_document`。这避免了项目 bootstrap 成功后又在桌面入口构造另一份无关文档状态。

`DesktopDocumentSession` 保留现有的有界异步语义：

- save 不会在提交时提前清除 UI dirty state；
- load 在 controller 有未保存改动时拒绝覆盖；
- receipt 完整校验后才安装 UI/renderer 文档；
- failed/cancelled save 保留 reconciliation 约束；
- retry 使用保留的精确 bytes、revision 和 durable base，不重新捕获当前 UI；
- `durable_revision()` 暴露最后成功保存或安装的 durable revision。

Scene Project bootstrap 自身使用有界超时。超时后会请求取消，并要求消费到 terminal cancellation；若操作没有按协议终止，则以 protocol error 失败，而不是继续返回可能仍在变化的会话。

## 6. CLI 合约

当前二进制支持：

```text
--desktop
--create-project <path>
--open-project <path>
--frames <u64>
--width <u32>
--height <u32>
```

约束如下：

- create/open 互斥，一个进程启动只接受一个 project action；
- project action、width、height 只能与 `--desktop` 一起使用；
- 重复参数、未知参数、缺值、把下一个 option 当作值、非法整数、零值和越界尺寸均返回明确错误和失败退出码；
- desktop 尺寸范围为 `1..=32768`；
- headless 和 desktop 都使用正数 frame limit；
- CLI 路径不使用 `panic`/`expect` 处理用户输入错误。

示例：

```sh
cargo run -p prism_editor_app --features desktop -- \
  --desktop --create-project /absolute/path/to/sample.prism --frames 1

cargo run -p prism_editor_app --features desktop -- \
  --desktop --open-project /absolute/path/to/sample.prism --width 1920 --height 1080
```

## 7. 安全边界

当前 create/open 明确拒绝：

- 路径中的 `..` parent traversal；
- symlink 项目根；
- symlink manifest；
- symlink Scene 根；
- 目录位置出现普通文件，或文件位置出现目录等类型错配；
- 在非空目录中创建项目；
- 覆盖已经发布的 manifest 或 Scene；
- manifest 截断、尾随数据、错误 magic、未知版本、nil GUID 或不支持的 cell 布局；
- 缺失 durable Scene cell；
- 超过 document/provider/controller 资源上限的数据。

当前实现面向单个本地文件系统项目根。它不承诺网络文件系统的跨主机一致性，也没有把多进程协作编辑升级为项目级锁协议。

## 8. 验证证据

2026-10-21 在独立 `CARGO_TARGET_DIR`、offline 模式下验证：

```sh
cargo test --offline -p prism_editor_app --features desktop --test scene_project
# 9 passed

cargo test --offline -p prism_editor_app --features desktop --test document
# 20 passed

cargo test --offline -p prism_editor_app --features desktop --test ecs_document_bridge
# 8 passed

cargo test --offline -p prism_editor_app --features desktop --bin prism_editor_app
# 5 passed
```

覆盖内容包括 durable revision 0、保存后 revision 1 重开、项目目录移动后的 GUID 稳定性、重复创建不覆盖、非空目录保护、symlink 拒绝、缺失 Scene、manifest 各类损坏，以及 document/ECS bridge/CLI 回归。

编译输出仍包含仓库其他 crate 的既有 warning；上述测试本身全部通过。

## 9. 仍未完成的 Editor 项

后续必须继续作为独立切片完成，不能由本实现代替：

1. 生产级 ECS → `DesktopScene` adapter，以及 `ECS 映射 → 视口渲染 → 选中对象 → 属性修改 → 回写数据模型` 的完整运行时闭环。
2. World Project Session：多 cell/world 的装载、驻留、切换、保存与项目级生命周期。
3. recent projects、项目选择/创建 UI、异常退出后的 crash recovery 与用户可见 reconciliation 流程。
4. 明确的多进程项目占用/协作策略；当前只保证 manifest 发布不覆盖以及 document revision fencing。
5. 项目格式迁移工具、未来 manifest version 升级策略和面向真实大型项目的兼容性矩阵。
