# 节点图 pointer 交互

本切片在 `prism_editor_ui` 的真实 GraphEditor/Loom 面板上增加 pointer capture，复用稳定节点、端口和边 ID，以及内核原子命令和逆历史。实现文件为 `src/graph_gesture.rs`、`src/graph.rs`、`src/graph_view.rs`，测试为 `tests/graph_gesture.rs`。没有增加依赖，没有引入 Bevy，也没有内置材质或项目节点类型。

## 宿主接线

UI 根模块声明 `mod graph_gesture;`，导出 `GraphPointerButton`、`GraphPointerHit`、`GraphGesturePreview`。这些根文件由主代理集成，本切片不编辑。

`BoundGraphPanel::hit_test([x,y])` 接收 **canvas 局部逻辑像素**。宿主从真实 layout 的 `graph:canvas` 左上角扣除全局逻辑坐标；物理像素先除以窗口 scale factor。工具栏不属于 canvas。宿主应缓存本次实际发布的 BoundGraphPanel，不能把旧画面的输入与新模型的 build 混用。

| API | 行为 |
| --- | --- |
| `pointer_press(&view, position, button, selection_mode)` | 命中实际绘制的节点/端口/背景，返回 `GraphPointerHit`；获取单 pointer capture。`Pan` 可由中键或宿主修饰键映射。 |
| `pointer_move(position)` | 更新预览；不改变文档、revision、稳定 ID 或 undo/redo。成功后 build 并发布新 view。 |
| `pointer_release(&view, position)` | 使用最新已发布 view 验证目标，并消费最终坐标；成功则一次提交，失败也结束 capture。 |
| `cancel_pointer()` | 幂等取消，返回是否曾有 capture；focus lost、Escape、窗口销毁时调用。 |
| `pointer_active()` / `gesture_preview()` | 宿主捕获状态和小型预览描述。 |
| `zoom_at(&view, position, factor)` | 以 pointer 下文档点为锚点缩放；已有 capture 时返回 `GestureState`，宿主先明确结束或取消。 |

`BoundGraphPanel::viewport()` 给出包含 pan 预览的实际绘制 viewport，controller 的 `viewport()` 保持已提交值。`preview_rects()` 给出最多四个覆盖层矩形，供布局检查与宿主诊断。

宿主只对自己的活动 capture 转发 move/release；成功 press 后应拦截对应的语义 Activate，避免一次 native 点击同时执行端口按钮连接和 pointer 连接。每次成功 move、cancel、release 或语义操作后重建并发布。重复 press 返回 `GestureState`。release 成功或失败后都应释放 OS pointer capture 并重建视图。此切片不修改 native app 的窗口事件路由；native 接入不能仅凭本控制器测试声称完成。

## 命中、坐标与 publication 围栏

命中使用已经裁剪并实际绘制的 f32 控件矩形，边界为左/上包含、右/下不包含。视口本身同样半开，非有限值和视口外 press/release 没有命中。端口在所属节点之上，较后绘制的节点及其端口覆盖较早节点；连线与预览覆盖层不获取 pointer。反向命中只遍历有界的可见节点/端口，不遍历完整文档。

viewport 尺寸每轴 64..4096，zoom 0.25..4，origin 每轴绝对值不超过一百万。capture 的变换和起点在 press 固定；节点增量按初始 zoom 换算并四舍五入为整数文档坐标，不累计 move 误差。节点最终坐标每轴绝对值不超过一百万。move 可以位于 canvas 外，但有限且绝对值不超过一百万逻辑像素；非法 move 保留最后一次有效预览。矩形预览端点钳制到 canvas；外部 release 仍取消，不悄悄提交钳制结果。

view 以 controller 私有 Arc publication 身份围栏绑定；另持有 GraphEditor snapshot token。每次有效 move 都使旧 view 失效。语义编辑、undo/redo、选择或导航成功后取消旧 capture，即使 undo 后值看似相同，也不能使用之前的 publication。过期或其他 controller 的 view 返回 `StaleView`，失败 release 清除 capture，文档保持不变。失败的普通语义操作不取消仍有效的 capture。

## 预览与提交

- 节点：press 已选节点时保留组选区；Replace 点击未选节点则单选，Add/Toggle 使用已有选择规则。Toggle 移除所点节点后不获取 drag。capture 只保存有界选区的 `(NodeId, 原位置)`，move 只更新一个 delta。build 用 delta 绘制节点、端口及两端相关的真实连线；文档仍保留原值。release 将全部目标位置组成 **一个** `GraphCommand::Edit`，真实 Undo/Redo 一次恢复/重做整组。零 delta 不写历史。取消保留 press 时产生的选择，但不移动节点。
- 端口：可从输入或输出开始，橡皮筋最多三条正交段；预览不创建边。release 必须命中实际绘制的端口，并按起始方向规范化 output/input。创建边使用内核 next ID 和 Connect 命令，实际验证 direction、项目提供的类型 key、input cardinality、DAG cycle。失败不推进 ID、revision 或历史。
- 背景：显示四条裁剪矩形边框，release 才进行 Replace/Add/Toggle 多选。空间查询不完整或超选择预算则整体拒绝，原选择不变。
- 平移：只更新候选 viewport，build 随候选值重绘；release 才提交导航值，cancel 恢复。平移、缩放和选择均不产生文档 undo 历史。

## 成本与界限

每次 move 最多检查选区预算的节点位置（硬上限 1024），不克隆图、节点 payload 或整份文档。一次 capture 保存 O(selection) 原位置；有效 move 会分配新的 publication Arc，不能宣称零分配。

build 用现有 BVH 查询原视口与逆向平移视口的包围矩形，保证原先在屏外的选中节点能进入预览；该保守查询仍受 visit 与结果数量预算限制，不保证截断后全部节点可见。硬上限为 8192 次树访问、256 个可见节点、2048 个端口、512 条边（各最多三段），手势覆盖层最多四个矩形；`stats.truncated` 明确报告截断。端口成员检查使用有界选区二分查找。模型提交后沿用内核校验和全局空间索引重建，release 不承诺 O(visible)。

## 验证与未覆盖范围

2026-10-21，使用独立轻量 Cargo workspace 指向真实源码，offline、单 job、debug=0、incremental=0：

- `graph_gesture` **13/13**：400 次 group move 文档 token/位置/历史不变、release 单次 Undo/Redo；从 input 反向连接及真实拓扑评估；精确类型/方向/基数/环错误；矩形三种选择；cancel/零移动/NaN/越界 release；不同 controller、旧 move publication、语义编辑/undo/redo 围栏；pan/anchor zoom；裁剪与绘制顺序命中；查询/选择预算拒绝；屏外组选节点拖入；真实 Loom layout 与 CPU raster 的橡皮筋像素。
- 既有 `graph_panel` **8/8**。
- 本切片 `cargo clippy -p prism_editor_ui --tests --no-deps -- -D warnings` 通过。含依赖的首次 Clippy 遇到范围外 `prism_math/src/ops.rs:1` 的 `doc_markdown` 错误，未修改该文件。

隔离 harness 验证 graph 控制器、view、真实内核与 Loom/CPU backend，并非整个 UI crate 或 native app 的 all-targets 结果；共享根联测由主代理负责。未执行 GPU raster 或 OS pointer capture 测试。材质域 native authoring 接入、多 pointer/touch、自动滚动、节点 resize 和路由编辑不在本切片内。
