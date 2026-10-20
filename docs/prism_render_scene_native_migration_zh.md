# prism_render_scene 原生迁移切片

## 目标与真实边界

本切片把 retained scene 的 CPU 核心迁移到 Prism 原生契约，入口是
pkg/prism_render_scene/src/native_scene.rs 的 NativeRenderScene。它只依赖
prism_ecs::Entity、prism_app::{App, Plugin} 和
prism_render_architecture::gpu_scene，不导入任何 bevy_* API。

端到端调用链为：

1. Prism ECS Entity 进入 NativeRenderScene，分配稳定的 SceneHandle；
2. Entity 操作被封装为 SceneTransaction，并由 CpuRenderScene 原子应用；
3. CPU 镜像产生 DirtySceneSlot，UploadPlanner 选择稀疏、连续或全量策略；
4. native API 提供 instance、transform、bounds ABI 行与 GpuSceneSnapshot；
5. 删除先发布 Destroy，再以 GPU completion value retire；只有
   reclaim_completed 之后才允许代际递增和槽位复用。

增量更新的 dirty 集合按 SceneHandle 存在有序 BTreeMap 中：同一帧对同一
对象的多次字段修改以 O(log n) 合并，生成上传计划时再按句柄顺序物化为
确定性列表，避免高频对象更新触发线性去重扫描。

## 已验证契约

- 同一 frame epoch 内多次 transform 更新不会滚动 previous_transform；下一帧的
  第一次更新才把上一帧 current_transform 写入 previous_transform。
- CpuRenderScene 拒绝 stale generation 和非法操作；NativeRenderScene 不会在
  CPU 应用失败时发布 Entity 映射。
- 插件只向 prism_app 主 world 注册 NativeRenderScene resource，不创建 Bevy
  RenderApp、ExtractSchedule 或隐式渲染系统。
- 上传计划和 ABI 行读取来自同一个 CPU 镜像，避免 Entity 映射、dirty 集合和
  GPU 行之间出现第二份状态。

## 尚未迁移的直接依赖

pkg/prism_render_scene/Cargo.toml 仍直接依赖以下 Bevy crate，因为旧适配器
仍在生产模块中编译：

- bevy_app、bevy_asset、bevy_ecs、bevy_derive、bevy_math；
- bevy_material、bevy_image、bevy_mesh、bevy_camera、bevy_color；
- bevy_core_pipeline、bevy_log、bevy_pbr、bevy_platform、bevy_render；
- bevy_transform、bevy_shader；
- 开发依赖 bevy_tasks。

这些依赖由 consumer、material、extract、geometry、gpu、opaque、plugin、
visibility、water 及剩余 shader/GPU 测试路径使用。当前切片没有删掉它们，
也没有用 feature 开关把旧实现伪装成已迁移状态。

## 下一步迁移顺序

下一步应把 NativeRenderScene 接到 prism_render_app 的 render sub-app：先建立
Entity/SceneTransaction 的单向 extract，再用 prism_render_driver 的 buffer/queue
契约消费 UploadPlan，最后迁移 geometry、visibility 和 opaque queue。每一步都应
保持一个可编译的 native vertical slice，再删除对应的 Bevy 依赖；在这些调用路径
完成前，不应宣称整个 prism_render_scene 已经 Bevy-free。
