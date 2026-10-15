# Water Render Fabric 接线说明

状态：已接入架构层水体帧计划与 scene 侧 resident water body；本切片不修改 `docs/refactoring-dev/`、Surface ABI 或 VT。

## 已完成

- `WaterFabricPlan` 由 `WaterShadingPlan` 构造，并随 `WaterFramePlan` 按 body 稳定顺序生成。
- Fabric 声明包含水体服务闭包、simulation、surface raster、MaskBlend 和 Composition 四个常驻 pass，以及水面、泡沫、喷雾、场景颜色、深度和 response composition 平面资源访问。
- scene prepare 将 fabric plan 携带到 `WaterGpuBody`；surface draw 在录制命令前执行 `is_link_complete` 门禁，避免声明不完整时提交绘制。
- `MaskBlend` 保留为常驻兼容 pipeline，当前真实 raster 入口仍为 `water_surface_fs_hybrid`。没有伪造 response-plane producer，也没有声明 RHI 不支持的 driver binary import/export。
- CPU/link 门禁覆盖服务闭包、四个常驻 pass、队列类型、关键资源读写、历史帧喷雾读取、composition typed planes 和 legacy hybrid 保留条件。

## 明确未宣称

本切片没有把现有 legacy hybrid raster 输出伪装成两臂 response-plane 生产结果；因此 Fabric 的 `MaskBlend` 资源访问保持为空并标记为 compatibility-only。只有完成等价 producer、ABI 和真实 RHI 能力验证后，才可以增加 response-plane 写入声明并移除 legacy 入口。
