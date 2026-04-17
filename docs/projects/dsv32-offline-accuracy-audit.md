# DSV3.2 Offline Accuracy Audit

**Created**: 2026-04-16
**Status**: archived-summary

## Scope
- 目标是定位 `dsv32_forward_full_ep8` 的 logits 对齐风险，覆盖 MLA、MoE、FP8、DeepEP、FlashMLA 等关键链路。
- 该文档保留结论摘要，不再保留逐步调试日志。

## Key Findings
- 当前主路径是 decode-style prefill，sparse prefill 不在该测试路径内。
- 剩余主要风险集中在：MLA 吸收权重与 scale 布局、FlashMLA split/scheduler 约束、MoE 路由与 combine 语义一致性。
- DeepEP 相关参数接线和 combine 关键错误已完成修复，后续关注点是数值一致性而非死锁。

## Cleanup Note
- 2026-04-17：离线精度工具 `tools/dsv32_ref/`、中间产物 `test_data/dsv32_*`、以及本审计文档中的执行日志已清理。
- 需要回溯细节时，以 git 历史为准。
