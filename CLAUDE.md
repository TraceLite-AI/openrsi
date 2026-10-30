# OpenRSI

组件式的 agent 自我改进框架和开放协议。项目说明见 README.md，协议见 spec/PROTOCOL.md。

## 审查

被要求审查（代码、PR、版本、整体状态）时，先读 REVIEW.md，带着里面的三个问题出发：

1. 我们的组件有哪些
2. 可自进化的组件有哪些
3. 是否能够兼容其他 RSI 框架

先按当前代码回答这三个问题，并和 REVIEW.md 里的基线对比，再看具体改动。

## 开发

- 测试：`pytest`（pytest.ini 已配置各包路径）
- 代码检查：`ruff check .`
- 改了 `sdk/python/openrsi/types.py` 或 `protocol.py` 后运行 `python -m openrsi.export_spec spec` 重新生成协议文件，测试会检查它是否最新
- 新增或修改组件后，更新它的 `rsi-component.yaml`，并在 REVIEW.md 的基线里记下变化
- README 有英文（README.md）和简体中文（README.zh-CN.md）两版，结构一一对应；改一版要同步另一版。实验结果填进 `<!-- RESULTS:... -->` 标记下面的表，两版都要填
