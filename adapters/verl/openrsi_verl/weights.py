"""权重可进化体已经移到独立的 openrsi-weights 包；这里保留旧名字，openrsi_verl:WeightsEvolvable 等写法照常可用。

用 verl 训练时，把 WeightsEvolvable 的 trainer 写成 verl 的命令模板，见 openrsi_verl 的模块说明。
"""
from openrsi_weights import DistillProposer, PreferenceProposer, WeightsEvolvable

__all__ = ["DistillProposer", "PreferenceProposer", "WeightsEvolvable"]
