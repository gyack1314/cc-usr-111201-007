# 现场数据抽样的校准权重与整群方差

这是现场机器人应用分析的合成资料项目。`data`保存业务记录，`analysis_data.io`负责严格JSON读取和原始字节指纹，`analysis_data.domain`提供已有的基础汇总，`analysis_data.calibration`实现带比例界的熵校准（raking）、对偶/KKT核对与分层 delete-one-PSU jackknife 方差，`analysis_data.synthetic`生成校准核对所需的合成资料。字段与数学口径见[资料约定](docs/data-contract.md)。所有厂区、设备、收益与结果均为合成，不代表任何实际项目的效果。

Python 3.11及以上版本可直接运行，当前无第三方Python依赖（凸优化用标准库实现的对偶牛顿法）。项目根目录中的常用命令：

```sh
python -m unittest discover -s tests -v
python -m compileall -q analysis_data tests
python -m analysis_data.cli inspect data/normal.json
python -m analysis_data.cli calibrate data/normal.json
python -m analysis_data.cli generate-materials data
```

`calibrate` 输出求解状态、逐约束校准残差、目标/对偶目标与对偶间隙、对偶乘数与界激活、加权异常比例、权重有效样本量（ESS）、最大权重占比，以及分层 jackknife 标准误和正态近似区间。互斥的诊断状态区分目标不一致、正目标无样本支撑、线性相关约束（一致则去冗余、矛盾则判不可行）、带界无可行解与未收敛；失败或未收敛时不返回点估计，也不靠放宽边际通过。单 PSU 层或任何副本不可行/不收敛时保留点估计但不报有效区间。

`data` 中的资料分工：`normal` 为含相关边际与线性相关约束的基线；`bound-active` 有触界但可行的解且全部 jackknife 副本可重校准；`bounded-infeasible` 无界可行但带界不可行；`inconsistent-totals` 边际合计不等于总量；`replicate-support-loss` 满样本可行但删 H1P1 后副本失撑；`single-psu-stratum` 含单 PSU 层；`infeasible-margins` 相关约束目标矛盾；`invalid-*` 保存不合法输入。结果写入其他目录时应保留输入来源指纹，原始资料不覆盖。

容器可执行 `docker build -t field-analysis .` 后用 `docker run --rm field-analysis` 查看资料，或 `docker run --rm field-analysis python -m unittest discover -s tests -v` 运行全部检查。运行期间无外网或账号要求。
