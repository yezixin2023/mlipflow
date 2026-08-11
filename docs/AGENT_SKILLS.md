# Agent Skills 使用说明

九个仓库 Skills 位于 `.agents/skills/`。它们面向 Codex 等能读仓库规则、调用 CLI 和解析 JSON 的 tool-using Agent。

| Skill | 何时使用 | 对应计算插件 |
|---|---|---|
| `$mlip-workflow` | 拆解和监督端到端流程 | 组合全部插件 |
| `$high-entropy-structure` | 构造无序/SQS 高熵候选 | `high-entropy-structure` |
| `$pes-sampling` | 监督 DIRECT 代表构型选择、用户自备 LASP/SSW 本地执行及历史 archive 归一化 | `pes-sampling` |
| `$dft-labeling` | DFT 能量/力/应力标注 | `dft-labeling` |
| `$mlip-training` | 多框架训练/微调计划 | `mlip-training` |
| `$mlip-benchmark` | 产生机器可读 benchmark/ranking | `mlip-benchmark` |
| `$ionic-transport` | MD→MSD→D/电导/Arrhenius | `ionic-transport` |
| `$composition-screening` | 大超胞组分筛选与 top-k 验证 | `composition-screening` |
| `$electrochemical-voltage` | Li 含量能量与电压曲线 | `electrochemical-voltage` |

Skills 是监督说明，不是计算实现。更新 Skill 时，应以当前 CLI 帮助、plugin manifest 和 schema 为接口事实；旧 claw skills 已发现命令名和参数漂移，不可直接复制。

`$pes-sampling` 中的 LASP/SSW 能力只覆盖 local、`shell=False` 外部执行契约和既有
`allstr.arc`/`best.arc`/`md.arc` 后处理。它不实现 SSW、不提交 scheduler，也不代表
LASP 势训练或 DFT→LASP `TrainStr`/`TrainFor` 导出已经完成。

开发或更新后，用仓库环境可用的 Python 运行官方 skill-creator 验证器：

```bash
python /path/to/skill-creator/scripts/quick_validate.py .agents/skills/<skill-name>
```

每个 `agents/openai.yaml` 的默认提示必须以对应 `$skill-name` 开头。
