# 高熵硫化物离线 replay 示例

这是一个**合成的接口 fixture**，用于演示 MLIPFlow 的完整 DAG、逐计划审批、结果采集和任务感知模型路由。所有数值、结构 ID 和“结果”均为虚构；它们不是来自真实材料计算，不能用于科学结论。

示例体系的 M 子晶格严格为 `Mn/Fe/Ni/Cu/Zn`，完整元素集合为 `Li/Mn/Fe/Ni/Cu/Zn/P/S`，不包含 Co。

## DAG

```text
structure-replay
  → sampling-replay
  → labeling-replay
  → training-replay
  → benchmark-replay
      ├→ transport-replay
      └→ screening-replay
            → high-fidelity-validation-replay
                → voltage-replay
```

九个节点覆盖八个确定性计算插件；`dft-labeling` 同时用于初始数据标注和筛选后的高保真验证。所有节点都采用 `mode: replay`，只引用现有小文件，不调用插件数值实现、网络、调度器或外部程序。

## 文件与模型 family

- `project.yaml`：九节点 DAG 和两套任务路由策略；
- `model_registry.yaml`：满足当前 model-registry schema 的 artifact、training data、compatibility 与 benchmark evidence 引用；
- `replay/*-result.json`：每个节点的小型结果 manifest；
- `replay/*.csv`/`training_models.json`：带 `synthetic`/`fixture-only` 标记的小型 artifact；
- `data/README.md`：为什么不分发结构、数据集或权重。

训练阶段仅声明三个原型 family：DeepMD、MACE、CHGNet。M3GNet 被列作一个 `unseen-transfer-family`，只带不同 scenario 的探针证据，因此在主示例路由中会被拒绝，而不会伪装成已经训练或完成公平比较的模型。

## 运行顺序

先从仓库根目录复制示例并初始化：

```bash
DEMO_ROOT="$(mktemp -d)"
cp -R examples/high_entropy_sulfide "$DEMO_ROOT/"
DEMO_PROJECT="$DEMO_ROOT/high_entropy_sulfide"
mlipflow --project "$DEMO_PROJECT" init
```

每次变更都先查看计划，再复制该计划自己的完整摘要批准。例如第一个节点：

```bash
mlipflow --project "$DEMO_PROJECT" run structure-replay --dry-run
mlipflow --project "$DEMO_PROJECT" run structure-replay \
  --approve 'sha256:<上一条计划的摘要>'

mlipflow --project "$DEMO_PROJECT" advance --dry-run
mlipflow --project "$DEMO_PROJECT" advance \
  --approve 'sha256:<上一条 advance 计划的摘要>'
```

保持相同的“检查 dry-run → 精确批准”步骤，依次执行：

1. `sampling-replay`
2. `labeling-replay`
3. `training-replay`
4. `benchmark-replay`
5. `transport-replay` 与 `screening-replay`（benchmark 后两者同时变为 `READY`）
6. `high-fidelity-validation-replay`
7. `voltage-replay`

每一层成功后再批准一次 `advance`；`advance` 只更新依赖状态，不执行节点。最终检查：

```bash
mlipflow --project "$DEMO_PROJECT" status
```

预期九个节点全部为 `OK`。

## 路由验证

```bash
mlipflow --project "$DEMO_PROJECT" route \
  --task ionic-transport \
  --elements Li Mn Fe Ni Cu Zn P S \
  --scenario synthetic-high-entropy-sulfide-v1

mlipflow --project "$DEMO_PROJECT" route \
  --task electrochemical-voltage \
  --elements Li Mn Fe Ni Cu Zn P S \
  --scenario synthetic-high-entropy-sulfide-v1
```

合成证据应分别选择 `deepmd-demo` 与 `chgnet-demo`。这是当前 fixture 指标和任务权重的结果，不是 MLIPFlow 核心中的全局模型偏好。
