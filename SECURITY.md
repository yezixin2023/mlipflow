# 安全策略

MLIPFlow 0.1.x 是 alpha 版本。安全修复优先作用于最新的 `main` 和最新发布版；目前不承诺旧版本的长期支持或响应时限。

## 私下报告漏洞

请使用代码托管平台的 **Security / Advisories / Report a vulnerability** 私下提交报告。若该入口尚未启用，请通过项目发布页列出的维护者私密联系方式报告。不要在公开 issue 中披露可利用细节、凭据或集群信息。

报告请包含：受影响版本/提交、最小复现、预期与实际行为、影响范围、是否涉及本地/SLURM/SSH+SLURM，以及可行的缓解建议。请使用虚构主机名和已撤销的测试凭据。

## 重点风险

以下问题属于安全范围：

- 只读命令产生写入、外部进程、网络或调度器副作用；
- execution approval 可被绕过、重放到另一计划或审批后被替换；
- argv、路径、manifest 或 SSH profile 导致命令注入/路径逃逸；
- 日志、manifest 或 provenance 泄露令牌、密码、私钥或敏感集群路径；
- 插件发现或读取配置时执行不受信任代码；
- artifact 处理覆盖项目外文件或加载不安全的模型序列化格式。

单纯的科学精度争议通常不是安全漏洞，但可能严重影响研究结论；请按普通 bug 报告并附可复现证据。若科学错误可触发任意代码执行、数据破坏或凭据泄露，则按安全漏洞私下报告。

## 使用者责任

- 只运行受信任的项目、插件、脚本和模型；模型文件可能包含可执行序列化载荷。
- 把 Python adapter 当作构建脚本而不是数据；`run --dry-run` 会执行其模块顶层及计划代码。当前版本没有第三方插件 sandbox。
- 把 SSH 认证留在系统/站点配置中，不写入 `project.yaml`。
- 批准前检查计划中的命令、后端、远端目录、资源、实际输入和 staged scripts。
- 使用最小权限的集群账户与目录权限，并保留调度器和 MLIPFlow 审计记录。
- 公开日志或 manifest 前先检查敏感信息。

## 当前强制边界

- local 外部程序使用 argv 与 `shell=False`，只继承白名单环境变量；凭据型参数、Bearer 值和带用户口令 URL 会被拒绝。
- Adapter 产物和 replay/completion 文件限制在项目/attempt 内，拒绝绝对逃逸、`..` 与符号链接。
- Python adapter 是受信任的构建/执行代码；其输出在 schema 边界解析一次。核心仍独立保护路径、凭据、子进程、远端 staging 和外部数据。
- 每个 attempt 保存执行时的 node/plan snapshot；终态 attempt 不随之后的项目配置修改而改变，当前项目的无关修改也不冻结整个状态库。
- SLURM completion 必须绑定 project/node/attempt 与成功退出状态；staging/fetch 使用内容校验保护 SSH transport，之后仍必须通过固定的科学 checker。
- 外部或 metadata-only benchmark evidence 默认不能参与模型路由。

这些边界不能替代科学软件本身的安全审计。用户提供的外部 wrapper、模型反序列化和站点 SLURM 脚本仍须按可信代码审查。
