# fsmcheck — 有限状态显式模型检查库

一个仅用 Python 3.14 标准库实现的小型显式状态模型检查器，用于在小型并发模型上发现：

- **安全违例**（safety violation）
- **死锁**（deadlock）
- **活性反例**（某条无限执行只访问 accept 有限次，即可以永远绕开 accept 成环）

结果严格区分 `PASS` / `FAIL` / `UNKNOWN`：只有完整探索后才给 `PASS`；
预算耗尽时如实返回 `UNKNOWN` 及已探索的状态/边计数；`FAIL` 一定附带
可独立逐步重放验证的反例。纯标准库，无第三方依赖、无网络、无 GUI、无 eval。

## 运行方法

```bash
python demo.py                                  # 演示（约 0.2 秒，远低于 8 秒预算）
python -m unittest discover -s tests -v         # 运行全部测试
```

## 建模

状态由 **1–6 个具名有界整数变量**组成，初态唯一；模型包含 **1–20 条
带唯一字符串 ID 的迁移**。迁移 = 布尔守卫 + 对变量的**同时赋值**。

表达式是一棵小 AST（也允许在叶子处直接写 `int` / `bool` / 变量名字符串，
构造时会被转换）：

| 类别 | 节点 |
|------|------|
| 整数 | `Int(n)`、`Var("x")`、`Add(a,b)`、`Sub(a,b)` |
| 比较 | `Lt Le Gt Ge Eq Ne`（操作数为整数，结果为布尔） |
| 布尔 | `Bool(b)`、`And(a,b)`、`Or(a,b)`、`Not(a)` |

构造模型时会预检：变量/迁移数量与重名、初值完整性与越界、表达式类型
（int/bool）、未知变量、AST 深度 ≤ 32；非法时抛 `ModelError`。
守卫和赋值右值都只接受 AST，不接受任何回调，求值由库自身遍历完成。

```python
from fsmcheck import (Model, Variable, Transition,
                      Var, Add, Lt, Eq, And, Not)

model = Model(
    variables=[Variable("p", 0, 2), Variable("q", 0, 2), Variable("L", 0, 1)],
    initial={"p": 0, "q": 0, "L": 0},
    transitions=[
        Transition("p_req",   Eq(Var("p"), 0), {"p": 1}),
        Transition("p_enter", And(Eq(Var("p"), 1), Eq(Var("L"), 0)),
                   {"p": 2, "L": 1}),
        Transition("p_exit",  Eq(Var("p"), 2), {"p": 0, "L": 0}),
        # ... q 侧对称
    ],
    accept=None,          # 可选：布尔表达式，标记 accepting 状态
    terminal=None,        # 可选：布尔表达式，标记合法终止状态
)
```

### 迁移语义

- 迁移按 **ID 字典序**枚举。
- 一条迁移的所有赋值**右值都读取旧状态**（同时赋值，可用于交换变量）。
- 守卫为真但任一赋值结果超出目标变量的 `[lower, upper]` 时，该迁移
  **被禁用**（不出现在后继中）。
- **死锁**：某可达状态没有任何启用迁移，且不满足 `terminal` 谓词。

## 三个检查

所有检查都返回 `CheckResult`（字段 `status` / `counterexample` /
`states_explored` / `edges_explored` / `state_budget` / `edge_budget` /
`message`），预算默认 `state_budget=10000`、`edge_budget=50000`。

```python
from fsmcheck import check_safety, check_deadlock, check_liveness

r1 = check_safety(model, Not(And(Eq(Var("p"), 2), Eq(Var("q"), 2))))
r2 = check_deadlock(model)
r3 = check_liveness(model)          # 使用模型的 accept 谓词
```

- `check_safety(model, predicate)`：要求每个可达状态都满足 `predicate`。
- `check_deadlock(model)`：寻找非终止的停滞状态。
- `check_liveness(model)`：要求**每条无限执行都无限次访问 accept**
  （不假设公平性）。算法先构造完整可达图，再在其中取 **非 accept 状态的
  诱导子图**，用 Kosaraju 求 **有环 SCC**；存在即说明可以从初态到达后
  永远成环、再不 accept。**有限终止不是此活性违例。**

### 反例与重放

- 安全/死锁反例是**迁移数最少、ID 序列字典序最小**的路径
  （BFS 按 ID 顺序枚举保证），允许**初态即失败的空路径**。
  `ce.prefix.transitions` 是 ID 序列，`ce.prefix.states` 是逐步状态
  （`states[i] --transitions[i]--> states[i+1]`）。
- 活性反例 = `ce.prefix`（从初态到坏 SCC 中某状态，**允许经过 accept**）
  + `ce.loop`（非空闭合环，环内无 accept）。
- 每个返回的反例都已用 `replay` / `replay_prefix_loop` **独立重放**：
  重新检查每个守卫、同时赋值与边界、以及终点违例（安全谓词为假 /
  确无启用迁移且非终止 / 环闭合且环内非 accept）。你也可以自己调用：

```python
from fsmcheck import replay, replay_prefix_loop
trace = replay(model, result.counterexample.prefix.transitions)
```

### UNKNOWN

预算在完成探索前耗尽时返回 `status == "unknown"`，并在 `message` 中
说明是哪个预算、在 `states_explored` / `edges_explored` 给出计数。
此时既不报 `PASS` 也不猜测 `FAIL`（初态即失败的空路径不需要预算，
即使预算为 0 也会直接 `FAIL`）。

## 范围限制

示例/测试模型均为小状态空间（≤100 状态量级）；不支持完整 LTL、
偏序规约或 GUI；不做长时间压力测试（随机差分测试使用固定种子、
60 个小模型）。

## 模块组织

| 文件 | 内容 |
|------|------|
| `fsmcheck/ast.py` | 表达式 AST、深度/遍历（迭代实现） |
| `fsmcheck/model.py` | `Variable` / `Transition` / `Model`、类型与深度预检、显式求值、迁移语义 |
| `fsmcheck/graph.py` | BFS 可达图（预算）、`CheckResult`/`Trace`/`Counterexample`、独立重放 |
| `fsmcheck/checker.py` | 安全、死锁、活性（Kosaraju SCC）检查 |
| `demo.py` | 一个正确协议的 PASS + 真实安全/死锁/活性失败 + UNKNOWN 演示 |
| `tests/test_model_checker.py` | 规格要求的全部场景 + 校验/重放/固定种子随机差分测试 |
