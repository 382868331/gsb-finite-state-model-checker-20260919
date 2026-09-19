# fsmc — 有限状态模型检查库

一个仅用 Python 3.14 标准库实现的小型显式状态模型检查器，用于在有界并发模型上查找：

- **安全违例**（不变量被某个可达状态破坏）；
- **死锁**（无启用迁移且不满足终止谓词的可达状态）；
- **活性反例**（某条无限执行只访问 accept 有限次；不假设公平性）。

所有反例都带逐步状态，可从初态**独立重放**（重新检查守卫、同时赋值、终点违例）。探索受状态/边预算约束，未完成时明确返回 **unknown**，而不是误报通过。

## 环境与运行

Windows 原生 Python 3.14.7，仅标准库，无第三方依赖、无外部服务、无 Docker。

```bash
python demo.py                                  # 演示（含一个 PASS 和多个真实 FAIL）
python -m unittest discover -s tests -v         # 全部测试
```

## 建模

模型由 **≤6 个具名有界整数变量**、唯一初态和 **≤20 条唯一 ID 迁移**组成。迁移包含一个布尔守卫和一组同时赋值；所有右值都读取**旧状态**，赋值结果越界则该迁移被禁用。可达迁移按 ID 字典序枚举。

```python
from fsmc import Model, Transition
from fsmc import const, variable, add, sub, lt, le, eq, ne, ge, gt
from fsmc import land, lor, lnot, true, false

m = Model()
m.declare("x", 0, 3, initial=0)          # 名字, 下界, 上界, 唯一初值
m.add_transition(Transition(
    "inc",                                # 唯一 ID
    lt(variable("x"), const(3)),          # 布尔守卫 AST
    {"x": add(variable("x"), const(1))},  # 同时赋值（右值读旧状态）
))
```

AST 节点（不可变 tuple，**禁止 eval/任意回调**）：

| 类别 | 构造函数 |
|---|---|
| 整数 | `const(k)`、`variable("x")`、`add(a,b)`、`sub(a,b)` |
| 比较（得布尔） | `lt le eq ne ge gt` |
| 布尔 | `lnot(a)`、`land(...)`、`lor(...)`、`true()`、`false()` |

`Model.check()`（各检查函数内部也会调用）预检：AST 形状、int/bool 类型、未知变量、深度 ≤ 32，失败抛 `ModelError`（AST 问题抛 `AstError`）。

## 检查接口

```python
from fsmc import Budgets, Status, check_safety, check_deadlock, check_liveness

# 每个可达状态都必须满足不变量
check_safety(model, invariant, budgets=Budgets(), *, property_name="safety")

# 死锁 = 无启用迁移 且 not terminal(state)；terminal 是合法有限终止
check_deadlock(model, terminal, budgets=Budgets())

# 每条无限执行都无限次访问 accept
check_liveness(model, accept, budgets=Budgets())
```

- `Budgets(max_states=10000, max_edges=50000)`：默认状态预算 10000、边预算 50000。超限返回 `Status.UNKNOWN`，并附 `states_explored` / `edges_explored` 计数和 `reason`；若超预算前已找到可重放反例，仍返回 `FAIL`；完整探索后才返回 `PASS`。
- **活性**需要完整可达迁移图（故预算耗尽必为 unknown）：先求全部可达状态，再在非 accept 状态的诱导子图中用 Tarjan 找有环 SCC；找到可达的有环 SCC 即 `FAIL`。反例 = 初态到环入口的**前缀**（允许经过 accept）+ 一条非空、始终避开 accept 的**闭环**。非 accept 的有限汇点（有限终止）不是违例。

## 结果与反例重放

`Result.status` 为 `Status.PASS / FAIL / UNKNOWN`。

安全/死锁反例 `result.counterexample`：

- `transition_ids`：迁移 ID 序列，**迁移数最少**，平局取**字典序最小**；初态即失败时为空序列；
- `steps`：每个 `Step(transition_id, source, target)`；
- `states`：逐步状态，`len(states) == len(steps) + 1`；
- `bad_state`：违例终点。

活性反例：`prefix` / `prefix_states` 与 `cycle` / `cycle_states`（闭环首尾状态相同、环内无 accept）。

返回前库会从初态独立重放：逐个重算守卫、同时赋值与越界禁用，并复核终点违例/闭环形状；任何对不上都会抛 `ReplayError`，而不是给出不可信的证据。

## 代码结构

```
fsmc/
  __init__.py   公开 API
  astx.py       安全表达式 AST、静态类型/变量/深度检查、无 eval 求值
  model.py      Model / Transition / Bounds，同时赋值与越界禁用语义
  checker.py    BFS 建图（带预算）、最短字典序反例、迭代 Tarjan 活性分析、独立重放
demo.py         可运行演示
tests/          unittest 测试（含固定种子的小规模随机交叉验证）
```

明确**不支持**：完整 LTL、偏序规约、GUI、网络服务与长压力测试；示例模型规模 ≤100 个状态。
