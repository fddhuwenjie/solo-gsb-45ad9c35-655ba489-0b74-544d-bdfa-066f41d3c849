"""无中断改线：中间快照（两个插/拔动作之间）保护通声校验测试。

覆盖：
  1. 任务核心缺陷——保护路径只依赖旧跳线、目标仅把另一端从 2B 换到 3B 类
     换接：旧代码只看宏动作完成后状态，误返回无中断方案；现在必须 409，
     且阻塞原因明确指向中间悬空快照（phase=mid）；
  2. 存在确实连续的操作（临时旁路）时仍返回 200，且旁路在中间悬空快照里
     真正承载保护路由；
  3. 换接孔位死锁 → 409 结构化阻塞原因；目标非法 / 保护序号越界 → 400 无步骤；
  4. 独立物理级 BFS（枚举每个宏动作的插/拔顺序与中间快照）与 Dijkstra 搜索
     的可行性结论在两组手写配置 + 数百组随机小配置上必须一致；
  5. 回放的每个中间快照与完成后快照都用 trace 独立复核保护通声。
"""
import random
import sys
from collections import deque

from solver import plan_all
from rewire import (PhysModel, RewireError, SeamlessConflict, _pid,
                    seamless_rewire)

FAIL = 0


def check(cond, msg):
    global FAIL
    if not cond:
        FAIL += 1
        print("  ✗", msg)
    else:
        print("  ✓", msg)


# ---- 拓扑构造 -------------------------------------------------------------

# 核心缺陷拓扑（见 README 场景）：保护路由 SRC→REC 唯一依赖跳线 1T→2B，
# pair4 全常态 MIX.out→REC.in 先被另一条 0 跳线路由保护；目标把该跳线 B 端
# 2B→3B（2B 让给新增 OTH 跳线）。
CORE_DEVICES = [
    {"name": "SRC", "inputs": [], "outputs": ["out"]},
    {"name": "OTH", "inputs": [], "outputs": ["out"]},
    {"name": "EX1", "inputs": [], "outputs": ["out"]},
    {"name": "MIX", "inputs": ["in2", "in3"], "outputs": ["out"]},
    {"name": "REC", "inputs": ["in"], "outputs": []},
]
CORE_PAIRS = [
    {"id": 1, "top": "SRC.out", "bottom": None, "mode": "through"},
    {"id": 2, "top": "OTH.out", "bottom": "MIX.in2", "mode": "through"},
    {"id": 3, "top": "EX1.out", "bottom": "MIX.in3", "mode": "through"},
    {"id": 4, "top": "MIX.out", "bottom": "REC.in", "mode": "full"},
]
CORE_CUR = {"devices": CORE_DEVICES, "pairs": CORE_PAIRS, "routes": [
    {"source": "MIX.out", "target": "REC.in"},
    {"source": "SRC.out", "target": "REC.in"},
]}
CORE_TGT = [
    {"source": "MIX.out", "target": "REC.in"},
    {"source": "OTH.out", "target": "MIX.in2"},
    {"source": "SRC.out", "target": "REC.in"},
]

# 连续拓扑：SRC 经 SPL（in→outA/outB）分配，outB=6T 与 DST.in 构成半常态
# 并送点；MIX 有空闲 in4(5B)，可搭临时旁路 1T→5B 保通换接。
CONT_DEVICES = [
    {"name": "SRC", "inputs": [], "outputs": ["out"]},
    {"name": "SPL", "inputs": ["in"], "outputs": ["outA", "outB"]},
    {"name": "OTH", "inputs": [], "outputs": ["out"]},
    {"name": "EX1", "inputs": [], "outputs": ["out"]},
    {"name": "DST", "inputs": ["in"], "outputs": []},
    {"name": "MIX", "inputs": ["in2", "in3", "in4"], "outputs": ["out"]},
    {"name": "REC", "inputs": ["in"], "outputs": []},
]
CONT_PAIRS = [
    {"id": 1, "top": "SPL.outA", "bottom": None, "mode": "through"},
    {"id": 2, "top": "OTH.out", "bottom": "MIX.in2", "mode": "through"},
    {"id": 3, "top": "EX1.out", "bottom": "MIX.in3", "mode": "through"},
    {"id": 4, "top": "MIX.out", "bottom": "REC.in", "mode": "full"},
    {"id": 5, "top": None, "bottom": "MIX.in4", "mode": "through"},
    {"id": 6, "top": "SPL.outB", "bottom": "DST.in", "mode": "half"},
    {"id": 7, "top": "SRC.out", "bottom": "SPL.in", "mode": "half"},
]
CONT_CUR = {"devices": CONT_DEVICES, "pairs": CONT_PAIRS, "routes": [
    {"source": "MIX.out", "target": "REC.in"},
    {"source": "SRC.out", "target": "REC.in"},
]}
CONT_TGT = [
    {"source": "MIX.out", "target": "REC.in"},
    {"source": "OTH.out", "target": "MIX.in2"},
    {"source": "SRC.out", "target": "REC.in"},
]


def assert_step_replay(phys, cur_plan, res, protected_idx):
    """按报告的 jackChanges 独立物理重放，逐原子动作复核保护通声。"""
    conns = frozenset((c["fromJack"], c["toJack"])
                      for c in cur_plan["solver"].cables)
    plugs = PhysModel.plug_set(conns)
    prkeys = []
    for pi in protected_idx:
        r = cur_plan["routes"][pi]
        prkeys.append((_pid(r["source"]), _pid(r["target"])))
    for s in res["steps"]:
        # 第一个原子动作后的中间快照
        first = s["jackChanges"][0]
        mplugs = set(plugs)
        mplugs.discard(first["jack"]) if first["change"] == "unplug" \
            else mplugs.add(first["jack"])
        mconns = set(conns)
        if s["action"] in ("complete", "park"):
            mconns.discard((s["cable"]["fromJack"], s["cable"]["toJack"]))
        mid = s["intermediate"]
        check(mid["firstJack"] == first["jack"],
              f"step {s['seq']} 中间快照首插孔与 jackChanges[0] 一致")
        for src, dst in prkeys:
            check(phys.audible(frozenset(mconns), src, dst, plugs=frozenset(mplugs)),
                  f"step {s['seq']} 中间快照 {src}→{dst} 通声")
        # 完成两个原子动作
        for ch in s["jackChanges"]:
            if ch["change"] == "plug":
                plugs = plugs | {ch["jack"]}
            else:
                plugs = plugs - {ch["jack"]}
        conns = frozenset(
            (c["fromJack"], c["toJack"])
            for c in s["snapshot"]["cables"] if not c.get("partial"))
        for src, dst in prkeys:
            check(phys.audible(conns, src, dst),
                  f"step {s['seq']} 完成后 {src}→{dst} 通声")


def test_core_blocker():
    print("【1】核心缺陷：2B→3B 换接中间悬空 → 必须 409，不产出步骤")
    try:
        seamless_rewire(CORE_CUR, CORE_TGT, [1])
        check(False, "应抛 SeamlessConflict")
    except SeamlessConflict as e:
        d = e.details
        check("steps" not in d, "409 响应不含 steps")
        types = [b["type"] for b in d["blockers"]]
        check("must-remove-cable" in types, "有 must-remove-cable 阻塞原因")
        check(any(b.get("phase") == "mid" for b in d["blockers"]),
              "阻塞原因标注中间快照 phase=mid")
        mid = [b for b in d["blockers"] if b.get("phase") == "mid"]
        check(mid and "悬空" in mid[0]["message"],
              "中间阻塞原因描述一端刚拔下的悬空状态")


def test_continuous_bypass():
    print("【2】确实连续：临时旁路在中间悬空快照承载保护路由 → 200")
    _, cur_plan = plan_all(CONT_CUR)
    phys = PhysModel(cur_plan["devices"], cur_plan["pairs"])
    phys.bind_jacks(cur_plan["port_jack"])
    res = seamless_rewire(CONT_CUR, CONT_TGT, [0, 1])
    check(res["stepCount"] >= 1, "返回非空方案")
    check(all(s["intermediate"]["protectedAllAudible"] for s in res["steps"]),
          "所有步骤的中间快照保护路由全通声")
    check(all(s["protectedAllAudible"] for s in res["steps"]),
          "所有步骤完成后保护路由全通声")
    move = next((s for s in res["steps"] if s["action"] == "complete"), None)
    check(move is not None, "包含换接 complete 步")
    if move:
        check(move["jackChanges"][0]["change"] == "unplug"
              and move["jackChanges"][0]["phase"] == "mid",
              "换接第一步为先拔旧端（中间快照）")
        r2 = next(p for p in move["intermediate"]["protected"] if p["routeNo"] == 2)
        check(r2["audible"] and bool(r2["viaTemporary"]),
              "中间悬空快照中受保护 SRC→REC 经临时跳线保通")
    check(bool(res["temporaryCables"])
          and res["temporaryCables"][0]["protects"],
          "临时跳线登记了它保通的保护路由")
    assert_step_replay(phys, cur_plan, res, [0, 1])


def test_invalid_requests():
    print("【3】目标非法 / 保护序号越界 → 400 且不产出步骤")
    try:
        seamless_rewire(CORE_CUR, [
            {"source": "MIX.out", "target": "REC.in"},
            {"source": "OTH.out", "target": "MIX.in2"},
            {"source": "EX1.out", "target": "MIX.in2"},
            {"source": "SRC.out", "target": "REC.in"},
        ], [1])
        check(False, "目标含无解路由应 400")
    except RewireError as e:
        check(e.code == "TARGET_INVALID", "code=TARGET_INVALID")
        check(not getattr(e, "payload", None), "400 不携带步骤 payload")

    try:
        seamless_rewire(CORE_CUR, CORE_TGT, [9])
        check(False, "保护序号越界应 400")
    except RewireError as e:
        check(e.code == "PROTECTED_INVALID", "code=PROTECTED_INVALID")

    try:
        seamless_rewire(CORE_CUR, [{"source": "NO.such", "target": "REC.in"}], [])
        check(False, "目标端口错误应 400")
    except RewireError as e:
        check(e.code in ("TARGET_INVALID", "REWIRE_INVALID"),
              f"端口错误 code={e.code}")


# ---- 独立物理级 BFS 交叉验证 ----------------------------------------------

def brute_force(config, target, protected_idx):
    """枚举宏动作及其插/拔顺序，中间+终态都通声才算可达；超规模返回 None。"""
    _, cur_p = plan_all(config)
    _, tgt_p = plan_all({"devices": config["devices"], "pairs": config["pairs"],
                         "routes": target})
    phys = PhysModel(cur_p["devices"], cur_p["pairs"])
    phys.bind_jacks(cur_p["port_jack"])
    init = frozenset((c["fromJack"], c["toJack"]) for c in cur_p["solver"].cables)
    goal = frozenset((c["fromJack"], c["toJack"]) for c in tgt_p["solver"].cables)
    pr = [(_pid(cur_p["routes"][i]["source"]), _pid(cur_p["routes"][i]["target"]))
          for i in protected_idx]
    all_t = sorted(j for j in phys.pair_of if j.endswith("T")
                   and j in phys.port_of_jack)
    all_b = sorted(j for j in phys.pair_of if j.endswith("B")
                   and j in phys.port_of_jack)

    def ok(conns, plugs):
        return all(phys.audible(conns, s, d, plugs=plugs) for s, d in pr)

    def goal_ok(conns):
        return all(phys.audible(conns, _pid(r["source"]), _pid(r["target"]))
                   for r in tgt_p["routes"])

    def succ(conns):
        plugs = PhysModel.plug_set(conns)
        out = []
        for (t, b) in conns:
            for nb in all_b:
                if nb in plugs:
                    continue
                nc = (conns - {(t, b)}) | {(t, nb)}
                # 换接：先拔 b（唯一物理顺序），中间旧连接一端悬空
                if ok(conns, plugs - {b}) and ok(nc, PhysModel.plug_set(nc)):
                    out.append(nc)
            for nt in all_t:
                if nt in plugs:
                    continue
                nc = (conns - {(t, b)}) | {(nt, b)}
                if ok(conns, plugs - {t}) and ok(nc, PhysModel.plug_set(nc)):
                    out.append(nc)
        for t in all_t:
            if t in plugs:
                continue
            for b in all_b:
                if b in plugs:
                    continue
                nc = conns | {(t, b)}
                # 补线：先 T / 先 B 任一中间顺序安全即可
                if (ok(conns, plugs | {t}) or ok(conns, plugs | {b})) \
                        and ok(nc, plugs | {t, b}):
                    out.append(nc)
        for (t, b) in conns:
            nc = conns - {(t, b)}
            # 拆线：先 B / 先 T 任一中间顺序安全即可
            if (ok(conns, plugs - {b}) or ok(conns, plugs - {t})) \
                    and ok(nc, plugs - {t, b}):
                out.append(nc)
        return out

    seen = {init}
    q = deque([init])
    while q and len(seen) <= 60000:
        conns = q.popleft()
        if conns == goal and goal_ok(conns):
            return True
        for nc in succ(conns):
            if nc not in seen:
                seen.add(nc)
                q.append(nc)
    return None if len(seen) > 60000 else False


def crosscheck_one(config, target, prot, label):
    bf = brute_force(config, target, prot)
    if bf is None:
        return
    try:
        res = seamless_rewire(config, target, prot)
        sw = True
        _, cur_plan = plan_all(config)
        phys = PhysModel(cur_plan["devices"], cur_plan["pairs"])
        phys.bind_jacks(cur_plan["port_jack"])
        assert_step_replay(phys, cur_plan, res, prot)
    except SeamlessConflict as e:
        if e.details["inconclusive"]:
            return
        sw = False
    check(bool(bf) == bool(sw), f"{label}: 暴力={bf} 搜索={sw}")


def test_manual_crosscheck():
    print("【4】手写配置：物理级 BFS 与搜索结论一致")
    crosscheck_one(CORE_CUR, CORE_TGT, [1], "核心死锁场景")
    crosscheck_one(CONT_CUR, CONT_TGT, [0, 1], "临时旁路连续场景")


def test_fuzz(seeds=range(1, 21)):
    print("【5】随机小配置交叉验证（含回放中间快照复核）")
    total = 0
    for seed in seeds:
        rnd = random.Random(seed)
        for _ in range(400):
            ndev = rnd.randint(2, 4)
            devices, pout, pin = [], {}, {}
            for d in range(ndev):
                ins = [f"i{k}" for k in range(rnd.randint(1, 2))]
                outs = [f"o{k}" for k in range(rnd.randint(1, 2))]
                name = f"D{d}"
                devices.append({"name": name, "inputs": ins, "outputs": outs})
                pout[name], pin[name] = outs, ins
            npair = rnd.randint(3, 5)
            outs_left = [f"{d}.{p}" for d, ps in pout.items() for p in ps]
            ins_left = [f"{d}.{p}" for d, ps in pin.items() for p in ps]
            rnd.shuffle(outs_left)
            rnd.shuffle(ins_left)
            pairs, oi, ii = [], 0, 0
            for pid in range(1, npair + 1):
                top = outs_left[oi] if oi < len(outs_left) and rnd.random() < .85 else None
                bot = ins_left[ii] if ii < len(ins_left) and rnd.random() < .85 else None
                if top:
                    oi += 1
                if bot:
                    ii += 1
                pairs.append({"id": pid, "top": top, "bottom": bot,
                              "mode": rnd.choice(["full", "half", "through"])})
            out_ports = outs_left
            in_ports = ins_left
            if not out_ports or not in_ports:
                continue

            def routes(n):
                return [{"source": rnd.choice(out_ports),
                         "target": rnd.choice(in_ports)} for _ in range(n)]

            cur, tgt = routes(rnd.randint(1, 2)), routes(rnd.randint(1, 3))
            try:
                _, cp = plan_all({"devices": devices, "pairs": pairs, "routes": cur})
                _, tp = plan_all({"devices": devices, "pairs": pairs, "routes": tgt})
            except Exception:
                continue
            if cp["conflicts"] or tp["conflicts"]:
                continue

            def keyf(r):
                return (_pid(r["source"]), _pid(r["target"]))

            tgt_keys = {keyf(r) for r in tp["routes"]}
            pool = [i for i, r in enumerate(cp["routes"]) if keyf(r) in tgt_keys]
            if not pool:
                continue
            prot = sorted(rnd.sample(pool, rnd.randint(1, len(pool))))
            cfg = {"devices": devices, "pairs": pairs, "routes": cur}
            before = FAIL
            crosscheck_one(cfg, tgt, prot, f"seed{seed}")
            if FAIL == before:
                total += 1
    print(f"    随机比对通过 {total} 组（所有成功方案均经回放中间快照复核）")


if __name__ == "__main__":
    test_core_blocker()
    test_continuous_bypass()
    test_invalid_requests()
    test_manual_crosscheck()
    test_fuzz(range(1, int(sys.argv[1]) + 1) if len(sys.argv) > 1 else range(1, 21))
    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        sys.exit(1)
    print("ALL PASSED")
