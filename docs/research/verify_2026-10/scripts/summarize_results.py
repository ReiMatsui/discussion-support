"""議論ごとの分布、感度分析、判断基準の機械的な適用をまとめる。"""
import json, os, sys, csv, collections, glob
root = sys.argv[1]
R = lambda n: list(csv.DictReader(open(os.path.join(root, 'results', n), encoding='utf-8-sig')))
tp = R('timepoint_all.csv'); pu = R('pairs_unique.csv'); sc = R('audit_scenes.csv'); ap = R('audit_pairs.csv'); mi = R('audit_misses.csv')
sup = json.load(open(os.path.join(root, 'results', 'audit_support_eval.json')))
MS = sorted({r['meeting'] for r in tp}, key=lambda m: ({'GAP': 0, 'ES2': 1, 'IS1': 1, 'TS3': 1, 'DEL': 2}[m[:3]], m))
rows = []
for m in MS:
    t = [r for r in tp if r['meeting'] == m]; p = [r for r in pu if r['meeting'] == m and r['has_main'] == 'True']
    s1 = [r for r in sc if r['meeting'] == m and r['auditor'] == 'AU1' and r['main'] == 'True']
    s2 = [r for r in sc if r['meeting'] == m and r['auditor'] == 'AU2' and r['main'] == 'True']
    rows.append({'議論': m, '課題': t[0]['source'], '判定時点': len(t), '支援': sum(r['decision'] == 'support' for r in t),
                 '控える': sum(r['decision'] == 'withhold' for r in t), '判断不能': sum(r['decision'] == 'undecidable' for r in t),
                 '主分析の対（重複除き）': len(p), 'うち最初の時点で既に明示': sum(r['explicit_at_first_main'] == 'True' for r in p),
                 '推測のみの対': sum(1 for r in pu if r['meeting'] == m and r['has_main'] == 'False'),
                 '全体分析の主場面 AU1/AU2': f'{len(s1)}/{len(s2)}',
                 '人が言葉にした AU1/AU2': f"{sum((r['explicit_turn'] or 'なし') != 'なし' for r in s1)}/{sum((r['explicit_turn'] or 'なし') != 'なし' for r in s2)}",
                 '判断に支障あり AU1/AU2': f"{sum(r['impairment'] == 'あった' for r in s1)}/{sum(r['impairment'] == 'あった' for r in s2)}",
                 '見落とし（支援条件を満たす）AU1/AU2': f"{sum(r['meeting'] == m and r['auditor'] == 'AU1' and r.get('support_condition_met') == 'はい' for r in mi)}/{sum(r['meeting'] == m and r['auditor'] == 'AU2' and r.get('support_condition_met') == 'はい' for r in mi)}"})
with open(os.path.join(root, 'results', 'per_discussion.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
for r in rows: print(r)
# 段階2〜4（全体分析の主場面）
for au in ('AU1', 'AU2'):
    ss = [r for r in sc if r['auditor'] == au and r['main'] == 'True']
    print(au, '主場面', len(ss), {k: dict(collections.Counter(r[k] for r in ss)) for k in ('stage2', 'stage3', 'stage4', 'self_handled', 'impairment')})
# 支援出力の基準判定
def clean(e): return all(e['B'].get(k, [''])[0] == '満たす' for k in ('B1', 'B2', 'B4'))
def critical(e): return any(e['B'].get(k, [''])[0] == '満たさない' for k in ('B1', 'B2', 'B4'))
def unnec(e): return any(str(e['C'].get(k, [''])[0]) in ('当たる', 'はい') for k in ('C1', 'C2', 'C3'))
def unnec_or_unc(e): return any(str(e['C'].get(k, [''])[0]) in ('当たる', 'はい', '不確か') for k in ('C1', 'C2', 'C3'))
print('== 支援出力')
for m, d in sup.items():
    for au, lst in d.items():
        for e in lst: print(m, au, e['checkpoint'], 'clean(B0-2,4満たす)', clean(e), 'critical', critical(e), 'unnecessary', unnec(e), 'unnec_or_uncertain', unnec_or_unc(e))
