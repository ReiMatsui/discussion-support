"""事後評価（AU1・AU2）の集計。分かれた評価は統合せずに両方残し、保守側・寛容側の両方で数える。"""
import json, os, sys, glob, collections, csv
root = sys.argv[1]
A = collections.defaultdict(dict); F = collections.defaultdict(dict)
for p in glob.glob(os.path.join(root, 'audit', '*.json')):
    b = os.path.basename(p)[:-5]
    au, rest = b.split('_', 1)
    if rest.endswith('_full'): F[rest[:-5]][au] = json.load(open(p))
    else: A[rest][au] = json.load(open(p))
MS = sorted(set(F) | set(A))
def v(x): return x[0] if isinstance(x, list) and x else (x if isinstance(x, str) else '')
# 全体分析
scene_rows = []
for m in MS:
    for au, d in F[m].items():
        for s in d.get('scenes', []):
            lv = [r.get('level') for r in s.get('reasons', [])]
            main = all(l in ('E1', 'E2') for l in lv) and len(lv) >= 2
            scene_rows.append({'meeting': m, 'auditor': au, 'scene': s.get('scene_id'), 'turns': s.get('turns'), 'item': s.get('choice_item'),
                               'main': main, 'explicit_turn': s.get('explicit_turn'), 'reconciled': s.get('reconciled'),
                               'stage2': v(s.get('stage2')), 'stage3': v(s.get('stage3')), 'stage4': v(s.get('stage4')),
                               'self_handled': s.get('self_handled'), 'impairment': v(s.get('impairment')),
                               'impairment_reason': s.get('impairment')[1] if isinstance(s.get('impairment'), list) and len(s.get('impairment')) > 1 else ''})
# 対の評価
pair_rows = []
for m in MS:
    for au, d in A[m].items():
        for e in d.get('pairs_eval', []):
            pair_rows.append({'meeting': m, 'auditor': au, 'pair_uid': e.get('pair_uid'), 'B1': v(e.get('B1')), 'B2': v(e.get('B2')), 'B3': v(e.get('B3')), 'B6': v(e.get('B6')),
                              'explicit_judgement': v(e.get('explicit_judgement')), 'first_explicit_turn': e.get('first_explicit_turn'), 'scene': e.get('full_analysis_scene')})
miss_rows = []
for m in MS:
    for au, d in A[m].items():
        for e in d.get('misses', []):
            miss_rows.append({'meeting': m, 'auditor': au, **{k: (json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x) for k, x in e.items()}})
sup = {m: {au: d.get('support_eval', []) for au, d in A[m].items()} for m in MS}
os.makedirs(os.path.join(root, 'results'), exist_ok=True)
def w(name, rows):
    if not rows: return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(os.path.join(root, 'results', name), 'w', newline='', encoding='utf-8-sig') as f:
        ww = csv.DictWriter(f, fieldnames=keys); ww.writeheader(); [ww.writerow(r) for r in rows]
w('audit_scenes.csv', scene_rows); w('audit_pairs.csv', pair_rows); w('audit_misses.csv', miss_rows)
json.dump(sup, open(os.path.join(root, 'results', 'audit_support_eval.json'), 'w'), ensure_ascii=False, indent=1)
# 集計
print('== 全体分析: 議論ごとの場面（主分析＝両方E1/E2）')
for m in MS:
    for au in ('AU1', 'AU2'):
        rs = [r for r in scene_rows if r['meeting'] == m and r['auditor'] == au]
        mm = [r for r in rs if r['main']]
        print(m, au, 'scenes', len(rs), 'main', len(mm), 'explicit', sum((r['explicit_turn'] or 'なし') != 'なし' for r in mm),
              'self', sum(r['self_handled'] == 'はい' for r in mm), 'st2', sum(r['stage2'] == '起きた' for r in mm), 'st3', sum(r['stage3'] == '起きた' for r in mm),
              'st4', sum(r['stage4'] == '起きた' for r in mm), 'impair', collections.Counter(r['impairment'] for r in mm))
print('== 対の評価（AU別）')
for au in ('AU1', 'AU2'):
    rs = [r for r in pair_rows if r['auditor'] == au]
    print(au, len(rs), {k: dict(collections.Counter(r[k] for r in rs)) for k in ('B1', 'B2', 'B3', 'B6', 'explicit_judgement')})
# 一致
byk = collections.defaultdict(dict)
for r in pair_rows: byk[r['pair_uid']][r['auditor']] = r
for k in ('B1', 'B2', 'B3', 'B6', 'explicit_judgement'):
    both = [x for x in byk.values() if 'AU1' in x and 'AU2' in x]
    print('agree', k, sum(x['AU1'][k] == x['AU2'][k] for x in both), '/', len(both))
print('== 見落とし', collections.Counter((r['auditor'], r.get('support_condition_met')) for r in miss_rows))
