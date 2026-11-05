"""事例 JSON を1つの CSV にまとめ、時刻と元の発話 ID を付ける。集計表も作る。
使い方: python3 build_cases.py <precheck_root> <explore_root>
"""
import json, os, sys, csv, collections
pre, ex = sys.argv[1], sys.argv[2]
MS = ['GAP01', 'GAP07', 'GAP08', 'GAP18', 'GAP24', 'ES2002c', 'ES2003c', 'ES2004c']
rows, segs, props, issues = [], [], [], []
for m in MS:
    turns = json.load(open(os.path.join(pre, 'data', f'{m}.turns.json')))['turns']; byt = {t['t']: t for t in turns}
    d = json.load(open(os.path.join(ex, 'cases', f'{m}.json')))
    cov = collections.Counter()
    for s in d['segments']:
        a, b = s['turns']
        for t in range(a, b + 1): cov[t] += 1
        segs.append({'meeting': m, 'seg': s['seg'], 'turn_from': a, 'turn_to': b, 'start_s': byt[a]['start'], 'end_s': byt[b]['end'], 'item': s['item'], 'phase': '・'.join(s['phase'])})
    miss = [t for t in byt if cov[t] != 1]
    if miss: issues.append(f'{m}: 区間の重複・漏れ {miss[:10]}（{len(miss)}）')
    segids = {s['seg'] for s in d['segments']}
    for c in d['cases']:
        a, b = c['turns']
        if a not in byt or b not in byt: issues.append(f"{c['case']}: ターン範囲が不正 {c['turns']}"); continue
        if c['seg'] not in segids: issues.append(f"{c['case']}: 区間 {c['seg']} がない")
        rows.append({'meeting': m, 'source': 'GAP' if m.startswith('GAP') else 'AMI', 'coder': d.get('coder', ''), 'case': c['case'], 'seg': c['seg'],
                     'turn_from': a, 'turn_to': b, 'start_s': byt[a]['start'], 'end_s': byt[b]['end'],
                     'source_ids': ' '.join(sid for t in range(a, b + 1) for sid in byt[t]['source_ids']) if b - a <= 40 else ' '.join(byt[a]['source_ids'][:1] + ['...'] + byt[b]['source_ids'][-1:]),
                     'kind': c['kind'], 'codes': ' '.join(c['codes']), 'context': c['context'], 'observed': c['observed'], 'interpretation': c['interpretation'],
                     'alternative': c['alternative'], 'followup': c['followup'], 'support': c['support'], 'support_note': c.get('support_note', ''), 'uncertainty': c['uncertainty']})
    for p in d.get('new_code_proposals', []) or []:
        props.append({'meeting': m, **{k: (json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v) for k, v in p.items()}})
od = os.path.join(ex, 'cases')
def w(name, data):
    keys = []
    for r in data:
        for k in r:
            if k not in keys: keys.append(k)
    with open(os.path.join(od, name), 'w', newline='', encoding='utf-8-sig') as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); [wr.writerow(r) for r in data]
w('cases_all.csv', rows); w('segments_all.csv', segs); w('new_code_proposals.csv', props)
# 集計
kinds = ['困難', '有用な行為', '人が自力で解消', '他者の働きかけで前進', '停滞に見えて問題なし', '再検討せず合理的', '判断不能', '困難なし区間']
tab = []
for m in MS:
    rs = [r for r in rows if r['meeting'] == m]
    row = {'meeting': m, 'segments': sum(s['meeting'] == m for s in segs), 'cases': len(rs)}
    for k in kinds: row[k] = sum(r['kind'] == k for r in rs)
    for k in ['役立ちそう', '不要', '判断不能']: row['支援:' + k] = sum(r['support'] == k for r in rs)
    cc = collections.Counter(code for r in rs for code in r['codes'].split())
    for code in ['K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9']: row[code] = cc.get(code, 0)
    tab.append(row)
w('summary_by_meeting.csv', tab)
open(os.path.join(od, 'integrity_issues.txt'), 'w').write('\n'.join(issues) + '\n')
print('cases', len(rows), 'segments', len(segs), 'proposals', len(props)); print('\n'.join(issues))
for r in tab: print(r)
