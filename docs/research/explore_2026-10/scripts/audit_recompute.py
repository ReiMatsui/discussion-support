"""旧事前検証（precheck_2026-10）の集計を、判定ログから独立に再計算する。
旧ファイルは読むだけで、書き込みは explore/audit/recompute/ にだけ行う。
使い方: python3 audit_recompute.py <precheck_root> <outdir>
"""
import json, os, sys, csv, base64, zlib, collections, statistics
root, outdir = sys.argv[1], sys.argv[2]
SPL = {'dev': ['GAP01', 'ES2002c'], 'test': ['GAP07', 'GAP08', 'GAP18', 'GAP24', 'ES2003c', 'ES2004c']}
def jl(p): return [json.loads(l) for l in open(p)]
rows = []; issues = []
for split, ms in SPL.items():
    for m in ms:
        run = os.path.join(root, 'runs', split, m)
        turns = json.loads(zlib.decompress(base64.b64decode(open(os.path.join(run, '.sealed'), 'rb').read())))['turns']
        byt = {t['t']: t for t in turns}
        js = jl(os.path.join(run, 'judgments.jsonl'))
        shown = open(os.path.join(run, 'shown.log')).read().splitlines()
        # 表示記録と判定の対応: 表示は元の順序どおりか
        shown_t = [int(l.split(']')[0][2:]) for l in shown]
        if shown_t != list(range(1, len(turns) + 1)): issues.append(f'{m}: shown.log の順序が元の順序と一致しない')
        poss = [j['pos'] for j in js]
        if poss != sorted(poss): issues.append(f'{m}: 判定の pos が単調増加でない')
        opened = set(); evcount = collections.Counter()
        for j in js:
            for e in j.get('episode_events') or []:
                evcount[e.get('event')] += 1
                if e.get('event') == 'open': opened.add(e.get('ep_id'))
        A = [(j['pos'], j['candidate']) for j in js if j.get('candidate')]
        B = [(j['pos'], j['policy_b']) for j in js if j.get('policy_b')]
        eps = collections.defaultdict(list)
        for pos, c in A: eps[c['ep_id']].append(c)
        cand_ok = [c for _, c in A if c['present_ok'] == 'yes']
        ep_ok = [e for e, cs in eps.items() if any(c['present_ok'] == 'yes' for c in cs)]
        ep_gen = [e for e, cs in eps.items() if any(c['generatable'] == 'yes' for c in cs)]
        cand_ep_not_opened = sorted(set(eps) - opened)
        gen_noprop = [c.get('cand_id') for _, c in A if c['generatable'] == 'yes' and not c['proposal_ja']]
        dup_ids = [k for k, v in collections.Counter(c.get('cand_id') for _, c in A).items() if v > 1]
        if cand_ep_not_opened: issues.append(f'{m}: open イベントのないエピソードに候補 {cand_ep_not_opened}')
        if gen_noprop: issues.append(f'{m}: 生成可能 yes なのに文面なし {gen_noprop}')
        if dup_ids: issues.append(f'{m}: cand_id の重複 {dup_ids}')
        # 第2段階
        s2p = os.path.join(run, 'stage2_output.json')
        s2 = {i['id']: i for i in json.load(open(s2p))['items']} if os.path.exists(s2p) else {}
        okids = [c.get('cand_id') for c in cand_ok]
        pre = [i for i in okids if s2.get(i, {}).get('labels', {}).get('S1')]
        upd = [i for i in okids if s2.get(i, {}).get('labels', {}).get('S2') or s2.get(i, {}).get('labels', {}).get('S3')]
        sub = [i for i in upd if s2[i].get('update_type') == 'substantive']
        rows.append({'split': split, 'meeting': m, 'turns': len(turns), 'minutes': round((turns[-1]['end'] - turns[0]['start']) / 60, 1),
                     'judgments': len(js), 'episodes_opened': len(opened), 'episodes_with_candidate': len(eps),
                     'candidate_records': len(A), 'cand_present_ok': len(cand_ok), 'ep_generatable': len(ep_gen), 'ep_present_ok': len(ep_ok),
                     'B_checkpoints': len(B), 'B_gen_yes': sum(b['generatable'] == 'yes' for _, b in B),
                     'ok_preempt_S1': len(pre), 'ok_update_S2S3': len(upd), 'ok_update_substantive': len(sub),
                     'stage2_items': len(s2), 'events': dict(evcount)})
with open(os.path.join(outdir, 'per_meeting_recomputed.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
open(os.path.join(outdir, 'integrity_issues.txt'), 'w').write('\n'.join(issues) + '\n')
# 旧 per_meeting.csv との突き合わせ
old = {r['meeting']: r for r in csv.DictReader(open(os.path.join(root, 'out', 'per_meeting.csv'), encoding='utf-8-sig'))}
cmp = []
for r in rows:
    o = old[r['meeting']]
    for a, b in [('episodes_opened', 'episodes_opened'), ('episodes_with_candidate', 'episodes_with_candidate'), ('candidate_records', 'candidate_records'),
                 ('ep_generatable', 'ep_generatable'), ('ep_present_ok', 'ep_present_ok'), ('B_checkpoints', 'B_checkpoints'), ('B_gen_yes', 'B_generatable'),
                 ('ok_preempt_S1', 'okcand_preempted_S1'), ('ok_update_S2S3', 'okcand_update_info_S2S3'), ('ok_update_substantive', 'update_substantive')]:
        cmp.append({'meeting': r['meeting'], 'item': a, 'recomputed': r[a], 'old': o[b], 'match': str(r[a]) == str(o[b])})
with open(os.path.join(outdir, 'old_vs_recomputed.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(cmp[0].keys())); w.writeheader(); [w.writerow(r) for r in cmp]
print('mismatches', [c for c in cmp if not c['match']])
print('\n'.join(issues))
for r in rows: print(r)
