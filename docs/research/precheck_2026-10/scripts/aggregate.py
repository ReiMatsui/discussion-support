"""集計: 全候補 CSV、議論ごとの集計、方針Bの集計、品質評価の集計を作る。
使い方: python3 aggregate.py  （runs/dev と runs/test を読み、out/ に書く）
"""
import json, os, csv, base64, zlib, collections, statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'out'); os.makedirs(OUT, exist_ok=True)
SPLITS = {'dev': ['GAP01', 'ES2002c'], 'test': ['GAP07', 'GAP08', 'GAP18', 'GAP24', 'ES2003c', 'ES2004c']}
QN = [str(i) for i in range(1, 9)]


def sealed(run):
    return json.loads(zlib.decompress(base64.b64decode(open(os.path.join(run, '.sealed'), 'rb').read())))


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def conservative(a, b):
    vs = [x for x in (a, b) if x]
    if not vs: return ''
    if '満たさない' in vs: return '満たさない'
    if '不確か' in vs or len(vs) < 2: return '不確か' if '不確か' in vs else vs[0]
    return '満たす'


rows, mrows, brows = [], [], []
qagree = collections.Counter(); qtotal = collections.Counter()
for split, meetings in SPLITS.items():
    for m in meetings:
        run = os.path.join(ROOT, 'runs', split, m)
        d = sealed(run); turns = d['turns']; byt = {t['t']: t for t in turns}
        js = jl(os.path.join(run, 'judgments.jsonl'))
        s2 = {}
        p2 = os.path.join(run, 'stage2_output.json')
        if os.path.exists(p2):
            s2 = {i['id']: i for i in json.load(open(p2))['items']}
        qj = {}
        for J in ('J1', 'J2'):
            p = os.path.join(run, f'quality_{J}.json')
            if os.path.exists(p):
                qj[J] = {i['id']: i['q'] for i in json.load(open(p))['items']}
        eps_opened = set(); ep_events = collections.defaultdict(list)
        for j in js:
            for e in j.get('episode_events') or []:
                ep_events[e.get('ep_id')].append((j['pos'], e.get('event')))
                if e.get('event') == 'open': eps_opened.add(e.get('ep_id'))
        def stage2_fields(cid, pos):
            o = s2.get(cid)
            if not o: return {}
            L = o['labels']; f = {}
            for k in ('S1', 'S2', 'S3', 'S4', 'S5'):
                v = L.get(k)
                f[k + '_turn'] = v if isinstance(v, int) else ''
                f[k + '_dturns'] = (v - pos) if isinstance(v, int) else ''
                f[k + '_dsec'] = round(byt[v]['start'] - byt[pos]['end'], 1) if isinstance(v, int) and v in byt else ''
            s6 = L.get('S6')
            f['S6'] = f"{s6['type']}@t{s6['turn']}" if isinstance(s6, dict) else ''
            f['S7'] = bool(L.get('S7'))
            f['answers_question'] = (o.get('answers_question') or {}).get('value', '')
            f['update_type'] = o.get('update_type', '')
            f['update_example_offline_ja'] = o.get('update_example_ja') or ''
            f['stage2_note'] = o.get('note', '')
            return f
        def quality_fields(cid):
            f = {}
            if not qj: return f
            for q in QN:
                a = qj.get('J1', {}).get(cid, {}).get(q, ['', ''])
                b = qj.get('J2', {}).get(cid, {}).get(q, ['', ''])
                f[f'Q{q}_J1'] = a[0]; f[f'Q{q}_J2'] = b[0]
                f[f'Q{q}_final'] = conservative(a[0], b[0])
                f[f'Q{q}_reasons'] = f'J1:{a[1]} / J2:{b[1]}'
                if a[0] and b[0]:
                    qtotal[q] += 1; qagree[q] += (a[0] == b[0])
            fin = [f.get(f'Q{q}_final', '') for q in QN]
            f['Q_fail_items'] = ','.join(q for q, v in zip(QN, fin) if v == '満たさない')
            f['Q_uncertain_items'] = ','.join(q for q, v in zip(QN, fin) if v == '不確か')
            return f
        cands = []
        for j in js:
            pos = j['pos']; c = j.get('candidate')
            if c:
                r = {'split': split, 'meeting': m, 'id': c.get('cand_id'), 'kind': 'A', 'pos': pos,
                     'time_s': byt[pos]['end'], 'ep_id': c['ep_id'], 'choice_item': c['choice_item'],
                     'options': ' | '.join(map(str, c['options'])),
                     'ground_turns': ','.join(str(g.get('turn')) for g in c['shared_grounds'] if isinstance(g, dict)),
                     'ground_source_ids': ','.join(sid for g in c['shared_grounds'] if isinstance(g, dict) and g.get('turn') in byt for sid in byt[g['turn']]['source_ids']),
                     'grounds_quote_original': ' || '.join(f"t{g.get('turn')}: {g.get('quote')}" for g in c['shared_grounds'] if isinstance(g, dict)),
                     'grounds_gloss_ja': ' || '.join(f"t{g.get('turn')}: {g.get('gloss')}" for g in c['shared_grounds'] if isinstance(g, dict)),
                     'explicit_conditions': ' | '.join(map(str, c['explicit_conditions'])),
                     'unconfirmed_conditions': ' | '.join(map(str, c['unconfirmed_conditions'])),
                     'human_already_comparing': c['human_already_comparing'], 'uncompared_content': c['uncompared_content'],
                     'generatable': c['generatable'], 'present_ok': c['present_ok'], 'withhold_reason': c['withhold_reason'],
                     'proposal_ja': c['proposal_ja'] or '', 'proposal_len': len(c['proposal_ja'] or ''),
                     'duplicate_of': c['duplicate_of'] or '', 'human_comparing_now': ''}
                r.update(stage2_fields(r['id'], pos)); r.update(quality_fields(r['id']))
                rows.append(r); cands.append(r)
            b = j.get('policy_b')
            if b:
                bid = f"{m}-B{int(b['checkpoint_s'])}"
                r = {'split': split, 'meeting': m, 'id': bid, 'kind': 'B', 'pos': pos, 'time_s': byt[pos]['end'],
                     'ep_id': '', 'choice_item': b.get('choice_item', ''), 'options': ' | '.join(map(str, b.get('options') or [])),
                     'ground_turns': ','.join(str(g.get('turn')) for g in (b.get('shared_grounds') or []) if isinstance(g, dict)),
                     'ground_source_ids': ','.join(sid for g in (b.get('shared_grounds') or []) if isinstance(g, dict) and g.get('turn') in byt for sid in byt[g['turn']]['source_ids']),
                     'grounds_quote_original': ' || '.join(f"t{g.get('turn')}: {g.get('quote')}" for g in (b.get('shared_grounds') or []) if isinstance(g, dict)),
                     'grounds_gloss_ja': ' || '.join(f"t{g.get('turn')}: {g.get('gloss')}" for g in (b.get('shared_grounds') or []) if isinstance(g, dict)),
                     'generatable': b['generatable'], 'present_ok': '', 'withhold_reason': b.get('reason_if_not', ''),
                     'proposal_ja': b['proposal_ja'] or '', 'proposal_len': len(b['proposal_ja'] or ''),
                     'human_comparing_now': b.get('human_comparing_now', ''), 'checkpoint_s': b['checkpoint_s']}
                r.update(stage2_fields(bid, pos)); r.update(quality_fields(bid))
                rows.append(r); brows.append(r)
        # 議論単位
        prim = [c for c in cands if not c['duplicate_of']]
        by_ep = collections.defaultdict(list)
        for c in cands: by_ep[c['ep_id']].append(c)
        ep_gen = [e for e, cs in by_ep.items() if any(c['generatable'] == 'yes' for c in cs)]
        ep_ok = [e for e, cs in by_ep.items() if any(c['present_ok'] == 'yes' for c in cs)]
        ep_unc = [e for e, cs in by_ep.items() if e not in ep_ok and any(c['present_ok'] == 'uncertain' for c in cs)]
        ep_hold_h = [e for e, cs in by_ep.items() if e not in ep_ok and any(str(c['withhold_reason']).startswith('H') for c in cs)]
        ep_hold_r = [e for e, cs in by_ep.items() if e not in ep_ok and any(str(c['withhold_reason']).startswith('R') for c in cs)]
        ok_c = [c for c in cands if c['present_ok'] == 'yes']
        pre = [c for c in ok_c if c.get('S1_turn') != '']
        upd = [c for c in ok_c if c.get('S2_turn') != '' or c.get('S3_turn') != '']
        sub = [c for c in upd if c.get('update_type') == 'substantive']
        reph = [c for c in upd if c.get('update_type') == 'rephrase']
        dur = turns[-1]['end'] - turns[0]['start']
        bg = [b for b in brows if b['meeting'] == m]
        mrows.append({'split': split, 'meeting': m, 'source': d['meta']['source'], 'speakers': len(set(t['spk'] for t in turns)),
                      'minutes': round(dur / 60, 1), 'turns': len(turns), 'source_units': sum(len(t['source_ids']) for t in turns),
                      'episodes_opened': len(eps_opened), 'episodes_with_candidate': len(by_ep),
                      'candidate_records': len(cands), 'duplicate_records': len(cands) - len(prim),
                      'ep_generatable': len(ep_gen), 'ep_present_ok': len(ep_ok), 'ep_present_uncertain_only': len(ep_unc),
                      'ep_withheld_human_comparing': len(ep_hold_h), 'ep_withheld_resolved_deferred': len(ep_hold_r),
                      'okcand_preempted_S1': len(pre),
                      'okcand_preempt_dturns': ';'.join(str(c['S1_dturns']) for c in pre),
                      'okcand_preempt_dsec': ';'.join(str(c['S1_dsec']) for c in pre),
                      'okcand_update_info_S2S3': len(upd), 'update_substantive': len(sub), 'update_rephrase': len(reph),
                      'B_checkpoints': len(bg), 'B_generatable': sum(b['generatable'] == 'yes' for b in bg),
                      'B_generatable_while_human_comparing': sum(b['generatable'] == 'yes' and b['human_comparing_now'] == 'yes' for b in bg),
                      'present_ok_times_s': ';'.join(str(round(c['time_s'])) for c in ok_c)})

cols = []
for r in rows:
    for k in r:
        if k not in cols: cols.append(k)
with open(os.path.join(OUT, 'candidates_all.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in rows]
with open(os.path.join(OUT, 'per_meeting.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(mrows[0].keys())); w.writeheader(); [w.writerow(r) for r in mrows]
json.dump({q: {'agree': qagree[q], 'pairs': qtotal[q]} for q in QN}, open(os.path.join(OUT, 'quality_agreement.json'), 'w'), ensure_ascii=False, indent=1)
print('rows', len(rows))
for r in mrows: print(r)
print({q: f'{qagree[q]}/{qtotal[q]}' for q in QN})
