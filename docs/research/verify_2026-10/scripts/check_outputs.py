"""時点ごとの出力を固定の処理で検査する（B0、形式、時点外の発言の参照）。
使い方: python3 check_outputs.py <verify_root>
"""
import json, os, re, sys, glob, collections, csv
root = sys.argv[1]
cps = {f"{c['meeting']}_cp{c['k']:02d}": c for c in json.load(open(os.path.join(root, 'sealed', 'checkpoints.json')))}
full = {}
def norm(s): return re.sub(r'\s+', ' ', s.lower()).strip()
REQ = ['checkpoint', 'reasons', 'pairs', 'inferred_pairs', 'decision', 'withhold_reason', 'target_pair', 'presentation_ja']
rows = []; problems = []
for name, cp in sorted(cps.items()):
    p = os.path.join(root, 'timepoint', name + '.json')
    if not os.path.exists(p): problems.append(f'{name}: 出力なし'); continue
    try: o = json.load(open(p))
    except Exception as e: problems.append(f'{name}: JSON不正 {e}'); continue
    miss = [k for k in REQ if k not in o]
    if miss: problems.append(f'{name}: キー不足 {miss}')
    m = cp['meeting']
    if m not in full: full[m] = {t['t']: t for t in json.load(open(os.path.join(root, 'sealed', 'full', f'{m}.json')))['turns']}
    T = full[m]; last = cp['last_t'] or 0
    quotes = []
    for r in o.get('reasons', []): quotes.append(('reason', r.get('turn'), r.get('quote', '')))
    for key in ('pairs', 'inferred_pairs'):
        for pr in o.get(key, []) or []:
            for pm in pr.get('premises', []):
                for ev in pm.get('evidence', []): quotes.append((key, ev.get('turn'), ev.get('quote', '')))
            for ev in pr.get('explicit_evidence', []) or []: quotes.append((key + ':explicit', ev.get('turn'), ev.get('quote', '')))
    n_ok = n_bad = n_future = 0
    for kind, turn, q in quotes:
        try: t = int(str(turn).lstrip('t'))
        except: n_bad += 1; problems.append(f'{name}: 発言IDが不正 {turn}'); continue
        if t > last: n_future += 1; problems.append(f'{name}: 判定時点より後の発言 t{t}（最後 t{last}）')
        qq = norm(q.split('...')[0].split('…')[0])
        if qq and t in T and qq[:40] in norm(T[t]['text']): n_ok += 1
        else: n_bad += 1; problems.append(f'{name}: 引用が t{t} に見つからない: {q[:60]}')
    tp = o.get('target_pair'); pres = o.get('presentation_ja')
    if o.get('decision') == 'support' and (not tp or not pres): problems.append(f'{name}: 支援なのに対または文面なし')
    rows.append({'checkpoint': name, 'meeting': m, 'source': cp['source'], 'k': cp['k'], 'label': cp['label'], 'n_turns': cp['n_turns'],
                 'decision': o.get('decision'), 'withhold_reason': o.get('withhold_reason'), 'n_reasons': len(o.get('reasons', [])),
                 'n_pairs_main': len(o.get('pairs', []) or []), 'n_pairs_inferred': len(o.get('inferred_pairs', []) or []),
                 'pairs_explicit': sum(bool(p.get('already_explicit_by_participants')) for p in o.get('pairs', []) or []),
                 'quotes_ok': n_ok, 'quotes_bad': n_bad, 'quotes_future': n_future, 'judge': o.get('judge'),
                 'presentation_ja': pres or '', 'note': o.get('note', '')})
os.makedirs(os.path.join(root, 'results'), exist_ok=True)
with open(os.path.join(root, 'results', 'timepoint_all.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
open(os.path.join(root, 'results', 'fixed_checks.txt'), 'w').write(f'出力 {len(rows)} / 時点 {len(cps)}\n' + '\n'.join(problems) + '\n')
print('outputs', len(rows), 'of', len(cps))
print(collections.Counter(r['decision'] for r in rows))
print(collections.Counter(r['withhold_reason'] for r in rows))
print('quotes ok', sum(r['quotes_ok'] for r in rows), 'bad', sum(r['quotes_bad'] for r in rows), 'future', sum(r['quotes_future'] for r in rows))
print('\n'.join(problems[:30]))
for r in rows:
    if r['n_pairs_main'] or r['n_pairs_inferred'] or r['decision'] != 'withhold': print(r['checkpoint'], r['decision'], r['withhold_reason'], 'main', r['n_pairs_main'], 'expl', r['pairs_explicit'], 'inf', r['n_pairs_inferred'])
