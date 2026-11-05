"""前提差の対を、同じ議論の中で証拠発言 ID の重なりによってまとめる（基準 6節）。
感度分析用に、選択事項の文字列が同じものをまとめる規則も出す。
"""
import json, os, sys, glob, collections, csv
root = sys.argv[1]
cps = {f"{c['meeting']}_cp{c['k']:02d}": c for c in json.load(open(os.path.join(root, 'sealed', 'checkpoints.json')))}
items = []
for name, cp in sorted(cps.items(), key=lambda x: (x[1]['meeting'], x[1]['k'])):
    o = json.load(open(os.path.join(root, 'timepoint', name + '.json')))
    for kind in ('pairs', 'inferred_pairs'):
        for p in o.get(kind) or []:
            ev = set()
            for pm in p.get('premises', []):
                for e in pm.get('evidence', []):
                    try: ev.add(int(str(e.get('turn')).lstrip('t')))
                    except: pass
            items.append({'meeting': cp['meeting'], 'source': cp['source'], 'checkpoint': name, 'k': cp['k'], 'kind': 'main' if kind == 'pairs' else 'inferred',
                          'pair_id': p.get('pair_id'), 'choice_item': p.get('choice_item', ''), 'ev': ev,
                          'levels': [pm.get('evidence_level') for pm in p.get('premises', [])], 'types': [pm.get('type') for pm in p.get('premises', [])],
                          'premises': [pm.get('premise_ja', '') for pm in p.get('premises', [])],
                          'explicit': bool(p.get('already_explicit_by_participants')), 'compatible': bool(p.get('compatible_or_reconciled')),
                          'presentation': bool(p.get('during_presentation')),
                          'targeted': o.get('decision') == 'support' and o.get('target_pair') == p.get('pair_id')})
# union-find by evidence overlap within meeting & kind group (main and inferred pooled so that E3 versions merge)
groups = []
for it in items:
    hit = [g for g in groups if g['meeting'] == it['meeting'] and g['ev'] & it['ev']]
    if hit:
        g = hit[0]
        for h in hit[1:]: g['members'] += h['members']; g['ev'] |= h['ev']; groups.remove(h)
        g['members'].append(it); g['ev'] |= it['ev']
    else:
        groups.append({'meeting': it['meeting'], 'ev': set(it['ev']), 'members': [it]})
out = []
for i, g in enumerate(sorted(groups, key=lambda g: (g['meeting'], min(m['k'] for m in g['members'])))):
    ms = g['members']; first = min(ms, key=lambda m: m['k'])
    main = [m for m in ms if m['kind'] == 'main']
    out.append({'pair_uid': f"{g['meeting']}-U{i + 1:02d}", 'meeting': g['meeting'], 'source': ms[0]['source'],
                'first_checkpoint': first['checkpoint'], 'checkpoints': ' '.join(sorted({m['checkpoint'] for m in ms})),
                'has_main': bool(main), 'main_first_checkpoint': min(main, key=lambda m: m['k'])['checkpoint'] if main else '',
                'explicit_at_first_main': main and min(main, key=lambda m: m['k'])['explicit'],
                'ever_unexplicit_main': any(not m['explicit'] and not m['compatible'] for m in main),
                'targeted': any(m['targeted'] for m in ms),
                'choice_item': first['choice_item'], 'premises_first': ' / '.join(first['premises']),
                'types': ' '.join(sorted({t or '' for m in ms for t in m['types']})), 'levels_first': ' '.join(l or '' for l in first['levels']),
                'evidence_turns': ' '.join(f't{t}' for t in sorted(g['ev']))})
for r in out: r['ev_set'] = None
with open(os.path.join(root, 'results', 'pairs_unique.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    keys = [k for k in out[0] if k != 'ev_set']
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow({k: r[k] for k in keys}) for r in out]
print('unique groups', len(out), 'main', sum(r['has_main'] for r in out))
print('by meeting', collections.Counter(r['meeting'] for r in out if r['has_main']))
print('main ever unexplicit', [(r['pair_uid'], r['checkpoints']) for r in out if r['has_main'] and r['ever_unexplicit_main']])
print('targeted', [r['pair_uid'] for r in out if r['targeted']])
# sensitivity: choice item string grouping
ci = collections.Counter((it['meeting'], it['choice_item'].strip().lower()) for it in items if it['kind'] == 'main')
print('sens: unique (meeting, choice_item) main', len(ci))
