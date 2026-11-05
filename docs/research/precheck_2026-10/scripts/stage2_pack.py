"""第2段階の材料を作る: 固定済みの候補（方針A）とチェック時点（方針B）ごとに、
候補時点までの直近の文脈と、その後 WINDOW ターンを並べる。
出力: <run>/stage2_input.json
"""
import json, sys, os, base64, zlib
WINDOW = int(os.environ.get('WINDOW', 30))
CTX = 12
run = sys.argv[1]
d = json.loads(zlib.decompress(base64.b64decode(open(os.path.join(run, '.sealed'), 'rb').read())))
turns = d['turns']; byt = {t['t']: t for t in turns}
items = []
for l in open(os.path.join(run, 'judgments.jsonl')):
    j = json.loads(l)
    pos = j['pos']
    fol = [t for t in turns if pos < t['t'] <= pos + WINDOW]
    ctx = [t for t in turns if pos - CTX < t['t'] <= pos]
    def pk(x): return [{'t': t['t'], 'spk': t['spk'], 'start': t['start'], 'end': t['end'], 'text': t['text']} for t in x]
    c = j.get('candidate')
    if c:
        items.append({'id': c.get('cand_id'), 'kind': 'A', 'pos': pos, 'pos_end_s': byt[pos]['end'], 'candidate': c, 'context': pk(ctx), 'following': pk(fol)})
    b = j.get('policy_b')
    if b and b.get('generatable') == 'yes':
        items.append({'id': f"{d['meeting']}-B{int(b['checkpoint_s'])}", 'kind': 'B', 'pos': pos, 'pos_end_s': byt[pos]['end'], 'candidate': b, 'context': pk(ctx), 'following': pk(fol)})
json.dump({'meeting': d['meeting'], 'window_turns': WINDOW, 'items': items}, open(os.path.join(run, 'stage2_input.json'), 'w'), ensure_ascii=False, indent=1)
print(d['meeting'], len(items), 'items')
