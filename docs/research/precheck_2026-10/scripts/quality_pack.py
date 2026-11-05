"""比較案の品質評価の材料: 候補ごとに「候補時点までの会話」と「比較案」だけを別ファイルにする。
判定者は生成に関わっていない別のエージェント。後続の発話は含めない。
出力: <run>/quality/<id>.txt と <run>/quality/index.json
"""
import json, sys, os, base64, zlib
run = sys.argv[1]
d = json.loads(zlib.decompress(base64.b64decode(open(os.path.join(run, '.sealed'), 'rb').read())))
turns = d['turns']
qd = os.path.join(run, 'quality'); os.makedirs(qd, exist_ok=True)
idx = []
for l in open(os.path.join(run, 'judgments.jsonl')):
    j = json.loads(l); pos = j['pos']
    recs = []
    c = j.get('candidate')
    if c and c.get('proposal_ja'):
        recs.append((c['cand_id'], 'A', c['proposal_ja']))
    b = j.get('policy_b')
    if b and b.get('proposal_ja'):
        recs.append((f"{d['meeting']}-B{int(b['checkpoint_s'])}", 'B', b['proposal_ja']))
    for cid, kind, prop in recs:
        lines = [f"[t{t['t']}] {t['spk']}: {t['text']}" for t in turns if t['t'] <= pos]
        open(os.path.join(qd, f'{cid}.txt'), 'w').write('# 候補時点までの会話（これより後は含まない）\n' + '\n'.join(lines) + f'\n\n# 比較案（この時点で AI が音声で提示する文面）\n{prop}\n')
        idx.append({'id': cid, 'kind': kind, 'pos': pos, 'file': os.path.join(qd, f'{cid}.txt')})
json.dump(idx, open(os.path.join(qd, 'index.json'), 'w'), ensure_ascii=False, indent=1)
print(d['meeting'], len(idx))
