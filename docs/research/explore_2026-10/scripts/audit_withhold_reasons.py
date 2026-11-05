"""旧判定ログで、妥当とされなかった候補の控える理由に H1/H4 が含まれる件数を数える（監査 2節）。"""
import json, sys, os
pre = sys.argv[1]
for name, ms in [('AMI', ['ES2003c', 'ES2004c']), ('GAP', ['GAP07', 'GAP08', 'GAP18', 'GAP24'])]:
    n = h1 = h4 = 0
    for m in ms:
        for l in open(os.path.join(pre, 'runs', 'test', m, 'judgments.jsonl')):
            c = json.loads(l)['candidate']
            if c and c['present_ok'] != 'yes':
                n += 1; h1 += 'H1' in c['withhold_reason']; h4 += 'H4' in c['withhold_reason']
    print(name, '妥当でない候補', n, 'H1を含む', h1, 'H4を含む', h4)
