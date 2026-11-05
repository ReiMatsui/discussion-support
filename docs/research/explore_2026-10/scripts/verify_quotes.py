"""事例の observed 欄にある「tNN 話者 '引用'」の引用が、その番号のターンに実在するかを機械的に確かめる。
引用の先頭20文字（省略記号の前まで）を、前後1ターンを含む本文で探す。
"""
import csv, json, os, re, sys, collections
pre, ex = sys.argv[1], sys.argv[2]
R = list(csv.DictReader(open(os.path.join(ex, 'cases', 'cases_all.csv'), encoding='utf-8-sig')))
tx = {}
def norm(s): return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
res = collections.Counter(); bad = []
for r in R:
    m = r['meeting']
    if m not in tx: tx[m] = {t['t']: t['text'] for t in json.load(open(os.path.join(pre, 'data', f'{m}.turns.json')))['turns']}
    for t, q in re.findall(r"t(\d+)[^'’]{0,40}?'([^']{6,})'", r['observed']):
        t = int(t); q = norm(q.split('...')[0])[:20]
        if not q: continue
        hay = ' '.join(norm(tx[m].get(x, '')) for x in (t - 1, t, t + 1))
        ok = q in hay
        res[(r['coder'][:4], ok)] += 1
        if not ok: bad.append(f"{r['case']} t{t}: '{q}'")
open(os.path.join(ex, 'cases', 'quote_check.txt'), 'w').write(f'{dict(res)}\n' + '\n'.join(bad) + '\n')
print(dict(res)); print('\n'.join(bad[:30]))
