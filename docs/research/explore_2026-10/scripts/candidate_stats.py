"""支援候補ごとの根拠事例・反例の集計（事例の記号から機械的に数える）。
C1 前提をそろえる: K1 を含む事例
C2 未確認の情報を扱う: K2 を含む事例
C3 決まったこと・残りを保つ: K4 を含む事例
参考: 人が自力で理由を尋ねる・異議・統合（P3/P4/P9）を含み、種類が『人が自力で解消／他者の働きかけで前進／有用な行為』の事例
"""
import csv, collections, sys, os
ex = sys.argv[1]
R = list(csv.DictReader(open(os.path.join(ex, 'cases', 'cases_all.csv'), encoding='utf-8-sig')))
MS = ['GAP01', 'GAP07', 'GAP08', 'GAP18', 'GAP24', 'ES2002c', 'ES2003c', 'ES2004c']
out = []
for name, code in [('C1', 'K1'), ('C2', 'K2'), ('C3', 'K4')]:
    rs = [r for r in R if code in r['codes'].split()]
    row = {'candidate': name, 'code': code, 'cases': len(rs), 'meetings_with_case': len({r['meeting'] for r in rs}),
           '役立ちそう': sum(r['support'] == '役立ちそう' for r in rs), '不要': sum(r['support'] == '不要' for r in rs), '判断不能': sum(r['support'] == '判断不能' for r in rs),
           '人が自力で解消・他者で前進': sum(r['kind'] in ('人が自力で解消', '他者の働きかけで前進') for r in rs)}
    for m in MS: row[m] = sum(r['meeting'] == m for r in rs)
    row['case_ids'] = ' '.join(r['case'] for r in rs)
    out.append(row)
rs = [r for r in R if set(r['codes'].split()) & {'P3', 'P4', 'P9'} and r['kind'] in ('人が自力で解消', '他者の働きかけで前進', '有用な行為')]
row = {'candidate': '参考: 人が理由を尋ねる・異議・統合で進めた', 'code': 'P3/P4/P9', 'cases': len(rs), 'meetings_with_case': len({r['meeting'] for r in rs})}
for m in MS: row[m] = sum(r['meeting'] == m for r in rs)
out.append(row)
keys = []
for r in out:
    for k in r:
        if k not in keys: keys.append(k)
with open(os.path.join(ex, 'cases', 'candidate_stats.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in out]
for r in out: print(r)
print('全事例', len(R), collections.Counter(r['support'] for r in R))
print('coder別 役立ちそう', collections.Counter((r['coder'][:4], r['support']) for r in R))
