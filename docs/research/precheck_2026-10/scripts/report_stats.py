"""報告書の追加集計（候補単位の内訳、比較案の形の機械的コーディング、方針Bのデータ別集計、5分区間の分布）。
使い方: python3 report_stats.py  （aggregate.py の後に実行）
比較案の形は最後の文の語で分ける（分析者の機械的な分類で、解釈を含む）:
  F1 二択の優先・前提を問う: 「どちら」と「重く/優先/先に/どちらで/どちらの前提/どちらを」
  F2 理由・用途を尋ねる: 「想定/何に使/何をする/理由/どんな場面/どういう考え」
  F3 条件・方針を確かめる: 上記以外で「前提/ありますか/いますか/ありそう/ですか/ますか/しますか」
"""
import csv, re, collections, statistics, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = list(csv.DictReader(open(os.path.join(ROOT, 'out', 'candidates_all.csv'), encoding='utf-8-sig')))
def form(p):
    if not p: return 'none'
    s = [x for x in re.split(r'(?<=[。？?])', p.strip()) if x.strip()][-1]
    if 'どちら' in s and re.search('重く|優先|先に|どちらで|どちらの前提|どちらを', s): return 'F1'
    if re.search('想定|何に使|何をする|理由|どんな場面|どういう考え', s): return 'F2'
    if re.search('前提|ありますか|いますか|ありそう|ですか|ますか|しますか', s): return 'F3'
    return 'F4'
T = [r for r in R if r['split'] == 'test']
ok = [r for r in T if r['kind'] == 'A' and r['present_ok'] == 'yes']
print('方針A 候補', sum(r['kind'] == 'A' for r in T), '妥当', len(ok))
print('妥当 S6', collections.Counter(r['S6'].split('@')[0] for r in ok if r['S6']))
offs = [int(r['S6'].split('@t')[1]) - int(r['pos']) for r in ok if r['S6']]
print('決定・転換までのターン 中央値', statistics.median(offs))
print('先取り', [(r['id'], r['S1_dturns'], r['S1_dsec']) for r in ok if r['S1_turn']])
print('更新', collections.Counter((r['meeting'], r['update_type']) for r in ok if r['S2_turn'] or r['S3_turn']))
print('問いへの答え', collections.Counter(r['answers_question'] for r in ok))
print('形 妥当A', collections.Counter(form(r['proposal_ja']) for r in ok))
print('形 B', collections.Counter(form(r['proposal_ja']) for r in T if r['kind'] == 'B' and r['proposal_ja']))
for q in range(1, 9):
    print('Q', q, '妥当A', dict(collections.Counter(r[f'Q{q}_final'] for r in ok)), 'B', dict(collections.Counter(r[f'Q{q}_final'] for r in T if r['kind'] == 'B' and r['proposal_ja'])))
B = [r for r in T if r['kind'] == 'B' and r['generatable'] == 'yes']
for ds in ('GAP', 'ES'):
    s = [r for r in B if r['meeting'].startswith(ds)]
    print('B', ds, len(s), '比較中', sum(r['human_comparing_now'] == 'yes' for r in s), 'S1', sum(r['S1_turn'] != '' for r in s), '答え', dict(collections.Counter(r['answers_question'] for r in s)), '実質', sum(r['update_type'] == 'substantive' for r in s))
L = [int(r['proposal_len']) for r in T if r['proposal_ja']]
print('文面の長さ 中央値', statistics.median(L), '最大', max(L))
