"""支援と判定した時点について、比較用の3種類の提示文を固定の処理で作る（基準 4節）。
静的案内: 固定文。引用のみ: 提案手法の対の証拠引用を、話者付きで並べる。提案手法: 判定者の文面。
"""
import json, os, sys, glob
root = sys.argv[1]
STATIC = '理由によっては、どんな状況を想定しているかで重みが変わることがあります。それぞれの理由がどんな想定に立っているか、確かめてから決めても構いません。'
full = {}
out = []
for p in sorted(glob.glob(os.path.join(root, 'timepoint', '*.json'))):
    o = json.load(open(p))
    if o.get('decision') != 'support': continue
    name = os.path.basename(p)[:-5]; m = name.split('_cp')[0]
    if m not in full: full[m] = {t['t']: t for t in json.load(open(os.path.join(root, 'sealed', 'full', f'{m}.json')))['turns']}
    pair = [x for x in o['pairs'] if x['pair_id'] == o['target_pair']][0]
    parts = []
    for pm in pair['premises']:
        for e in pm['evidence']:
            t = int(str(e['turn']).lstrip('t'))
            parts.append(f"{full[m][t]['spk']}さん（t{t}）「{e['quote']}」")
    quote_only = '。'.join(parts) + '。'
    out.append({'checkpoint': name, 'static': STATIC, 'quote_only': quote_only, 'proposed': o['presentation_ja'],
                'len_static': len(STATIC), 'len_quote_only': len(quote_only), 'len_proposed': len(o['presentation_ja'])})
json.dump(out, open(os.path.join(root, 'results', 'variants.json'), 'w'), ensure_ascii=False, indent=1)
for v in out: print(json.dumps(v, ensure_ascii=False, indent=1))
