"""最小処理（実データで動く範囲）。

入力: その時点までの話者付き履歴（`ターンID|開始秒|話者|発話` の行）
出力: 証拠付きの理由・前提対応、支援する／控える／判断不能、短い提示文

役割の分担:
- LLM に任せる部分: 理由の抽出、前提の推定（種類・証拠水準）、明示済みかの判定、提示文の作成。
  この環境では API を直接呼べないため、`prompt` サブコマンドで判定用の入力を作り、LLM（今回の検証では Claude のサブエージェント）の出力 JSON を受け取る。
- 固定ルールの部分（このスクリプト）:
  1. 履歴の切り出し（判定時点より後の発話を入れない）
  2. 出力の形式検査
  3. 引用の実在と発言 ID の検査（判定時点以前の発話か）
  4. 支援判定のゲート（主分析の対があり、未明示、両立済みでない、発表中でない、引用の検査を通ったときだけ「支援する」を残す。それ以外は「控える」か「判断不能」に落とす）
  5. 3種類の提示文の生成（静的案内、引用のみ、提案手法）と長さの検査

使い方:
  python3 minimal_pipeline.py cut <全文.txt> <最後のターン番号> > history.txt
  python3 minimal_pipeline.py prompt <history.txt> > request.md
  python3 minimal_pipeline.py gate <history.txt> <llm_output.json>
  python3 minimal_pipeline.py batch <verify_root>   # 今回の84時点の出力にゲートを当て直して集計する

話者情報・音声・実時間性が何に必要か:
- 話者情報: 理由を「異なる話者」の対にすること、提示文で発言を話者に帰属させることに必須。話者の取り違えは、そのまま帰属の誤りになる。
- 音声: 対面の議論で提示を全員に同時に届ける手段の候補。ログの検証では不要。音声で出す場合、今回の提案手法の文（313〜394字）は長すぎる。
- 実時間性: 判定時点で履歴を切り出せれば足りる。120秒ごとの判定なら、数秒の遅延は問題にならない。
- 今回の性能は公開転記でのもので、実時間の音声認識を通した性能は未検証。
"""
import json, os, re, sys, collections

STATIC = '理由によっては、どんな状況を想定しているかで重みが変わることがあります。それぞれの理由がどんな想定に立っているか、確かめてから決めても構いません。'
MAX_PROPOSED = 200  # 運用上の目安（音声で一度に聞ける量として今回決めた値。先行研究の値ではない）


def parse_history(text):
    turns = {}
    for line in text.splitlines():
        if not line.startswith('t'): continue
        parts = line.split('|', 3)
        if len(parts) < 4: continue
        try: t = int(parts[0][1:])
        except ValueError: continue
        turns[t] = {'spk': parts[2], 'text': parts[3]}
    return turns


def norm(s): return re.sub(r'\s+', ' ', s.lower()).strip()


def check_quote(turns, turn, quote, last):
    try: t = int(str(turn).lstrip('t'))
    except ValueError: return 'IDが不正'
    if t > last: return '判定時点より後'
    if t not in turns: return 'IDが履歴にない'
    q = norm(quote.split('...')[0].split('…')[0])[:40]
    if not q or q not in norm(turns[t]['text']): return '引用が見つからない'
    return 'ok'


def gate(turns, o):
    last = max(turns) if turns else 0
    issues = []
    for k in ('reasons', 'pairs', 'decision', 'presentation_ja'):
        if k not in o: issues.append(f'キー不足:{k}')
    pairs = {p.get('pair_id'): p for p in o.get('pairs') or []}
    target = pairs.get(o.get('target_pair'))
    result = {'llm_decision': o.get('decision'), 'final': None, 'gate_reasons': [], 'variants': None}
    if o.get('decision') != 'support':
        result['final'] = o.get('decision'); return result, issues
    if target is None:
        result['final'] = 'undecidable'; result['gate_reasons'].append('対象の対がない'); return result, issues
    bad = []
    for pm in target.get('premises', []):
        if pm.get('evidence_level') not in ('E1', 'E2'): bad.append('推測（E3）の前提を含む')
        if pm.get('type') not in ('T1', 'T2'): bad.append(f"前提の種類が {pm.get('type')}")
        for ev in pm.get('evidence', []):
            r = check_quote(turns, ev.get('turn'), ev.get('quote', ''), last)
            if r != 'ok': bad.append(f"証拠 {ev.get('turn')}: {r}")
    spk = set()
    for pm in target.get('premises', []):
        for ev in pm.get('evidence', []):
            try: spk.add(turns[int(str(ev['turn']).lstrip('t'))]['spk'])
            except Exception: pass
    if len(spk) < 2: bad.append('証拠の話者が1人')
    if target.get('already_explicit_by_participants'): bad.append('既に明示')
    if target.get('compatible_or_reconciled'): bad.append('両立済み')
    if target.get('during_presentation'): bad.append('発表中')
    result['gate_reasons'] = bad
    result['final'] = 'withhold' if bad else 'support'
    quote_only = '。'.join(f"{turns[int(str(ev['turn']).lstrip('t'))]['spk']}さん「{ev['quote']}」" for pm in target.get('premises', []) for ev in pm.get('evidence', []) if check_quote(turns, ev.get('turn'), ev.get('quote', ''), last) == 'ok') + '。'
    prop = o.get('presentation_ja') or ''
    result['variants'] = {'static': STATIC, 'quote_only': quote_only, 'proposed': prop,
                          'len': {'static': len(STATIC), 'quote_only': len(quote_only), 'proposed': len(prop)},
                          'proposed_too_long': len(prop) > MAX_PROPOSED}
    return result, issues


def main():
    cmd = sys.argv[1]
    if cmd == 'cut':
        text = open(sys.argv[2]).read(); last = int(sys.argv[3])
        out = [l for l in text.splitlines() if not l.startswith('t') or int(l.split('|')[0][1:]) <= last]
        print('\n'.join(out))
    elif cmd == 'prompt':
        hist = open(sys.argv[2]).read()
        spec = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prompts', 'judge_timepoint.md')).read()
        print(spec.split('## 判定（ファイルごと）')[1].split('## 出力')[0])
        print('## 履歴\n' + hist)
    elif cmd == 'gate':
        turns = parse_history(open(sys.argv[2]).read()); o = json.load(open(sys.argv[3]))
        r, issues = gate(turns, o); print(json.dumps({'result': r, 'format_issues': issues}, ensure_ascii=False, indent=1))
    elif cmd == 'batch':
        root = sys.argv[2]
        cps = json.load(open(os.path.join(root, 'sealed', 'checkpoints.json')))
        full = {}
        cnt = collections.Counter(); rows = []
        for c in cps:
            m = c['meeting']; name = f"{m}_cp{c['k']:02d}"
            if m not in full: full[m] = {t['t']: {'spk': t['spk'], 'text': t['text']} for t in json.load(open(os.path.join(root, 'sealed', 'full', f'{m}.json')))['turns']}
            turns = {t: v for t, v in full[m].items() if t <= (c['last_t'] or 0)}
            o = json.load(open(os.path.join(root, 'timepoint', name + '.json')))
            r, issues = gate(turns, o)
            cnt[(r['llm_decision'], r['final'])] += 1
            rows.append({'checkpoint': name, 'llm': r['llm_decision'], 'final': r['final'], 'gate_reasons': '; '.join(r['gate_reasons']), 'format_issues': '; '.join(issues),
                         'len_proposed': r['variants']['len']['proposed'] if r['variants'] else '', 'too_long': r['variants']['proposed_too_long'] if r['variants'] else ''})
        import csv
        with open(os.path.join(root, 'results', 'minimal_pipeline_batch.csv'), 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(x) for x in rows]
        print(dict(cnt)); [print(x) for x in rows if x['llm'] == 'support' or x['format_issues']]


if __name__ == '__main__':
    main()
