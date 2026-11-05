"""検証用データを選定規則どおりに選び、判定時点ごとの履歴を切り出して、時点ごとに別の合言葉で暗号化する。
内容は表示しない（件数と時刻だけ表示）。
使い方: python3 select_and_seal.py <precheck_root> <deli_dir> <outdir>
出力:
  <outdir>/selection.json   選ばれた議論と選定理由（内容なし）
  <outdir>/checkpoints.json 判定時点の一覧（議論、番号、履歴のターン数、最後のターン ID、ファイル名）
  <outdir>/cp/*.enc         暗号化した履歴（openssl aes-256-cbc -pbkdf2）
  <outdir>/passwords.tsv    合言葉（主作業者が後で暗号化して平文を消す）
  <outdir>/full/*.json      全文（後で暗号化して平文を消す）
"""
import json, os, sys, subprocess, secrets, collections
sys.path.insert(0, os.path.join(sys.argv[1], 'scripts'))
import prep  # 旧作業の前処理（同じ話者の連続をターンにまとめる）
pre, deli_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(os.path.join(out, 'cp'), exist_ok=True); os.makedirs(os.path.join(out, 'full'), exist_ok=True)
TASK = {
 'GAP': '冬の生存課題（Winter Survival Task）。小型機が冬の北方の森（1月）に墜落し、乗客は生き残った。持ち出せた15品目（ナイフ、ライター（燃料なし）、コンパス、地図、衣類、ウイスキー、スチールウール、懐中電灯、救急用品、拳銃、ロープ、新聞、チョコレート、ショートニング、スキーのストック）を、生き延びるための重要度で1〜15位に順位付けする。各自が個人で順位を付けた後、3人で最大15分話し合い、一つの順位に合意する。英語の対面会話の書き起こし。',
 'AMI': 'AMI 会議コーパスのシナリオ会議。4人（PM、工業デザイナー、UI デザイナー、マーケティング担当）が、模擬企業で新しいテレビ用リモコンを設計する。この会議は4回のうち3回目の概念設計で、各担当の発表の後、部品（電源、ケースの素材・形、チップ、操作部など）の方向を決める。英語の対面会議の書き起こし。',
 'DELI': 'DeliData（Wason のカード選択課題）。参加者は、表に文字か数字が書かれた4枚のカードと、「ある規則」（例: 片面が母音なら、もう片面は偶数）を見せられ、規則が成り立つかを確かめるために裏返す必要があるカードを選ぶ。各自が一度選んだ後、オンラインの文字チャットで相談し、各自が答えを提出する（何度でも提出できる）。[提出] の行は提出の記録。英語の文字チャット。',
}
sel = []
# GAP
conv = json.load(open(os.path.join(pre, 'raw', 'gap', 'gap-corpus', 'conversations.json')))
used = {1, 7, 8, 18, 24}
gap_ids = sorted(int(m['Group Number']) for m in conv.values() if m['Meeting Size'] == 3.0 and m['Meeting Length in Minutes'] >= 5 and int(m['Group Number']) not in used)[:6]
meetings = []
for g in gap_ids:
    turns, meta = prep.gap(g)
    meetings.append((f'GAP{g:02d}', 'GAP', 'spoken', turns)); sel.append({'meeting': f'GAP{g:02d}', 'rule': 'GAP: 3人・5分以上・未使用・番号順6件'})
for a in ['ES2005c', 'IS1000c', 'TS3003c']:
    turns, meta = prep.ami(a)
    meetings.append((a, 'AMI', 'spoken', turns)); sel.append({'meeting': a, 'rule': 'AMI: c回、未使用、ES2005c と各収録地点の最初'})
# DeliData
d = os.path.join(deli_dir, 'deli-corpus')
utts = collections.defaultdict(list)
for l in open(os.path.join(d, 'utterances.jsonl')):
    u = json.loads(l); utts[u['conversation_id']].append(u)
cands = []
for cid in sorted(utts):
    us = utts[cid]
    spk = {u['speaker'] for u in us if u['meta'].get('message_type') == 'MESSAGE'}
    nmsg = sum(u['meta'].get('message_type') == 'MESSAGE' for u in us)
    if len(spk) == 3 and nmsg >= 30: cands.append(cid)
for i, cid in enumerate(cands[:4]):
    us = utts[cid]; turns = []
    for k, u in enumerate(us, 1):
        mt = u['meta'].get('message_type')
        if mt not in ('MESSAGE', 'SUBMIT'): continue
        text = u['text'] if mt == 'MESSAGE' else '[提出] ' + str(u['text'])
        turns.append({'t': len(turns) + 1, 'spk': str(u['speaker'])[:8], 'start': None, 'end': None, 'text': text, 'source_ids': [u['id']], 'is_msg': mt == 'MESSAGE'})
    mid = f'DELI{i + 1:02d}'
    meetings.append((mid, 'DELI', 'chat', turns)); sel.append({'meeting': mid, 'source_conversation_id': cid, 'rule': 'DeliData: 3人・発言30件以上・会話ID辞書順4件'})
cps = []; pw = []
for mid, src, medium, turns in meetings:
    json.dump({'meeting': mid, 'source': src, 'medium': medium, 'task': TASK[src], 'turns': turns}, open(os.path.join(out, 'full', f'{mid}.json'), 'w'), ensure_ascii=False)
    points = []
    if medium == 'spoken':
        last_start = turns[-1]['start']; k = 1
        while k * 120 <= last_start:
            hist = [t for t in turns if t['end'] <= k * 120]; points.append((k, f'{k * 120}s', hist)); k += 1
    else:
        msg_idx = [i for i, t in enumerate(turns) if t['is_msg']]; k = 1
        while k * 10 <= len(msg_idx):
            hist = turns[:msg_idx[k * 10 - 1] + 1]; points.append((k, f'{k * 10}msg', hist)); k += 1
    for k, label, hist in points:
        def fmt(t):
            ts = f"{t['start']:.0f}s" if t['start'] is not None else '-'
            return f"t{t['t']}|{ts}|{t['spk']}|{t['text']}"
        body = f"# 議論 {mid}（判定時点 {k}: {label}）\n# 課題: {TASK[src]}\n# 形式: ターンID|開始秒|話者|発話。この時点までの履歴だけを含む。\n" + '\n'.join(fmt(t) for t in hist) + '\n'
        name = f'{mid}_cp{k:02d}'
        p = secrets.token_urlsafe(18)
        subprocess.run(['openssl', 'enc', '-aes-256-cbc', '-pbkdf2', '-salt', '-pass', 'pass:' + p, '-out', os.path.join(out, 'cp', name + '.enc')], input=body.encode(), check=True)
        cps.append({'meeting': mid, 'source': src, 'k': k, 'label': label, 'n_turns': len(hist), 'last_t': hist[-1]['t'] if hist else None, 'file': f'cp/{name}.enc'})
        pw.append(f'{name}\t{p}')
json.dump(sel, open(os.path.join(out, 'selection.json'), 'w'), ensure_ascii=False, indent=1)
json.dump(cps, open(os.path.join(out, 'checkpoints.json'), 'w'), ensure_ascii=False, indent=1)
open(os.path.join(out, 'passwords.tsv'), 'w').write('\n'.join(pw) + '\n')
c = collections.Counter(x['meeting'] for x in cps)
print('meetings', [m[0] for m in meetings]); print('checkpoints per meeting', dict(c)); print('total', len(cps), 'max k', max(x['k'] for x in cps))
print('turns', {m[0]: len(m[3]) for m in meetings})
