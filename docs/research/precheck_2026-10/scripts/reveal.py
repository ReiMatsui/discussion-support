"""逐次判定用の表示スクリプト（未読の発話を表示しない）。

使い方:
  python3 reveal.py init --meeting GAP01 --run runs/GAP01 [--b-interval 180]
  python3 reveal.py step --run runs/GAP01 --judgment '<JSON>'     # 2回目以降は判定必須
  python3 reveal.py step --run runs/GAP01 --judgment-file j.json
  python3 reveal.py status --run runs/GAP01

仕組み:
- init で対象会議の発話列を圧縮・符号化して run ディレクトリの .sealed に置く。
  判定者には「.sealed と data/ を読まない」ことを指示する（暗号ではなく手続き上の封印）。
- step は、直前に表示した位置への判定 JSON を受け取って judgments.jsonl に追記してから、
  次の1ターンを表示する。判定がなければ次を表示しない。
- 相槌だけのターン（3語以下で相槌語のみ）は、直前の実質ターンと一緒に表示する。
  これは次の実質発話を先に見せないための束ね方で、相槌の存在だけが先に分かる。
- 方針Bのチェック時点: 表示済みの最後のターン終了時刻が b_interval の倍数を初めて
  超えた時点。そこでの判定には policy_b フィールドを必須にする。
"""
import argparse, base64, json, os, re, sys, zlib, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BC = set("yeah yes yep yea mm mm-hmm mhm hmm uh um uh-huh okay ok right oh ah so well sure no huh cool alright true exactly totally".split())

REQUIRED = ['pos', 'state_note', 'episode_events', 'candidate']
CAND_KEYS = ['ep_id', 'choice_item', 'options', 'shared_grounds', 'explicit_conditions', 'unconfirmed_conditions',
             'human_already_comparing', 'uncompared_content', 'generatable', 'present_ok', 'withhold_reason',
             'proposal_ja', 'duplicate_of']
B_KEYS = ['generatable', 'options', 'shared_grounds', 'proposal_ja', 'reason_if_not']


def isbc(t):
    w = re.findall(r"[a-z'-]+", t['text'].lower())
    return len(w) <= 3 and all(x in BC for x in w)


def load_sealed(run):
    return json.loads(zlib.decompress(base64.b64decode(open(os.path.join(run, '.sealed'), 'rb').read())))


def units(turns):
    out, i = [], 0
    while i < len(turns):
        j = i + 1
        while j < len(turns) and isbc(turns[j]):
            j += 1
        out.append(list(range(i, j)))
        i = j
    return out


def fmt(t):
    return f"[t{t['t']}] {t['start']:.1f}-{t['end']:.1f}s {t['spk']}: {t['text']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('--meeting')
    ap.add_argument('--run', required=True)
    ap.add_argument('--b-interval', type=float, default=180)
    ap.add_argument('--judgment')
    ap.add_argument('--judgment-file')
    a = ap.parse_args()
    run = a.run
    if a.cmd == 'init':
        os.makedirs(run, exist_ok=True)
        d = json.load(open(os.path.join(ROOT, 'data', f'{a.meeting}.turns.json')))
        open(os.path.join(run, '.sealed'), 'wb').write(base64.b64encode(zlib.compress(json.dumps(d).encode())))
        st = {'meeting': a.meeting, 'unit': 0, 'b_interval': a.b_interval, 'next_b': a.b_interval, 'b_due': False,
              'last_t': 0, 'done': False, 'started': time.time()}
        json.dump(st, open(os.path.join(run, 'state.json'), 'w'))
        open(os.path.join(run, 'judgments.jsonl'), 'w').close()
        open(os.path.join(run, 'shown.log'), 'w').close()
        spk = sorted(set(t['spk'] for t in d['turns']))
        print(f"meeting={a.meeting} source={d['meta']['source']} speakers={spk} turns_total=非表示")
        print("最初のターンを表示するには: step --run", run, "（初回は判定不要）")
        return
    st = json.load(open(os.path.join(run, 'state.json')))
    d = load_sealed(run)
    turns = d['turns']
    us = units(turns)
    if a.cmd == 'status':
        print(json.dumps({k: st[k] for k in ('meeting', 'unit', 'last_t', 'b_due', 'done')}, ensure_ascii=False))
        return
    if a.cmd != 'step':
        sys.exit('unknown cmd')
    if st['done']:
        print('END: 全ターン表示済み。判定は保存済み。')
        return
    if st['unit'] > 0:
        raw = a.judgment if a.judgment else (open(a.judgment_file).read() if a.judgment_file else None)
        if not raw:
            sys.exit(f"ERROR: t{st['last_t']} までの判定 JSON が必要です（--judgment）。次は表示しません。")
        try:
            j = json.loads(raw)
        except Exception as e:
            sys.exit(f'ERROR: JSON 解析失敗: {e}')
        miss = [k for k in REQUIRED if k not in j]
        if miss:
            sys.exit(f'ERROR: 必須キー不足 {miss}')
        if j['pos'] != st['last_t']:
            sys.exit(f"ERROR: pos は {st['last_t']} であるべき（受け取った {j['pos']}）")
        if j['candidate'] is not None:
            m = [k for k in CAND_KEYS if k not in j['candidate']]
            if m:
                sys.exit(f'ERROR: candidate のキー不足 {m}')
        if st['b_due']:
            if 'policy_b' not in j or not isinstance(j['policy_b'], dict):
                sys.exit('ERROR: この時点は方針Bのチェック時点です。policy_b を付けてください。')
            m = [k for k in B_KEYS if k not in j['policy_b']]
            if m:
                sys.exit(f'ERROR: policy_b のキー不足 {m}')
            j['policy_b']['checkpoint_s'] = st['next_b'] - st['b_interval']
        j['_saved_at'] = time.time()
        j['_b_due'] = st['b_due']
        with open(os.path.join(run, 'judgments.jsonl'), 'a') as f:
            f.write(json.dumps(j, ensure_ascii=False) + '\n')
    if st['unit'] >= len(us):
        st['done'] = True
        json.dump(st, open(os.path.join(run, 'state.json'), 'w'))
        print('END: 全ターン表示済み。最後の判定を保存しました。')
        return
    idxs = us[st['unit']]
    lines = [fmt(turns[i]) for i in idxs]
    with open(os.path.join(run, 'shown.log'), 'a') as f:
        f.write('\n'.join(lines) + '\n')
    st['unit'] += 1
    st['last_t'] = turns[idxs[-1]]['t']
    end_s = turns[idxs[-1]]['end']
    st['b_due'] = False
    if end_s >= st['next_b']:
        st['b_due'] = True
        while st['next_b'] <= end_s:
            st['next_b'] += st['b_interval']
    json.dump(st, open(os.path.join(run, 'state.json'), 'w'))
    print('\n'.join(lines))
    tail = f"次の判定は pos={st['last_t']}"
    if st['b_due']:
        tail += f"。ここは方針Bのチェック時点（{st['next_b'] - st['b_interval']:.0f}s 経過）: policy_b 必須"
    if st['unit'] >= len(us):
        tail += '。これが最後のターンです'
    print('--', tail)


if __name__ == '__main__':
    main()
