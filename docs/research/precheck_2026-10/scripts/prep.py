"""公開コーパスから分析用の発話列（ターン）を作る。
- 同一話者の連続した発話単位を1ターンに結合する（発話の細分化で候補を増やさないため）。
- 元の発話ID（GAP: utterance id、AMI: dialogue act id）を source_ids に保持する。
出力: data/<meeting>.turns.json  [{"t": ターン番号(1始まり), "spk", "start", "end", "text", "source_ids"}]
"""
import json, sys, re, glob, os, collections, math
import xml.etree.ElementTree as ET
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'raw')
NITE = '{http://nite.sourceforge.net/}'

def mmss(s):
    m, sec = s.split(':'); return int(m) * 60 + float(sec)

def merge(units):
    turns = []
    for u in units:
        if turns and turns[-1]['spk'] == u['spk']:
            t = turns[-1]; t['text'] += ' ' + u['text']; t['end'] = max(t['end'], u['end']); t['source_ids'].append(u['id'])
        else:
            turns.append({'spk': u['spk'], 'start': u['start'], 'end': u['end'], 'text': u['text'], 'source_ids': [u['id']]})
    for i, t in enumerate(turns, 1):
        t['t'] = i; t['start'] = round(t['start'], 2); t['end'] = round(t['end'], 2)
    return turns

def gap(group):
    d = os.path.join(RAW, 'gap', 'gap-corpus')
    conv = json.load(open(os.path.join(d, 'conversations.json')))
    cid = [c for c, m in conv.items() if str(m['Group Number']) == str(group)][0]
    units = []
    for l in open(os.path.join(d, 'utterances.jsonl')):
        u = json.loads(l)
        if u['conversation_id'] != cid: continue
        txt = u['text'].strip().strip('"')
        st = mmss(u['timestamp']); en = mmss(u['meta']['End'])
        units.append({'id': u['id'], 'spk': u['speaker'].split('.')[1], 'start': st, 'end': en, 'text': txt})
    units.sort(key=lambda x: (x['start'], x['end']))
    return merge(units), {'source': 'GAP', 'conversation_id': cid, 'meta': conv[cid]}

def ami(meeting):
    d = os.path.join(RAW, 'ami')
    units = []
    for f in sorted(glob.glob(os.path.join(d, 'words', f'{meeting}.*.words.xml'))):
        spk = f.split('.')[-3]
        words = {}
        order = []
        for w in ET.parse(f).getroot():
            wid = w.get(NITE + 'id')
            order.append(wid)
            words[wid] = (w.tag, w.text, w.get('starttime'), w.get('endtime'), w.get('punc'))
        idx = {w: i for i, w in enumerate(order)}
        dafile = os.path.join(d, 'dialogueActs', f'{meeting}.{spk}.dialog-act.xml')
        for da in ET.parse(dafile).getroot():
            ch = da.find(NITE + 'child')
            if ch is None: continue
            m = re.findall(r'id\(([^)]+)\)', ch.get('href'))
            a = idx[m[0]]; b = idx[m[-1]]
            toks, sts, ens = [], [], []
            for wid in order[a:b + 1]:
                tag, text, st, en, punc = words[wid]
                if tag == 'w' and text:
                    toks.append(text)
                    if st: sts.append(float(st))
                    if en: ens.append(float(en))
            if not toks or not sts: continue
            txt = ' '.join(toks)
            txt = re.sub(r" ([.,?!'])", r'\1', txt)
            units.append({'id': da.get(NITE + 'id'), 'spk': spk, 'start': min(sts), 'end': max(ens), 'text': txt})
    units.sort(key=lambda x: (x['start'], x['end']))
    return merge(units), {'source': 'AMI', 'meeting': meeting}

if __name__ == '__main__':
    out = os.path.join(ROOT, 'data')
    for name in sys.argv[1:]:
        kind, key = name.split(':')
        turns, meta = gap(key) if kind == 'gap' else ami(key)
        mid = f'GAP{int(key):02d}' if kind == 'gap' else key
        json.dump({'meeting': mid, 'meta': meta, 'turns': turns}, open(os.path.join(out, f'{mid}.turns.json'), 'w'), ensure_ascii=False, indent=1)
        dur = turns[-1]['end'] - turns[0]['start']
        print(mid, 'turns', len(turns), 'units', sum(len(t['source_ids']) for t in turns), 'min', round(dur / 60, 1), 'spk', sorted(set(t['spk'] for t in turns)))
