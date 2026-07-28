# 実会話マイク直検証 実行手順書（2026-07-17 準備）

> 作成: Claude Fable 5。handoff_2026-07-14_unregistered_speakers.md §18.4 の
> 本命タスク「実会話（マイク直）検証」を、機材の前に座ったら迷わず回せる形に
> 段取り化したもの。プロトコルの原典は同 handoff §4（検証項目）と
> §15.3（ゲート方式・実用ライン）。**測定方法論は変更していない。**

## 0. 位置づけ — なぜこれが本命か

これまでの検証は全て本来ドメインの近似条件だった:
YouTubeスピーカー再生（劣化・対象外判定済み, §12）、CallHome（8kHz電話・
2話者）、Chiba（対面3人だがヘッドセット由来のクリーン音声, §15.2）。
**「対面・実マイク・室内残響・登録者ゼロ・3人」の本来条件は未測定**。
また replay で反復できるのは声紋層のみで、クラスタ層（ヒステリシス・
constrain・クラスタ確定バー0.65）の総合挙動はライブでしか測れない。

## 1. 事前準備（5分）

```bash
# ブランチ取り込み（未実施なら）
git checkout main && git merge refactor/attribution-cleanup
git branch -d try/pyannote-live1 fix/code-review-2026-07-07  # 整理（§18.3）

# 環境確認
uv run pytest -q          # 891件グリーンが基準
uv run python eval/replay_attribution.py   # 79%/未確定3%/誤帰属18% を確認
```

- `.env` に `SONIOX_API_KEY`（必須）。介入を切って測るなら OPENAI は不要
- 参加者3人・実マイク1本（会議想定の距離）。**登録者ゼロ**（voices.json の
  実名プロファイルは activate しない）で開始
- 静かすぎる部屋より、実際に使う会議室相当が望ましい（残響耐性が今回の主眼）

## 2. 実施メニュー（handoff §4 の現行版）

すべて推奨構成（ハイブリッド）で実行:

```bash
uv run das listen-soniox --hybrid --max-speakers 3 \
  --vp-mint-cluster-link --soniox-args "--no-agent"
```

**`--vp-mint-cluster-link` を必ず付ける**（2026-07-25 追加）。二重帳簿の根治
（handoff §21）の opt-in で、既定は無効。付けないと「同じ人が2席を取り実在者が
締め出される」現象がそのまま再現し、この検証で見たい数字が取れない。
A/B が必要なら記録は1本でよく、後述の診断スクリプトが同じ記録を
on/off で流し直す（実会話を2本録る必要はない）。

**想定話者数（`--max-speakers`）は実際の参加人数に合わせる。** 2026-07-25 の
実会話3本は上限1のまま録れており（§20-3）、当時は設定が記録に残っていなかった
ため事後まで気づけなかった。現在は diag の1行目に構成が残るので、録音直後に
`head -1 transcripts/$SESSION.diag.jsonl` で確認できる。

（介入込みの通し確認をしたい場合は `--no-agent` を外す。ただし帰属測定の
本数を稼ぐ間は切っておくとAPIコストと交絡を避けられる）

| ラン | 条件 | 合格基準（その場で見る） |
|---|---|---|
| A. 本命 | 登録者ゼロ・3人・10分以上の自然な議論 | 参加者ラベル総数 ≤ 実人数+1、主要話者が概ね同一ラベルに集まる |
| B. 登録併用 | 本人＋登録済み1名（activate して2人会話） | 表示が「登録名2つ＋未確定」のみになる |
| C. 回帰 | 単独発話（1人で数分） | 参加者が増えない（ヒステリシス回帰） |
| D. 操作系 | ラン中に stdin で `1=名前` の実名化と /rename | リネーム後に旧名の人格が復活しない（P2 の実地確認） |

- 録音・turns・diag は transcripts/ に自動保存される。**セッション名を控える**
  （`ls -t transcripts/*.turns.jsonl | head -1`）
- A は可能なら2本（会話の重なり方が違う日・メンバーで）。1本の数字で
  結論を出さない（§15.3 の過学習警戒と同じ姿勢）

## 2.5 録音直後にまず流す（アノテーション不要・1分）

GT を付ける前に、この1本から取れる判断材料をまとめて出す:

```bash
SESSION=$(basename "$(ls -t transcripts/*.turns.jsonl | head -1)" .turns.jsonl)
uv run python eval/diagnose_live_session.py --session $SESSION
```

出るもの（詳細はスクリプトの docstring）:

- **A. 再生の忠実性** — 記録に判定の入力が揃っているか。揃っていれば
  `eval/replay_live_attribution.py --session $SESSION` の「記録との自己一致」が
  100% になるはずで、**そうなって初めて ID空間の統合（§24.3）に着手できる**
- **B. クラスタ分裂** — 実質クラスタ数 ÷ 想定話者数。**項目2（土台が2つとも
  壊れている問題）に決着をつける数字**。1.0 に近ければ土台は健全で改善は
  声紋層・門番側で効く。大きく上回るなら分裂が支配的で、門番を増やしても
  頭打ち＝入力側（マイク配置・複数マイク・DOA）の問題
- **C. 席の圧迫** — 未確定率と、上限で落ちたキーの内訳
- **D. 声紋層の健全性** — ラベル不純率・クラスタ確定の見送り・鋳造リンクの発火

鋳造リンクの A/B（同じ記録を on/off で流し直す）:

```bash
uv run python eval/diagnose_live_session.py --session $SESSION --ab
```

Phase 0 の予測は GT11本平均で 実質 +9pt / 未確定 −15pt（§21）。実会話で同じ
向きなら opt-in を既定化してよい、という判断材料になる。

## 3. 終了後の採点（1本あたり: アノテーション20-30分＋採点1分）

```bash
SESSION=<セッション名>   # 例: 2026-07-18_1030

# 1) 正解を付ける（ブラウザが開く。音声もサーバから配られるので選択不要）
uv run python eval/annotate.py $SESSION
# → 1〜9 で話者、S で直前と同じ、0 不明、M 複数人。
#    eval/gt_$SESSION.json へ自動保存されるので、途中でやめて再開できる。

# 2) 採点
uv run python eval/eval_speaker_gt.py eval/gt_$SESSION.json

# 3) 声紋層だけのオフライン反復が必要になったら（チューニングは凍結中なので
#    原則は原因分析用途のみ。GTがそのセッションの区切りなのでそのまま回る）
uv run python eval/replay_attribution.py --gt eval/gt_$SESSION.json
```

**見る指標（この順）**:

1. **相槌除外（実質発話）の精度と誤帰属** — 実用ライン（§15.3 暫定合意）:
   精度 ≥80%・誤帰属 ≤10%
2. ラベル爆発なし（参加者総数 ≤ 実人数+1）
3. 見出しの全発話精度は参考値（相槌の未確定はルールなので下がって正常）
4. 「判定したうちの正解率」= 精度 ÷ (精度+誤帰属)。議事録の信頼度の実感値

## 4. diag 分析（次の判断の材料集め）

`transcripts/$SESSION.diag.jsonl` から2つの分布を取る:

```bash
# a) クラスタ確定・名寄せイベントと nearest_sim 分布（merge_sim 校正材料, P4）
grep '"type": "cluster_naming"' transcripts/$SESSION.diag.jsonl \
  | python3 -c "import sys,json
for l in sys.stdin:
    d=json.loads(l)
    print(d.get('match'), d.get('kind',''), d.get('nearest',''), d.get('nearest_sim',''))"

# b) 未確定の内訳（P1 再判断の材料）: final_key が ? の発話の kind を集計
python3 -c "import json,collections
c=collections.Counter()
for l in open('transcripts/$SESSION.diag.jsonl'):
    d=json.loads(l)
    if d.get('final_key')=='?': c[d.get('kind','(なし)')]+=1
print(c.most_common())"
```

読み方:

- **a)** 正しい確定が 0.65 バー付近で足止めされていないか（§15.11 の分離帯
  誤≤0.62/正≥0.66 が実会話でも保たれるか）。nearest_sim の同一人物/別人の
  帯が分離して見えるなら merge_sim 復活の校正材料になる（現状は §15.12 で
  既定無効のまま）
- **b)** 未確定の主因が「相槌」なら想定内（DOA等の将来課題）。
  「スロット満杯後の新クラスタ」（speaker_source が pyannote 系で
  final_key だけ ? が続く）が主因なら **P1（§16 保存設計）の再判断材料**。
  P1 の差し込み点はコード整理済みで `_attribution._anonymous_cluster_key`
  内部＋相槌 pending 分岐に閉じる（§18.1）

## 5. 分岐（§15.3 のゲート方式を踏襲）

- **実質発話で実用ライン達成（2本とも）** → git tag で確定版。チューニング
  凍結のまま9月パイロットへ。残るは介入込みの通しリハーサル
- **未確定が主因で未達** → §4-b の内訳を根拠に P1 を再判断（挙動変更なので
  実装は別承認・別ブランチ。検証はユニットテスト＋本手順の再実施）
- **誤帰属が主因で未達** → diag の sim 分布から該当門番を1つだけ特定して
  校正（1変更=1コミット=1測定、同時に2変更しない）
- **区切り（混在・範囲外）が主因** → §11 選択肢b の再検討（最後の手段）

## 6. してはいけないこと（既存合意の再掲）

- chiba0132 以外での閾値チューニング（§15.3: ホールドアウト保護）
- 相槌を音響情報だけで回収しようとする再挑戦（§15.6 で撤退済み・DOA案件）
- クラスタ同士の統合（名寄せ）の安易な再有効化（§15.12: 分離可能な閾値が
  存在しないことが実測済み。復活はこの手順の a) 分布を見てから）
