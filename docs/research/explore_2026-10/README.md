# explore_2026-10: 支援対象の探索的再分析

既存の事前検証（`../precheck_2026-10/`）を監査したうえで、介入案を前提にせず、8議論を読み直した分析。旧資料・旧基準は変更していない。

## ファイル一覧

| ファイル | 内容 |
|---|---|
| `00_報告.md` | 最終報告（結論、観測、解釈、残す・変える・捨てる、次の試行、成果物、できなかったこと） |
| `01_監査結果.md` | 作業1: 既存検証の監査 |
| `audit/` | 監査の再計算（`recompute/`）、ハッシュの照合、控える理由の内訳 |
| `02_分析対象と選定理由.md` | 対象8議論と選定理由、データの利用条件 |
| `read/codebook_v1.md` | 探索的な分類 v1 |
| `read/codebook_v2_changelog.md` | 分類の変更履歴（v2） |
| `read/coder_prompt*.md` | サブエージェントへの記録の依頼文 |
| `cases/` | 事例（`cases_all.csv`）、区間、議論ごとの集計、候補ごとの集計、引用の確認 |
| `04_事例一覧の読み方と代表例.md` | 集計表と代表例 |
| `06_支援候補の比較と推奨.md` | 作業3・4: 最大3候補の比較と、方向 A〜D の推奨 |
| `07_予備試行資料/` | 作業5: 課題と説明、研究者の判断手順と提示例、研究上の問いと観察・面接項目、符号化表、切り分けと判断基準 |
| `08_出典と制約.md` | データと文献の出典（確認度つき）、制約 |
| `scripts/` | 再現用スクリプト |

## 再現手順

元データの取得と前処理は、旧検証の手順に従う（`../precheck_2026-10/scripts/README.md`）。

- 前処理の出力 `data/<議論>.turns.json` は、旧スクリプト `prep.py` で作る。
- 以下で `<pre>` は旧作業フォルダ（`data/`、`runs/`、`out/` を含む）、`<ex>` はこのフォルダを指す。

```
# 監査: 旧集計の再計算と、控える理由の内訳
python3 -I scripts/audit_recompute.py <pre> <ex>/audit/recompute
python3 -I scripts/audit_withhold_reasons.py <pre>

# 事例の統合（時刻・元の発話 ID の付与）と集計
python3 -I scripts/build_cases.py <pre> <ex>
python3 -I scripts/candidate_stats.py <ex>
python3 -I scripts/verify_quotes.py <pre> <ex>
```

- `audit_recompute.py` は、旧検証の封印ファイル（`runs/*/.sealed`）を必要とする。Mac の `precheck_2026-10` には同梱されていない（全文の文字起こしなので）。旧手順で再生成できるが、判定ログとの対応を保つため、旧作業フォルダが必要になる。
- 文字起こしの全文（`read/transcripts/`）は、元データの再配布を避けるため同梱していない。`prep.py` の出力から次の形で作れる。
  - 形式: `ターン番号|開始秒|話者|発話`

## 記録者

- GAP01 と ES2003c は、主作業者（Claude）が全文を読んで記録した。
- 他の6議論は、分類 v1 と記入例を渡したサブエージェント（Claude）が記録した。
- 人による確認はしていない。
