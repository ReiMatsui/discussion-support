# ローカル Sortformer をライブで使う（--diarization sortformer）

いつものハイブリッド構成の「話者分離だけ」を、pyannote Live-1（クラウド）から
ローカルの NVIDIA Streaming Sortformer に差し替える opt-in モード。
画面・操作・声紋名前付け・介入はすべて従来どおり。既定の挙動は変わらない
（`--diarization sortformer` を付けたときだけ有効）。

実測での位置づけ（docs/design/sortformer_feasibility_2026-07-22.md）:
クリーン音源（千葉/電話）では現行構成を大差で上回る一方、マイク＋部屋残響の
音では 41〜44% まで崩れる（現行 79%）。**未確定はほぼ出さないが、その分
「間違った名前が自信満々に付く」方向に倒れる**。この目視検証のためのモード。

## 1. セットアップ（Mac、初回のみ）

NeMo は重い依存なので、本体の環境とは別の専用 venv に入れる。
ディスクを 3〜4GB ほど使う。

```
python3 -m venv ~/.venvs/sortformer
~/.venvs/sortformer/bin/pip install -U pip
~/.venvs/sortformer/bin/pip install "nemo_toolkit[asr]"
```

モデルの事前ダウンロード（初回のライブ起動を軽くするため推奨）:

```
~/.venvs/sortformer/bin/python -c "from nemo.collections.asr.models import SortformerEncLabelModel; SortformerEncLabelModel.from_pretrained('nvidia/diar_streaming_sortformer_4spk-v2.1')"
```

動作確認（手元の録音を1本流してイベントが出れば OK）:

```
cd ~/discussion-support
ffmpeg -loglevel error -i transcripts/2026-07-14_142016.wav -f s16le -ar 16000 -ac 1 - | ~/.venvs/sortformer/bin/python scripts/sortformer_worker.py | head
```

`{"e": "ready"}` に続いて `{"e": "start", ...}` が出れば動いている。

## 2. 使い方

```
uv run das listen-soniox --hybrid --diarization sortformer --max-speakers 3
```

`--hybrid` の pyannote 指定を後勝ちで sortformer に差し替える書き方。
声紋クラスタ名前付け（--vp-cluster-naming）は自動で引き継がれる。

処理が重い/遅いと感じたら Apple Silicon の GPU（MPS）を使う:

```
uv run das listen-soniox --hybrid --diarization sortformer --max-speakers 3 \
    --soniox-args "--sortformer-device mps"
```

venv を別の場所に作った場合は環境変数で教える:

```
export SORTFORMER_PYTHON=/path/to/venv/bin/python
```

## 3. 既知の制約・期待値

- **話者は最大4人**（モデル仕様）。5人以上の会議では使えない。
- レイテンシ約1秒（pyannote Live-1 は 0.3 秒）。画面の話者確定がわずかに遅れる。
- 会議リセット/STT再接続のたびにワーカーがモデルを読み直す（数十秒）。
- 処理速度: 検証用クラウドの非力な2コアCPUでは実時間の約2.3倍かかった
  （＝ライブに追いつかない）。M系チップの CPU では数倍速い見込みだが、
  **初回に必ず上の動作確認コマンドで所要時間を測ること**。4分の音源が
  4分未満で処理できなければ、`--sortformer-device mps` を使う。
- ワーカーが起動失敗・途中死しても本体は落ちない（以後 Soniox＋声紋のみで
  継続し、その旨をログに出す）。
- 帰属ロジック側の扱いは pyannote と同一: 新規クラスタの参加者化に
  3秒ヒステリシス、@diar 採番、声紋による名前確定、0.65 確定バー。

## 4. 実装の場所

- ワーカー: `scripts/sortformer_worker.py`（NeMo venv 側で動く。stdin=PCM16、
  stdout=JSON Lines。低遅延プリセットは HF モデルカード公表値）
- プロバイダ: `src/das/asr/live/_sortformer_diarization.py`（サブプロセス管理）
- 配線: `_bootstrap.py`（構築）、`_session_state.py`（ヒステリシス）、
  `src/das/asr/live/__init__.py` / `src/das/cli/_listen.py`（CLI）
- テスト: `tests/unit/live/test_sortformer_provider.py`（フェイクワーカー）
- ワーカー実音声E2E: クラウドで 142016 を流し 39.4%（バッチ推論 42.3% と
  整合。差は低遅延構成の代償）を確認済み
