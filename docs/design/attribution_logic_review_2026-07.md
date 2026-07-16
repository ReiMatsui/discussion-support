# 話者帰属ロジック 徹底レビュー（2026-07-16）

> 作成: Claude Fable 5 (claude-fable-5), 2026-07-16。ブランチ try/pyannote-live1（HEAD 3f7c5b5）。
> 依頼: 度重なる修正・チューニングで複雑化した話者帰属ロジック（ヒステリシス、
> クラスタ間名寄せ、ラベル継続、short_bonus、constrain、最近傍統合、遡及rekey…）を
> 先入観なしで調査・評価する。実装は含まない（提案まで）。

## 0. 調査方法

- ドキュメント精読: handoff_2026-07-14_unregistered_speakers.md 全節、pyannote_live1_trial_2026-07-09.md §8-9・§12
- コードトレース: _recv_loop.py / _session_state.py / _cluster_naming.py / _voice_profiles.py / _diarization.py / _constants.py の帰属全経路（サブエージェント2体で独立トレース後、重大主張は本体が行番号レベルで裏取り）
- 数値検証: eval/replay_attribution.py でベースライン再現（79%/未確定3%/誤帰属18%、ドキュメント§10と完全一致・決定的）＋ ablation 24条件（§3.2）
- 注意: replay ハーネスは**声紋層のみ**を再生する。ヒステリシス・constrain・名寄せ・最近傍統合・SpeakerResolver は乗らない（§5.1 に検証可否マップ）。GT は単一録音（2026-07-14_142016、オーバーフィット許容が明示された録音）であり、数値の一般化には限界がある。

## 1. 総合判定 (a)

**3役分業（Soniox=区切り・文字 / pyannote=クラスタ / 声紋=名前付け・名寄せ）というアーキテクチャ自体は妥当。**
盲検裁定2セッション（クラスタ一貫性78% vs 断片単位名前付け44%、trial §8.3-8.4）という実測根拠があり、§12「ボトルネックは区切り」とも整合する。帰属単位が Soniox の発話区切りである以上、区切り品質が天井を決めるのは責務配置の帰結として正しく、矛盾はない。

**一方、実装には構造的な歪みが2つある。**

1. **「昇格の厳格化」（handoff §3-2）は設計意図どおり動いていない。**
   ヒステリシス→@diar発行→constrain→最近傍統合という4機構の縫い合わせに順序の欠陥があり、最近傍統合は各生ラベルの初回キー発行の一瞬しか発動機会がない。スロット満杯後の新ラベルは「不可視の@diar:N」として恒久登録され、以後永久に未確定へ落ち続ける（矛盾C1、コードで確認済み）。パッチの積み重ねの歪みが最も集中している箇所。

2. **話者IDの台帳が3つ並存し、整合はflush時のその場しのぎに依存している。**
   ①STTラベル→sp_map（声紋層）、②pyannote生クラスタ→@diar:N/エイリアス（セッション層）、③人物N/実名プロファイル（声紋層）の3空間を、rekey・_confirmed・diarization_speaker_keys がバラバラに持つ。rekey は records/colors/names しか更新しないため、UI /rename や stdin fix の後にクラスタ確定名が腐り、リネームしたはずの人物が別人格として復活し得る（矛盾C3）。

**声紋層のGTチューニング（§10）は概ね健全だが、簡約可能。**
ablation の結果、短発話厳格化3点セット（strict_sec / short_bonus / short_margin_mult）のうち実効成分は short_bonus のみで、「3秒未満は閾値+0.08」という1行に等価簡約できる（出力ビット単位一致で実証、§3.2）。ラベル継続・自動登録・dedupe合流は削除すると劣化するか、削除根拠が不足しており維持が妥当。

**過去の判断で覆すべきもの**: handoff §8 の「短発話の積み上げでは最近傍統合が発動する前に昇格する（その後の名寄せ＋遡及rekeyで回収される設計）」という記述。回収は match_profile 成立（=登録プロファイルが照合できた場合）にしか起きず、**登録者ゼロ運用では回収経路が実質存在しない**。根拠は §4 C1/C2。

## 2. アーキテクチャ妥当性の詳細（観点1）

責務配置の評価:

| 役割 | 担当 | 評価 |
|---|---|---|
| 発話の区切り・文字 | Soniox | 妥当。ただし帰属単位＝Soniox区切りなので、区切り品質が全体の天井（§11-12で実証済み）。ここは本レビューの対象外の既知結論 |
| 同一声の束ね | pyannote Live-1 | 妥当（盲検裁定でクラスタ一貫性が声紋断片照合より優位） |
| クラスタへの名前付け | 声紋 match_profile | 妥当。5秒蓄積→照合は断片照合の弱点を正しく回避 |
| クラスタ間の名寄せ | 声紋埋め込みのコサイン比較 | 方向は妥当だが、閾値が無校正の流用（C6）で、統合の発動窓が狭すぎる（C1） |
| 発話単位の即時判定 | 声紋 classify | **責務が過積載**。照合・ラベル継続・自動登録・合流・AIエコー検出を1関数で担い、独自のID空間（sp_map・人物N）を持つ。3つ目の台帳が生まれる原因 |

矛盾がないかという問いへの答え: 分業の「切り方」は正しいが、「縫い目」（ID空間の統合点）が設計されておらず、flush() 内の手続き的な if 連鎖（_recv_loop.py:295-436）が事実上の統合層になっている。ここが唯一のアーキテクチャ上の欠落。

## 3. ヒューリスティック棚卸し (b)（観点2）

### 3.1 一覧表

判定凡例: ✅=維持 / 🔧=簡約・修正 / ❌=削除候補 / ⚠️=このデータでは判定不能

| 仕組み | 値・定数 | 位置 | 何の問題への対処か | 今も必要か | 重複・干渉 |
|---|---|---|---|---|---|
| 参加者化ヒステリシス | 3.0s (PYANNOTE_PARTICIPANT_HYSTERESIS_S) | _session_state.py:412-446 | pyannote序盤のラベル揺れで偽参加者乱造（trial §8.1） | ✅ 必要 | C1: @diar発行がconstrain前で最近傍統合を殺す。C4: 相槌もpending加算。C7: stt_fallbackにも波及 |
| ヒステリシスpending合算 | — | _session_state.py:448-460 | クラスタ分裂で参加者化が二重に遅れる | ✅ 必要（名寄せの付属品） | 単独では問題なし |
| クラスタ照合の最小蓄積 | 5.0s (CLUSTER_NAMING_MIN_SEC) | _cluster_naming.py:173 | 断片照合44% vs クラスタ一貫78% | ✅ 必要 | C2: ヒステリシス3s との時定数ずれ（3s<5s） |
| クラスタ間名寄せ | dedupe流用 (redimnet 0.50) | _cluster_naming.py:183-206 | 登録者ゼロでのクラスタ分裂→参加者乱造（handoff §3-1） | ✅ 必要（登録者ゼロ対応の本質） | C6: 閾値が無校正流用。C9: 確定後の回復手段なし |
| 最近傍統合 | dedupe下限 | _recv_loop.py:91-102 | max-speakers超過時の参加者増殖（handoff §3-2） | 🔧 意図は必要だが**現状ほぼ死んでいる** | C1: 発動窓が初回発行の一瞬のみ。C10: 成立しても非永続 |
| constrain（スロット選別） | max_speakers、辞書順sorted[:N] | _session_state.py:357-397 | 上限超過の匿名話者を未確定へ | 🔧 必要だが選別基準が脆弱 | C1: @diar発行後に落とすため不可視キーを量産。C5: ラベル文字再利用で先着逆転 |
| プロファイル済みキーのconstrain除外 | — | _session_state.py:382-383 | 人物2が辞書順選別で全滅した実バグ（handoff §9） | ✅ 必要（実害の再発防止） | なし |
| 重複発話ゲート | ratio 0.2 (OVERLAP_MIN_RATIO) | _diarization.py:168-196 | 混合音声によるバッファ汚染・誤帰属 | ✅ 必要 | なし |
| strict_sec（中尺厳格化） | 3.0s | _voice_profiles.py:448-451 | 1-2.5sの誤一致 sim 0.43-0.49（§10） | 🔧 効果は実証済みだが実効成分はshort_bonusのみ | 3点セットは1行に簡約可（§3.2 実験2=3=10） |
| short_bonus | +0.08 | _voice_profiles.py:450,494 | 同上 | ✅ 必要（外すと-25pt） | 同上 |
| short_margin_mult | ×2.0 | _voice_profiles.py:451,496 | 似た声の誤マッチ防止 | ❌ 削除候補（外して出力完全同一） | marginと二重の防御。どちらも本GTで不発 |
| short_floor（0.45-1.0s独立照合経路） | 0.45s | _voice_profiles.py:479-508 | 短発話を追従に落とさない | ⚠️ 無効化して出力完全同一。ただし本GTに0.45-1.0s帯の判定機会が少ない可能性 | 中尺経路とほぼ同条件で経路だけ別。AIエコーチェック欠落の非対称（C13） |
| hybrid（既知1人でも短発話照合） | set_hybrid | _voice_profiles.py:487 | 蓄積期の短発話を照合対象に（§10） | ⚠️ off で出力完全同一。ただし本来の対象シナリオ（登録1名）がGTに無い | なし |
| ラベル継続 | — | _voice_profiles.py:460-516 | 一度の照合失敗で#ラベルと人物Nに分裂（§10、44%→54%） | ✅ 必要（実測根拠あり） | C14: 継続中の音声が登録プールに入る非対称 |
| 人物別閾値 _person_th | median−0.12、履歴≥3 | _voice_profiles.py:365-370 | 吸収帯と本人帯の分離（吸収率91%→0%） | ✅ 必要 | C15: classify成功のみで学習し、クラスタ照合の閾値を一方的に動かす |
| margin（2位差） | 0.05 | _voice_profiles.py:451 | 似た声の誤マッチ防止 | ⚠️ 0にして出力完全同一（本GTで一度も発火せず）。多人数保険として維持推奨 | short_margin_multと重複 |
| dedupe（合流） | 0.50 | _voice_profiles.py:154 | 同一人物の重複登録防止 | ✅ 必要（0.40に緩めると-13pt過剰合流。1.0=無効でも本GTでは無風だが分裂セッションの保険） | C6: 4文脈流用 |
| 自動登録 | 45字 (enroll_min_total_chars) 等 | _voice_profiles.py:472-478,298-345 | 未登録者の人物N昇格 | ✅ 必要（20字に緩めると-9pt） | C14 |
| 相槌の未確定化 | _BACKCHANNEL_RE | _recv_loop.py:151,408-409 | 相槌の誤帰属（前話者追従28%の教訓） | ✅ 必要 | C4: pendingには加算される。デッドコードあり（D1） |
| voiceprint_high_confidence | 0.70 | _diarization.py:88 | Resolver優先順位の閾値 | ❌ デッド（呼び出し側が常にconf=1.0） | C8 |
| STTフォールバック参加者化 | — | _recv_loop.py:390-399 | diarization欠落時の受け皿 | ✅ 必要 | C7: pyannote用ヒステリシスが意図不明なまま適用 |

### 3.2 ablation 実測値（replay_attribution.py、GT=2026-07-14_142016、71発話採点、1発話≈1.4pt）

| 条件 | 精度 | 未確定 | 誤帰属 | Δ |
|---|---|---|---|---|
| ベースライン | **79%** | 3% | 18% | — |
| strict_sec=0 | 54% | 3% | 32% | **-25pt** |
| short_bonus=0 | 54% | 3% | 32% | **-25pt**（strict_sec=0と出力ビット単位一致） |
| short_margin_mult=1.0 | 79% | 3% | 18% | ±0（完全同一） |
| short_floor=1.0（短発話経路消滅） | 79% | 3% | 18% | ±0（完全同一） |
| --no-hybrid | 79% | 3% | 18% | ±0（完全同一） |
| margin=0 / 0.10 | 79% | — | — | ±0（一度も発火せず） |
| dedupe=0.40 | 66% | — | — | -13pt（過剰合流） |
| dedupe=0.45〜0.60 / 1.0 | 79% | — | — | ±0 |
| enroll_min_total_chars=20 | 70% | 10% | 11% | -9pt（早期登録が枠を食い潰す） |
| enroll_min_total_chars=90 | 79% | — | — | ±0 |
| strict_sec sweep 2.0/3.0/4.0/5.0 | 79% | — | — | 2.0以上で平坦 |
| short_bonus=0.04 / 0.12 | 66% / 79% | — | — | 0.08で飽和 |

重要な帰結: **実験2・3・10（3点セット同時無効）の出力が完全一致** → 厳格化パスの実効成分は short_bonus のみ。「wav < 3.0s なら採用閾値に +0.08」という1行と等価。margin系ガード2つ（margin、short_margin_mult）は本セッションで完全に不発。

限界: 単一録音・3話者・71発話。margin/hybrid/dedupe合流の「±0」は「このGTで発火機会がなかった」ことしか意味しない。削除判断は「発火機会が構造的に無い」（short_margin_mult: strict帯でmarginの2倍を要求するが、strict帯の実効判定はボーナス側で先に決まる）ものに限定するのが安全。

## 4. 発見した矛盾・破綻 (c)（観点3）

重大度順。C1-C3 が構造的、C4-C10 が個別、C11以降が軽微。

### C1. 最近傍統合の死角（重大）— handoff §3-2 の設計意図が未達

状態遷移で追うと:

```
新ラベル raw 出現（スロット満杯状態）
→ [1発話目〜] key_for_diarization_speaker: pending加算、UNSURE（ヒステリシス3s）
→ [累積3s到達] _merged_diarization_speaker_key:91 の条件
     「raw not in diarization_speaker_keys ∧ budget_exhausted」は真だが、
     nearest_cluster() は各クラスタの代表埋め込み（≥5s蓄積+照合失敗後に生成）が
     まだ無ければ None → 経路3へ
→ [経路3] key_for_diarization_speaker が @diar:N を発行し
     diarization_speaker_keys[raw] に恒久登録（_session_state.py:440-446）
→ [同じflush内] constrain が UNSURE に落とす（:395-396）。しかし登録は残る
→ [以後の全発話] raw in diarization_speaker_keys が真（:432-433）のため
     最近傍統合の前提条件（:91「raw 未キー」）が永久に偽 → 永久に未確定
```

つまり最近傍統合が発動し得るのは「ヒステリシス消化の瞬間に、たまたま統合先の代表埋め込みが既に存在し、sim≥dedupe」という狭い窓だけ。しかも代表埋め込みの生成には5秒蓄積が要る（C2）ので、序盤はほぼ確実に窓を外す。**スロット満杯後の新クラスタは実質全て未確定行き**であり、「超過分は最も近い既存参加者へ統合」（handoff §3-2）は動いていない。実地テスト（§12）で未確定50%だった一因の可能性が高い。

### C2. ヒステリシス3s < 埋め込み5s の時定数ずれ（既知課題の悪化版）

handoff §8 は「昇格が先行しても、その後の名寄せ＋遡及rekeyで回収される」としたが、回収経路は2つとも条件付き:
- match_profile 成立による確定＋遡及rekey → **登録プロファイルがある場合のみ**
- クラスタ間名寄せ → match_profile 不成立時に発動するが、統合先の代表埋め込みが必要（これも5s＋照合失敗を経ないと生成されない）

登録者ゼロ・スロット満杯では両方が塞がり、C1と合流して「回収されない昇格」だけが残る。§8 の記述は登録者ありの条件でのみ正しい。

### C3. rekey が3つの台帳を更新しない（リネーム後の人格復活）

`rekey()`（_session_state.py:529-549）は records/colors/names/anonymous_labels のみ更新。以下は残留する:

- `ClusterVoiceNamer._confirmed`: クラスタ→"人物2" 確定後に UI /rename で人物2→実名にすると、profiles からは人物2が消える（enroll がpop）が、_confirmed は"人物2"のまま。以後そのクラスタの発話は observe() が古い"人物2"を返し（_cluster_naming.py:165-167）、constrain では profiles に無いため匿名扱い→ max_speakers 判定→ disp_name が新しい「参加者X」文字を割当て。**リネームした人物が別人格として復活・分裂する**。stdin fix（tracker.remap）も同型。
- `diarization_speaker_keys`: 同様に古い確定名のまま残る（更新箇所は _recv_loop.py:349 と :86 のみ）。
- `tracker.sp_map`: rekey とは独立に tracker 側が更新するが、同期の保証はない。

### C4. 相槌がヒステリシスpendingを加算する

相槌（_is_backchannel）でも resolver→key_for_diarization_speaker は通常どおり呼ばれ、pending に加算される。最終レコードは constrain 前に UNSURE 化されるのに、**相槌の積み上げだけで @diar:N が発行され得る**。「幻ラベルの参加者化防止」というヒステリシスの趣旨と逆行。前話者追従を廃止した判断（相槌は帰属根拠にしない、c50d1a9）とも不整合。

### C5. constrain の辞書順スロット選別は構造的に脆弱

`sorted(labels)[:max_speakers]`（:391-392）は「若い文字＝先着」を仮定するが、ラベル文字は実名化で解放・再利用される（:294-302）。後着の話者が解放済みの若い文字を拾うと先着話者が上限外へ転落する逆転が起き得る。人物2事件（handoff §9）は profiles 除外で塞いだが、匿名同士の逆転の根は残っている。

### C6. dedupe 閾値（0.50）の4文脈流用

「3発話プロファイル同士」で校正した値（_voice_profiles.py:113-114 のコメント）を、①_commit_profile の合流（本来の文脈）②activate ③クラスタ間名寄せ（5-20s連結埋め込み vs クラスタ代表）④最近傍統合（代表同士）に流用。③④の独立校正記録はない。handoff §8 自身が「20秒連結音声同士では保守側に倒れる可能性、ラベル総数超過時の第一容疑」と指摘済みのまま放置されている。

### C7. stt_fallback にも pyannote ヒステリシスが掛かる

`_uses_pyannote_hysteresis()` は provider 名しか見ないため、pyannote 使用時は Soniox の STTフォールバックラベル（"stt:N"）にも3秒ヒステリシスが適用される。docstring の趣旨（pyannote のラベル揺れ対策）とラベルの出所が一致していない。意図的なら文書化が、非意図なら修正が要る。

### C8. voiceprint_high_confidence=0.70 はデッドな校正値

呼び出し側（_recv_loop.py:301）が信頼4種のとき常に conf=1.0 を渡すため、閾値比較（_diarization.py:104-114）は恒真。「声紋はdiarizationに無条件で勝つ」が実態であり、Resolver の confidence 抽象は形骸化。将来実simを渡す変更をすると挙動が暗黙に変わる罠。

### C9. クラスタ確定の永久凍結（誤確定の回復手段なし）

_confirmed に入ると overlapped でも再照合なしで確定名を返し続ける。最初の5秒の照合が誤っていた場合（劣化音声で混合クラスタが registered profile に誤マッチ等）、クラスタ単位の回復手段がない。trial §9.1 の残課題として認識済みだが、C3と組み合わさると被害が拡大する。

### C10. 最近傍統合が非永続

統合成立時に alias も diarization_speaker_keys[raw] も記録しないため、毎発話 nearest_cluster を再計算する。埋め込みの揺れで同じ raw が発話ごとに別参加者へ振れ得る（C1により現状は発動自体が稀なため実害は潜在的）。

### C11-C15. 軽微・観測性

- C11: rekey が colors を pop するため、html_color のインデックスがずれ、以後の話者の HTML 色が変わる（「再読み込みでも色がぶれない」コメントと矛盾）
- C12: set_diarization_max_speakers のコメント「次回リセットで反映」と実態（constrain は即時反映）の乖離。会議中に減らすと既存参加者が突然未確定化
- C13: 短発話経路（0.45-1.0s）に AI声紋チェックがない（中尺経路 L437-442 との非対称）
- C14: ラベル継続中の照合失敗音声が登録プールに入る。照合側は 0.42+0.08 を要求するのに、登録側の一貫性判定は 0.42 相当・1.5s窓埋め込みで、strict_sec の「中尺埋め込みは信用しない」原則が登録側に及んでいない
- C15: _person_th は classify 成功のみで学習し、match_profile（クラスタ照合）の閾値を一方的に引き上げる。2系統の判定器が閾値を共有して片方だけ学習する構図

### D. デッドコード

- D1: _recv_loop.py:434 `sp_id = UNSURE_SPEAKER`（直後:436で必ず上書き。生きているのは:435のbcフラグのみ）
- D2: _cluster_naming.py `confirmed_name()`（src内呼び出しなし）、`nearest_cluster(exclude=)`（未使用引数）
- D3: kind文字列「相槌追従」は追従しない経路のセンチネルとして生存（実態は「ラベル継続」。stats内訳・diag の読み手を誤導）
- D4: tests/unit/live/test_recv_loop_hybrid_follow.py の docstring は廃止済みkind「相槌未確定」前提のまま（フェイク注入でテストは通るが仕様記述が乖離）
- D5: observe() 内の名寄せループは nearest_cluster() とほぼ同一のインライン重複実装（片方だけ直す事故の温床）

## 5. 簡素化提案 (d)（観点4、優先順）

### P1. 匿名キー解決を単一の状態機械に統合し、「発行」と「表示可否」を分離しない【最優先・効果大・リスク中】

対象: ヒステリシス／@diar発行／constrainスロット判定／最近傍統合（C1・C2・C4・C10 を一括解消）。

現状は「発行してから落とす」ため不可視キーが量産される。これを「発行前に判定する」1本の流れに畳む:

```
未キーの raw の解決（1関数）:
  1. 相槌・重なり発話 → pending加算せず UNSURE（C4解消）
  2. pending < 3s → UNSURE
  3. 名寄せ成立済み → canonical キー（現行どおり）
  4. スロットに空きあり → @diar:N 発行
  5. スロット満杯 → 最近傍統合を試み、成立なら
     diarization_speaker_keys[raw] = nearest_key を永続記録（C10解消）。
     不成立なら**キーを発行せず** UNSURE のまま pending を保持
     （後続発話で埋め込みが育ってから再試行できる＝C1の死角解消、
      C2の時定数ずれも「待てる」ことで実質吸収）
```

- 期待効果: handoff §3-2 の設計意図（超過分の最近傍統合）が初めて実際に機能する。登録者ゼロ・スロット満杯での「永久未確定」が解消し、実地テストの未確定50%（§12）の一部を回収できる見込み。constrain は「closed roster と profiles 素通し」だけに縮小でき、辞書順選別（C5）も廃止可能
- リスク: 中。replay では検証不可のため、ライブ再実験（--wav 再生）＋ eval_speaker_gt.py のタイムライン方式で before/after を測る。ヒステリシス・名寄せ関連の既存テスト（pending合算等）の書き換えが必要
- 規模: _recv_loop.py の _merged_diarization_speaker_key と _session_state.py の key_for_diarization_speaker / constrain の再編。既存挙動の温存が必要なのは「pyannote 以外の provider は即時発行」のみ

### P2. rekey を状態一貫性の単一入口にする【効果大・リスク小〜中】

対象: C3・C9・C11。

rekey(old, new) 時に diarization_speaker_keys の値・ClusterVoiceNamer（_confirmed/_aliases/_embeddings のキー）・必要なら tracker.sp_map へ変更を伝搬する（SessionState が namer/tracker への参照を既に持つので、明示呼び出しで足りる。オブザーバ機構は不要）。html_color は colors のインデックス依存をやめ、キーのハッシュか採番順の別辞書にする。

- 期待効果: UI /rename・stdin fix 後の人格復活バグ（C3）の根絶。C9 の部分緩和（確定名の付け替えが正しく波及する）
- リスク: 小〜中。ユニットテストで再現・検証可能（「rename後に同クラスタの発話が旧名で復活しない」）。replay 対象外だがライブ再実験も不要な純粋な整合性修正
- 規模: rekey 本体＋namer に rename_confirmed(old, new) 追加程度

### P3. 短発話厳格化3点セットを1行に簡約【実証済み・リスク最小】

対象: strict_sec / short_bonus / short_margin_mult（＋short_floor 独立経路）。

「wav < 3.0s なら採用閾値に +0.08」に置換し、short_margin_mult を削除。0.45-1.0s の独立照合経路（L479-508）も中尺経路と同一条件になるため統合できる（AIエコーチェックの非対称 C13 も自然解消）。margin（2位差 0.05）は多人数時の保険として1本だけ残す。

- 期待効果: 精度変化なし（replay で出力ビット単位一致を確認済み）。_classify の分岐が4バンド→2バンド（照合する/しない）に減り、可読性が大幅に向上
- リスク: 最小。ただし hybrid フラグと short_floor の削除判断は保留を推奨（本GTに発火機会がなかっただけで、「登録1名＋短い相槌が多い実会話」が本来の対象シナリオ。実会話検証後に再判定）
- 検証: replay で before/after 完全一致を確認するだけ

### P4. dedupe の文脈分離【リスク小】

クラスタ間名寄せ・最近傍統合用の閾値を別定数（例: CLUSTER_MERGE_SIM、初期値は現行と同じ0.50）に分離し、replay とは別の校正手段（gt付きライブ再実験時に diag の cluster_naming イベントで sim 分布を取る）を用意する。handoff §8 の「第一容疑」を検証可能にする観測性整備であり、値の変更はしない。

- 期待効果: P1 で最近傍統合が生き返った後のチューニングを安全にする前提整備
- リスク: 小（値は変えない）

### P5. デッドコード・名称の掃除【リスクほぼゼロ】

D1-D5 の削除・改名（「相槌追従」→「ラベル継続待ち」等）、voiceprint_high_confidence の削除（または実simを渡して閾値を生かすかの明示判断、C8）、observe() 内の名寄せループを nearest_cluster() 呼び出しに一本化（D5）、C7（stt_fallback へのヒステリシス適用）の意図の文書化 or 修正、C12 のコメント修正。

- 期待効果: 挙動不変のまま誤読リスクを除去。テスト docstring の乖離（D4）も解消
- リスク: ほぼゼロ（テストで挙動不変を担保）

### P6.（長期・任意）話者IDレジストリの一本化

P1・P2 が入れば実害は概ね解消するが、根治は「キーの発行・統合・改名・表示名を一元管理する SpeakerRegistry」に3台帳を寄せること。flush() の if 連鎖は「観測を registry に渡し、確定キーを受け取る」だけになる。9月パイロット前の大手術になるため、実会話検証（handoff §12 の次アクション）で現構成の実力を見てから判断することを推奨。

### 実施順の提案

1. P5（掃除、無風）→ 2. P3（実証済み簡約）→ 3. P2（整合性修正、テストで担保）→ 4. P1（構造修正、ライブ再実験で検証）→ 5. P4（P1後の校正整備）→ 6. P6は実会話検証の結果待ち

## 6. 検証方法メモ

- 声紋層（P3）: eval/replay_attribution.py で before/after（決定的・数秒）。sandbox 実行時は Linux 用 venv が別途必要（リポジトリ .venv は macOS-arm64）
- セッション層（P1・P2）: replay に乗らないため、(1) ユニットテスト（状態遷移の直接検証）、(2) `--wav transcripts/2026-07-14_142016.wav` のライブ再実行＋ eval/eval_speaker_gt.py（タイムライン方式、§11）で未確定率の変化を測る。ただしこの録音は §12 で「対象外」判定済みの劣化音源なので、合否は未確定率の相対比較に留め、最終判定は実会話で行う
- 統計上の注意: 71発話採点では±4pt（3発話）が誤差帯。sweep の僅差追いは過学習
