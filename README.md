# Rock-Paper-Scissors 画像分類 — NNC学習からSpresense推論までの記録

じゃんけん画像分類モデルを Sony Neural Network Console (NNC) で学習し、
Spresenseマイコンに書き込んでリアルタイム推論するまでの過程を、
試行錯誤・失敗・対策まで含めて記録したものです。

---

## 1. プロジェクト概要

| 項目 | 内容 |
|---|---|
| タスク | じゃんけん（paper / rock / scissors）の3クラス画像分類 |
| 学習環境 | Sony Neural Network Console (NNC) |
| 推論環境 | Spresense（メイン基板 + カメラ拡張ボード） |
| データセット | Laurence Moroney 作の CGI ベース Rock-Paper-Scissors（Kaggle配布版） |
| 最終目的 | Spresenseのカメラでじゃんけんの手をリアルタイム識別 |

---

## 2. データセット

### 2.1 元データ

Kaggleで配布されている `Rock-Paper-Scissors` データセット。
Laurence Moroney 氏が CGI レンダリングで生成したもので、
背景・光源条件が学習用と評価用で微妙に異なることが知られている
（汎化が難しい性質を持つ）。

| 種別 | 枚数 | 解像度 | クラス |
|---|---|---|---|
| train | 各クラス 840枚（計2520枚） | 300×300 RGBA PNG | paper / rock / scissors |
| test  | 各クラス 124枚（計372枚） | 300×300 RGBA PNG | paper / rock / scissors |
| validation | 計33枚（命名規則が異なる） | 300×300 PNG | 混在 |

### 2.2 NNCで読み込める形式への変換

NNCは画像のパスとラベルを書いたCSVインデックスを読み込む方式。
仕様：

- 1行目はヘッダ `x:image,y:label`
- パスはCSVからの相対パス
- ヘッダの `:` 左側が変数名（Input層の `Dataset` 欄、CategoricalCrossEntropy層の `T.Dataset` 欄に紐づく）

クラスIDは `paper=0, rock=1, scissors=2`（アルファベット順）。

#### 変換スクリプト `make_nnc_dataset.py`

主要な処理：

- 元の300×300画像を **指定サイズの正方形にリサイズ**（デフォルト64×64）
- RGBA→RGB変換（透過部は白背景に合成）
- `train/`, `test/` 配下の画像を読み込み、`nnc_dataset[_<size>]/` に出力
- `train.csv`, `test.csv` を生成

実行例：
```bash
python make_nnc_dataset.py             # 64x64 → ./nnc_dataset/
python make_nnc_dataset.py --size 32   # 32x32 → ./nnc_dataset_32/
```

`validation/` フォルダの33枚は train/test と命名規則が違うため除外。

### 2.3 生成された出力

```
nnc_dataset/         (64x64 RGB, 16MB, 2892枚)  ← 通常学習用
nnc_dataset_32/      (32x32 RGB, 11MB, 2892枚)  ← Spresense向け
```

---

## 3. GitHub への公開

学習に必要な要素を切り出して公開：
https://github.com/ruiname1218/janken_dataset

含めたもの：
- `nnc_dataset/`（必須）
- `make_nnc_dataset.py`（再現用）
- `.gitignore`（元の300×300は除外）

含めていないもの：
- 元の300×300画像（容量・配布元の事情）
- ネストしている重複フォルダ

---

## 4. 試行錯誤の記録

ここからが本題。**何を試して、何が失敗したか、どう直したか**。

---

### 試行1: 標準的なCNNを組んで学習

#### アーキテクチャ
```
Input (3, 64, 64)
Conv(16, 3x3, pad=1) → ReLU → MaxPool(2x2)
Conv(32, 3x3, pad=1) → ReLU → MaxPool(2x2)
Conv(64, 3x3, pad=1) → ReLU → MaxPool(2x2)
Affine(128) → ReLU → Dropout(0.5)
Affine(3) → Softmax → CategoricalCrossEntropy
```

設定：Adam (lr=0.001), Batch=64, Epoch=30, Image Normalization 1/255。

#### 結果: **過学習**

- Training Error: 順調に減少
- Validation Error: ほぼ下がらず、頭打ち

このデータセットは train/test 間に分布のズレがあるため、
**何もしないと典型的な過学習になる**ことが分かった。

---

### 試行2: 標準的な学習設定を調査

複数の参考実装（TensorFlow Datasets, Laurence Moroney原典, Medium記事）を調査。
共通していた標準設定：

| 項目 | 標準値 |
|---|---|
| 画像サイズ | **150×150 RGB** |
| アーキテクチャ | Conv 4ブロック (64→64→128→128) + Dense512 + Dense3 |
| Optimizer | Adam (lr=0.001) |
| Batch Size | 32〜128 |
| Epochs | 25〜32 |
| Augmentation | rotation 40°, shift 0.2, zoom 0.2, shear 0.2, horizontal flip |

期待精度: train/val ともに95〜99%。

#### 学んだこと

1. **画像サイズ 64×64 は標準より小さく、情報量不足の可能性**
2. **データ拡張なしではこのデータセットは汎化しない**（Medium記事でも明示）
3. ImageAugmentation層を入れることが事実上必須

---

### 試行3: Spresenseへ書き込もうとして DNN load error

学習がそこそこ動いた段階で、推論をSpresenseに移植しようとした。

#### 推論スケッチの構成（`spresense_inference/spresense_inference.ino`）

- SDカードから `model.nnb` をロード
- カメラ QVGA (320×240) をキャプチャ
- 中央240×240を切り出して64×64にリサイズ
- YUV422 → RGB565 → CHWレイアウトのfloat配列に変換（0〜1正規化）
- `dnnrt.forward()` で推論
- シリアルに結果出力

#### 起きた問題: `DNN load error`

`dnnrt.begin()` が失敗。最初は層の互換性やSDカードを疑ったが、
**根本原因はメモリ不足**だった。

---

### 試行4: 原因はNNBサイズ過大

エクスポートされた `model.nnb` は **約2000KB（2MB）**。
一方 Spresense本体のRAMは **1.5MB（1536KB）** しかない。
NNB本体に加えて推論時の作業メモリも必要なので、
**NNBサイズは300〜500KB以下**に収めるのが現実的目標。

#### なぜ2MBになったか

Conv部の重みは大したことないが、
**Flatten後のAffine層が巨大だった**：

```
最終Conv出力: 64ch × 8 × 8 = 4096
Affine(4096 → 128) = 524,288 重み × 4byte ≈ 2MB
```

このAffine層だけで2MB。これが容量の主犯。

---

### 試行5: 量子化で2MBのまま動かせるか検討

NNCは Float16 / Fixed16 / Fixed8 で書き出せる。

| 形式 | 期待サイズ | Spresense対応 |
|---|---|---|
| Float32（既定） | 2000KB | ◯（だが入らない） |
| Float16 | 約1000KB | ◯ |
| Fixed8 | 約500KB | △ |

最初はFloat16で試す案を考えたが、本質的な解決にならない（モデル自体が肥大化していて精度・容量バランスが悪い）ため、**構造そのものを再設計する方針**に切り替えた。

---

### 最終解: モデル構造の見直し

容量削減・過学習防止・精度維持を**同時に**達成する戦略：

#### 鍵は「Global Average Pooling (GAP)」

Flatten + Affine(4096→128) を、**GAP + Affine(64→3)** に置き換える。

| 効果 | 内容 |
|---|---|
| 容量削減 | Affineが約2MB → 約1KBに激減 |
| 過学習抑制 | GAPは強い正則化として働く |
| 精度維持 | 画像分類ではGAPで精度がほぼ落ちない |

#### あわせて入れた対策

- 入力を 32×32 に縮小（Convの計算量・特徴マップを削減）
- 各Convの後に **BatchNormalization** を追加（学習安定 + 正則化）
- **ImageAugmentation層** を入れる（このデータセットの汎化には必須）
- Dropout は 0.5 → 0.3 に緩める（小型モデルで強すぎるとアンダーフィット）
- Optimizer に **Weight Decay 0.0001** を追加

---

## 5. 最終アーキテクチャ

### ネットワーク

```
Input (3, 32, 32)
ImageAugmentation                                 ← 学習時のみ
Convolution(16, 3x3, pad=1)
BatchNormalization
ReLU
MaxPooling(2x2)                                   → 16, 16, 16
Convolution(32, 3x3, pad=1)
BatchNormalization
ReLU
MaxPooling(2x2)                                   → 32,  8,  8
Convolution(64, 3x3, pad=1)
BatchNormalization
ReLU
MaxPooling(2x2)                                   → 64,  4,  4
GlobalAveragePooling                              → 64           ★ Affine地獄を回避
Dropout(0.3)
Affine(3) → Softmax
CategoricalCrossEntropy
```

期待NNBサイズ: **100〜150KB**
期待精度: **Validation 95%前後**

### Augmentation 設定

| 項目 | 値 |
|---|---|
| MinScale / MaxScale | 0.8 / 1.2 |
| Angle | 0.26（約±15°） |
| AspectRatio | 1.3 |
| FlipLR | True |
| Brightness | 0.2 |
| Contrast | 1.2 |
| Noise | 0.05 |

### 学習設定

| 項目 | 値 |
|---|---|
| Optimizer | Adam (lr=0.001) |
| Weight Decay | 0.0001 |
| Batch Size | 64 |
| Max Epoch | 40 |
| Image Normalization | 1/255（DATASETタブで有効化） |

---

## 6. Spresenseへのデプロイ

### 手順

1. NNCで学習完了後、結果を右クリック → **Export → NNB**
2. データ型は **Float32**（容量を100〜200KB台に抑えられたら不要に量子化しなくてOK）
3. 生成された `.nnb` を SDカード ルートに配置
4. Arduino IDE で `spresense_inference/spresense_inference.ino` を開く
5. **ツール → Memory → 1536KB**（最大）に設定
6. 書き込み → シリアルモニタ 115200 baud

### スケッチの主要パラメータ

`spresense_inference.ino` で変更が必要：

```cpp
#define DNN_IMG_W   32      // 32x32モデルに合わせる
#define DNN_IMG_H   32
```

クラスラベルは `train.csv` と同順にする：
```cpp
static const char* kLabels[] = {"paper", "rock", "scissors"};
```

### 出力例

```
pred=rock  scores: paper=0.012 rock=0.973 scissors=0.015
```

---

## 7. 学んだこと（Lessons Learned）

### このデータセット固有の知見

1. **train/test の分布シフトが大きい**ので、Augmentationなしでは絶対に汎化しない
2. **CGIベースの画像**なので実カメラ画像への直接の汎化は限定的
3. **背景の単色（緑など）に強く依存**してしまう

### Spresenseに載せるための知見

1. **NNBサイズの主犯は Flatten 後の Affine**。CNN本体ではない
2. **GAP（Global Average Pooling）は組み込み機械学習の必須テクニック**
3. メモリ設定は Arduino IDE の **ツール → Memory** で 1536KB（最大）に
4. 入力サイズを下げると劇的に容量が減る。32×32 でも3クラス分類なら十分

### NNC運用の知見

1. CSVヘッダの `x:image, y:label` の **`x` / `y`** が、Input層の `Dataset` と
   CategoricalCrossEntropy層の `T.Dataset` に対応する
2. **Image Normalization 1/255** をDATASETタブで必ず有効化
3. Run Time 用ネットワークを別途用意するとエクスポート時のトラブルが減る
4. 学習結果を改善するときは「Augmentation → BatchNorm → Dropout → 構造変更」の順に試すと効率的

### 一般的な深層学習の失敗パターン

| 症状 | 真の原因 | 対策 |
|---|---|---|
| Validation Error が下がらない | 過学習・分布シフト | Augmentation, BN, GAP |
| 推論器に載らない | Affineが巨大 | GAPで置換、入力縮小 |
| 量子化で逃げたくなる | モデル設計が悪い | まず構造を見直すべき |
| メモリが足りない | Arduino IDE設定が不十分 | Memory設定を最大に |

---

## 8. ディレクトリ構成

```
Rock-Paper-Scissors/
├── README.md                       ← この記録
├── make_nnc_dataset.py             ← データセット変換スクリプト
├── .gitignore
│
├── train/                          ← 元の300×300画像（git管理外）
├── test/
├── validation/
├── Rock-Paper-Scissors/            ← 重複フォルダ（git管理外）
│
├── nnc_dataset/                    ← 64×64 RGB（汎用学習用）
│   ├── train/{paper,rock,scissors}/*.png
│   ├── test/{paper,rock,scissors}/*.png
│   ├── train.csv
│   └── test.csv
│
├── nnc_dataset_32/                 ← 32×32 RGB（Spresense向け）
│   └── （上と同構成）
│
└── spresense_inference/
    └── spresense_inference.ino     ← Spresense推論スケッチ
```

---

## 9. 参考資料

- [TensorFlow Datasets — rock_paper_scissors](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors)
- [Laurence Moroney datasets](https://laurencemoroney.com/datasets.html)
- [Rock-Paper-Scissors Image Classification with Keras (Medium)](https://medium.com/@sdwiulfah/rock-paper-scissors-image-classification-with-keras-tensorflow-1c29ba0fe14d)
- [trekhleb/machine-learning-experiments — rock_paper_scissors_cnn](https://github.com/trekhleb/machine-learning-experiments/blob/master/experiments/rock_paper_scissors_cnn/rock_paper_scissors_cnn.ipynb)
- [Sony Neural Network Console](https://dl.sony.com/)
- [Spresense Arduino Reference](https://developer.sony.com/spresense/)
