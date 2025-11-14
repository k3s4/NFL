# NFL Big Data Bowl 2026 - 幾何学的ニューラルネットワーク アンサンブル

## 🏆 プロジェクト概要

このプロジェクトは、**NFL Big Data Bowl 2026**向けの最先端な機械学習ソリューションです。2つの強力なニューラルネットワークモデルをアンサンブルして、NFLプレイヤーの将来位置を高精度で予測します。

### 🎯 核心となる洞察

フットボール選手の動きは**幾何学的ルール**に従い、モデルは**修正項**のみを学習することで最適化されます：

- **レシーバー** → ボール着地点（幾何学的）
- **ディフェンダー** → レシーバーをミラーリング（幾何学的結合）  
- **その他** → 運動量（物理学的）
- **モデル** → カバレッジ、衝突、境界の修正のみ学習

## 🏗 アーキテクチャ

### 1. nfl_gnn.py - Geometric Neural Breakthrough
- **Spatio-Temporal Transformer**による空間-時系列モデリング
- **154の実証済み特徴量** + **13の幾何学的特徴量** = **167特徴量**
- **ResidualMLP Head**による高度な非線形変換
- **TemporalHuber損失**による時間重み付け学習

**主要コンポーネント：**
- `compute_geometric_endpoint()` - 幾何学的エンドポイント計算
- `STTransformer` - Transformer + 注意機構プーリング
- `ResidualMLPHead` - 残差ブロック付きMLP
- 10-fold Cross Validation

### 2. nfl_gru.py - Sequence Model
- **GRU + Attention**による時系列パターン認識
- 複数の特徴量グループによるマルチモーダル学習
- TTA (Test Time Augmentation) による予測安定化

### 3. アンサンブル統合
```python
pred_ensemble = (pred_gru + pred_gnn) / 2
```

## 📊 特徴量エンジニアリング

### 基本特徴量 (154次元)
- **位置・速度**: x, y, s, a, o, dir
- **身体特性**: 身長、体重、BMI
- **運動学**: velocity_x/y, acceleration_x/y, 運動量
- **ボール関係**: 距離、角度、接近速度
- **対戦相手**: 最近接距離、包囲状況
- **時系列**: ラグ特徴量、移動平均、EMA
- **GNN組み込み**: 近隣選手との相互作用

### 幾何学的特徴量 (13次元) 🎯
- **geo_endpoint_x/y**: 幾何学的終点座標
- **geo_vector_x/y**: 終点への方向ベクトル  
- **geo_distance**: 終点までの距離
- **geo_required_vx/vy**: 必要速度
- **geo_velocity_error**: 速度誤差
- **geo_required_ax/ay**: 必要加速度
- **geo_alignment**: 幾何学的経路との整合性

## 🚀 実行方法

### 学習・推論
```python
# NFLPredictor クラスが自動で以下を実行：
# 1. データロード
# 2. 特徴量エンジニアリング  
# 3. 10-fold Cross Validation
# 4. モデル学習・保存
# 5. 推論準備

predictor = NFLPredictor()
```

### Kaggle評価サーバー
```python
inference_server = kaggle_evaluation.nfl_inference_server.NFLInferenceServer(predict)
inference_server.serve()  # Competition mode
# or
inference_server.run_local_gateway(...)  # Local test
```

## 🎯 期待性能

- **目標スコア**: 0.54-0.56 Leaderboard RMSE
- **Cross Validation**: 各fold毎にRMSE計算
- **アンサンブル効果**: 2モデルの予測値平均化

## 📂 ファイル構成

```
/Users/keitosaegusa/dev/NFL/NFL/
├── ensemble-nfl-big-data-bowl-2026-v1.ipynb  # メインノートブック
├── nfl_gnn.py           # 幾何学的ニューラルネットワーク
├── nfl_gru.py           # シーケンスモデル  
├── src/                 # nfl_gru用モジュール群
└── CLAUDE.md           # この説明書
```

## 🔧 主要パラメータ

### nfl_gnn.py 設定
```python
N_FOLDS = 10           # Cross Validation数
BATCH_SIZE = 256       # バッチサイズ
EPOCHS = 200           # 最大エポック数
PATIENCE = 30          # Early Stopping
LEARNING_RATE = 1e-3   # 学習率
WINDOW_SIZE = 10       # 入力フレーム数
HIDDEN_DIM = 128       # 隠れ層次元
N_HEADS = 4           # Attention Head数
N_LAYERS = 2          # Transformer層数
MLP_HIDDEN_DIM = 256  # MLP隠れ次元
```

## 🏈 NFL固有の考慮事項

1. **フィールド境界**: 0-120ヤード（x軸）、0-53.3ヤード（y軸）
2. **プレイ方向**: left/rightでフィールド座標を統一
3. **ポジション役割**: Targeted Receiver, Defensive Coverage, Passer等
4. **時間制約**: 最大94フレーム（9.4秒）の予測

## 🔍 重要な実装詳細

- **幾何学的ベースライン**: 学習前に決定論的な位置計算
- **Temporal Attention**: 時間減衰による重み付け
- **Group K-Fold**: play_idベースの適切なvalidation分割
- **Gradient Clipping**: 勾配爆発防止
- **Device Auto-Detection**: CUDA/CPU自動選択

このソリューションは、従来のブラックボックス手法ではなく、フットボール戦術の物理的制約を活用した**解釈可能なAI**アプローチを採用しています。