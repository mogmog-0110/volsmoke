# volsmoke — 3D ボリュメトリック煙＆炎の GPU シミュレーション

3D グリッド上で流体方程式 (Navier-Stokes) を GPU で解き、密度・温度場を
**ボリュームレイマーチ**で立体描画する DirectX 12 のリアルタイムデモ。
炎・煙の見た目をスライダーでその場で作り変えられる。

自作ゲームエンジン [MitiruEngine](https://github.com/mogmog-0110/MitiruEngine) の
DX12 コンピュート基盤 (`Dx12ComputeContext`) の上に実装している。

![presets](media/presets.png)

![montage](media/montage.gif)

> 注: 流体ソルバ自体は古典的な手法 (Stam 1999 / Fedkiw 2001-2002, GPU 化は GPU Gems 3, 2007)。
> 本リポジトリはそれを自作 DX12 エンジン上にフルスクラッチ実装したもの。

## 仕組み

毎フレーム、128³ のグリッドに対してコンピュートシェーダで以下を順に解く:

| 段 | 内容 |
|---|---|
| 注入 (inject) | 火源に温度・密度・上向き速度を与える |
| 移流 (advect) | 速度場で速度・密度・温度を semi-Lagrangian に運ぶ |
| 浮力 (buoyancy) | 温度に比例した上昇力を加える |
| 渦 (vorticity confinement) | 数値拡散で失われた小スケールの渦を復元し乱流を取り戻す |
| 圧力 (pressure projection) | 発散を計算 → Jacobi 法 40 反復 → 勾配を引いて非圧縮化 |
| 描画 (raymarch) | カメラから光線を飛ばし、温度→黒体放射色で発光、密度を吸収 + 自己影 |

- 温度を**黒体放射ふうの色**(暗赤→赤→橙→黄→白)に写像して炎を発光させる。
- 煙は**光方向へ 2 段目のレイマーチ**を打って自己影を付け、立体感を出す。
- 全フィールドは `RWTexture3D` (UAV) で読み書きし、パス間は UAV バリアで同期する。

## ビルド

Windows + Visual Studio 2022 (C++20)。

```bash
cmake -B build -G Ninja -DMITIRU_ROOT=E:/user/MitiruEngine
cmake --build build
```

`MITIRU_ROOT` は MitiruEngine のチェックアウト先 (ヘッダと vendored ImGui を参照)。

## 実行

```bash
# ライブ調整 (窓 + スライダー + プリセット切替)
volsmoke --interactive

# プリセット指定 / 窓サイズ指定
volsmoke --interactive --preset blue --width 1600 --height 900

# 複数を横並びで同時比較 (ライブ)
volsmoke --interactive --compare fire,blue,smoke --width 1500 --height 720

# ヘッドレスで静止画 1 枚 (N フレーム回した最終フレーム)
volsmoke --out fire.png --frames 130 --preset fire

# ヘッドレスで横並び比較を 1 枚
volsmoke --compare fire,blue,smoke --out compare.png --frames 140 --width 1500

# 連番ダンプ (アニメーション素材)
volsmoke --seq frames --frames 175 --seqfrom 25 --stride 2 --preset fire
```

プリセット: `fire` / `blue` (青いガス炎) / `torch` (松明) / `smoke` / `ink` (色付きの煙)。
`--compare a,b,c` は各プリセットを独立にシミュレートして横タイルに並べる。既定の窓サイズは 1280×720。

## ライセンス / 帰属

Copyright (c) 2026 川村優弥 (Shiggy)。MitiruEngine の一部として開発。
