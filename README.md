# NYOSU

Apex Legends向けの外部AIエイムアシスト（Windows / Python + PyQt6）。
画面をキャプチャしてYOLOv8 (ONNX) で敵を検出し、マウス / 仮想ゲームパッドへ照準補正を出力します。

## 実行

```
pip install -r requirements.txt
python run.py
```

## 注意

- 検出モデル（.onnx）はこのリポジトリに含まれていません。`apex_v8s_3class.onnx` などを同じフォルダに置いてください（無いと検出できません）。
- 外部ツールの使用はゲームの利用規約に抵触する可能性があります。利用は自己責任で。
