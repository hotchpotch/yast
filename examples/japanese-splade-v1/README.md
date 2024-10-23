

# 日本語 SPLADE モデルの学習方法

yast のルートディレクトリで実行します。

```
poetry install
```

## toy データセットの学習

サンプルデータセットの学習です。データセットが小さすぎて、きちんとしたモデルは作れませんが、データのサンプルとして。

```
poetry run python -m yast.run ./examples/japanese-splade-v1/toy.yaml
```

## japanese-splade-base-v1 の学習

```
poetry run python -m yast.run ./examples/japanese-splade-v1/japanese-splade-base-v1.yaml
```

## japanese-splade-base-v1-mmarco-only の学習

```
poetry run python -m yast.run ./examples/japanese-splade-v1/japanese-splade-base-v1-mmarco-only.yaml
```

## japanese-splade-base-v1 と toy データセットの学習

データセットを作った場合、データセットを混ぜての学習が可能です。

```
poetry run python -m yast.run ./examples/japanese-splade-v1/japanese-splade-base-v1-with-toy.yaml
```

