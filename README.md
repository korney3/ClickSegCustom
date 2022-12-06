

## D585_ZERO Dataset inference
```python /Users/Alisa.Alenicheva/Documents/supervisely/ClickSEG/scripts/evaluate_model.py FocalClick --model_dir=./experiments/focalclick/hr18ss1/ --checkpoint=last_checkpoint --infer-size=256 --datasets=D585_ZERO --n-clicks=20 --target-iou=0.90 --thresh=0.50 --vis --cpu```
Eval results for model: last_checkpoint

|  Pipeline   |  Dataset  | NoC@80% | NoC@85% | NoC@90% |>=20@85% |>=20@90% | SPC,s |  Time   |
|-------------|-----------|---------|---------|---------|---------|---------|-------|---------|
| FocalClick  | D585_ZERO |  4.42   |  5.62   |  8.08   |   53    |   98    | 0.197 | 0:15:29 |

## Zurich dataset

|  Pipeline   |  Dataset  | NoC@80% | NoC@85% | NoC@90% |>=20@85% |>=20@90% | SPC,s |  Time   |
|-------------|-----------|---------|---------|---------|---------|---------|-------|---------|
| FocalClick  |  ZURICH   |  19.62  |  19.86  |  20.00  |   49    |   50    | 0.317 | 0:05:16 |

## Human segmentation dataset
[link](https://github.com/VikramShenoy97/Human-Segmentation-Dataset)

|  Pipeline   |  Dataset  | NoC@80% | NoC@85% | NoC@90% |>=20@85% |>=20@90% | SPC,s |  Time   |
|-------------|-----------|---------|---------|---------|---------|---------|-------|---------|
| FocalClick  |   HUMAN   |  1.46   |  1.54   |  1.78   |    0    |    0    | 0.197 | 0:01:41 |

## Crack Dataset

```python train.py models/focalclick/hrnet18s_S1_crack.py --ngpus=1 --workers=0 --batch-size=64 --exp-name=hrnet18s_S1_crack```

|  Pipeline   |  Dataset  | NoC@80% | NoC@85% | NoC@90% |>=20@85% |>=20@90% | SPC,s |  Time   |
|-------------|-----------|---------|---------|---------|---------|---------|-------|---------|
| FocalClick  |   CRACK   |  16.20  |  17.27  |  18.35  |  1386   |  1491   | 0.162 | 1:23:54 |



1. Try\Except в train - не дает дебажить
2. Нет EarlyStopping - на задаче с трещинами уже на 10 эпохе ошибка почти не падает
3. Имена датасетов - лучше было бы добваить какой-нибудь Enum class с вариантами