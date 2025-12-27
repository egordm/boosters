# Full Benchmark Report

Generated: 2025-12-27T06:04:37.637546

## Environment

| Property | Value |
|----------|-------|
| Machine | arm |
| Cores | 10 |
| Memory | 32.0 GB |
| OS | Darwin 25.1.0 |
| Git SHA | e542228 |

## Library Versions

| Library | Version |
|---------|---------|
| Python | 3.12.9 |
| boosters | 0.1.0 |
| xgboost | 3.1.2 |
| lightgbm | 4.6.0 |
| numpy | 2.3.5 |

## Configuration

- **Suite**: full
- **Seeds**: 5
- **n_estimators**: 100
- **learning_rate**: 0.1
- **max_depth**: 6
- **growth_strategy**: leafwise
- **max_bins**: 256
- **min_samples_leaf**: 1
- **reg_lambda (L2)**: 0.0
- **reg_alpha (L1)**: 0.0
- **linear_l2**: 0.01
- **booster_types**: gbdt, gblinear, linear_trees

## Results

### Regression

**california** (primary: rmse)

| Booster      | Library   | rmse          | train_time_s      |
|:-------------|:----------|:--------------|:------------------|
| gbdt         | boosters  | 0.4677±0.0057 | 0.0923±0.0004     |
| gbdt         | lightgbm  | 0.4718±0.0068 | 0.1353±0.0012     |
| gbdt         | xgboost   | 0.4721±0.0067 | 0.1275±0.0014     |
| gblinear     | boosters  | 0.8495±0.0263 | 0.0175±0.0009     |
| gblinear     | xgboost   | 0.8507±0.0278 | 0.0454±0.0010     |
| linear_trees | boosters  | 0.4747±0.0139 | **0.1384±0.0006** |
| linear_trees | lightgbm  | 1.0324±0.6164 | 0.2121±0.0020     |

**synthetic_reg_medium** (primary: rmse)

| Booster      | Library   | rmse            | train_time_s      |
|:-------------|:----------|:----------------|:------------------|
| gbdt         | boosters  | 33.5357±0.6317  | **0.5935±0.0041** |
| gbdt         | lightgbm  | 39.7303±0.5938  | 0.8648±0.0892     |
| gbdt         | xgboost   | 39.7124±0.6946  | 1.6421±0.0094     |
| gblinear     | boosters  | 0.1021±0.0006   | 0.0302±0.0008     |
| gblinear     | xgboost   | 0.1019±0.0006   | 0.2260±0.0011     |
| linear_trees | boosters  | 108.2527±1.8559 | **0.6388±0.0008** |
| linear_trees | lightgbm  | 18.0266±0.3594  | 0.8831±0.0028     |

**synthetic_reg_small** (primary: rmse)

| Booster      | Library   | rmse           | train_time_s      |
|:-------------|:----------|:---------------|:------------------|
| gbdt         | boosters  | 39.3658±2.7246 | **0.1761±0.0003** |
| gbdt         | lightgbm  | 46.8610±2.7768 | 0.2742±0.0031     |
| gbdt         | xgboost   | 47.2707±2.6052 | 0.6128±0.0031     |
| gblinear     | boosters  | 0.1038±0.0019  | 0.0084±0.0005     |
| gblinear     | xgboost   | 0.1033±0.0021  | 0.0230±0.0002     |
| linear_trees | boosters  | 42.3795±0.9839 | **0.1853±0.0023** |
| linear_trees | lightgbm  | 30.8853±1.0862 | 0.3137±0.0105     |

### Binary Classification

**breast_cancer** (primary: logloss)

| Booster      | Library   | logloss           | train_time_s   |
|:-------------|:----------|:------------------|:---------------|
| gbdt         | boosters  | 0.0980±0.0343     | 0.0311±0.0010  |
| gbdt         | lightgbm  | 0.0866±0.0270     | 0.0198±0.0003  |
| gbdt         | xgboost   | 0.0978±0.0427     | 0.0453±0.0012  |
| gblinear     | boosters  | 1.0628±0.5817     | 0.0052±0.0005  |
| gblinear     | xgboost   | 0.1252±0.0397     | 0.0050±0.0001  |
| linear_trees | boosters  | 0.0985±0.0326     | 0.0316±0.0012  |
| linear_trees | lightgbm  | **0.0911±0.0207** | 0.0232±0.0005  |

**synthetic_bin_medium** (primary: logloss)

| Booster      | Library   | logloss           | train_time_s      |
|:-------------|:----------|:------------------|:------------------|
| gbdt         | boosters  | 0.1716±0.0114     | **0.5492±0.0013** |
| gbdt         | lightgbm  | 0.1745±0.0125     | 0.5760±0.0075     |
| gbdt         | xgboost   | 0.1745±0.0130     | 0.9871±0.0124     |
| gblinear     | boosters  | 0.5063±0.0040     | 0.0320±0.0007     |
| gblinear     | xgboost   | 0.5063±0.0040     | 0.2249±0.0010     |
| linear_trees | boosters  | **0.1743±0.0122** | 0.6391±0.1079     |
| linear_trees | lightgbm  | 0.1781±0.0121     | **0.6266±0.0075** |

**synthetic_bin_small** (primary: logloss)

| Booster      | Library   | logloss           | train_time_s      |
|:-------------|:----------|:------------------|:------------------|
| gbdt         | boosters  | 0.2984±0.0502     | **0.1501±0.0016** |
| gbdt         | lightgbm  | 0.2768±0.0405     | 0.1518±0.0058     |
| gbdt         | xgboost   | **0.2722±0.0349** | 0.2683±0.0081     |
| gblinear     | boosters  | 0.6017±0.0258     | 0.0085±0.0005     |
| gblinear     | xgboost   | 0.6017±0.0258     | 0.0237±0.0002     |
| linear_trees | boosters  | 0.3023±0.0489     | **0.1563±0.0017** |
| linear_trees | lightgbm  | **0.2850±0.0520** | 0.1622±0.0040     |

### Multiclass Classification

**covertype** (primary: mlogloss)

| Booster      | Library   | mlogloss      | train_time_s      |
|:-------------|:----------|:--------------|:------------------|
| gbdt         | boosters  | 0.3815±0.0053 | 9.7193±0.0789     |
| gbdt         | lightgbm  | 0.4285±0.0074 | **1.9054±0.0897** |
| gbdt         | xgboost   | 0.4698±0.0064 | 2.1948±0.0246     |
| gblinear     | boosters  | 0.7539±0.0122 | **1.8790±0.0328** |
| gblinear     | xgboost   | 0.7535±0.0084 | 4.6472±0.0241     |
| linear_trees | boosters  | 0.3739±0.0072 | 10.5046±0.1037    |
| linear_trees | lightgbm  | 0.4053±0.0091 | **2.7250±0.0725** |

**iris** (primary: mlogloss)

| Booster      | Library   | mlogloss          | train_time_s      |
|:-------------|:----------|:------------------|:------------------|
| gbdt         | boosters  | 0.1744±0.1042     | 0.0013±0.0001     |
| gbdt         | lightgbm  | **0.1423±0.0872** | 0.0086±0.0003     |
| gbdt         | xgboost   | 0.1846±0.1158     | 0.0081±0.0007     |
| gblinear     | boosters  | 0.3162±0.0379     | 0.0087±0.0002     |
| gblinear     | xgboost   | 0.4261±0.0304     | 0.0021±0.0002     |
| linear_trees | boosters  | 0.1700±0.1060     | **0.0016±0.0001** |
| linear_trees | lightgbm  | **0.1063±0.0652** | 0.0129±0.0009     |

**synthetic_multi_small** (primary: mlogloss)

| Booster      | Library   | mlogloss          | train_time_s      |
|:-------------|:----------|:------------------|:------------------|
| gbdt         | boosters  | 0.6916±0.1175     | **0.4955±0.0052** |
| gbdt         | lightgbm  | 0.6668±0.1138     | 0.7189±0.0710     |
| gbdt         | xgboost   | **0.6654±0.1010** | 1.6788±0.0240     |
| gblinear     | boosters  | 1.0560±0.0461     | 0.0852±0.0009     |
| gblinear     | xgboost   | 1.0546±0.0451     | 0.1106±0.0004     |
| linear_trees | boosters  | **0.6771±0.1295** | **0.5220±0.0050** |
| linear_trees | lightgbm  | 0.6876±0.1082     | 0.6909±0.0082     |

## Reproducing

```bash
boosters-eval full
```

---

*Best values per metric are **bolded**. Lower is better for loss/time metrics.*