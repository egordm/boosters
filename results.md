# Full Benchmark Report

Generated: 2025-12-27T12:17:54.635520

## Environment

| Property | Value |
|----------|-------|
| Machine | arm |
| Cores | 10 |
| Memory | 32.0 GB |
| OS | Darwin 25.1.0 |
| Git SHA | 2ec8335 |

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
| gbdt         | boosters  | 0.4677±0.0057 | **0.1214±0.0035** |
| gbdt         | lightgbm  | 0.4718±0.0068 | 0.1772±0.0026     |
| gbdt         | xgboost   | 0.4721±0.0067 | 0.1650±0.0010     |
| gblinear     | boosters  | 0.8495±0.0263 | **0.0195±0.0001** |
| gblinear     | xgboost   | 0.8507±0.0278 | 0.0585±0.0003     |
| linear_trees | boosters  | 0.4747±0.0139 | **0.1825±0.0013** |
| linear_trees | lightgbm  | 1.0324±0.6164 | 0.2779±0.0029     |

**synthetic_reg_medium** (primary: rmse)

| Booster      | Library   | rmse               | train_time_s      |
|:-------------|:----------|:-------------------|:------------------|
| gbdt         | boosters  | **33.5357±0.6317** | **0.7792±0.0044** |
| gbdt         | lightgbm  | 39.7303±0.5938     | 1.0453±0.0052     |
| gbdt         | xgboost   | 39.7124±0.6946     | 2.1929±0.0947     |
| gblinear     | boosters  | 0.1021±0.0006      | **0.0353±0.0008** |
| gblinear     | xgboost   | 0.1019±0.0006      | 0.2939±0.0015     |
| linear_trees | boosters  | 108.2527±1.8559    | **0.8926±0.1144** |
| linear_trees | lightgbm  | **18.0266±0.3594** | 1.2442±0.1244     |

**synthetic_reg_small** (primary: rmse)

| Booster      | Library   | rmse               | train_time_s      |
|:-------------|:----------|:-------------------|:------------------|
| gbdt         | boosters  | **39.3658±2.7246** | **0.2299±0.0007** |
| gbdt         | lightgbm  | 46.8610±2.7768     | 0.3573±0.0034     |
| gbdt         | xgboost   | 47.2707±2.6052     | 0.7963±0.0049     |
| gblinear     | boosters  | 0.1038±0.0019      | **0.0084±0.0008** |
| gblinear     | xgboost   | 0.1033±0.0021      | 0.0300±0.0002     |
| linear_trees | boosters  | 42.3795±0.9839     | **0.2411±0.0003** |
| linear_trees | lightgbm  | **30.8853±1.0862** | 0.4071±0.0134     |

### Binary Classification

**breast_cancer** (primary: logloss)

| Booster      | Library   | logloss           | train_time_s      |
|:-------------|:----------|:------------------|:------------------|
| gbdt         | boosters  | 0.0980±0.0343     | 0.0405±0.0011     |
| gbdt         | lightgbm  | 0.0866±0.0270     | **0.0261±0.0004** |
| gbdt         | xgboost   | 0.0978±0.0427     | 0.0581±0.0013     |
| gblinear     | boosters  | 1.0628±0.5817     | **0.0061±0.0003** |
| gblinear     | xgboost   | **0.1252±0.0397** | 0.0066±0.0001     |
| linear_trees | boosters  | 0.0985±0.0326     | 0.0415±0.0012     |
| linear_trees | lightgbm  | 0.0911±0.0207     | **0.0302±0.0003** |

**synthetic_bin_medium** (primary: logloss)

| Booster      | Library   | logloss       | train_time_s      |
|:-------------|:----------|:--------------|:------------------|
| gbdt         | boosters  | 0.1716±0.0114 | **0.7215±0.0044** |
| gbdt         | lightgbm  | 0.1745±0.0125 | 0.7543±0.0122     |
| gbdt         | xgboost   | 0.1745±0.0130 | 1.2954±0.0323     |
| gblinear     | boosters  | 0.5063±0.0040 | **0.0396±0.0009** |
| gblinear     | xgboost   | 0.5063±0.0040 | 0.2972±0.0033     |
| linear_trees | boosters  | 0.1743±0.0122 | **0.7811±0.0044** |
| linear_trees | lightgbm  | 0.1781±0.0121 | 0.8195±0.0073     |

**synthetic_bin_small** (primary: logloss)

| Booster      | Library   | logloss       | train_time_s      |
|:-------------|:----------|:--------------|:------------------|
| gbdt         | boosters  | 0.2984±0.0502 | 0.1965±0.0026     |
| gbdt         | lightgbm  | 0.2768±0.0405 | 0.1982±0.0065     |
| gbdt         | xgboost   | 0.2722±0.0349 | 0.3517±0.0095     |
| gblinear     | boosters  | 0.6017±0.0258 | **0.0091±0.0003** |
| gblinear     | xgboost   | 0.6017±0.0258 | 0.0306±0.0001     |
| linear_trees | boosters  | 0.3023±0.0489 | 0.2623±0.1236     |
| linear_trees | lightgbm  | 0.2850±0.0520 | 0.2232±0.0258     |

### Multiclass Classification

**covertype** (primary: mlogloss)

| Booster      | Library   | mlogloss          | train_time_s      |
|:-------------|:----------|:------------------|:------------------|
| gbdt         | boosters  | **0.3815±0.0053** | 13.0296±0.4501    |
| gbdt         | lightgbm  | 0.4285±0.0074     | **2.4977±0.1030** |
| gbdt         | xgboost   | 0.4698±0.0064     | 3.0219±0.2782     |
| gblinear     | boosters  | 0.7539±0.0122     | **2.3276±0.0525** |
| gblinear     | xgboost   | 0.7535±0.0084     | 6.0968±0.0872     |
| linear_trees | boosters  | **0.3739±0.0072** | 13.7795±0.1187    |
| linear_trees | lightgbm  | 0.4053±0.0091     | **3.6310±0.1004** |

**iris** (primary: mlogloss)

| Booster      | Library   | mlogloss          | train_time_s      |
|:-------------|:----------|:------------------|:------------------|
| gbdt         | boosters  | 0.1744±0.1042     | **0.0017±0.0001** |
| gbdt         | lightgbm  | 0.1423±0.0872     | 0.0108±0.0006     |
| gbdt         | xgboost   | 0.1846±0.1158     | 0.0104±0.0005     |
| gblinear     | boosters  | **0.3162±0.0379** | 0.0097±0.0003     |
| gblinear     | xgboost   | 0.4261±0.0304     | **0.0026±0.0001** |
| linear_trees | boosters  | 0.1700±0.1060     | **0.0020±0.0001** |
| linear_trees | lightgbm  | 0.1063±0.0652     | 0.0168±0.0011     |

**synthetic_multi_small** (primary: mlogloss)

| Booster      | Library   | mlogloss      | train_time_s      |
|:-------------|:----------|:--------------|:------------------|
| gbdt         | boosters  | 0.6916±0.1175 | **0.6518±0.0054** |
| gbdt         | lightgbm  | 0.6668±0.1138 | 0.9008±0.0161     |
| gbdt         | xgboost   | 0.6654±0.1010 | 2.2801±0.1196     |
| gblinear     | boosters  | 1.0560±0.0461 | **0.1021±0.0010** |
| gblinear     | xgboost   | 1.0546±0.0451 | 0.1453±0.0002     |
| linear_trees | boosters  | 0.6771±0.1295 | **0.6889±0.0058** |
| linear_trees | lightgbm  | 0.6876±0.1082 | 0.9037±0.0100     |

## Reproducing

```bash
boosters-eval full
```

---

*Best values per metric are **bolded**. Lower is better for loss/time metrics.*