# Full Benchmark Report

Generated: 2025-12-27T17:54:15.213719

## Environment

| Property | Value |
|----------|-------|
| Machine | arm |
| Cores | 10 |
| Memory | 32.0 GB |
| OS | Darwin 25.1.0 |
| Git SHA | ef939ee |

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
| gbdt         | boosters  | 0.4677±0.0057 | **0.1201±0.0011** |
| gbdt         | lightgbm  | 0.4718±0.0068 | 0.1794±0.0045     |
| gbdt         | xgboost   | 0.4721±0.0067 | 0.1661±0.0020     |
| gblinear     | boosters  | 0.8495±0.0263 | **0.0314±0.0005** |
| gblinear     | xgboost   | 0.8507±0.0278 | 0.0597±0.0024     |
| linear_trees | boosters  | 0.4726±0.0074 | 0.3779±0.0675     |
| linear_trees | lightgbm  | 1.0324±0.6164 | **0.2788±0.0034** |

**synthetic_reg_medium** (primary: rmse)

| Booster      | Library   | rmse               | train_time_s      |
|:-------------|:----------|:-------------------|:------------------|
| gbdt         | boosters  | **33.5357±0.6317** | **0.8111±0.0821** |
| gbdt         | lightgbm  | 39.7303±0.5938     | 1.0437±0.0037     |
| gbdt         | xgboost   | 39.7124±0.6946     | 2.1240±0.0080     |
| gblinear     | boosters  | 0.1021±0.0006      | **0.0994±0.0011** |
| gblinear     | xgboost   | 0.1019±0.0006      | 0.2948±0.0009     |
| linear_trees | boosters  | 18.7762±0.9837     | **0.9244±0.0052** |
| linear_trees | lightgbm  | 18.0266±0.3594     | 1.1524±0.0038     |

**synthetic_reg_small** (primary: rmse)

| Booster      | Library   | rmse               | train_time_s      |
|:-------------|:----------|:-------------------|:------------------|
| gbdt         | boosters  | **39.3658±2.7246** | **0.2294±0.0007** |
| gbdt         | lightgbm  | 46.8610±2.7768     | 0.3584±0.0035     |
| gbdt         | xgboost   | 47.2707±2.6052     | 0.7986±0.0063     |
| gblinear     | boosters  | 0.1038±0.0019      | **0.0202±0.0004** |
| gblinear     | xgboost   | 0.1033±0.0021      | 0.0301±0.0001     |
| linear_trees | boosters  | **28.1962±1.8004** | **0.2614±0.0010** |
| linear_trees | lightgbm  | 30.8853±1.0862     | 0.4082±0.0138     |

### Binary Classification

**breast_cancer** (primary: logloss)

| Booster      | Library   | logloss           | train_time_s      |
|:-------------|:----------|:------------------|:------------------|
| gbdt         | boosters  | 0.0980±0.0343     | 0.0405±0.0012     |
| gbdt         | lightgbm  | 0.0866±0.0270     | **0.0263±0.0007** |
| gbdt         | xgboost   | 0.0978±0.0427     | 0.0584±0.0011     |
| gblinear     | boosters  | 1.0628±0.5817     | 0.0070±0.0009     |
| gblinear     | xgboost   | **0.1252±0.0397** | 0.0066±0.0001     |
| linear_trees | boosters  | 0.0981±0.0318     | 0.0446±0.0012     |
| linear_trees | lightgbm  | 0.0911±0.0207     | **0.0303±0.0004** |

**synthetic_bin_medium** (primary: logloss)

| Booster      | Library   | logloss       | train_time_s      |
|:-------------|:----------|:--------------|:------------------|
| gbdt         | boosters  | 0.1716±0.0114 | **0.7190±0.0024** |
| gbdt         | lightgbm  | 0.1745±0.0125 | 0.7493±0.0102     |
| gbdt         | xgboost   | 0.1745±0.0130 | 1.2807±0.0183     |
| gblinear     | boosters  | 0.5063±0.0040 | **0.1034±0.0012** |
| gblinear     | xgboost   | 0.5063±0.0040 | 0.2957±0.0005     |
| linear_trees | boosters  | 0.1770±0.0136 | 0.8826±0.0036     |
| linear_trees | lightgbm  | 0.1781±0.0121 | **0.8103±0.0075** |

**synthetic_bin_small** (primary: logloss)

| Booster      | Library   | logloss       | train_time_s      |
|:-------------|:----------|:--------------|:------------------|
| gbdt         | boosters  | 0.2984±0.0502 | 0.1962±0.0020     |
| gbdt         | lightgbm  | 0.2768±0.0405 | 0.1974±0.0069     |
| gbdt         | xgboost   | 0.2722±0.0349 | 0.3497±0.0095     |
| gblinear     | boosters  | 0.6017±0.0258 | **0.0211±0.0005** |
| gblinear     | xgboost   | 0.6017±0.0258 | 0.0305±0.0002     |
| linear_trees | boosters  | 0.2982±0.0491 | 0.2153±0.0027     |
| linear_trees | lightgbm  | 0.2850±0.0520 | 0.2108±0.0049     |

### Multiclass Classification

**covertype** (primary: mlogloss)

| Booster      | Library   | mlogloss          | train_time_s      |
|:-------------|:----------|:------------------|:------------------|
| gbdt         | boosters  | **0.3815±0.0053** | 12.7138±0.1147    |
| gbdt         | lightgbm  | 0.4285±0.0074     | **2.4757±0.0819** |
| gbdt         | xgboost   | 0.4698±0.0064     | 2.8685±0.0200     |
| gblinear     | boosters  | 0.7539±0.0122     | **3.4402±0.1022** |
| gblinear     | xgboost   | 0.7535±0.0084     | 6.0934±0.0643     |
| linear_trees | boosters  | **0.3754±0.0064** | 15.5947±0.0404    |
| linear_trees | lightgbm  | 0.4053±0.0091     | **3.5768±0.0580** |

**iris** (primary: mlogloss)

| Booster      | Library   | mlogloss          | train_time_s      |
|:-------------|:----------|:------------------|:------------------|
| gbdt         | boosters  | 0.1744±0.1042     | **0.0017±0.0000** |
| gbdt         | lightgbm  | 0.1423±0.0872     | 0.0111±0.0010     |
| gbdt         | xgboost   | 0.1846±0.1158     | 0.0106±0.0004     |
| gblinear     | boosters  | **0.3162±0.0379** | 0.0097±0.0003     |
| gblinear     | xgboost   | 0.4261±0.0304     | **0.0026±0.0001** |
| linear_trees | boosters  | 0.1740±0.0974     | **0.0025±0.0002** |
| linear_trees | lightgbm  | 0.1063±0.0652     | 0.0166±0.0011     |

**synthetic_multi_small** (primary: mlogloss)

| Booster      | Library   | mlogloss      | train_time_s      |
|:-------------|:----------|:--------------|:------------------|
| gbdt         | boosters  | 0.6916±0.1175 | **0.6500±0.0045** |
| gbdt         | lightgbm  | 0.6668±0.1138 | 0.9659±0.1624     |
| gbdt         | xgboost   | 0.6654±0.1010 | 2.1991±0.0348     |
| gblinear     | boosters  | 1.0560±0.0461 | **0.1404±0.0007** |
| gblinear     | xgboost   | 1.0546±0.0451 | 0.1455±0.0003     |
| linear_trees | boosters  | 0.6726±0.1213 | **0.7015±0.0024** |
| linear_trees | lightgbm  | 0.6876±0.1082 | 0.9023±0.0062     |

## Reproducing

```bash
boosters-eval full
```

---

*Best values per metric are **bolded**. Lower is better for loss/time metrics.*