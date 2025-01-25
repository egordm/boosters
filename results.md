# Full Benchmark Report

Generated: 2025-12-27T12:52:24.950150

## Environment

| Property | Value |
|----------|-------|
| Machine | arm |
| Cores | 10 |
| Memory | 32.0 GB |
| OS | Darwin 25.1.0 |
| Git SHA | 8a19f31 |

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
| gbdt         | boosters  | 0.4677±0.0057 | **0.0933±0.0009** |
| gbdt         | lightgbm  | 0.4718±0.0068 | 0.1401±0.0012     |
| gbdt         | xgboost   | 0.4721±0.0067 | 0.1292±0.0022     |
| gblinear     | boosters  | 0.8495±0.0263 | **0.0176±0.0005** |
| gblinear     | xgboost   | 0.8507±0.0278 | 0.0456±0.0007     |
| linear_trees | boosters  | 0.4726±0.0074 | 0.2704±0.0012     |
| linear_trees | lightgbm  | 1.0324±0.6164 | **0.2172±0.0038** |

**synthetic_reg_medium** (primary: rmse)

| Booster      | Library   | rmse               | train_time_s      |
|:-------------|:----------|:-------------------|:------------------|
| gbdt         | boosters  | **33.5357±0.6317** | **0.5976±0.0069** |
| gbdt         | lightgbm  | 39.7303±0.5938     | 0.8075±0.0036     |
| gbdt         | xgboost   | 39.7124±0.6946     | 1.6503±0.0051     |
| gblinear     | boosters  | 0.1021±0.0006      | **0.0299±0.0011** |
| gblinear     | xgboost   | 0.1019±0.0006      | 0.2276±0.0027     |
| linear_trees | boosters  | 18.7762±0.9837     | **0.7232±0.0081** |
| linear_trees | lightgbm  | 18.0266±0.3594     | 0.9075±0.0060     |

**synthetic_reg_small** (primary: rmse)

| Booster      | Library   | rmse               | train_time_s      |
|:-------------|:----------|:-------------------|:------------------|
| gbdt         | boosters  | **39.3658±2.7246** | **0.1787±0.0013** |
| gbdt         | lightgbm  | 46.8610±2.7768     | 0.3149±0.0747     |
| gbdt         | xgboost   | 47.2707±2.6052     | 0.6250±0.0053     |
| gblinear     | boosters  | 0.1038±0.0019      | **0.0081±0.0006** |
| gblinear     | xgboost   | 0.1033±0.0021      | 0.0231±0.0002     |
| linear_trees | boosters  | **28.1962±1.8004** | **0.2019±0.0022** |
| linear_trees | lightgbm  | 30.8853±1.0862     | 0.3157±0.0103     |

### Binary Classification

**breast_cancer** (primary: logloss)

| Booster      | Library   | logloss           | train_time_s      |
|:-------------|:----------|:------------------|:------------------|
| gbdt         | boosters  | 0.0980±0.0343     | 0.0315±0.0013     |
| gbdt         | lightgbm  | 0.0866±0.0270     | **0.0206±0.0003** |
| gbdt         | xgboost   | 0.0978±0.0427     | 0.0454±0.0008     |
| gblinear     | boosters  | 1.0628±0.5817     | 0.0049±0.0005     |
| gblinear     | xgboost   | **0.1252±0.0397** | 0.0050±0.0001     |
| linear_trees | boosters  | 0.0981±0.0318     | 0.0347±0.0008     |
| linear_trees | lightgbm  | 0.0911±0.0207     | **0.0238±0.0004** |

**synthetic_bin_medium** (primary: logloss)

| Booster      | Library   | logloss       | train_time_s      |
|:-------------|:----------|:--------------|:------------------|
| gbdt         | boosters  | 0.1716±0.0114 | 0.5966±0.0782     |
| gbdt         | lightgbm  | 0.1745±0.0125 | 0.5900±0.0099     |
| gbdt         | xgboost   | 0.1745±0.0130 | 1.0112±0.0109     |
| gblinear     | boosters  | 0.5063±0.0040 | **0.0338±0.0016** |
| gblinear     | xgboost   | 0.5063±0.0040 | 0.2287±0.0020     |
| linear_trees | boosters  | 0.1770±0.0136 | 0.6872±0.0056     |
| linear_trees | lightgbm  | 0.1781±0.0121 | **0.6367±0.0043** |

**synthetic_bin_small** (primary: logloss)

| Booster      | Library   | logloss       | train_time_s      |
|:-------------|:----------|:--------------|:------------------|
| gbdt         | boosters  | 0.2984±0.0502 | 0.1529±0.0022     |
| gbdt         | lightgbm  | 0.2768±0.0405 | 0.1560±0.0043     |
| gbdt         | xgboost   | 0.2722±0.0349 | 0.2756±0.0059     |
| gblinear     | boosters  | 0.6017±0.0258 | **0.0085±0.0005** |
| gblinear     | xgboost   | 0.6017±0.0258 | 0.0235±0.0003     |
| linear_trees | boosters  | 0.2982±0.0491 | 0.1666±0.0031     |
| linear_trees | lightgbm  | 0.2850±0.0520 | 0.1632±0.0053     |

### Multiclass Classification

**covertype** (primary: mlogloss)

| Booster      | Library   | mlogloss          | train_time_s      |
|:-------------|:----------|:------------------|:------------------|
| gbdt         | boosters  | **0.3815±0.0053** | 9.9084±0.0752     |
| gbdt         | lightgbm  | 0.4285±0.0074     | **1.8965±0.0160** |
| gbdt         | xgboost   | 0.4698±0.0064     | 2.3041±0.1055     |
| gblinear     | boosters  | 0.7539±0.0122     | **1.8716±0.0424** |
| gblinear     | xgboost   | 0.7535±0.0084     | 4.7794±0.0812     |
| linear_trees | boosters  | **0.3754±0.0064** | 12.0711±0.1321    |
| linear_trees | lightgbm  | 0.4053±0.0091     | **2.7450±0.0401** |

**iris** (primary: mlogloss)

| Booster      | Library   | mlogloss          | train_time_s      |
|:-------------|:----------|:------------------|:------------------|
| gbdt         | boosters  | 0.1744±0.1042     | **0.0013±0.0001** |
| gbdt         | lightgbm  | 0.1423±0.0872     | 0.0087±0.0005     |
| gbdt         | xgboost   | 0.1846±0.1158     | 0.0082±0.0003     |
| gblinear     | boosters  | **0.3162±0.0379** | 0.0087±0.0003     |
| gblinear     | xgboost   | 0.4261±0.0304     | **0.0021±0.0001** |
| linear_trees | boosters  | 0.1740±0.0974     | **0.0020±0.0002** |
| linear_trees | lightgbm  | 0.1063±0.0652     | 0.0131±0.0009     |

**synthetic_multi_small** (primary: mlogloss)

| Booster      | Library   | mlogloss      | train_time_s      |
|:-------------|:----------|:--------------|:------------------|
| gbdt         | boosters  | 0.6916±0.1175 | **0.4875±0.0040** |
| gbdt         | lightgbm  | 0.6668±0.1138 | 0.6726±0.0114     |
| gbdt         | xgboost   | 0.6654±0.1010 | 1.6491±0.0293     |
| gblinear     | boosters  | 1.0560±0.0461 | **0.0861±0.0012** |
| gblinear     | xgboost   | 1.0546±0.0451 | 0.1097±0.0004     |
| linear_trees | boosters  | 0.6726±0.1213 | **0.5306±0.0096** |
| linear_trees | lightgbm  | 0.6876±0.1082 | 0.6841±0.0139     |

## Reproducing

```bash
boosters-eval full
```

---

*Best values per metric are **bolded**. Lower is better for loss/time metrics.*