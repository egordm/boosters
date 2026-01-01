# Full Benchmark Report

Generated: 2026-01-01T14:40:10.049043

## Environment

| Property | Value |
|----------|-------|
| Machine | arm |
| Cores | 10 |
| Memory | 32.0 GB |
| OS | Darwin 25.1.0 |
| Git SHA | 55e032d |

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
| gbdt         | boosters  | 0.4699±0.0057 | **0.1137±0.0009** |
| gbdt         | lightgbm  | 0.4718±0.0068 | 0.1768±0.0032     |
| gbdt         | xgboost   | 0.4721±0.0067 | 0.1637±0.0011     |
| gblinear     | boosters  | 0.8507±0.0278 | **0.0269±0.0000** |
| gblinear     | xgboost   | 0.8507±0.0278 | 0.0592±0.0019     |
| linear_trees | boosters  | 0.5242±0.0769 | **0.2151±0.0024** |
| linear_trees | lightgbm  | 1.0324±0.6164 | 0.2758±0.0025     |

**synthetic_reg_medium** (primary: rmse)

| Booster      | Library   | rmse               | train_time_s      |
|:-------------|:----------|:-------------------|:------------------|
| gbdt         | boosters  | **33.2517±0.7987** | **0.7483±0.0073** |
| gbdt         | lightgbm  | 39.7303±0.5938     | 1.0605±0.0394     |
| gbdt         | xgboost   | 39.7124±0.6946     | 2.1194±0.0097     |
| gblinear     | boosters  | 0.1019±0.0006      | **0.1350±0.0006** |
| gblinear     | xgboost   | 0.1019±0.0006      | 0.2963±0.0025     |
| linear_trees | boosters  | 17.6713±1.1195     | **0.8670±0.0072** |
| linear_trees | lightgbm  | 18.0266±0.3594     | 1.1481±0.0026     |

**synthetic_reg_small** (primary: rmse)

| Booster      | Library   | rmse               | train_time_s      |
|:-------------|:----------|:-------------------|:------------------|
| gbdt         | boosters  | **38.2871±3.1588** | **0.2293±0.0015** |
| gbdt         | lightgbm  | 46.8610±2.7768     | 0.3576±0.0021     |
| gbdt         | xgboost   | 47.2707±2.6052     | 0.7963±0.0063     |
| gblinear     | boosters  | 0.1033±0.0021      | **0.0139±0.0001** |
| gblinear     | xgboost   | 0.1033±0.0021      | 0.0302±0.0001     |
| linear_trees | boosters  | **28.0423±1.1177** | **0.2477±0.0012** |
| linear_trees | lightgbm  | 30.8853±1.0862     | 0.4063±0.0108     |

### Binary Classification

**breast_cancer** (primary: logloss)

| Booster      | Library   | logloss       | train_time_s      |
|:-------------|:----------|:--------------|:------------------|
| gbdt         | boosters  | 0.0983±0.0307 | 0.0410±0.0010     |
| gbdt         | lightgbm  | 0.0866±0.0270 | **0.0258±0.0003** |
| gbdt         | xgboost   | 0.0978±0.0427 | 0.0580±0.0011     |
| gblinear     | boosters  | 0.1252±0.0397 | **0.0026±0.0000** |
| gblinear     | xgboost   | 0.1252±0.0397 | 0.0067±0.0002     |
| linear_trees | boosters  | 0.0948±0.0319 | 0.0441±0.0009     |
| linear_trees | lightgbm  | 0.0911±0.0207 | **0.0306±0.0006** |

**synthetic_bin_medium** (primary: logloss)

| Booster      | Library   | logloss       | train_time_s      |
|:-------------|:----------|:--------------|:------------------|
| gbdt         | boosters  | 0.1736±0.0107 | **0.6865±0.0053** |
| gbdt         | lightgbm  | 0.1745±0.0125 | 0.7498±0.0117     |
| gbdt         | xgboost   | 0.1745±0.0130 | 1.3201±0.1142     |
| gblinear     | boosters  | 0.5063±0.0040 | **0.1369±0.0003** |
| gblinear     | xgboost   | 0.5063±0.0040 | 0.2959±0.0017     |
| linear_trees | boosters  | 0.1787±0.0145 | 0.8176±0.0034     |
| linear_trees | lightgbm  | 0.1781±0.0121 | **0.8081±0.0058** |

**synthetic_bin_small** (primary: logloss)

| Booster      | Library   | logloss       | train_time_s      |
|:-------------|:----------|:--------------|:------------------|
| gbdt         | boosters  | 0.2910±0.0456 | 0.1941±0.0025     |
| gbdt         | lightgbm  | 0.2768±0.0405 | 0.1965±0.0075     |
| gbdt         | xgboost   | 0.2722±0.0349 | 0.3469±0.0096     |
| gblinear     | boosters  | 0.6017±0.0258 | **0.0143±0.0001** |
| gblinear     | xgboost   | 0.6017±0.0258 | 0.0307±0.0002     |
| linear_trees | boosters  | 0.2926±0.0461 | **0.2026±0.0025** |
| linear_trees | lightgbm  | 0.2850±0.0520 | 0.2091±0.0052     |

### Multiclass Classification

**covertype** (primary: mlogloss)

| Booster      | Library   | mlogloss          | train_time_s      |
|:-------------|:----------|:------------------|:------------------|
| gbdt         | boosters  | **0.4137±0.0092** | **2.2908±0.1071** |
| gbdt         | lightgbm  | 0.4285±0.0074     | 2.4857±0.0288     |
| gbdt         | xgboost   | 0.4698±0.0064     | 2.9410±0.0723     |
| gblinear     | boosters  | 0.7532±0.0117     | **2.7619±0.0298** |
| gblinear     | xgboost   | 0.7535±0.0084     | 6.0757±0.0409     |
| linear_trees | boosters  | 0.4069±0.0080     | 4.7096±0.1592     |
| linear_trees | lightgbm  | 0.4053±0.0091     | **3.5385±0.0405** |

**iris** (primary: mlogloss)

| Booster      | Library   | mlogloss          | train_time_s      |
|:-------------|:----------|:------------------|:------------------|
| gbdt         | boosters  | 0.1738±0.1038     | **0.0018±0.0001** |
| gbdt         | lightgbm  | 0.1423±0.0872     | 0.0160±0.0115     |
| gbdt         | xgboost   | 0.1846±0.1158     | 0.0104±0.0004     |
| gblinear     | boosters  | **0.3106±0.0374** | **0.0007±0.0001** |
| gblinear     | xgboost   | 0.4261±0.0304     | 0.0026±0.0000     |
| linear_trees | boosters  | 0.1770±0.0970     | **0.0024±0.0002** |
| linear_trees | lightgbm  | 0.1063±0.0652     | 0.0165±0.0009     |

**synthetic_multi_small** (primary: mlogloss)

| Booster      | Library   | mlogloss      | train_time_s      |
|:-------------|:----------|:--------------|:------------------|
| gbdt         | boosters  | 0.6864±0.1123 | **0.6532±0.0059** |
| gbdt         | lightgbm  | 0.6668±0.1138 | 0.8929±0.0143     |
| gbdt         | xgboost   | 0.6654±0.1010 | 2.2020±0.0554     |
| gblinear     | boosters  | 1.0560±0.0461 | **0.0740±0.0001** |
| gblinear     | xgboost   | 1.0546±0.0451 | 0.1446±0.0006     |
| linear_trees | boosters  | 0.6566±0.1056 | **0.6657±0.0057** |
| linear_trees | lightgbm  | 0.6876±0.1082 | 0.8943±0.0114     |

## Reproducing

```bash
boosters-eval full
```

---

*Best values per metric are **bolded**. Lower is better for loss/time metrics.*