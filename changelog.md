# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- [x] Period 1~2: lacking the `step` and `tmcs` shapfile

- [ ] 计算不同城市之间的距离变化情况
- [ ] 绘制kde变化叠加图

  ```
  # 全部变量的KDE分布图
  dist_cols = 6
  dist_rows = len(test_data.columns)
  plt.figure(figsize=(4*dist_cols, 4 * dist_rows))

  i = 1
  for col in test_data.columns:
      ax = plt.subplot(dist_rows, dist_cols, i)
      ax = sns.kdeplot(train_data[col], color='Red', shade=True)
      ax = sns.kdeplot(test_data[col], color='Blue', shade=True)
      ax.set_xlabel(col)
      ax.set_ylabel("Freanquency")
      ax.legend(['train', 'test'])

      i+=1
  plt.show()
  ```

## [1.0.04] - 2020-06-30

### Added

- `trip_crawler` to crawl data from the map provider.
- `data_process_step_spark` reorgnize, and move some funs from eda_step

## [1.0.03] - 2020-06-30

### Added

- Extract bridge congestion index from `step` csv or shapefile with Spark.

## [1.0.02] - 2020-06-01

### Added

- 线路热力图
- spark 参与计算，提升速度

## [1.0.01] - 2021-04-18

### Added

- build the project
- `PathSet` Class: To store the path between cities
- 预处理数据（path, steps）
