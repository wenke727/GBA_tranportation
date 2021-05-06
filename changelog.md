# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
- [ ] Period 1~2: lacking the `step` and `tmcs` shapfile
  How to extract traffic condition information from `GBA_step_190322.csv` 
  ```
  tripID,instruction,orientation,road,distance,tolls,toll_distance,toll_road,duration,action,assistant_action
  220000001,向东北行驶768米右转,东北, ,768,0,0,[],71,右转,[]
  220000001,沿启成街途径启华街向东行驶227米右转,东,启成街,227,0,0,[],66,右转,[]
  220000001,沿宏光道向南行驶116米向左前方行驶进入左转专用道,南,宏光道,116,0,0,[],28,向左前方行驶,进入左转专用道
  ```
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
- [ ] 
## [1.0.01] - 2021-04-18
### Added
- build the project
- `PathSet` Class: To store the path between cities. 
- 预处理数据（path, steps）
