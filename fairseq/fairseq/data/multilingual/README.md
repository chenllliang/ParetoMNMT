代码注解


## multilingual_data_manager.py
一次epoch中数据集生成的逻辑

1. 计算原始数据集的数据量的比例 ratio
2. 计算温度采样后的新比例
3. 使用新比例，保持原始数据集中最多的数据集的样本不变，对其他数据集进行上采样，不会进行下采样
4. 将采样完的数据集拼接(*)
5. 进行batch数据生成

无法准确控制每个batch中来自不同数据集的数据数量





## HotpotNMT 的数据集生成逻辑

Main Course 数据集 + Cusine 数据集s + recipe

1. 数据集不进行混合，生成一个batch时根据recipe从各个数据集采样
2. Main Course 完成一个 epoch 作为标志
3. 进行比较精准的控制

Todos:





```python
# sss
def _establish_virtual_datasets(self):
    if self.sample_ratios is None and self._cur_indices is not None:
        # not a samping dataset, no need to resample if indices are already established
        return
    self._reset_cached_properties()


    # indices are index for each example in the total dataset ?

    start_time = time.time()
    # Generate a weighted sample of indices as a function of the
    # random seed and the current epoch.
    rng = np.random.RandomState(
        [
            int(
                hashlib.sha1(
                    str(self.__class__.__name__).encode("utf-8")
                ).hexdigest(),
                16,
            )
            % (2**32),
            self.seed % (2**32),  # global seed
            self._cur_epoch,  # epoch index,
        ]
    )
    self._clean_if_not_none(
        [self.cumulated_sizes, self.virtual_size_per_dataset, self._sizes]
    )
    self._sizes = None

    indices, cumulated_sizes, virtual_size_per_dataset = self.get_virtual_indices(
        rng, self.datasets, self.sample_ratios, self.virtual_size
    ) # generate datasets for current epoch
    self._cur_indices = indices
    self.cumulated_sizes = cumulated_sizes
    self.virtual_size_per_dataset = virtual_size_per_dataset

    raw_sizes = [len(d) for d in self.datasets]
    sampled_sizes = self.virtual_size_per_dataset
    logger.info(
        f"[{self.split}] Raw sizes: {str(dict(zip(self.keys, raw_sizes)))}; "
        f"raw total size: {sum(raw_sizes)}"
    )
    logger.info(
        f"[{self.split}] Resampled sizes: {str(dict(zip(self.keys, sampled_sizes)))}; "
        f"resampled total size: {sum(sampled_sizes)}"
    )
    if self.sample_ratios is not None:
        logger.info(
            f"[{self.split}] Upsampling ratios: {str(dict(zip(self.keys, self.sample_ratios)))}"
        )
    else:
        logger.info(f"[{self.split}] A concat dataset")
    logger.info(
        f"[{self.split}] virtual dataset established time: {get_time_gap(start_time, time.time())}"
    )
```


```python
def get_virtual_indices(self, rng, datasets, sample_ratios, virtual_size):
    def get_counts(sample_ratios):
        counts = np.array([virtual_size * r for r in sample_ratios], dtype=np.int64)
        # number of example in each sampled dataset
        diff = virtual_size - counts.sum()

        assert diff >= 0
        # due to round-offs, the size might not match the desired sizes
        if diff > 0:
            dataset_indices = rng.choice(
                len(sample_ratios), size=diff, p=sample_ratios
            )
            for i in dataset_indices:
                counts[i] += 1
        return counts

    def get_in_dataset_indices(datasets, sizes, sample_ratios):
        counts = get_counts(sample_ratios)
        # uniformally sample desired counts for each dataset
        # if the desired counts are large, sample with replacement:
        indices = [
            self.random_choice_in_dataset(rng, d, c)
            for c, d in zip(counts, datasets)
        ]
        #generate a list of indices for each dataset
        return indices

    sizes = [len(d) for d in datasets]
    if sample_ratios is None:
        # default back to concating datasets
        in_dataset_indices = [list(range(s)) for s in sizes]
        virtual_sizes_per_dataset = sizes
    else:
        ratios = sample_ratios / sample_ratios.sum()
        in_dataset_indices = get_in_dataset_indices(datasets, sizes, ratios)
        virtual_sizes_per_dataset = [len(d) for d in in_dataset_indices]
    virtual_sizes_per_dataset = np.array(virtual_sizes_per_dataset, np.int64)
    cumulative_sizes = np.cumsum(virtual_sizes_per_dataset)
    assert sum(virtual_sizes_per_dataset) == virtual_size
    assert cumulative_sizes[-1] == virtual_size
    if virtual_size < sum(sizes):
        logger.warning(
            f"virtual data size ({virtual_size}) is less than real data size ({sum(sizes)})."
            " If virtual size << real data size, there could be data coverage issue."
        )
    in_dataset_indices = np.hstack(in_dataset_indices)
    return in_dataset_indices, cumulative_sizes, virtual_sizes_per_dataset

```


```python
def setup_sampling(self, sample_ratios, virtual_size):
    sizes = [len(d) for d in self.datasets]
    if sample_ratios is None:
        # default back to concating datasets
        self.sample_ratios = None
        self.virtual_size = sum(sizes)
    else:
        if not isinstance(sample_ratios, np.ndarray):
            sample_ratios = np.array(sample_ratios)
        self.sample_ratios = sample_ratios
        virtual_size = (
            default_virtual_size_func if virtual_size is None else virtual_size
        )
        self.virtual_size = (
            virtual_size(self.datasets, self.sample_ratios)
            if callable(virtual_size)
            else virtual_size
        )
        # compute the size of total datasets after sampling
```


```python


def default_virtual_size_func(datasets, ratios, max_scale_up=1.5):

    # give the origial datasets and the sampling ratios
    # return the virtual size of sampled total dataset

    sizes = [len(d) for d in datasets]
    if ratios is None:
        return sum(sizes)
    largest_idx = np.argmax(sizes)  # find max size dataset's index
    largest_r = ratios[largest_idx] # find ratio of the max size dataset
    largest_s = sizes[largest_idx]  # find size of the max size dataset
    
    virtual_sizes = [(r / largest_r) * largest_s for r in ratios] # set virtual sizes relative to the largest dataset
    # for example
    # if largest_r = 0.5, largest_s = 1000, and all rs = [0.5,0.25.0.125]
    # virtual_sizes = [0.5/0.5*1000, 0.25/0.5*1000, 0.125/0.5*1000]
    
    vsize = sum(virtual_sizes)
    max_size = sum(sizes) * max_scale_up # max size of the total dataset, set a limit of max examples in the total dataset, not exceed 1.5 times of the original datasets
    return int(vsize if vsize < max_size else max_size)

```