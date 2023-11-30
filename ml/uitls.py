import itertools


def get_all_index(n, min_len, max_len):
    numbers = list(range(0, n))
    combinations = []

    for r in range(min_len, min(n, max_len) + 1):  # 从5开始，只保留长度为5及以上的组合
        combinations.extend(list(itertools.combinations(numbers, r)))

    return combinations


if __name__ == '__main__':

    import numpy as np

    print(len(get_all_index(30, 5, 5)))

    exit()
    # 生成随机数组
    np.random.seed(0)  # 设置随机种子以确保结果可重现
    arr = np.random.rand(100, 5)  # 生成一个形状为(100, 5)的随机数组
    print(type(arr), arr.shape)
    for i in get_all_index(5):

        selected_features = arr[:, i]
        print(selected_features.shape)
