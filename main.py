# import numpy as np
# from sklearn.cluster import KMeans

# def run():
#     N, K, t = map(int, input().split())
#     heights = [int(input()) for _ in range(N)]
#
#     val = heights[t-1]
#
#     heights.sort()
#
#     min_diff = float('inf')
#
#     for i in range(N - K + 1):
#         max_height = heights[i + K - 1]
#         min_height = heights[i]
#         diff = max_height - min_height
#
#         if heights[i] <= val <= heights[i + K - 1]:
#             min_diff = min(min_diff, max_height - min_height)
#
#
#     print(min_diff)

def max_coins(orders):
    cur_time = 0
    tips = 0
    for a, b in orders:
        cur_time += b
        tips += a - cur_time
    return tips


def binary_search(arr, target):
    low = 0
    high = len(arr) - 1

    while low <= high:
        mid = (low + high) // 2
        guess = arr[mid]

        if guess == target:
            return mid
        elif guess[1] < target[1]:
            low = mid + 1
        else:
            high = mid - 1

    return -1


def binary_search2(arr, target):
    low = 0
    high = len(arr) - 1

    while low <= high:
        mid = (low + high) // 2
        guess = arr[mid][1]

        if mid == 0 and target <= guess:
            return 0
        if mid == len(arr) - 1 and target >= guess:
            return len(arr)
        if mid + 1 < len(arr) and guess < target and target <= arr[mid + 1][1]:
            return mid + 1
        elif guess < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1


def run():
    n, q = map(int, input().split())

    orders_orig = [list(map(int, input().split())) for _ in range(n)]

    orders_sorted = orders_orig.copy()
    orders_sorted.sort(key=lambda x: x[1])

    print(max_coins(orders_sorted))

    for _ in range(q):
        i, a, b = map(int, input().split())
        old_el = orders_orig[i - 1]
        orders_orig = orders_orig[:i - 1] + [[a, b]] + orders_orig[i:]
        idx_in_sorted = orders_orig.index(old_el)

        orders_sorted = orders_sorted[:idx_in_sorted] + orders_sorted[idx_in_sorted + 1:]
        new_idx_in_sorted = binary_search2(orders_sorted, b)

        orders_sorted = orders_sorted[:new_idx_in_sorted] + [[a, b]] + orders_sorted[new_idx_in_sorted:]
        print(max_coins(orders_sorted))

# def calc_profit(data_, clusters_centers, labels):
#     c_cost = 10
#     prof = 25000
#     for lab_ in np.unique(labels):
#         res = sum(c_cost*(((((data_[labels == lab_] - clusters_centers[lab_])**2).sum(axis=1))**0.5)**0.25 + 1))/(len(labels == lab_))
#         #print(res)
#         prof -= res
#     return prof
#
# def task_B(data_arr):
#
#
#     wcss = []
#     profit_arr = []
#     # for loop
#     for i in range(1200, 2000):
#         # k-mean cluster model for different k values
#         kmeans = KMeans(n_clusters=i, init='k-means++')
#         kmeans.fit(data_arr)
#
#         # inertia method returns wcss for that model
#         wcss.append(kmeans.inertia_)
#         res = calc_profit(data_arr, kmeans.cluster_centers_, kmeans.labels_)
#         print(res, i)
#         profit_arr.append([res, i])
#
#     profit_arr = np.array(profit_arr)
#     print(profit_arr[profit_arr[:, 0] == profit_arr.max()])

with open("kinopoisk_input.txt", "r") as file:
    n, m, q = map(int, file.readline().split())
    matrix = [list(map(int, file.readline().split())) for _ in range(n)]
    queries = [line.strip().split() for line in file.readlines()]

# Функция для определения Евклидова расстояния между двумя векторами
def euclidean_distance(user1, user2):
    return sum((a - b) ** 2 for a, b in zip(user1, user2)) ** 0.5


def kino():
    answered_queries = 0
    output = []

    for query in queries:
        print(query)
        if query[0] == "u":
            user_query = int(query[1])
            best_similarity = float('inf')  # Начальное значение для сравнения
            best_movie = -1  # Начальное значение для лучшего фильма
            for i, user in enumerate(matrix):
                if i != user_query - 1:
                    similarity = euclidean_distance(matrix[user_query - 1], user)
                    if similarity < best_similarity:
                        best_similarity = similarity
                        best_movie = user.index(max(user))
            answered_queries += 1
            output.append(best_movie + 1)  # Добавляем 1, так как индексация начинается с 0
        elif query[0] == "m":
            movie_query = int(query[1])
            best_similarity = float('inf')  # Начальное значение для сравнения
            best_movie = -1  # Начальное значение для лучшего фильма

            movie_vector = [row[movie_query - 1] for row in matrix]

            for j in range(m):
                if j != movie_query - 1:
                    column = [row[j] for row in matrix]
                    similarity = euclidean_distance(movie_vector, column)
                    if similarity < best_similarity:
                        best_similarity = similarity
                        best_movie = column.index(max(column))
            answered_queries += 1
            output.append(best_movie + 1)

if __name__ == '__main__':
    pass
    # import sys
    #
    # import pandas as pd
    #
    # # Прочитаем данные
    # df = pd.read_csv(sys.stdin, sep=";")
    #
    # df_bad_service = df[df['оценка_качества_предоставленной_услуги'] == 'плохо']
    #
    # # Группируем данные по 'мирный_путешественник' и 'расстояние_кат'
    # grouped = df_bad_service.groupby(['мирный_путешественник', 'расстояние_кат']).size().reset_index(name='counts')
    #
    # # Группируем исходные данные по тем же признакам
    # total_grouped = df.groupby(['мирный_путешественник', 'расстояние_кат']).size().reset_index(name='total_counts')
    #
    # # Объединяем группы для расчета доли
    # result = pd.merge(grouped, total_grouped, on=['мирный_путешественник', 'расстояние_кат'])
    #
    # # Рассчитываем долю и преобразуем в проценты с округлением до одного знака после запятой
    # result['процент'] = (result['counts'] / result['total_counts'] * 100).round(1)
    #
    # print(pd.DataFrame(result).round(1))