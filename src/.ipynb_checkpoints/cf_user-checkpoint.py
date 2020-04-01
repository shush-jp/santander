import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm


def calc_item_user_mat(data, score_matrix_item, item_list, user_list):
    """return calculated item x user matrix by data

    Arguments:
        data {pandas.DataFrame} -- DataFrame columns: [user_id, item_id, score]
        score_matrix_item {2d array} -- calculated by data
        item_list {list} -- item list to be used for matrix row size
        user_list {list} -- user list to be used for matrix col size

    Returns:
        2d array -- item x user matrix matrix
    """
    score_matrix_item = np.zeros([len(item_list), len(user_list)])

    for item_id in tqdm(range(1, score_matrix_item.shape[0])):
        user_list_item = data[data['item_id'] == item_id].user_id.unique()
        for user_id in user_list_item:
            try:
                user_score = data[(data['item_id'] == item_id) &
                                  (data['user_id'] == user_id)].loc[:, 'score']
            except:
                user_score = 0
            score_matrix_item[item_id - 1, user_id - 1] = user_score
    return score_matrix_item


def calc_similarity_matrix(score_matrix_item, metric):
    """return similarity matrix

    Arguments:
        score_matrix_item {2d array} -- item x user matrix
        metric {str} -- "cosine", "jaccard"

    Returns:
        2d array -- similarity matrix
    """
    similarity_matrix = 1 - \
        pairwise_distances(score_matrix_item, metric=metric)
    # 対角成分の値はゼロにする
    np.fill_diagonal(similarity_matrix, 0)
    return similarity_matrix


def reccomend_itembase(similarity_matrix, score_matrix_item, user_id, top_n):
    """return top_n reccomended item list

    Arguments:
        similarity_matrix {2d array} -- item x item simirality matrix
        score_matrix_item {2d array} -- item x user matrix
        user_id {int} -- user id
        top_n {int} -- # of reccomended items

    Returns:
        list -- top_n reccomended item list
    """
    # item x user matrix scored：1, not scored: 0
    is_score_matrix = score_matrix_item.copy()
    is_score_matrix[is_score_matrix != 0] = 1
    is_not_score_matrix = np.abs(is_score_matrix - 1)

    # user_id=user_idの評価値を抜き出し「類似度×評価点」を算出
    score_matrix_user = score_matrix_item[:, user_id - 1]
    pred_score_user = similarity_matrix * score_matrix_user
    # アイテム（行）ごとに「類似度×評価点」を合計
    pred_score_user = pred_score_user.sum(axis=1)

    # ユーザが既に評価したアイテムのスコアはゼロに直す
    pred_score_user_item = pred_score_user * \
        is_not_score_matrix[:, user_id - 1]

    # レコメンド アイテムリストtop_n個を返す. [::-1]は配列をリバース
    reccomend_list = np.argsort(pred_score_user_item)[::-1][:top_n] + 1
    return reccomend_list


def calc_precision(reccomend_list, actual_list):
    """retrun precision

    Arguments:
        reccomend_list {list} -- reccomended item list
        actual_list {list} -- actual purchased list

    Returns:
        real -- precision
    """
    hits = 0
    n = len(reccomend_list)

    for item_id in reccomend_list:
        if item_id in actual_list:
            hits += 1
    precision = hits / n
    return precision


if __name__ == '__main__':
    # item_id x user_idの行列に読み込み ------------------------------------------------
    try:
        with open('../model/item_user_score.dat', 'rb') as f:
            score_matrix_item = pickle.load(f)
    except:
        print("File can not be opened.")

    # 類似度の計算 ------------------------------------------------------------------
    similarity_matrix = calc_similarity_matrix(score_matrix_item, "cosine")

    # user_id=100でレコメンドの結果確認 ----------------------------------------------
    user_id = 100
    top_n = 10

    reccomend_list = reccomend_itembase(
        similarity_matrix, score_matrix_item, user_id, top_n)
    print(reccomend_list)
    '''
    # 精度の計算 -------------------------------------------------------------------
    purchase_list_user = u_data_test[u_data_test.user_id == user_id].\
        loc[:, 'item_id'].unique()

    precision = calc_precision(reccomend_list, purchase_list_user)

    print('Recommend list:', reccomend_list)
    print('Rated list:', purchase_list_user)
    print('Precision:', str(precision))

    # テストuserにおけるリコメンドの結果確認・精度の計算 -------------------------------
    top_n = 10

    precision_list = []
    user_list_test = u_data_test.sort_values('user_id').user_id.unique()

    for user_id in tqdm(user_list_test):
        reccomend_list = reccomend_itembase(
            similarity_matrix, score_matrix_item, user_id, top_n)

        purchase_list_user = u_data_test[u_data_test.user_id == user_id].\
            loc[:, 'item_id'].unique()

        precision = calc_precision(reccomend_list, purchase_list_user)

        precision_list.append(precision)

    # 全体の精度平均値の計算 --------------------------------------------------------
    precision = sum(precision_list) / len(precision_list)
    print('Precision:', precision)
    '''
