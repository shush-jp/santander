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


class MatrixFactorization():
    def __init__(self, R, X, Y, k, steps=200, alpha=0.01, lambda_=0.001, threshold=0.001):
        self.R = R
        self.m = R.shape[0]
        self.n = R.shape[1]
        self.k = k
        # initialize U and V
        self.U = np.random.rand(self.m, self.k)
        self.V = np.random.rand(self.k, self.n)
        self.alpha = alpha
        self.lambda_ = lambda_
        self.threshold = threshold
        self.steps = 200

        # preserve user_id list and item_id list
        self.X = X
        self.Y = Y

    def shuffle_in_unison_scary(self, x, y):
        rng_state = np.random.get_state()
        np.random.shuffle(x)
        np.random.set_state(rng_state)
        np.random.shuffle(y)

    def fit(self):
        for step in range(self.steps):
            error = 0
            # shuffle the order of the entry
            self.shuffle_in_unison_scary(self.X, self.Y)

            # update U and V
            for i in self.X:
                for j in self.Y:
                    r_ij = self.R[i - 1, j - 1]

                    if r_ij > 0:
                        err_ij = r_ij - \
                            np.dot(self.U[i - 1, :], self.V[:, j - 1])

                        for q in range(self.k):
                            self.U[i - 1, q] += self.alpha * (
                                err_ij * self.V[q, j - 1] + self.lambda_ * self.U[i - 1, q])
                            self.V[q, j - 1] += self.alpha * (
                                err_ij * self.U[i - 1, q] + self.lambda_ * self.V[q, j - i])

            # approximation
            R_hat = np.dot(self.U, self.V)
            # calculate estimation error for observed values
            for i in self.X:
                for j in self.Y:
                    r_ij = self.R[i - 1, j - 1]
                    r_hat_ij = R_hat[i - 1, j - 1]
                    if r_ij > 0:
                        error += pow(r_ij - r_hat_ij, 2)
            # regularization
            error += (self.lambda_ * np.power(self.U, 2).sum()) / 2
            error += (self.lambda_ * np.power(self.V, 2).sum()) / 2
            print(error)

            if error < self.threshold:
                break
        return self.U, self.V


def reccomend_mfbase(latent_matrix, score_matrix_item, user_id, top_n):
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

    # user_id=user_idの評価値を抜き出す
    pred_score_user = latent_matrix[:, user_id - 1]

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
     # item_id x user_idの行列, 行列のインデックスとカスタマーID、商品名の辞書、submissionデータ読み込み ------------------------------------------------
    try:
        with open('../model/item_user_score.dat', 'rb') as f:
            score_matrix_item = pickle.load(f)

        with open('../model/dict_customerid.dat', 'rb') as f:
            dict_customerid = pickle.load(f)

        with open('../model/dict_itemname.dat', 'rb') as f:
            dict_itemname = pickle.load(f)

        df_submission = pd.read_csv("../input/sample_submission.csv")

    except:
        print("File can not be opened.")

    #  latent matrixの計算------------------------------------------------------
    item_id = [k+1 for k in dict_itemname.keys()]
    user_id = [k+1 for k in dict_customerid.keys()]
    
    k = 4
    steps = 150

    mf = MatrixFactorization(score_matrix_item, item_id, user_id, k, steps)

    U, V = mf.fit()
    latent_matrix = np.dot(U, V)

    # レコメンドアイテムの計算 ------------------------------------------
    num_calc = score_matrix_item.shape[1]
    top_n = 7

    result = []
    for i in range(num_calc):
        reccomend_list = reccomend_mfbase(
            latent_matrix, score_matrix_item, i, top_n)
        result.append(reccomend_list)

    # submission用のファイル作成
    added_products = []
    for i in range(len(result)):
        added_products.append(' '.join(map(str, [dict_itemname[item_id-1] for item_id in result[i]])))
        
    ncodpers = [i for i in dict_custmoerid.values()][0:num_calc]

    df_result = pd.DataFrame({"ncodpers": ncodpers, "added_products": added_products})
    
    df_submission = df_submission.loc[:, ['ncodpers']].merge(df_result, how='left', on='ncodpers').replace(np.nan, '')
    df_submission.to_csv("../submission/submission.csv", index=False)
