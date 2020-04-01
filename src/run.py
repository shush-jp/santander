import numpy as np
import pandas as pd

from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import pairwise_distances

import sys


# define function to calcurate recommendation matrix (user x item)
def svd(mat, k):
    """calcurate SVD of matrix
    
    Arguments:
        mat {2d array} -- score/purchase matrix (dimension: (user, item))
        k {int} -- parameter of dimension compression
    
    Returns:
        2d array -- recommendation matrix (dimension: (user, item))
    """
    print("SVD start")
    
    U, s, V = np.linalg.svd(mat, full_matrices=False)
    s = np.diag(s)[0:k, 0:k]
    U = U[:, 0:k]
    V = V[0:k, :]

    s_root = np.sqrt(s)

    Usk = np.dot(U, s_root)
    skV = np.dot(s_root, V)
    UsV = np.dot(Usk, skV)

    print("SVD end")
    return UsV


def nmf(mat, k):
    """calcurate Non-negative Matrix Factorization of matrix
    
    Arguments:
        mat {2d array} -- score/purchase matrix (dimension: (user, item))
        k {int} -- parameter of dimension compression
    
    Returns:
        2d array -- recommendation matrix (dimension: (user, item))
    """
    print("NMF start")
    
    model = NMF(n_components=k, init="random", random_state=0)  # n_componentsで特徴の次元を指定
    P = model.fit_transform(mat)
    Q = model.components_
    PQ = np.dot(P, Q)
    
    print("NMT end")
    return PQ


def cf_itembased(mat, metric="cosine"):   
    """calcurate recommendation matrix for item-based collaborative filtering
    
    Arguments:
        mat {2d array} -- purchase flag matrix (dimension: (user, item))
        metric {string} -- metric to use when calculating distance
    
    Returns:
        2d array -- recommendation matrix (dimension: (user, item))
    """
    print("user based CF start")
    
    similarity_matrix = 1 - pairwise_distances(mat.T, metric=metric)
    np.fill_diagonal(similarity_matrix, 0)
    recommendation_matrix = np.dot(mat, similarity_matrix)
    
    print("user based CF end")
    return recommendation_matrix
    

def mask_alreadybuy(recommend_mat, train_mat):
    """mask (convert to zero) recommendation matrix by train_mat (purchase flag matrix)
    
    Arguments:
        recommend_mat {2d array} -- recommendation matrix (dimension: (user, item))
        train_mat {2d array} -- train data (purchase flag) matrix (dimension: (user, item))
    
    Returns:
        2d array -- recommendation matrix masked items which is already purchased (dimension: (user, item))
    """
    return(recommend_mat * (1 - train_mat))    


# define functions to get recommendation items
def get_topn_items(recommendation_mat, item_dict, index_user, top_n, threshold):
    """get top n high score items for user (defined by index of user (index user))
 
    Arguments:
        recommendation_mat {2d array} -- recommendation matrix masked items which is already purchased (dimension: (user, item))
        item_dict {dict} -- key: index(int), value: item name(string)
        index_user {int} -- user index
        top_n {int} -- parameter which decides how many items to pick
        threshold {double} -- parameter which decides items whose score is over this parameter
    
    Returns:
        list -- list of recommendation item names
    """
    # get top n high score item indices for user (defined by index of user (index user))
    topn_item_indices = recommendation_mat[index_user, :].argsort()[-top_n:][::-1]
    # extract items whose score is over threshold
    result_items = [item_dict[i] for i in topn_item_indices if recommendation_mat[index_user, i] > threshold]
    
    return result_items


def recommend(train_mat, test, item_dict, user_dict, top_n, threshold, mode="cf_itembased"):
    """calcurate recommendation items and return submission formatted data frame
    
    Arguments:
        train_mat {2d array} -- train data (purchase flag) matrix (dimension: (user, item))
        test {pandas.DataFrame} -- test data
        item_dict {dict} -- key: index(int), value: item name(string)
        user_dict {dict} -- key: user id (int), value: user index for train data (int)
        top_n {int} -- parameter which decides how many items to pick
        threshold {int} -- parameter which decides items whose score is over this parameter
        mode {str} -- algorithm to calcurate recommendation matrix ("svd" or "nmf" or "cf_itembased")
    
    Returns:
        pandasDataFrame -- submission data
    """
    if mode == "svd":
        recommendation_mat = svd(train_mat, k=4)
    elif mode == "nmf":
        recommendation_mat = nmf(train_mat, k=4)
    elif mode == "cf_itembased":
        recommendation_mat = cf_itembased(train_mat)
    
    recommendation_mat = mask_alreadybuy(recommendation_mat, train_mat)

    pred = []
    for user in test["ncodpers"]:
        i = user_dict[user]
        p = get_topn_items(recommendation_mat, item_dict, i, top_n, threshold)  # [::-1] means reverse of array
        pred.append(" ".join(p))  # join with white space for submission format
        
    test["added_products"] = pred
    return test


if __name__ == "__main__":
    mode = sys.argv[1]

    usecols1 = [
                "fecha_dato", "ncodpers", "ind_ahor_fin_ult1", "ind_aval_fin_ult1", "ind_cco_fin_ult1",
                "ind_cder_fin_ult1", "ind_cno_fin_ult1", "ind_ctju_fin_ult1",
                "ind_ctma_fin_ult1", "ind_ctop_fin_ult1", "ind_ctpp_fin_ult1",
                "ind_deco_fin_ult1", "ind_deme_fin_ult1", "ind_dela_fin_ult1",
                "ind_ecue_fin_ult1", "ind_fond_fin_ult1", "ind_hip_fin_ult1",
                "ind_plan_fin_ult1", "ind_pres_fin_ult1", "ind_reca_fin_ult1",
                "ind_tjcr_fin_ult1", "ind_valo_fin_ult1", "ind_viv_fin_ult1",
                "ind_nomina_ult1", "ind_nom_pens_ult1", "ind_recibo_ult1"
                ]

    # read file & filter by date (use only 2016/5 data) for train data
    print("read file")
    train = pd.read_csv("../input/train_ver2.csv", usecols=usecols1)
    train = train.query("fecha_dato == '2016-05-28'").drop("fecha_dato", axis=1)
    
    test = pd.read_csv("../input/test_ver2.csv")
    test = test.loc[:, ["ncodpers"]]

    # make user dictionary (key: user id, value: index), item dictionary (key: index, value: item name) based on train
    users = train["ncodpers"]
    items = train.drop("ncodpers", axis=1).columns.tolist()

    user_dict = {}
    for i, user in enumerate(users):
        user_dict[user] = i
        
    item_dict = {}
    for i, item in enumerate(items):
        item_dict[i] = item

    # convert train data frame to matrix
    train = train.drop('ncodpers', axis=1)
    train_mat = np.array(train)

    # calcuration recommendation items and write submit csv
    print("start calculation")
    recommend(train_mat, test, item_dict, user_dict, 7, 0.001, mode)
    test.to_csv("../submission/submission_" + mode + ".csv", index=False)
