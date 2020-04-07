import numpy as np
import pandas as pd


if __name__ == "__main__":
    idcols = ["ncodpers"]
    featurecols = ["sexo", "age"]

    itemcols = ["ind_ahor_fin_ult1", "ind_aval_fin_ult1", "ind_cco_fin_ult1",
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
    train = pd.read_csv("../model/train_unpivot.csv")
    train["age"] = train["age"].clip(10, 100)  # clipping
    train["age"] = train["age"].fillna(train["age"].mean())  # fill na

    test = pd.read_csv("../input/test_ver2.csv", usecols=idcols + featurecols, na_values=[" NA", "     NA"])
    items = pd.DataFrame(np.zeros(shape=(len(test), len(itemcols))), columns=itemcols)
    test = pd.concat([test, items], axis=1)
    test = test.melt(id_vars=idcols + featurecols, value_vars=itemcols, var_name="item", value_name="target")

    # sampling target = 0 records for unballance target
    train_target1 = train.query("target == 1")
    train_target0 = train.query("target == 0").sample(n=len(train_target1))
    train = train_target1.append(train_target0, ignore_index=True)
    
    # transform to ffm format for train
    category_vars = ["ncodpers", "sexo", "item"]
    numeric_vars = ["age"]
    target_var = ['target']
    variables = numeric_vars + category_vars

    dict_field = {}
    dict_index = {}
    current_idx = 0
    for field, var in enumerate(variables):
        if var in numeric_vars:
            dict_field[var] = field
            dict_index[var] = current_idx
            train[var] = [str(field) + ":" + str(current_idx) + ":" + str(val) for val in train[var]]
            current_idx += 1
        if var in category_vars:
            categories = train[var].unique()  # unique categories in this variable
            idx_range = range(len(categories))  # num of unique categories (range(0, num))
            
            dict_field[var] = field
            dict_index[var] = {val: current_idx + idx for idx, val in zip(idx_range, categories)}
            train[var] = [str(field) + ":" + str(dict_index[var][val]) + ":1" for val in train[var]]
            current_idx += max(idx_range) + 1
    train = pd.concat([train.loc[:, ["target"]], train.loc[:, variables]], axis=1)
    
    # transform to ffm format for test
    is_keyerror = False
    for var in variables:
        if var in numeric_vars:
            test[var] = [str(dict_field[var]) + ":" + str(dict_index[var]) + ":" + str(val) for val in test[var]]
        if var in category_vars:
            for row, val in enumerate(test[var]):
                try:
                    idx = dict_index[var][val]
                except KeyError:
                    idx = current_idx
                    is_keyerror = True
                test.loc[row, var] = str(dict_field[var]) + ":" + str(idx) + ":1"
            
            if is_keyerror is True:
                current_idx += 1
                is_keyerror = False
    test = pd.concat([test.loc[:, ["target"]], test.loc[:, variables]], axis=1)
    
    # write csv
    train.to_csv("./model/train_ffm.csv", index=False)
    test.to_csv("./model/test_ffm.csv", index=False)
