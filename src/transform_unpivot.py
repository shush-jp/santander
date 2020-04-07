import numpy as np
import pandas as pd


if __name__ == "__main__":
    datecols = ["fecha_dato"]
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

    usecols = datecols + idcols + featurecols + itemcols

    # read file & filter by date (use only 2016/5 data) for train data
    print("read file")
    train = pd.read_csv("../input/train_ver2.csv", usecols=usecols, na_values=[" NA", "     NA"])
    train = train.query("fecha_dato == '2016-05-28'").drop(datecols, axis=1)
    train = train.melt(id_vars=idcols + featurecols, value_vars=itemcols, var_name="item", value_name="target")

    train.to_csv("../model/train_unpivot.csv", index=False)
    