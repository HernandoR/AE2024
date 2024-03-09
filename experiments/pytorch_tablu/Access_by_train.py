'''
Author: Zhen Liu lzhen.dev@outlook.com
CreateDate: Do not edit
LastEditors: Zhen Liu lzhen.dev@outlook.com
LastEditTime: 2024-03-07
Description: 

Copyright (c) 2024 by HernandoR lzhen.dev@outlook.com, All Rights Reserved. 
'''
# %%
import pandas as pd
from  utils import extractSubmision,loadmodel_and_predict,gen_submission,incr_act_top10,calc

target_cols=['ind_recommended','activation']



if __name__ == '__main__':
    idf=pd.read_parquet('data/Amex_ori/Amex Campus Challenge Train 3.parquet')
    bkp=pd.DataFrame()
    for target in target_cols:
        bkp[target]=idf[target]
        idf[target]=loadmodel_and_predict(f"checkpoints/{target}",idf)
    idf=gen_submission(idf)

    for target in target_cols:
        idf[target]=bkp[target]
    del bkp
# %%
    print(incr_act_top10(input_df = idf, pred_col = 'predicted_score'))
    idf['act_score']=idf.apply(lambda x:calc(x['ind_recommended'],x['activation']),axis=1)
    print(incr_act_top10(input_df = idf, pred_col = 'act_score'))
    
# %%
    idf=pd.read_parquet('data/Amex_ori/Amex Campus Challenge Round 1.parquet')
    for target in target_cols:
        idf[target]=loadmodel_and_predict(f"checkpoints/{target}",idf)
    idf=gen_submission(idf)
    
                        