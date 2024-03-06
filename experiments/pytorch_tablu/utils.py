import pandas as pd
from pytorch_tabular import TabularModel

from typing import Callable

def extractSubmision(input_df:pd.DataFrame,
                     pred_col: str,
                     cm_key='customer',
                     mct_key='merchant',
                     output_path='submission.csv')->None:
    '''
    function that extracts the submission file for the AMEX Singapore Hackathon 2024

    input_df : pandas Dataframe which has customer, merchant and pred_col
    pred_col : name of your prediction score variable
    cm_key : customer unique ID (do not change)
    mct_key : merchant unique ID (do not change)
    output_path : path to save the submission file

    Returns - None
    '''
    print('formate prodiction')
    if not __name__ == '__main__':
        LEN=len(input_df)
        if LEN!=12604600:
            raise f"miss matchlenth {LEN} "
    output_df=pd.DataFrame()
    output_df[[cm_key,mct_key,'predicted_score']]=input_df[[cm_key,mct_key,pred_col]].apply(pd.to_numeric,errors='coerce')
    output_df.to_csv(output_path,index=False)
    return output_df

def calc(rec:int,act:int):
    crit=rec*2+act
    match crit:
        case 3: # TT
            return 3
        case 0: # NN
            return 2
        case 2: # TN
            return 1
        case 1: # NT
            return 0
    return None
        
# add colum pre_score by func calc eat 'ind_recommended' and 'activaton'
        
def loadmodel_and_predict(modelpath:str,test_data:pd.DataFrame,data_preprocess:Callable = None) -> pd.DataFrame:
    print(f"loading model at {modelpath}")
    loadedmodel=TabularModel.load_model(modelpath)
    # idf=pd.read_csv(test_path)
    if data_preprocess:
        test_data=data_preprocess(test_data)
    print(f"predicting")
    return loadedmodel.predict(test_data)['prediction']

def gen_submission(pred_df:pd.DataFrame):
    print("generating_pred_score")
    for pred in ["ind_recommended","activation",'customer','merchant']:
        if pred not in pred_df.columns:
            print(f"missing col {pred}")


    pred_df['predicted_score']=pred_df.apply(lambda x:calc(x['ind_recommended'],x['activation']),axis=1)

    
    return extractSubmision(pred_df,'predicted_score')



def incr_act_top10(input_df: pd.DataFrame,
                   pred_col: str,
                   cm_key='customer',
                   treated_col='ind_recommended',
                   actual_col='activation'):
    '''
    Function that returns the incremental activation score for the AMEX Singapore Hackathon 2024

    input_df : pandas Dataframe which has customer, ind_recommended, activation and pred_col
    pred_col : name of your prediction score variable
    cm_key : customer unique ID (do not change)
    treated_col : indicator variable whether a merchant was recommended
    actual_col : whether a CM had transacted at a given merchant (target variable)

    Returns - incremental activation
    '''
    
	#for correcting variable types
    input_df[[treated_col, actual_col, pred_col]] = input_df[[treated_col, actual_col, pred_col]].apply(pd.to_numeric, errors='coerce')
	
    input_df['rank_per_cm1'] = input_df.groupby(cm_key)[pred_col].rank(method='first', ascending=False)
    
    input_df = input_df.loc[input_df.rank_per_cm1 <= 10,:]
    
    agg_df = input_df.groupby(treated_col,as_index=False).agg({actual_col:'mean'})
    agg_df.columns = [treated_col,'avg_30d_act']
    
    print(agg_df)
    recommended_avg_30d_act = float(agg_df.loc[agg_df[treated_col]==1,'avg_30d_act'])
    not_recommended_avg_30d_act = float(agg_df.loc[agg_df[treated_col]==0,'avg_30d_act'])
    
	
    return (recommended_avg_30d_act-not_recommended_avg_30d_act)
