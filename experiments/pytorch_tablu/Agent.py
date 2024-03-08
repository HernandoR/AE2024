# %%
"""
Author: Zhen Liu lzhen.dev@outlook.com
CreateDate: Do not edit
LastEditors: Zhen Liu lzhen.dev@outlook.com
LastEditTime: 2024-03-06
Description: 

Copyright (c) 2024 by HernandoR lzhen.dev@outlook.com, All Rights Reserved. 
"""
import time
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import random
import numpy as np
import pandas as pd
import wandb
from rich import print

from imblearn.combine import SMOTETomek, SMOTEENN  

# %%
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
    ExperimentConfig,
)
from pytorch_tabular.models.common.heads import LinearHeadConfig


from typing import Callable
from datetime import datetime


from pytorch_tabular.utils import (
    make_mixed_dataset,
    print_metrics,
    load_covertype_dataset,
)
from pytorch_tabular.utils import get_balanced_sampler, get_class_weighted_cross_entropy

import os

# %load_ext autoreload
# %autoreload 2

# %%
# wandb.login()

# %%
# data, cat_col_names, num_col_names = make_mixed_dataset(
#     task="classification",
#     n_samples=10000,
#     n_features=8,
#     n_categories=4,
#     weights=[0.8],
#     random_state=42)
# target_col='target'


# %%


# %% [markdown]
# ## Define the Configs
#


# %%


def prepare_Model(
    run_name: str,
    model_config,
    data_config,
    trainer_config,
    optimizer_config,
    head_config,
):

    experiment_config = ExperimentConfig(
        project_name="AMEX",
        run_name=run_name,
        exp_watch="gradients",
        # log_target="wandb", # wandb will raise error, too many indicat for 1-d data.
        log_logits=True,
    )

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        experiment_config=experiment_config,
        verbose=False,
        # suppress_lightning_logger=True,
    )

    return tabular_model


# %%

# %%
if __name__=="__main__":
    Feature_Explain = pd.read_excel("data/Amex_ori/Amex Campus.xlsx")
    Feature_target_cols = Feature_Explain["Feature Name"][
        Feature_Explain["Extended description"] == "Target"
    ].to_list()
    Feature_cat_col_names = Feature_Explain["Feature Name"][
        Feature_Explain["Feature Type"] == "categorical"
    ].to_list()
    Feature_num_col_names = Feature_Explain["Feature Name"][
        Feature_Explain["Variable Type"] == "numeric"
    ].to_list()

    DEVICE = (
        "mps"
        if torch.backends.mps.is_available()
        else ("gpu" if torch.cuda.is_available() else "cpu")
    )
    print(DEVICE)

    # %%
    data = pd.read_parquet("data/train.parquet")

    cols = data.columns.to_list()
    target_cols = [col for col in cols if col in Feature_target_cols]
    if len(target_cols) == 0:
        print("No target column found")
        exit()
    # elif len(target_cols) > 1:
    #     target_col=target_cols[1]
    cat_col_names = [col for col in cols if col in Feature_cat_col_names]
    num_col_names = [col for col in cols if col in Feature_num_col_names]

    print(f"{target_cols}")
    # target_col=cols.unite(Feature_target_cols)

    # target_col = 'activation'

    # %%

    needed_cols = cat_col_names + num_col_names + target_cols
    data = data[needed_cols]

    data.head()
    print(
        f"target_cols={target_cols}, cat_col_names={cat_col_names}, num_col_names={num_col_names}"
    )

    # %%
    # Data: DF, cat_col_names, num_col_names

    # 1. activate(0.57%) X recom(12.7%) -> 0~3
    # _,data=train_test_split(data,test_size=0.05)

    train, test = train_test_split(data, random_state=42)
    train, val = train_test_split(train, random_state=42)


    trainer_config = TrainerConfig(
        auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
        batch_size=1024*2,
        max_epochs=100,
        early_stopping="valid_loss",  # Monitor valid_loss for early stopping
        early_stopping_mode="min",  # Set the mode as min because for val_loss, lower is better
        early_stopping_patience=10,  # No. of epochs of degradation training will wait before terminating
        checkpoints="valid_loss",  # Save best checkpoint monitoring val_loss
        checkpoints_path="checkpoints",  # Save the checkpoint in the experiment directory
        checkpoints_save_top_k=2,
        progress_bar="simple",
        load_best=True,  # After training, load the best checkpoint
        accelerator=DEVICE,
        trainer_kwargs={
            "enable_progress_bar": True,
            # "max_steps":1000,
            # "progress_bar_refresh_rate":1
        },
    )
    optimizer_config = OptimizerConfig()

    head_config = LinearHeadConfig(
        layers="",  # No additional layer in head, just a mapping layer to output_dim
        dropout=0.1,
        initialization="kaiming",
    ).__dict__  # Convert to dict to pass to the model config (OmegaConf doesn't accept objects)


    model_config = CategoryEmbeddingModelConfig(
        task="classification",
        layers="256-128-64",  # Number of nodes in each layer
        activation="LeakyReLU",  # Activation between each layers
        head="LinearHead",  # Linear Head
        head_config=head_config,  # Linear Head Config
        learning_rate=1e-3,
        metrics=["f1_score", "accuracy"],
        metrics_params=[{"num_classes": 2}, {}],  # f1_score needs num_classes
        metrics_prob_input=[
            True,
            False,
        ],  # f1_score needs probability scores, while accuracy doesn't
    )

    target_col = target_cols[0]

    for target_col in target_cols:
        data_config = DataConfig(
            target=[target_col],
            # target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
            continuous_cols=num_col_names,
            categorical_cols=cat_col_names,
            num_workers=5,
        )

        tabular_model = prepare_Model(
            target_col, model_config, data_config, trainer_config, optimizer_config, head_config
        )
        sampler = get_balanced_sampler(train[target_col].values.ravel())

        tabular_model.fit(train=train, validation=val, train_sampler=sampler)
        # takes ~ 60min,13 Epoch, most time in preparing data, weird

        result = tabular_model.evaluate(test)
        # pred_df = tabular_model.predict(test)
        tabular_model.save_model(
            f"{trainer_config.checkpoints_path}/{target_col}"
        )  # save in the dir


# %%
# pred_df.head()


# # %%
# def extractSubmision(
#     input_df: pd.DataFrame,
#     pred_col: str,
#     cm_key="customer",
#     mct_key="merchant",
#     output_path="submission.csv",
# ) -> None:
#     """
#     function that extracts the submission file for the AMEX Singapore Hackathon 2024

#     input_df : pandas Dataframe which has customer, merchant and pred_col
#     pred_col : name of your prediction score variable
#     cm_key : customer unique ID (do not change)
#     mct_key : merchant unique ID (do not change)
#     output_path : path to save the submission file

#     Returns - None
#     """
#     print("formate prodiction")
#     LEN = len(input_df)
#     if LEN != 12604600:
#         raise f"miss matchlenth {LEN} "
#     output_df = pd.DataFrame()
#     output_df[[cm_key, mct_key, "predicted_score"]] = input_df[
#         [cm_key, mct_key, pred_col]
#     ].apply(pd.to_numeric, errors="coerce")
#     output_df.to_csv(output_path, index=False)
#     return output_df


# def calc(rec: int, act: int):
#     crit = rec * 2 + act
#     match crit:
#         case 3:  # TT
#             return 3
#         case 0:  # NN
#             return 2
#         case 2:  # TN
#             return 1
#         case 1:  # NT
#             return 0
#     return None


# # add colum pre_score by func calc eat 'ind_recommended' and 'activaton'


# def loadmodel_and_predict(
#     modelpath: str, test_data: pd.DataFrame, data_preprocess: Callable = None
# ) -> pd.DataFrame:
#     print(f"loading model at {modelpath}")
#     loadedmodel = TabularModel.load_model(modelpath)
#     # idf=pd.read_csv(test_path)
#     if data_preprocess:
#         test_data = data_preprocess(test_data)
#     print(f"predicting")
#     return loadedmodel.predict(test_data)["prediction"]


# def gen_submission(pred_df: pd.DataFrame):
#     print("generating_pred_score")
#     for pred in ["ind_recommended", "activation", "customer", "merchant"]:
#         if pred not in pred_df.columns:
#             print(f"missing col {pred}")

#     pred_df["predicted_score"] = pred_df.apply(
#         lambda x: calc(x["ind_recommended"], x["activation"]), axis=1
#     )

#     return extractSubmision(pred_df, "predicted_score")


# def preprocess(idf: pd.DataFrame):
#     cols = []
#     for col in idf.columns:
#         if (col not in cat_col_names) and (col not in num_col_names):
#             cols.append(col)
#     idf.drop(columns=cols)
#     return idf


# # %%

# idf = pd.read_csv("data/Amex_ori/Amex Campus Challenge Round 1.csv")
# idf.head()

# # %%

# for target in target_cols:
#     idf[target] = loadmodel_and_predict(
#         f"{trainer_config.checkpoints_path}/{target}", idf, preprocess
#     )


# # %%

# idf = gen_submission(idf)


# # %%
# def incr_act_top10(
#     input_df: pd.DataFrame,
#     pred_col: str,
#     cm_key="customer",
#     treated_col="ind_recommended",
#     actual_col="activation",
# ):
#     """
#     Function that returns the incremental activation score for the AMEX Singapore Hackathon 2024

#     input_df : pandas Dataframe which has customer, ind_recommended, activation and pred_col
#     pred_col : name of your prediction score variable
#     cm_key : customer unique ID (do not change)
#     treated_col : indicator variable whether a merchant was recommended
#     actual_col : whether a CM had transacted at a given merchant (target variable)

#     Returns - incremental activation
#     """

#     # for correcting variable types
#     input_df[[treated_col, actual_col, pred_col]] = input_df[
#         [treated_col, actual_col, pred_col]
#     ].apply(pd.to_numeric, errors="coerce")

#     input_df["rank_per_cm1"] = input_df.groupby(cm_key)[pred_col].rank(
#         method="first", ascending=False
#     )

#     input_df = input_df.loc[input_df.rank_per_cm1 <= 10, :]

#     agg_df = input_df.groupby(treated_col, as_index=False).agg({actual_col: "mean"})
#     agg_df.columns = [treated_col, "avg_30d_act"]

#     print(agg_df)
#     recommended_avg_30d_act = float(
#         agg_df.iloc[agg_df[treated_col] == 1, "avg_30d_act"]
#     )
#     not_recommended_avg_30d_act = float(
#         agg_df.iloc[agg_df[treated_col] == 0, "avg_30d_act"]
#     )

#     return recommended_avg_30d_act - not_recommended_avg_30d_act


# %%

# idf = pd.read_parquet("data/Amex_ori/Amex Campus Challenge Train 3.parquet")
# bkp = pd.DataFrame()
# for target in target_cols:
#     bkp[target] = idf[target]
#     idf[target] = loadmodel_and_predict(
#         f"{trainer_config.checkpoints_path}/{target}", idf, preprocess
#     )
# idf = gen_submission(idf)

# for target in target_cols:
#     idf[target] = bkp[target]
# del bkp


# incr_act_top10(input_df=idf, pred_col="predicted_score")


# %% [markdown]
#

# %% [markdown]
#

# %% [markdown]
#

# %% [markdown]
# ## Custom Sampler
#
# PyTorch Tabular also allows custom batching strategy through Custom Samplers  which comes in handy when working with imbalanced data.
#
# Although you can use any sampler, Pytorch Tabular has a few handy utility functions which takes in the target array and implements WeightedRandomSampler using inverse frequency sampling to combat imbalance. This is analogous to preprocessing techniques like Under or OverSampling in traditional ML systems.

# %%


# # %%
# tabular_model = TabularModel(
#     data_config=data_config,
#     model_config=model_config,
#     optimizer_config=optimizer_config,
#     trainer_config=trainer_config,
#     verbose=False,
# )
# sampler = get_balanced_sampler(train[target_col].values.ravel())

# tabular_model.fit(train=train, validation=val, train_sampler=sampler)


# # %%
# result = tabular_model.evaluate(test)

# # %%
# result = {k: float(v) for k, v in result[0].items()}
# result["mode"] = "Balanced Sampler"

# results.append(result)

# # %% [markdown]
# # ## Custom Weighted Loss
# #
# # If Samplers were like Over/Under Sampling, Custom Weighted Loss is similar to `class_weights`. Depending on the problem, one of these might help you with imbalance. You can easily make calculate the class_weights and provide them to the CrossEntropyLoss using the parameter `weight`. To make this easier, PyTorch Tabular has a handy utility method which calculates smoothed class weights and initializes a weighted loss. Once you have that loss, it's just a matter of passing it to the 1fit1 method using the `loss` parameter.

# # %%
# tabular_model = TabularModel(
#     data_config=data_config,
#     model_config=model_config,
#     optimizer_config=optimizer_config,
#     trainer_config=trainer_config,
#     verbose=False,
# )
# weighted_loss = get_class_weighted_cross_entropy(train["target"].values.ravel(), mu=0.1)

# tabular_model.fit(train=train, validation=val, loss=weighted_loss)


# # %%
# result = tabular_model.evaluate(test)

# # %%
# result = {k: float(v) for k, v in result[0].items()}
# result["mode"] = "Class Weights"

# results.append(result)

# # %%
# res_df = pd.DataFrame(results).T
# res_df.columns = res_df.iloc[-1]
# res_df = res_df.iloc[:-1].astype(float)
# res_df.style.highlight_min(color="lightgreen", axis=1)

# # %%


# %%


# %%
