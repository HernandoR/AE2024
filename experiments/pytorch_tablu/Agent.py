# %%
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import random
import numpy as np
import pandas as pd
import wandb

from pytorch_tabular.utils import make_mixed_dataset, print_metrics
import os

# %%
wandb.login()

# %%
data, cat_col_names, num_col_names = make_mixed_dataset(
    task="classification", 
    n_samples=10000, 
    n_features=8, 
    n_categories=4, 
    weights=[0.8], 
    random_state=42)
train, test = train_test_split(data, random_state=42)
train, val = train_test_split(train, random_state=42)

# %%
data.target.value_counts(normalize=True)

# %% [markdown]
# # Importing the Library

# %%
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig

# %%
results = []

# %% [markdown]
# ## Define the Configs
# 

# %%
data_config = DataConfig(
    target=['target'], #target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
    continuous_cols=num_col_names,
    categorical_cols=cat_col_names,
)
trainer_config = TrainerConfig(
    auto_lr_find=True, # Runs the LRFinder to automatically derive a learning rate
    batch_size=1024,
    max_epochs=100,
    early_stopping="valid_loss", # Monitor valid_loss for early stopping
    early_stopping_mode = "min", # Set the mode as min because for val_loss, lower is better
    early_stopping_patience=5, # No. of epochs of degradation training will wait before terminating
    checkpoints="valid_loss", # Save best checkpoint monitoring val_loss
    checkpoints_path = '../checkpoints/', # Save the checkpoint in the experiment directory
    load_best=True, # After training, load the best checkpoint
    accelerator="mps"
)
optimizer_config = OptimizerConfig()

head_config = LinearHeadConfig(
    layers="", # No additional layer in head, just a mapping layer to output_dim
    dropout=0.1,
    initialization="kaiming"
).__dict__ # Convert to dict to pass to the model config (OmegaConf doesn't accept objects)

EXP_PROJECT_NAME = "pytorch-tabular-covertype"


# %%

model_config = CategoryEmbeddingModelConfig(
    task="classification",
    layers="1024-512-512",  # Number of nodes in each layer
    activation="LeakyReLU", # Activation between each layers
    head = "LinearHead", #Linear Head
    head_config = head_config, # Linear Head Config
    learning_rate = 1e-3,
    # metrics=["f1_score","accuracy"], 
    # metrics_params=[{"num_classes":2},{}], # f1_score needs num_classes
    # metrics_prob_input=[True, False] # f1_score needs probability scores, while accuracy doesn't
)

experiment_config = ExperimentConfig(
    project_name=EXP_PROJECT_NAME,
    run_name="CategoryEmbeddingModel",
    exp_watch="gradients",
    log_target="wandb",
    log_logits=True
)



# %% [markdown]
# ## Training the Model 

# %%
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    experiment_config=experiment_config,
    verbose=False,
    suppress_lightning_logger=True,

)

# %%
tabular_model.fit(train=train, validation=val)

# %%
result = tabular_model.evaluate(test)

# %%
result = {k: float(v) for k,v in result[0].items()}
result["mode"] = "Normal"

results.append(result)

# %% [markdown]
# ## Custom Sampler
# 
# PyTorch Tabular also allows custom batching strategy through Custom Samplers  which comes in handy when working with imbalanced data.
# 
# Although you can use any sampler, Pytorch Tabular has a few handy utility functions which takes in the target array and implements WeightedRandomSampler using inverse frequency sampling to combat imbalance. This is analogous to preprocessing techniques like Under or OverSampling in traditional ML systems.

# %%
from pytorch_tabular.utils import get_balanced_sampler, get_class_weighted_cross_entropy

# %%
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    verbose=False
)
sampler = get_balanced_sampler(train['target'].values.ravel())

tabular_model.fit(train=train, validation=val, train_sampler=sampler)


# %%
result = tabular_model.evaluate(test)

# %%
result = {k: float(v) for k,v in result[0].items()}
result["mode"] = "Balanced Sampler"

results.append(result)

# %% [markdown]
# ## Custom Weighted Loss
# 
# If Samplers were like Over/Under Sampling, Custom Weighted Loss is similar to `class_weights`. Depending on the problem, one of these might help you with imbalance. You can easily make calculate the class_weights and provide them to the CrossEntropyLoss using the parameter `weight`. To make this easier, PyTorch Tabular has a handy utility method which calculates smoothed class weights and initializes a weighted loss. Once you have that loss, it's just a matter of passing it to the 1fit1 method using the `loss` parameter.

# %%
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    verbose=False
)
weighted_loss = get_class_weighted_cross_entropy(train["target"].values.ravel(), mu=0.1)

tabular_model.fit(train=train, validation=val, loss=weighted_loss)


# %%
result = tabular_model.evaluate(test)

# %%
result = {k: float(v) for k,v in result[0].items()}
result["mode"] = "Class Weights"

results.append(result)

# %%
res_df = pd.DataFrame(results).T
res_df.columns = res_df.iloc[-1]
res_df = res_df.iloc[:-1].astype(float)
res_df.style.highlight_min(color="lightgreen",axis=1)


