"""
This script show how to use this package to create a pytorch dataloader
It requires the installation of pytorch==2.6.0
"""

import sys
sys.path.insert(0, "..") # For local tests without pkg installation, to make challenge_welding module visible 

import challenge_welding.dataloaders
from challenge_welding.user_interface import ChallengeUI
from challenge_welding.dataloaders import ChallengeWeldingDataset


# Initiate the user interface

my_challenge_UI=ChallengeUI(cache_strategy="local",cache_dir="loadercache")

# Get list of available datasets

ds_list=my_challenge_UI.list_datasets()
print(ds_list)

# In this example we will choose a small dataset

ds_name="example_mini_dataset"

# Load all metadata of your dataset

meta_df=my_challenge_UI.get_ds_metadata_dataframe(ds_name)

# Initialize the ChallengeWeldingDataset
dataset = ChallengeWeldingDataset(
    user_ui=my_challenge_UI,  # The initialized ChallengeUI
    meta_df=meta_df[0:20],  # First 20 rows of metadata 
    resize=(256, 256),  # adjust if needed
    transform=None  # Optional transformations, set to None if no transformation
)

# Create your dataloader
dataloader = challenge_welding.dataloaders.create_pytorch_dataloader(
    dataset=dataset,
    batch_size=100,
    shuffle=False,
    num_workers=0
)
# Test your dataloader       
for i_batch, sample_batched in enumerate(dataloader):
    print("batch number", i_batch)
    print("batch content image",    sample_batched['image'].shape)
    print("batch content meta",sample_batched['meta'])

    # observe 4th batch and stop.
    if i_batch == 3:
        break