"""
This script is an example to show how to list available datasets, explore basic data properties, and load an image from the repository
"""
import sys
import challenge_welding
# sys.path.insert(0, "..") # Uncomment this line For local tests without pkg installation, to make challenge_welding module visible 

from challenge_welding.user_interface import ChallengeUI

# Initiate the user interface

# Cache strategy can be "local" or "remote". If it set on "local" each image read on repository will be stored first time in a local cache folder to be directly used in next requests involving it. cache_dir is the path to the folder you want to use locally to store cache images.

my_challenge_UI=ChallengeUI(cache_strategy="local",cache_dir="notebooks_cache")

# Get list of available datasets

ds_list=my_challenge_UI.list_datasets()
print(ds_list)

# In this example we will choose a small dataset

ds_name="example_mini_dataset"

# Load all metadata of your dataset as a pandas dataframe

meta_df=my_challenge_UI.get_ds_metadata_dataframe(ds_name)

# Open an image in your dataset

sample_idx=10 # We choose an sample in the dataset, the sample at index 10 for this example
sample_meta=meta_df.iloc[sample_idx] # Get metadata of the chosen sample

print("opening image metadata with idx ..", sample_idx)
print(sample_meta)

# You may want to see the different type resolution of image in the dataset

meta_df["resolution"]=meta_df["resolution"].astype(str)
print(meta_df["resolution"].value_counts())

# With this dataframe you can explore, and draw statistics. For example, you can compute the repartition of weld classes

print(meta_df["class"].value_counts())

# You may  want to see the class distribution for each welding-seams , or the blur distributoin

print(meta_df.groupby(["welding-seams","class"]).count()["sample_id"])

print(meta_df.groupby(["welding-seams","blur_class"]).count()["sample_id"])

# Or you may want ot see the distribution of blur level and luminosity overs each welding-seams

print(meta_df.groupby(["welding-seams"])[["blur_level","luminosity_level"]].describe())


# Now we open the selected image
img=my_challenge_UI.open_image(sample_meta["path"])

print("size of the opened image", img.shape)

# Check integrity of all files in your dataset . This step could may take a while. That is why the following line has been commented but you can uncomment it .

# anomlie_list=my_challenge_UI.check_integrity(ds_name)