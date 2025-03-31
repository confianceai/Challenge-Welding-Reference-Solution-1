
"""
# This script show how an AI component can be tested to ensure it will be compatible with the evaluation pipeline process
"""
import sys
sys.path.insert(0, "..") # Uncomment this line For local tests without pkg installation, to make challenge_welding module visible 
from challenge_welding.user_interface import ChallengeUI
from challenge_welding.inference_tools import TestAIComponent

#1) Load a dataset to test 

# In this example we will choose a small dataset

ds_name="example_mini_dataset"

# Load all metadata of your dataset as a pandas dataframe, (you can point to a local cache metafile instead of original one pointing on remote repository)

my_challenge_UI=ChallengeUI()
evaluation_ds_meta_df=my_challenge_UI.get_ds_metadata_dataframe(ds_name)

# Define path of AI component to test
AI_comp_path="../Challenge-Welding-Reference-Solution-1"
#AI_comp_path="https://github.com/confianceai/Challenge-Welding-Reference-Solution-1"

# Initialize test pipeline
my_test_pipeline=TestAIComponent(proposed_solution_path=AI_comp_path, # Set here the AI component path you want to evaluate
                              meta_root_path="starter_kit_test_pipeline_results", # set name of output directory that will store the pipeline results
                              cache_strategy="local", # In this on local, all image used for evaluation , will be locally stored in a cache directory
                              cache_dir="test_cache") # chosen directory for cache
                                                         
# Load the AI component in the test pipeline
my_test_pipeline.load_proposed_solution()

# Perform inference on AI component on evaluation dataset, inference results are stored as parquet file and as dataframe returned by the function too.

ds_name="example_mini_dataset"

# Load all metadata of your dataset as a pandas dataframe, (you can point to a local cache metafile instead of original one pointing on remote repository)

my_challenge_UI=ChallengeUI()
evaluation_ds_meta_df=my_challenge_UI.get_ds_metadata_dataframe(ds_name)

result_df=my_test_pipeline.perform_grouped_inference(evaluation_dataset=evaluation_ds_meta_df,
                                       results_inference_path=my_test_pipeline.meta_root_path+"/res_inference.parquet",
                                       batch_size=150) 

print("inference_results output")
print(result_df)
# Check the output dataframe contain columns corresponding to ouptut fields filled with correct values. 

# Here the reference component tested shall create columns named, predicted_states, scores_KO, scores OK, OOD_scores

# If your result dataframe is correct and the parquet is well created, then your AI comp is compatible with the evaluation pipeline
print("pipeline finished")
