
## Dataset metadescriptoin

All datasets accessible in the challenge are described with a parquet file containing the metadescription of a sample
A parquet file is a format  representing a dataframe. For each sample the following fields are acessible :

| **Field**             | **Description** |
|----------------------|----------------|
|sample_id|Unique identifier for the sample. It follows the template "data_X" |
|class|Real state of the welding present on the image, this is the ground_truth . Two values are possible OK or KO|
|timestamp |Datetime where the photo has been taken, this field shall not be useful  |
|welding-seams | Name of the welding seams whose welding belongs to . The welding-seams are named "c_X"|
|labelling_type | Type of human that annotated the data . two possible values : "expert" or "operator"|
|resolution | List contining the resolution of the image [width, height]|
|path | Internal path of the image in the challenge storage |
|sha256 | Sha256 of the image . It's a unique hexadecimal key representing the image data. This is used to detect alteration of corruption on the storage|
|storage_type |Type of storage where sample is stored : "s3" or "filesystem" |
|data-origin | Type of data. This field has two possible values (real or synthetic)|
|blur_level | Level of blur on the image. This measure has been made numerically using opencv library. Lesser is this value , blurer the image is .|
|blur_class | Class of blur deduced from blur-level field. Two class are considered "blur", and "clean".  The value is set to "blur" when blur level is inferior to 950|
|lumninosity_level | Percentage of luminosity of the image, mesured numerically|
|external_path | Url of the image. This url shall be used by Challengers to directly download the sample from the dataset from storage|

## Dataset example

### Example_mini_dataset
For now, a first example of dataset is provided . Purpose of this dataset named "example_mini_dataset" is to a give an overview of the final dataset that would be provided for the official start of this challenge.
This dataset contains 2857 images of weldings splitted into 3 different welding-seams (c102,c20,c33).
The metadata file of this dataset can be found here : [example_mini_dataset metadata](https://minio-storage.apps.confianceai-public.irtsysx.fr/challenge-welding/datasets/example_mini_dataset/metadata/ds_meta.parquet)

This an example of the first 9 rows of this metadescription file

![meta example](assets/meta_example.png)

The whole dataset can be downloaded directly as a zip file : [Download example_mini_dataset](https://minio-storage.apps.confianceai-public.irtsysx.fr/challenge-welding/datasets/example_mini_dataset.zip)]

#### Welding-detection-challenge-dataset

The final dataset provided with this challenge is named ```welding-detection-challenge-dataset```. It contains 22753 images of weldings covering three different welding-seams named c20, c102 anc c33.
The metadata file of this dataset can be found here : [welding-detection-challenge-dataset metadata](https://minio-storage.apps.confianceai-public.irtsysx.fr/challenge-welding/datasets/welding-detection-challenge-dataset/metadata/ds_meta.parquet)

The whole dataset can be downloaded directly as a zip file : [Download welding-detection-challenge-dataset](https://minio-storage.apps.confianceai-public.irtsysx.fr/challenge-welding/datasets/welding-detection-challenge-dataset.zip)]