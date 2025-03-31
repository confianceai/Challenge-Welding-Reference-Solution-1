# Expected solution specification

As it was explained in the main presentation, the aim of this challenge is to build an AI component which take an image of welding as input and return as output , the predicted class of the welding (OK , KO) or UNKNOWN . UNKNOWN is used to say that model is not sure about the predicted class. As we will see later in the evaluation process an unknown output can be less penalizing than a False Negative (meaning a true KO predicted as OK) that have a critical cost.

From the evaluation process point of view , the AI component is considered as a black box. It means that the full evaluation process is only based on inference results of the submitted AI component on different datasets. It is not needed to access to the AI component architecture, layers, or gradient of neural network.

Thus the minimal requirement for any solution submitted is to have the following requirement.

The submitted zip folder to codabench by participant shall have the following requirements:

A "challenge_solution "folder shall contain at least three files :
- ```__init__.py``` : This is a file that is necessary to ensure that all files in this directory will be integrated to the python package of your component when it will be built .
- ```AIComponent.py``` : The interface module of your AI component that shall contain a class named ```MyAIComponent``` describing the interface of your component.

This class shall have at least two main methods that will be used by the evaluation pipeline named :
- load_model() : load whatever is needed in the virtual env to be able to use the predict method .
- predict() : Perform predictions on a list of input images, and return the list of predictions.

- A ```requirements.txt``` file listing all dependencies with exact versions of needed packages.
- A ```setup.py``` file that will be used to build the python package of your AI component. This setup shall use the requirement.txt file.
- A ```MANIFEST.IN``` file that is needed to  list all additionnal files you may have added to be used in your  ```challenge_solution``` folder to ensure that they will be integrated in your AI component python package

This interface class statisfy the following abstract class given in file challenge-welding/AIComponent_interface.py

```
class AbstractAIComponent(ABC):
    """
    Abstract base class for AI components interface.
    """
    def __init__(self):
        self.AI_Component=None
        self.AI_component_meta_informations={}
    @abstractmethod
    def load_model(self,config_file=None):
        """
        Abstract method to load the model into the AI Component.
        Should be implemented by all subclasses.
        
        Parameters:
            config file : str
                A optional config file that could be used for the AI component loading
        
        """
        pass

    @abstractmethod
    def predict(self, input_images: list[np.ndarray], images_meta_informations: list[dict]) -> dict :
        """
        Abstract method to make predictions using the AI component.
        
        Parameters:
            input_images: list[np.ndarrays]
                The list of images numpy arrays
            images_meta_information: list[dict]
                The list of images metadata dictionaries 
                
        Returns:
            A dict containing 4 keys "predictions", "probabilities", "OOD_scores"(optional),"explainability"(optional). 
                predictions : A list of the predictions given by the AI component among 3 possible values [KO, OK UNKNOWN"]
                probabilities : A list of 3-values lists containing predicted scores for each sample in this order [proba KO, proba OK, proba UNKNOWN]. sum of proba shall be 1 for each lists  
                OOD_scores : A list of  OOD score predicted by the AI component for each sample. An ood score is a real positive number. The image is considered OOD when this score is >=1
                explainabilities  : A list of explainabilities for each sample . An explainability is an intensity matrix (a numpy array contaning only real numbers between 0 and 1) representing the importance level of each pixel on input image in the decision leading to the final prediction of your AI component. 
        """
        pass


```
To sum-up, your AI component shall have at least the following files and folder :

```
setup.py
MANIFEST.in
requirements.txt
challenge_solution/
    __init__.py
	AIComponent.py
```
You are free to add any other files you need to this structure in order to make your AI component working.

## Example of AI Component

An example of such AI component is provided in this repository in this folder ```reference-solutions/Solution-1/ ```.
This AI component is not designed to have good performance but just to show what is expected to be compatible with the evaluation pipeline
An example of script loading this AI component and test it via inference computations is given [here](../examples/03-Test_AIComponent.ipynb)

# Evaluation process

The submitted AI component will be evaluated according to different quality evaluation metrics like:

- Operationnal metrics: Measure the gain brought the AI component compared to a human only qualification process. This metrics is based on the confusion matrix and penalize strongly false negative predictions.

- Uncertainty metrics: Measure the ability of the AI component to produce a calibrated prediction confidence indicator expressing risk of model errors.

- Robustness metrics: Measure the ability of the AI component to have invariant output is images have slight perturbations (blut, luminosity, rotation, translation)

- Monitoring metrics: Measure the ability of the AI component to detect if an input image is ood, and gives the appropriate output ->Unknown

- Explainability metrics: Measure the ability of the AI component to give appropriate explanations
 

