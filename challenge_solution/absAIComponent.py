from abc import ABC, abstractmethod
import numpy as np

class AbstractAIComponent(ABC):
    """
    Abstract base class for AI components.
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
                explainabilities  : a list of expllainabilities for each sample . An explainability , is an intensity matrix (a numpy array contaning only real numbers between 0 and 1)
        """
        pass
