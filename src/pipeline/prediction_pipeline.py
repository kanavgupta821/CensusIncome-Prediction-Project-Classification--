import sys
import os
project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_directory)

from exception import CustomException
from logger import logging
from utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
        
        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        

# ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
# 'marital_status', 'occupation', 'relationship', 'race', 'sex',
# 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country'],
class CustomData:
    def __init__(self,
                 age:float,
                 workclass:str,
                 fnlwgt:float,
                 education:str,
                 education_num:int,
                 marital_status:str,
                 occupation:str,
                 relationship:str,
                 race:str,
                 sex:str,
                 capital_gain:float,
                 capital_loss:float,
                 hours_per_week:int,
                 native_country:str
                ):
        
        self.age = age
        self.workclass = workclass
        self.fnlwgt = fnlwgt
        self.education = education
        self.education_num = education_num
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week
        self.native_country = native_country
    
    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'age':[self.age],
                'workclass':[self.workclass], 
                'fnlwgt':[self.fnlwgt], 
                'education':[self.education], 
                'education_num':[self.education_num],
                'marital_status':[self.marital_status], 
                'occupation':[self.occupation], 
                'relationship':[self.relationship], 
                'race':[self.race], 
                'sex':[self.sex],
                'capital_gain':[self.capital_gain], 
                'capital_loss':[self.capital_loss], 
                'hours_per_week':[self.hours_per_week], 
                'native_country':[self.native_country]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)