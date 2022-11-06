import pandas as pd 
import numpy as np 
import pickle
import json


class TitanicSurvival():

    def __init__(self,PassengerId,Pclass,Gender,Age,SibSp,Parch,Fare,Embarked):
        self.PassengerId=PassengerId
        self.Pclass=Pclass
        self.Gender=Gender
        self.Age=Age
        self.SibSp=SibSp
        self.Parch=Parch
        self.Fare=Fare
        self.Embarked="Embarked_"+Embarked
        


    def Load_Files(self):

        with open("models/logistic_model_titanic.pkl","rb")as f:
            self.logistic_model=pickle.load(f) 

        with open("models/project_data_titanic.json","r")as f:
            self.project_data=json.load(f)      

    def Survival_prediction(self):

        self.Load_Files()
        array=np.zeros(len(self.project_data["columns"]))

        index_value=self.project_data["columns"].index(self.Embarked)

        array[0]=self.PassengerId
        array[1]=self.Pclass
        array[2]=self.project_data["Gender"][self.Gender]
        array[3]=self.Age
        array[4]=self.SibSp
        array[5]=self.Parch
        array[6]=self.Fare
        array[index_value]=1

        predection=self.logistic_model.predict([array])[0]
        print(predection)

        if predection==1:
            return "PERSON IS SURVIVED"

        else:
            return "PERSON IS NOT SURVIVED"    


        

    
if __name__=="__main__":
    PassengerId=1.00
    Pclass=3.00
    Gender="male"
    Age=22
    SibSp=1.00
    Parch=0.00
    Fare=7.25
    Embarked="C"

    pred_survival=TitanicSurvival(PassengerId,Pclass,Gender,Age,SibSp,Parch,Fare,Embarked)
    survival=pred_survival.Survival_prediction()
    print("preson survival  status in titanic",survival)