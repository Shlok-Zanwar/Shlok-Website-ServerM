from pickle import load
import numpy as np
import pandas as pd

workclass  = ["State-gov", "Self-emp-not-inc", "Private", "Federal-gov", "Local-gov", "Self-emp-inc", "Without-pay"]
education  = ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", "Assoc-acdm", "7th-8th", "Doctorate", "Assoc-voc", "Prof-school", "5th-6th", "10th", "Preschool", "12th", "1st-4th"]
marital_status  = ["Never-married", "Married-civ-spouse", "Divorced", "Married-spouse-absent", "Separated", "Married-AF-spouse", "Widowed"]
occupation  = ["Adm-clerical", "Exec-managerial", "Handlers-cleaners", "Prof-specialty", "Other-service", "Sales", "Transport-moving", "Farming-fishing", "Machine-op-inspct", "Tech-support", "Craft-repair", "Protective-serv", "Armed-Forces", "Priv-house-serv"]
relationship  = ["Not-in-family", "Husband", "Wife", "Own-child", "Unmarried", "Other-relative"]
race  = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
sex  = ["Male", "Female"]
native_country  = ["United-States", "Cuba", "Jamaica", "India", "Mexico", "Puerto-Rico", "Honduras", "England", "Canada", "Germany", "Iran", "Philippines", "Poland", "Columbia", "Cambodia", "Thailand", "Ecuador", "Laos", "Taiwan", "Haiti", "Portugal", "Dominican-Republic", "El-Salvador", "France", "Guatemala", "Italy", "China", "South", "Japan", "Yugoslavia", "Peru", "Outlying-US(Guam-USVI-etc)", "Scotland", "Trinadad&Tobago", "Greece", "Nicaragua", "Vietnam", "Hong", "Ireland", "Hungary", "Holand-Netherlands"]

model = load(open('Models/income_model.pkl', 'rb'))
transformer = load(open('Models/income_transformer.pkl', 'rb'))

def handleIncomePrediction(fieldsData):
    # fieldsData = [39, "State-gov", 77516, "Bachelors", 13, "Never-married", "Adm-clerical", "Not-in-family",
    #                    "White", "Male", 2174, 0, 40, "United-States"]
    array = np.array([ fieldsData ])
    column_values = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                     'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

    newDfToPredict = pd.DataFrame(data=array,
                                  columns=column_values)
    newDfToPredict["age"] = pd.to_numeric(newDfToPredict["age"])
    newDfToPredict["fnlwgt"] = pd.to_numeric(newDfToPredict["fnlwgt"])
    newDfToPredict["education-num"] = pd.to_numeric(newDfToPredict["education-num"])
    newDfToPredict["capital-gain"] = pd.to_numeric(newDfToPredict["capital-gain"])
    newDfToPredict["capital-loss"] = pd.to_numeric(newDfToPredict["capital-loss"])
    newDfToPredict["hours-per-week"] = pd.to_numeric(newDfToPredict["hours-per-week"])

    encoded = transformer.transform(newDfToPredict)
    print(encoded)
    return model.predict(encoded)[0]
