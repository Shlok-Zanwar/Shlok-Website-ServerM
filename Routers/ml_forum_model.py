from fastapi import APIRouter, File
# from Functions.income_functions import handleIncomePrediction
from Functions.mnist_functions import handleMnistPrediction
from pydantic import BaseModel

router = APIRouter(
    tags=['ML Forum Model']
)

@router.post("/models/mnist")
async def create_file(file: bytes = File(...) ):
    # time.sleep(1.5)
    return {"prediction": handleMnistPrediction(file)}


class IncomeModelSchema (BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

@router.post("/models/income_classification")
async def create_file_3(
        schema: IncomeModelSchema
):
    # time.sleep(1.5)
    return {"prediction": handleIncomePrediction(
        [
            schema.age,
            schema.workclass,
            schema.fnlwgt,
            schema.education,
            schema.education_num,
            schema.marital_status,
            schema.occupation,
            schema.relationship,
            schema.race,
            schema.sex,
            schema.capital_gain,
            schema.capital_loss,
            schema.hours_per_week,
            schema.native_country,
        ]
    )}