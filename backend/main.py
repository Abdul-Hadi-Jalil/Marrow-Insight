from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


from prediction import predict_single_image
from Resnet_predictor import resnet_predict_single_image
from Mobilenet_predictor import mobilenet_predict_single_image


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    model_name: str
    image_path: str


@app.post("/predict")
async def predict(request_data: PredictionRequest):
    if not request_data.model_name or not request_data.image_path:
        raise HTTPException(status_code=400, detail="Missing model_name or image_path")

    processed_path = request_data.image_path
    if not processed_path.startswith(('r"', "r'")):
        processed_path = r"{}".format(processed_path)

    if request_data.model_name == "Inception":
        prediction_result = predict_single_image(processed_path)
    elif request_data.model_name == "ResNet":
        prediction_result = resnet_predict_single_image(processed_path)
    elif request_data.model_name == "MobileNet":
        prediction_result = mobilenet_predict_single_image(processed_path)

    if "error" in prediction_result:
        raise HTTPException(status_code=400, detail=prediction_result["error"])

    return {
        "cell_type": prediction_result["cell_type"],
        "confidence": prediction_result["confidence"],
    }
