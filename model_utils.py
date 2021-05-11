import os

from datetime import datetime
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
from sklearn.pipeline import Pipeline

MODEL_PATH = f"{os.getcwd()}/model"


def build_model():
    iris = load_iris(as_frame=True)
    x = iris.data
    x.columns = [str(col).replace(" (cm)","") for col in x.columns]
    y = iris.target
    print(x.head(1))
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42, stratify=y)

    # Build the Model
    scaler = StandardScaler()
    model = LogisticRegression()
    model_pipeline = Pipeline([("scale", scaler), ("model", model)])

    model_pipeline.fit(x_train, y_train)

    y_test_pred = model_pipeline.predict(x_test)
    acc = accuracy_score(y_test, y_test_pred)
    print("accuracy_score", acc)

    model_name = "iris"
    training_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_location = f"{MODEL_PATH}/{model_name} | {training_date_time}.pkl"
    pickle.dump(model_pipeline, open(model_location, "wb"))

    msg = f"Model with {acc} accuracy, saved into {model_location}."
    print(msg)
    return msg


def get_latest_model(model_name: str):
    models = [filename for filename in os.listdir(MODEL_PATH) if "pkl" in filename and filename.startswith(model_name)]
    models.sort()
    latest_model_name = models[-1]
    latest_model = pickle.load(open(os.path.join(MODEL_PATH, latest_model_name), 'rb'))
    print(latest_model_name, "is loaded.")
    return latest_model


if __name__ == "__main__":
    build_model()


