import pandas as pd
from flask import Flask, request, json
from sklearn.datasets import load_iris

from model_utils import get_latest_model, build_model

app = Flask(__name__)


@app.route('/iris/build_model', methods=["post"])
def build_iris_model():
    return build_model()


@app.route('/iris/predict', methods=["get"])
def predict_iris():
    model = get_latest_model("iris")

    features = request.get_json()
    print(type(features), features)
    # Expected Columns: "sepal length"  "sepal width"  "petal length"  "petal width"
    # {"sepal length": 0.1,
    #  "sepal width": 0.1,
    #  "petal length": 0.1,
    #  "petal width": 0.1}

    x = pd.DataFrame(features, index=[0])[["sepal length", "sepal width", "petal length", "petal width"]]

    prediction = model.predict(x)[0]
    return load_iris().target_names[prediction]


if __name__ == '__main__':
    app.run(debug=True)
