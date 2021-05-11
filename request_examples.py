import json
import requests

url = "http://127.0.0.1:5000/"

data = {"sepal length": 0.1,
        "sepal width": 0.1,
        "petal length": 0.1,
        "petal width": 0.1}
data_json = json.dumps(data)

headers = {"content-type": "application/json",
           "Accept-Charset": "UTF-8",
           }

r = requests.get(
    url + "/iris/predict",
    data=data_json,
    headers=headers,
)

print(r.status_code, r.text)
# 200 setosa

