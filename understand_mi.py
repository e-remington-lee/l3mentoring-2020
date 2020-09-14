
import numpy as np


list1 = {1: 100, 2: 200, 3: 300}

features = {key:np.array(value) for key,value in dict(list1).items()}


predictions = [{
        "predictions": [1,2,3],
        "derek": 3
    }]


def idk(predictions):
    for item in predictions:
        x = item['predictions'][0]
        return x

print(idk(predictions))