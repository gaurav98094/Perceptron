from utils.model import Perceptron
from utils.all_utils import save_model,save_plot,prepare_data

import pandas as pd

AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,0,0,1],
}

df = pd.DataFrame(AND)


X,y = prepare_data(df)

ETA = 0.01 # 0 and 1
EPOCHS = 20

model = Perceptron()
model.fit(X, y,eta=ETA,epochs=EPOCHS)

save_model(model,'AND')

save_plot(df, "AND.png", model)