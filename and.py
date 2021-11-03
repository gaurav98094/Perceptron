
from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd

import os

import logging
logging_str = " [ %(asctime)s:%(levelname)s:%(module)s:%(message)s ] "
log_dir="logs"
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO,format=logging_str,filemode="a")


def main(data, modelName, plotName, eta, epochs):
    df = pd.DataFrame(data)
    X, y = prepare_data(df)
    model = Perceptron()
    model.fit(X, y,eta=eta, epochs=epochs)
    save_model(model, filename=modelName)
    save_plot(df, plotName, model)

if __name__ == '__main__':
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,0,0,1],
    }
    ETA = 0.3 # 0 and 1
    EPOCHS = 20
    try:
        main(data=AND, modelName="AND.model", plotName="AND.png", eta=ETA, epochs=EPOCHS)
        logging.info("Successful \n\n\n")
    except Exception as e:
        logging.exception(e)
        raise e