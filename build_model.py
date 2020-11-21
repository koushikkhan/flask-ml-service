import os 
from config import DATA_PATH, MODEL_PATH, MODEL_FILE_NAME
from algo import train_model, infer, load_data


def main():
    train_model(data_path=DATA_PATH, data_fname='iris_data.csv', model_path=MODEL_PATH, model_fname=MODEL_FILE_NAME)
    print("training completed")

    test = load_data(data_path=DATA_PATH, data_fname='test.csv')
    
    pred_proba = infer(model_path=MODEL_PATH, model_fname=MODEL_FILE_NAME, sample=test)
    print("prediction generated")
    print(pred_proba)


if __name__ == "__main__":
    main()