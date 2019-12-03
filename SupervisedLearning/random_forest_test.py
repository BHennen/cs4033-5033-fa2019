from data_processing import DataProcessor
import random_forest
import os

if __name__ == "__main__":
    cur_path = os.path.dirname(__file__)
    data_folder = "data"
    processed_data_folder = os.path.join(cur_path, data_folder)
    data_file_path = os.path.join(processed_data_folder, "data.csv") #TODO: update data filename
    data = DataProcessor(data_file_path, processed_data_folder)

    if data.check_if_processed():
        data.load_processed_data()
    else:
        data.process_data()

