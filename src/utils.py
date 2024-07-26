
import os
import pickle


def save_file_as_pickle(path,name):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path,"wb") as path:
        pickle.dump(name,path)
        print("\t\t\t\t\t\t\tFile saved in pickle format")


def load_pickle_file(path):
    with open(path,"rb") as path:
        return pickle.load(path)