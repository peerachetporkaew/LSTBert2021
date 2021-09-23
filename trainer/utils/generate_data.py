
import sys 
from pathlib import Path

def generate_sentence_boundary_dataset(path='./raw_data/lst20-sent'):

    """ Generate consecutive sentence dataset """
    ftrainS = open(Path.cwd() / path / "sent.train.th" ,"r").readlines()
    ftrainL = open(Path.cwd() / path / "sent.train.label" ,"r").readlines()

    otrainS = open(Path.cwd() / path / "sent1.train.th", "w")
    otrainL = open(Path.cwd() / path / "sent1.train.label", "w")

    





    print(len(ftrainS))



if __name__ == "__main__":
    generate_sentence_boundary_dataset()