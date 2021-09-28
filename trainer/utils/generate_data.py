
from os import remove
import sys 
from pathlib import Path
import random

def generate_sentence_boundary_dataset(path='./raw_data/lst20-sent'):

    for subset in ["train", "eval" ,"test"]:
        """ Generate consecutive sentence dataset """
        ftrainS = open(Path.cwd() / path / f"sent.{subset}.th" ,"r").readlines()
        ftrainL = open(Path.cwd() / path / f"sent.{subset}.label" ,"r").readlines()

        otrainS = open(Path.cwd() / path / f"sent1.{subset}.th", "w")
        otrainL = open(Path.cwd() / path / f"sent1.{subset}.label", "w")

        for i in range(0,len(ftrainS),2):
            new_src = ftrainS[i].strip() + " _ " + ftrainS[i+1].strip()
            new_lab = ftrainL[i].strip() + " MARK " + ftrainL[i+1].strip()

            otrainS.writelines(new_src + "\n")
            otrainL.writelines(new_lab + "\n")

            if subset != "test":

                otrainS.writelines(new_src + "\n")
                otrainL.writelines(new_lab + "\n")
        
        otrainS.close()
        otrainL.close()

def generate_shuffle_sentence_boundary_dataset(path='./raw_data/lst20-sent'):

    for subset in ["train", "eval" ,"test"]:
        """ Generate consecutive sentence dataset """
        ftrainS = open(Path.cwd() / path / f"sent.{subset}.th" ,"r").readlines()
        ftrainL = open(Path.cwd() / path / f"sent.{subset}.label" ,"r").readlines()

        dataset = list(zip(ftrainS,ftrainL))
        
        random.shuffle(dataset)

        otrainS = open(Path.cwd() / path / f"sent2.{subset}.th", "w")
        otrainL = open(Path.cwd() / path / f"sent2.{subset}.label", "w")

        for i in range(0,len(dataset),2):
            new_src = dataset[i][0].strip() + " _ " + dataset[i+1][0].strip()
            new_lab = dataset[i][1].strip() + " MARK " + dataset[i+1][1].strip()

            otrainS.writelines(new_src + "\n")
            otrainL.writelines(new_lab + "\n")

            if subset != "test":
                otrainS.writelines(new_src + "\n")
                otrainL.writelines(new_lab + "\n")
        
        otrainS.close()
        otrainL.close()

def remove_duplicate(fin):

    fp = open("./raw_data/lst20-sent2/" + fin,"r").readlines()
    fo = open("./raw_data/lst20-sent2_new/" + fin,"w")
    for id, line in enumerate(fp):
        if id % 2 == 0:
            fo.writelines(line)
    fo.close()

if __name__ == "__main__":

    remove_duplicate("sent2.train.label")
    remove_duplicate("sent2.train.th")

    remove_duplicate("sent2.eval.label")
    remove_duplicate("sent2.eval.th")

    remove_duplicate("sent2.test.label")
    remove_duplicate("sent2.test.th")

    #generate_sentence_boundary_dataset()
    #generate_shuffle_sentence_boundary_dataset()