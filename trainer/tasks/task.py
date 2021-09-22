import argparse

class Task(object):

    def __init__(self,args):
        pass

    def get_parser(self,args):
        return None

    def setup_task(self,args):
        return None
    
    def train(self,args):
        pass

    def evaluate(self,args, dataset="valid"):
        pass

    