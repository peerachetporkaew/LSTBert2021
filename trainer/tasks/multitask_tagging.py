from . import Task
from . import register_task
from ..models import build_model
from ..utils import build_dataloader
from pathlib import Path


import argparse

@register_task("multitask-tagging")
class MultiTaskTagging(Task):
    
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--datapath',type=str, default="./raw_data", help='datapath')

        return parser

    def __init__(self,args):

        pass

    def setup_task(self, args, parser):
        #Setup Model

        model = build_model(parser, args.model)
        ic()
        ic(model.num_layer)

        #Setup Criterion

        #Setup Optimizer

        #Setup Trainer

        #Setup Dataset
        self.datapath = args.datapath
        self.posdata = Path.cwd() / self.datapath / 'lst20-pos' / 'pos.eval.th'

        fp = open(self.posdata,"r").readlines()
        print(len(fp))
        return None
    
    def train(self,args):
        pass

    def evaluate(self,args, dataset="valid"):
        pass