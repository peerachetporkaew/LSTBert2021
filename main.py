import logging
logging.basicConfig(level=logging.INFO)

from trainer.utils import build_dataloader

from trainer.models import build_model
from trainer.tasks import build_task
from trainer.tasks import MultiTaskTagging

from options import get_parser

import random 
import numpy as np 
import torch 

from icecream import install
install()

def info(s):
    logging.info(s)

ic.configureOutput(outputFunction=info)

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    parser = get_parser()
    args, _ = parser.parse_known_args()
    print(args)

    task = build_task(args, parser, args.task) # Return task object
    if args.do == "train":
        task.train()
    elif args.do == "test":
        task.evaluate()
    elif args.do == "load":
        task.evaluate()
