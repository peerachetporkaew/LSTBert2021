from trainer.utils import build_dataloader
from trainer.models import MultiTaskTagger
from trainer.tasks import MultiTaskTagging

from options import get_parser

import random 
import numpy as np 
import torch 

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    print("HELLO")