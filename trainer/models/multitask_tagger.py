from ..utils import build_dataloader
from . import register_model

@register_model("multitask-tagger")
class MultiTaskTagger():

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--num-layer',type=int, default=8, help='number of layer')

        return parser

    def __init__(self,args):
        self.num_layer = args.num_layer


if __name__ == "__main__":
    print("hi")