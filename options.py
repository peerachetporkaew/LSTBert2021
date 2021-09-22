import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    #Task and Model Options
    parser.add_argument('--task', type=str, default="multitask-tagging",
                        help='select from {multitask-tagging, tagging}')
    parser.add_argument('--do', default=None,help='select from {train , test or valid }')
    parser.add_argument('--model', type=str, default="multitask-tagger",
                        help='type of model') # single task, multitask
    parser.add_argument('--pretrained', default="lst",
                        help='select from {lst , wangchan, mbert, thaibert }')

    #Training Options
    parser.add_argument('--max-epochs', metavar='N', type=int,
                        help='an integer for the maximum train epochs')
    parser.add_argument('--batch-size', metavar='N', type=int, default=32,
                        help='an integer for the batch size')
    parser.add_argument('--lr', type=float, help='learning rate', default=5e-5)
    
    
    #Checkpoint Options
    parser.add_argument('--reload-checkpoint', default=None,
                        help='restore weight from a specified checkpoint')

    return parser