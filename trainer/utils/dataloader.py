from pathlib import Path
from torch.utils.data import DataLoader


def build_dataloader(path,dataset="pos",batch_size,shuffle=True):
    sent  = path / f'lst20-{dataset}' / f'{dataset}.train.th'
    label = path / f'lst20-{dataset}' / f'{dataset}.train.label'
    sentTrainList  = [line.strip.split() for line in open(sent,"r").readlines()]
    labelTrainList = [line.strip.split() for line in open(label,"r").readlines()]
    trainCorpus = [[s,l] for s,l in zip(sentTrainList, labelTrainList)]
    train_dataloader = DataLoader(trainCorpus, batch_size=batch_size, shuffle=shuffle)

    sent  = path / f'lst20-{dataset}' / f'{dataset}.eval.th'
    label = path / f'lst20-{dataset}' / f'{dataset}.eval.label'
    sentTrainList  = [line.strip.split() for line in open(sent,"r").readlines()]
    labelTrainList = [line.strip.split() for line in open(label,"r").readlines()]
    evalCorpus = [[s,l] for s,l in zip(sentTrainList, labelTrainList)]
    valid_dataloader = DataLoader(evalCorpus, batch_size=batch_size, shuffle=False)

    sent  = path / f'lst20-{dataset}' / f'{dataset}.test.th'
    label = path / f'lst20-{dataset}' / f'{dataset}.test.label'
    sentTrainList  = [line.strip.split() for line in open(sent,"r").readlines()]
    labelTrainList = [line.strip.split() for line in open(label,"r").readlines()]
    testCorpus = [[s,l] for s,l in zip(sentTrainList, labelTrainList)]
    test_dataloader = DataLoader(testCorpus, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader