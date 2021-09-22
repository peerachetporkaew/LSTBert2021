from . import Task
from . import register_task
from ..models import build_model
from ..utils import build_dataloader, build_data_iterator, load_dictionaries
from pathlib import Path

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse

from fairseq.data.data_utils import collate_tokens

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res

def mycollate_tokens(pad_index):

    def myfunc(values):
        #ic("myfunc",len(values))

        return [collate_tokens([values[i][0] for i in range(len(values))],pad_idx=pad_index), 
                collate_tokens([values[i][1] for i in range(len(values))],pad_idx=pad_index),
                collate_tokens([values[i][2] for i in range(len(values))],pad_idx=pad_index)]

    return myfunc

def map_pos_to_bpe(model, batch): #model = roberta model
    src_raw = []
    tag_raw = []
    new_trg = []
    for i,s in enumerate(batch.src):
        sl = len(batch.src[i].split())
        tl = len(batch.trg[i].split())
        assert sl == tl
        trgi     = '<s> ' + batch.trg[i] + ' </s>'
        trgi     = trgi.split()
        bpe_srci = '<s> ' + model.bpe.encode(s) + ' </s>'
        
        src_raw.append(bpe_srci)
        tag_raw.append(batch.trg[i])

        bpe_srci = bpe_srci.split()
        bpe_trgi = ""
        k = 0
        for j in range(0,len(bpe_srci)):
            if not bpe_srci[j].endswith("@@"):
                bpe_trgi += trgi[k] + " "
                k += 1
            else:
                bpe_trgi += trgi[k] + " "

        new_trg.append(bpe_trgi.strip())
    return new_trg, src_raw, tag_raw

def _to_tensor(src,label,src_encoder,label_encoder):
    return src_encoder(src), label_encoder(label)

#Pytorch Lightning Trainer 

class MultiTaskTaggingModule(pl.LightningModule):

    def __init__(self, model, optimizer, criterion=None, traindata = None, validdata = None):
        super().__init__()
        self.automatic_optimization = True

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.traindata = traindata
        self.validdata = validdata
        
    def configure_optimizers(self):
        return self.optimizer
    
    def train_dataloader(self):
        return self.traindata 

    def val_dataloader(self):
        return self.validdata

    def training_step(self,train_batch, batch_idx):
        #self.optimizer.zero_grad()

        #ic(train_batch[0].size())
        #ic(train_batch[1].size())
        loss = torch.tensor([0.0]).sum()

        if True:
            predictions, _, _ = self.model.forward(train_batch[0])
        
            #Calcuate Loss
            trg = train_batch[1]
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = trg.view(-1).long()
            loss = self.criterion(predictions, tags)
            self.log('loss', loss.item(),prog_bar=True)

            #Calcuate Accuracy
            batch_prediction = predictions.argmax(dim = -1, keepdim = True) #[batch, len]
            batch_prediction = batch_prediction.squeeze(-1)

        return loss


@register_task("multitask-tagging")
class MultiTaskTagging(Task):
    
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--traindata',type=str, default="./raw_data", help='datapath')
        
        return parser

    def __init__(self,args):
        self.pad_index = 1
        pass

    def setup_task(self, args, parser):
    
        #Setup Dictionary and Pad Index
        pos_dict, ne_dict, sent_dict = load_dictionaries(args.traindata)
        self.pad_idx = pos_dict.pad_index
        outdim = [len(pos_dict.symbols), len(ne_dict.symbols), len(sent_dict.symbols) ]

        #Setup Model
        kwargs = {"output_dim" : outdim}
        model = build_model(parser, args.model, **kwargs)
        self.model = model
        ic()
        ic(model.dropout)

        #Setup Criterion
        criterion = nn.CrossEntropyLoss(ignore_index = self.pad_index)

        #Setup Optimizer
        LEARNING_RATE = args.lr
        optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE,betas=(0.9, 0.98), eps=1e-9)

        #Setup Dataset
        ic("Loading POS dataset ...")
        ic(args.batch_size)
        trainSetpos = build_data_iterator(args,args.traindata,dataset="pos",type="train")
        trainSetPos_tensor = self.convert_to_tensor(trainSetpos,
                                                    label_encoder=pos_dict.encode_line)
        trainPosData = DataLoader(trainSetPos_tensor, 
                                  batch_size=args.batch_size, 
                                  shuffle=True, 
                                  collate_fn = mycollate_tokens(self.pad_idx))

        validSetpos = build_data_iterator(args,args.traindata,dataset="pos",type="eval")
        validSetPos_tensor = self.convert_to_tensor(validSetpos,
                                                    label_encoder=pos_dict.encode_line)
        validPosData = DataLoader(validSetPos_tensor, 
                                  batch_size=args.batch_size, 
                                  shuffle=False, 
                                  collate_fn = mycollate_tokens(self.pad_idx))

        #Setup Trainer
        ic("Loading trainer ...")
        checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints/lstfinetune', monitor = 'val_loss',
                        save_top_k=5, every_n_val_epochs=1, filename="{epoch}-{step}-{val_loss:.3f}")

        earlystop_callback = EarlyStopping(monitor='val_loss', patience=8, mode='min', check_on_train_epoch_end=False,verbose=True)

        self.plmodel = MultiTaskTaggingModule(model, optimizer,criterion,trainPosData,validPosData)
        self.trainer = pl.Trainer(gpus=[0], val_check_interval=300, reload_dataloaders_every_n_epochs=1)
        return None

    def convert_to_tensor(self,data, label_encoder):
        """
        Convert String to Tensor
        """
        srcL , trgL, trgOriL = [] , [], [] #Source BPE, Target BPE, Original Target
        
        srcBPETensor = []
        trgBPETensor = []
        trgORITensor = []

        srcDict = self.model.bert.task.source_dictionary
        for idx,batch in enumerate(data):
            if idx % 100 == 0:
                ic(idx)

            trgBPEList, srcBPEList, trgORIList = map_pos_to_bpe(self.model.bert,batch)
            
            srcBPETensor.extend([srcDict.encode_line(lineT, append_eos=False, add_if_not_exist=False) for lineT in srcBPEList])

            trgBPETensor.extend([label_encoder(lineT,append_eos=False, add_if_not_exist=False) for lineT in trgBPEList])

            trgORITensor.extend([label_encoder(lineT, append_eos=False, add_if_not_exist=False) for lineT in trgORIList])

        return list(zip(srcBPETensor, trgBPETensor, trgORITensor))
    
    def train(self):
        self.trainer.fit(self.plmodel)
        return 

    def evaluate(self,args, dataset="valid"):
        pass