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
                collate_tokens([values[i][2] for i in range(len(values))],pad_idx=pad_index),
                collate_tokens([values[i][3] for i in range(len(values))],pad_idx=0)]       # masking

    return myfunc

def map_pos_to_bpe(model, batch): #model = roberta model
    src_raw = []
    tag_raw = []
    new_trg = []
    mask = []
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
        mask_trg = []
        k = 0
        for j in range(0,len(bpe_srci)):
            if not bpe_srci[j].endswith("@@"):
                bpe_trgi += trgi[k] + " "
                mask_trg.append(1)
                k += 1
            else:
                bpe_trgi += trgi[k] + " "
                mask_trg.append(0)

        new_trg.append(bpe_trgi.strip())

        #mask_trg  [0,1] keep (1) output of that position or not (0)
        #for calucating tagging accuracy 
        mask.append(mask_trg)               
    return new_trg, src_raw, tag_raw, mask

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
        loss = torch.tensor([0.0]).sum()
        if True:
            predictions, _, _ = self.model.forward(train_batch[0])
        
            #Calcuate Loss
            trg = train_batch[1]
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = trg.view(-1).long()
            loss = self.criterion(predictions, tags)
            self.log('train_loss', loss.item(),prog_bar=True)

            #Calcuate Accuracy
            #batch_prediction = predictions.argmax(dim = -1, keepdim = True) #[batch, len]
            #batch_prediction = batch_prediction.squeeze(-1)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = torch.tensor([0.0]).sum()
        if True:
            predictions, _, _ = self.model.forward(val_batch[0][:,0:500])
        
            #Calcuate Loss
            trg = val_batch[1]
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = trg[:,0:500].reshape((-1,)).long()
            loss = self.criterion(predictions, tags)
            self.log('val_loss', loss.item())
            #Calculate Accuracy
            batch_prediction = predictions.argmax(dim = -1, keepdim = True) #[batch, len]
            batch_prediction = batch_prediction.squeeze(-1).cpu()

            actual = tags.cpu()
            mask = val_batch[3][:,0:500].reshape((-1,))

            loss_item = loss.item()

            acc = batch_prediction == actual 
            acc = acc * mask.cpu()

            correct = acc.sum().item()
            all = mask.sum().item()

        return {'loss' : loss_item, 'pred' : batch_prediction, 
                'actual' : trg, 'correct' : correct, 'all' : all}

    def validation_epoch_end(self, val_step_outputs):
        correct = 0
        all = 0
        for out in val_step_outputs:
            correct += out["correct"]
            all += out["all"]

        acc = correct / all * 100
        self.log("val_acc",acc)

@register_task("multitask-tagging")
class MultiTaskTagging(Task):
    
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--traindata',type=str, default="./raw_data", help='datapath')
        parser.add_argument('--valid-interval',type=int, default="500", help='validation interval')
        parser.add_argument('--sample', action="store_true")
        return parser

    def __init__(self,args):
        self.pad_index = 1
        pass

    def setup_task(self, args, parser):
    
        self.loadall = not args.sample

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
        checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints/lstfinetune/', monitor = 'val_loss',
                        save_top_k=5, every_n_val_epochs=1, filename="{epoch}-{step}-{val_loss:.3f}-{val_acc:.3f}")

        earlystop_callback = EarlyStopping(monitor='val_loss', patience=8, mode='min', check_on_train_epoch_end=False,verbose=True)

        callbacks = [checkpoint_callback,earlystop_callback]
        #callbacks = []
        
        self.plmodel = MultiTaskTaggingModule(model, optimizer,criterion,trainPosData,validPosData)
        self.trainer = pl.Trainer(gpus=[0], val_check_interval=args.valid_interval, 
        reload_dataloaders_every_n_epochs=1,callbacks=callbacks)

        return None

    def convert_to_tensor(self,data, label_encoder):
        """
        Convert String to Tensor
        """
        srcL , trgL, trgOriL = [] , [], [] #Source BPE, Target BPE, Original Target
        
        srcBPETensor = []
        trgBPETensor = []
        trgORITensor = []
        mskBPETensor = []

        srcDict = self.model.bert.task.source_dictionary
        for idx,batch in enumerate(data):
            if idx % 100 == 0:
                ic(idx)

            if not self.loadall:
                if idx == 500:
                    break

            trgBPEList, srcBPEList, trgORIList, mskBPEList = map_pos_to_bpe(self.model.bert,batch)
            #ic(mskBPEList)
            
            srcBPETensor.extend([srcDict.encode_line(lineT, append_eos=False, add_if_not_exist=False) for lineT in srcBPEList])

            trgBPETensor.extend([label_encoder(lineT,append_eos=False, add_if_not_exist=False) for lineT in trgBPEList])

            trgORITensor.extend([label_encoder(lineT, append_eos=False, add_if_not_exist=False) for lineT in trgORIList])

            mskBPETensor.extend([torch.tensor(msk,dtype=torch.int32) for msk in mskBPEList])

        return list(zip(srcBPETensor, trgBPETensor, trgORITensor,mskBPETensor))
    
    def train(self):
        self.trainer.fit(self.plmodel)
        return 

    def evaluate(self,args, dataset="valid"):
        pass