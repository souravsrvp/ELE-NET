import warnings
warnings.filterwarnings("ignore")
import os
import torch
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch import loggers as pl_loggers
from ELE_Net import ELE_Net
from dataset2 import ELE_Dataset
from argparse import ArgumentParser
import torchmetrics
import pandas as pd
import torchvision
import io
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class IntHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        text = plt.matplotlib.text.Text(x0, y0, str(orig_handle))
        handlebox.add_artist(text)
        return text
    

class LITELE(L.LightningModule):
    def __init__(self, ELE_Net, lr):
        super().__init__()
        self.ELE_Net = ELE_Net
        self.lr = lr
        self.validation_step_outputs = []
        self.train_step_outputs = []
        # self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=5)
        # self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=5)
        # self.train_pr = torchmetrics.classification.Precision(task="multiclass", num_classes=5, average='weighted')
        # self.valid_pr = torchmetrics.classification.Precision(task="multiclass", num_classes=5, average='weighted')
        # self.train_re = torchmetrics.classification.Recall(task="multiclass", num_classes=5, average='weighted')
        # self.valid_re = torchmetrics.classification.Recall(task="multiclass", num_classes=5, average='weighted')
        # self.train_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=5, average='weighted')
        # self.valid_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=5, average='weighted')
        # self.confusion_val = torchmetrics.ConfusionMatrix(num_classes=5, task='multiclass')
        # self.confusion_train = torchmetrics.ConfusionMatrix(num_classes=5, task='multiclass')

        self.train_accl = torchmetrics.classification.Accuracy(task="multiclass", num_classes=5)
        self.valid_accl = torchmetrics.classification.Accuracy(task="multiclass", num_classes=5)
        self.train_prl = torchmetrics.classification.Precision(task="multiclass", num_classes=5, average='weighted')
        self.valid_prl = torchmetrics.classification.Precision(task="multiclass", num_classes=5, average='weighted')
        self.train_rel = torchmetrics.classification.Recall(task="multiclass", num_classes=5, average='weighted')
        self.valid_rel = torchmetrics.classification.Recall(task="multiclass", num_classes=5, average='weighted')
        self.train_f1l = torchmetrics.classification.F1Score(task="multiclass", num_classes=5, average='weighted')
        self.valid_f1l = torchmetrics.classification.F1Score(task="multiclass", num_classes=5, average='weighted')
        self.confusion_vall = torchmetrics.ConfusionMatrix(num_classes=5, task='multiclass')
        self.confusion_trainl = torchmetrics.ConfusionMatrix(num_classes=5, task='multiclass')

        self.train_accr = torchmetrics.classification.Accuracy(task="multiclass", num_classes=5)
        self.valid_accr = torchmetrics.classification.Accuracy(task="multiclass", num_classes=5)
        self.train_prr = torchmetrics.classification.Precision(task="multiclass", num_classes=5, average='weighted')
        self.valid_prr = torchmetrics.classification.Precision(task="multiclass", num_classes=5, average='weighted')
        self.train_rer = torchmetrics.classification.Recall(task="multiclass", num_classes=5, average='weighted')
        self.valid_rer = torchmetrics.classification.Recall(task="multiclass", num_classes=5, average='weighted')
        self.train_f1r = torchmetrics.classification.F1Score(task="multiclass", num_classes=5, average='weighted')
        self.valid_f1r = torchmetrics.classification.F1Score(task="multiclass", num_classes=5, average='weighted')
        self.confusion_valr = torchmetrics.ConfusionMatrix(num_classes=5, task='multiclass')
        self.confusion_trainr = torchmetrics.ConfusionMatrix(num_classes=5, task='multiclass')

    def training_step(self, batch, batch_idx):
        #x, y = batch
        x, y1, y2 = batch
        x = x.cuda()
        # y= y.cuda()
        y1 = y1.cuda()
        y2 = y2.cuda()
        #preds = self.ELE_Net(x)
        preds1, preds2 = self.ELE_Net(x)

        #loss = F.cross_entropy(preds, y)
        loss = F.cross_entropy(preds1, y1) + F.cross_entropy(preds2, y2)

        # self.train_acc(torch.argmax(preds, axis=1), y)
        # self.train_pr(torch.argmax(preds, axis=1), y)
        # self.train_re(torch.argmax(preds, axis=1), y)
        # self.train_f1(torch.argmax(preds, axis=1), y)


        # self.log("Training_loss "+self.lr, loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("Training Accuracy (1, 2, 3, 4) " + self.lr, self.train_acc, on_step=False, on_epoch=True, logger=True)
        # self.log("Training Precision (1, 2, 3, 4) " + self.lr, self.train_pr, on_step=False, on_epoch=True, logger=True)
        # self.log("Training Recall (1, 2, 3, 4) " + self.lr, self.train_re, on_step=False, on_epoch=True, logger=True)
        # self.log("Training F1_Score (1, 2, 3, 4) " + self.lr, self.train_f1, on_step=False, on_epoch=True, logger=True)
        # self.train_step_outputs.append({'preds': preds, 'target': y})

        self.train_accl(torch.argmax(preds1, axis=1), y1)
        self.train_prl(torch.argmax(preds1, axis=1), y1)
        self.train_rel(torch.argmax(preds1, axis=1), y1)
        self.train_f1l(torch.argmax(preds1, axis=1), y1)


        self.log("Training_loss "+self.lr, loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("Training Accuracy (1, 2, 3, 4) " + "Left", self.train_accl, on_step=False, on_epoch=True, logger=True)
        self.log("Training Precision (1, 2, 3, 4) " + "Left", self.train_prl, on_step=False, on_epoch=True, logger=True)
        self.log("Training Recall (1, 2, 3, 4) " + "Left", self.train_rel, on_step=False, on_epoch=True, logger=True)
        self.log("Training F1_Score (1, 2, 3, 4) " + "Left", self.train_f1l, on_step=False, on_epoch=True, logger=True)
        

        self.train_accr(torch.argmax(preds2, axis=1), y2)
        self.train_prr(torch.argmax(preds2, axis=1), y2)
        self.train_rer(torch.argmax(preds2, axis=1), y2)
        self.train_f1r(torch.argmax(preds2, axis=1), y2)
        
        self.log("Training Accuracy (1, 2, 3, 4) " + "Right", self.train_accr, on_step=False, on_epoch=True, logger=True)
        self.log("Training Precision (1, 2, 3, 4) " + "Right", self.train_prr, on_step=False, on_epoch=True, logger=True)
        self.log("Training Recall (1, 2, 3, 4) " + "Right", self.train_rer, on_step=False, on_epoch=True, logger=True)
        self.log("Training F1_Score (1, 2, 3, 4) " + "Right", self.train_f1r, on_step=False, on_epoch=True, logger=True)
        self.train_step_outputs.append({'preds1': preds1, 'target1': y1, 'preds2': preds2, 'target2': y2})
        return loss
    

    def on_train_epoch_end(self):
        tb = self.logger.experiment

        # preds = torch.cat([torch.argmax(tmp['preds'], axis=1) for tmp in self.train_step_outputs])
        # targets = torch.cat([tmp['target'] for tmp in self.train_step_outputs])
        # self.train_step_outputs = []
        # self.confusion_train(preds, targets)
        # computed_confusion = self.confusion_train.compute().detach().cpu().numpy().astype(int)

        # # confusion matrix
        # df_cm = pd.DataFrame(
        #     computed_confusion,
        #     index={0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}.values(),
        #     columns={0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}.values(),
        # )

        # fig, ax = plt.subplots(figsize=(10, 5))
        # fig.subplots_adjust(left=0.05, right=.65)
        # sns.set_theme(font_scale=1.2)
        # sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', ax=ax)
        # ax.legend(
        #     {0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}.values(),
        #     {0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}.keys(),
        #     handler_map={int: IntHandler()},
        #     loc='upper left',
        #     bbox_to_anchor=(1.2, 1)
        # )
        # buf = io.BytesIO()

        # plt.savefig(buf, format='jpeg', bbox_inches='tight')
        # buf.seek(0)
        # im = Image.open(buf)
        # im = torchvision.transforms.ToTensor()(im)
        # tb.add_image("Train_confusion_matrix", im, global_step=self.current_epoch)
        # self.confusion_train.reset()
        # buf.close()
        # plt.close()

        preds = torch.cat([torch.argmax(tmp['preds1'], axis=1) for tmp in self.train_step_outputs])
        targets = torch.cat([tmp['target1'] for tmp in self.train_step_outputs])
        
        self.confusion_trainl(preds, targets)
        computed_confusionl = self.confusion_trainl.compute().detach().cpu().numpy().astype(int)

        # confusion matrix
        df_cm = pd.DataFrame(
            computed_confusionl,
            index={0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}.values(),
            columns={0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}.values(),
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(left=0.05, right=.65)
        sns.set_theme(font_scale=1.2)
        sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', ax=ax)
        ax.legend(
            {0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}.values(),
            {0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}.keys(),
            handler_map={int: IntHandler()},
            loc='upper left',
            bbox_to_anchor=(1.2, 1)
        )
        buf = io.BytesIO()

        plt.savefig(buf, format='jpeg', bbox_inches='tight')
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        tb.add_image("Train_confusion_matrix_Left", im, global_step=self.current_epoch)
        self.confusion_trainl.reset()
        buf.close()
        plt.close()

        preds = torch.cat([torch.argmax(tmp['preds2'], axis=1) for tmp in self.train_step_outputs])
        targets = torch.cat([tmp['target2'] for tmp in self.train_step_outputs])
        
        self.confusion_trainr(preds, targets)
        computed_confusionr = self.confusion_trainr.compute().detach().cpu().numpy().astype(int)

        # confusion matrix
        df_cm = pd.DataFrame(
            computed_confusionr,
            index={0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}.values(),
            columns={0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}.values(),
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(left=0.05, right=.65)
        sns.set_theme(font_scale=1.2)
        sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', ax=ax)
        ax.legend(
            {0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}.values(),
            {0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}.keys(),
            handler_map={int: IntHandler()},
            loc='upper left',
            bbox_to_anchor=(1.2, 1)
        )
        buf = io.BytesIO()

        plt.savefig(buf, format='jpeg', bbox_inches='tight')
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        tb.add_image("Train_confusion_matrix_Right", im, global_step=self.current_epoch)
        self.confusion_trainr.reset()
        self.train_step_outputs = []
        buf.close()
        plt.close()
        
        
    def validation_step(self, batch, batch_idx):
        
        #x, y = batch
        x, y1, y2 = batch
        x = x.cuda()
        # y= y.cuda()
        y1 = y1.cuda()
        y2 = y2.cuda()
        #preds = self.ELE_Net(x)
        preds1, preds2 = self.ELE_Net(x)

        #loss = F.cross_entropy(preds, y)
        loss = F.cross_entropy(preds1, y1) + F.cross_entropy(preds2, y2)

        # self.valid_acc(torch.argmax(preds, axis=1), y)
        # self.valid_pr(torch.argmax(preds, axis=1), y)
        # self.valid_re(torch.argmax(preds, axis=1), y)
        # self.valid_f1(torch.argmax(preds, axis=1), y)

        # self.log("Validation_loss "+self.lr, loss, on_step=False, on_epoch=True, logger=True)
        # self.log("Validation Accuracy (1, 2, 3, 4) " + self.lr, self.valid_acc, on_step=False, on_epoch=True, logger=True)
        # self.log("Validation Precision (1, 2, 3, 4) " + self.lr, self.valid_pr, on_step=False, on_epoch=True, logger=True)
        # self.log("Validation Recall (1, 2, 3, 4) " + self.lr, self.valid_re, on_step=False, on_epoch=True, logger=True)
        # self.log("Validation F1_Score (1, 2, 3, 4) " + self.lr, self.valid_f1, on_step=False, on_epoch=True, logger=True)
        # self.validation_step_outputs.append({'preds': preds, 'target': y})

        self.valid_accl(torch.argmax(preds1, axis=1), y1)
        self.valid_prl(torch.argmax(preds1, axis=1), y1)
        self.valid_rel(torch.argmax(preds1, axis=1), y1)
        self.valid_f1l(torch.argmax(preds1, axis=1), y1)

        self.log("Validation_loss "+self.lr, loss, on_step=False, on_epoch=True, logger=True)
        self.log("Validation Accuracy (1, 2, 3, 4) " + "Left", self.valid_accl, on_step=False, on_epoch=True, logger=True)
        self.log("Validation Precision (1, 2, 3, 4) " + "Left", self.valid_prl, on_step=False, on_epoch=True, logger=True)
        self.log("Validation Recall (1, 2, 3, 4) " + "Left", self.valid_rel, on_step=False, on_epoch=True, logger=True)
        self.log("Validation F1_Score (1, 2, 3, 4) " + "Left", self.valid_f1l, on_step=False, on_epoch=True, logger=True)
        

        self.valid_accr(torch.argmax(preds2, axis=1), y2)
        self.valid_prr(torch.argmax(preds2, axis=1), y2)
        self.valid_rer(torch.argmax(preds2, axis=1), y2)
        self.valid_f1r(torch.argmax(preds2, axis=1), y2)

        
        self.log("Validation Accuracy (1, 2, 3, 4) " + "Right", self.valid_accr, on_step=False, on_epoch=True, logger=True)
        self.log("Validation Precision (1, 2, 3, 4) " + "Right", self.valid_prr, on_step=False, on_epoch=True, logger=True)
        self.log("Validation Recall (1, 2, 3, 4) " + "Right", self.valid_rer, on_step=False, on_epoch=True, logger=True)
        self.log("Validation F1_Score (1, 2, 3, 4) " + "Right", self.valid_f1r, on_step=False, on_epoch=True, logger=True)

        self.validation_step_outputs.append({'preds1': preds1, 'target1': y1, 'preds2': preds2, 'target2': y2})

    def on_validation_epoch_end(self):
        tb = self.logger.experiment
        # preds = torch.cat([torch.argmax(tmp['preds'], axis=1) for tmp in self.validation_step_outputs])
        # targets = torch.cat([tmp['target'] for tmp in self.validation_step_outputs])
        
        # self.confusion_val(preds, targets)
        # computed_confusion = self.confusion_val.compute().detach().cpu().numpy().astype(int)

        # # confusion matrix
        # df_cm = pd.DataFrame(
        #     computed_confusion,
        #     index={0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}.values(),
        #     columns={0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}.values(),
        # )

        # fig, ax = plt.subplots(figsize=(10, 5))
        # fig.subplots_adjust(left=0.05, right=.65)
        # sns.set_theme(font_scale=1.2)
        # sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', ax=ax)
        # ax.legend(
        #     {0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}.values(),
        #     {0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}.keys(),
        #     handler_map={int: IntHandler()},
        #     loc='upper left',
        #     bbox_to_anchor=(1.2, 1)
        # )
        # buf = io.BytesIO()

        # plt.savefig(buf, format='jpeg', bbox_inches='tight')
        # buf.seek(0)
        # im = Image.open(buf)
        # im = torchvision.transforms.ToTensor()(im)
        # tb.add_image("val_confusion_matrix", im, global_step=self.current_epoch)
        # buf.close()
        # plt.close()
        # self.validation_step_outputs = []
        # self.confusion_val.reset()

        preds = torch.cat([torch.argmax(tmp['preds1'], axis=1) for tmp in self.validation_step_outputs])
        targets = torch.cat([tmp['target1'] for tmp in self.validation_step_outputs])
        
        self.confusion_vall(preds, targets)
        computed_confusionl = self.confusion_vall.compute().detach().cpu().numpy().astype(int)

        # confusion matrix
        df_cm = pd.DataFrame(
            computed_confusionl,
            index={0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}.values(),
            columns={0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}.values(),
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(left=0.05, right=.65)
        sns.set_theme(font_scale=1.2)
        sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', ax=ax)
        ax.legend(
            {0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}.values(),
            {0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}.keys(),
            handler_map={int: IntHandler()},
            loc='upper left',
            bbox_to_anchor=(1.2, 1)
        )
        buf = io.BytesIO()

        plt.savefig(buf, format='jpeg', bbox_inches='tight')
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        tb.add_image("Val_confusion_matrix_Left", im, global_step=self.current_epoch)
        buf.close()
        plt.close()
        
        self.confusion_vall.reset()

        preds = torch.cat([torch.argmax(tmp['preds2'], axis=1) for tmp in self.validation_step_outputs])
        targets = torch.cat([tmp['target2'] for tmp in self.validation_step_outputs])
        
        self.confusion_valr(preds, targets)
        computed_confusionr = self.confusion_valr.compute().detach().cpu().numpy().astype(int)

        # confusion matrix
        df_cm = pd.DataFrame(
            computed_confusionr,
            index={0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}.values(),
            columns={0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}.values(),
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(left=0.05, right=.65)
        sns.set_theme(font_scale=1.2)
        sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', ax=ax)
        ax.legend(
            {0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}.values(),
            {0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}.keys(),
            handler_map={int: IntHandler()},
            loc='upper left',
            bbox_to_anchor=(1.2, 1)
        )
        buf = io.BytesIO()

        plt.savefig(buf, format='jpeg', bbox_inches='tight')
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        tb.add_image("Val_confusion_matrix_Right", im, global_step=self.current_epoch)
        buf.close()
        plt.close()
        self.validation_step_outputs = []
        self.confusion_valr.reset()
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.000333)
        #optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        return optimizer
    
    
def main(hparams):

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")

    #LeftELE = LITELE(ELE_Net(103, 64, 5).to(device), 'Left')
    #RightELE = LITELE(ELE_Net(103, 64, 5).to(device), "Right")
    LeftRightELE = LITELE(ELE_Net(103, 64, 5).to(device), 'Left_Right')

    #trainerL = L.Trainer(logger=tb_logger, gradient_clip_val=1, gradient_clip_algorithm="value", max_epochs=int(hparams.max_epochs))
    #trainerR = L.Trainer(logger=tb_logger, gradient_clip_val=1, gradient_clip_algorithm="value", max_epochs=int(hparams.max_epochs))
    trainerLR = L.Trainer(logger=tb_logger, gradient_clip_val=1, gradient_clip_algorithm="value", max_epochs=int(hparams.max_epochs))

    #left_dataset = ELE_Dataset(hparams.data_path, device, 'left', 3.5, 30)
    #right_dataset = ELE_Dataset(hparams.data_path, device, 'right', 3.5, 30)
    left_right_dataset = ELE_Dataset(hparams.data_path, device, 'left_right', 2.5, 30)

    #train_dataset_l, validation_dataset_l = torch.utils.data.random_split(left_dataset, [0.8, 0.2])
    #train_dataset_r, validation_dataset_r = torch.utils.data.random_split(right_dataset, [0.8, 0.2])
    train_dataset_lr, validation_dataset_lr = torch.utils.data.random_split(left_right_dataset, [0.8, 0.2])
    
    #trainloaderL = DataLoader(train_dataset_l, batch_size=int(hparams.batch_size), shuffle=False, num_workers=8, pin_memory=True)
    #trainloaderR = DataLoader(train_dataset_r, batch_size=int(hparams.batch_size), shuffle=False, num_workers=8, pin_memory=True)
    trainloaderLR = DataLoader(train_dataset_lr, batch_size=int(hparams.batch_size), shuffle=False, num_workers=8, pin_memory=True)

    #validloaderl = DataLoader(validation_dataset_l, batch_size=128, shuffle=False)
    #validloaderr = DataLoader(validation_dataset_r, batch_size=128, shuffle=False)
    validloaderlr = DataLoader(validation_dataset_lr, batch_size=128, shuffle=False)

    #trainerL.fit(LeftELE, trainloaderL, validloaderl)
    #trainerR.fit(RightELE, trainloaderR, validloaderr)
    trainerLR.fit(LeftRightELE, trainloaderLR, validloaderlr)
    

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", default=None)
    #parser.add_argument("--device", default=torch.cuda(0))
    
    device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--max_epochs", default=100)
    args = parser.parse_args()
    main(args)
    

    


