#!/usr/bin/env python
# coding: utf-8

# In[14]:


import platform
import numpy as np
import pandas as pd
import random

import torch 
from torch import optim 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn

from tqdm.notebook import tqdm
from transformers import AutoTokenizer

# enable tqdm in pandas
tqdm.pandas()

# select device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif 'arm64' in platform.platform():
    device = torch.device('mps') # 'mps'
else:
    device = torch.device('cpu')
print(f'device: {device.type}') 

# random seed
seed = 1234

# pytorch ignores this label in the loss
ignore_index = -100

# set random seed
if seed is not None:
    print(f'random seed: {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# which transformer to use
transformer_name = "bert-base-cased" # 'xlm-roberta-base' # 'distilbert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(transformer_name)


# In[15]:



# map labels to the first token in each word
def align_labels(word_ids, labels, label_to_index):
    label_ids = []
    previous_word_id = None
    for word_id in word_ids:
        if word_id is None or word_id == previous_word_id:
            # ignore if not a word or word id has already been seen
            label_ids.append(ignore_index)
        else:
            # get label id for corresponding word
            label_id = label_to_index[labels[word_id]]
            label_ids.append(label_id)
        # remember this word id
        previous_word_id = word_id
    
    return label_ids
            
# build a set of labels in the dataset            
def read_label_set(fn):
    labels = set()
    with open(fn) as f:
        for index, line in enumerate(f):
            line = line.strip()
            tokens = line.split()
            if tokens != []:
                label = tokens[-1]
                labels.add(label)
    return labels

# converts a two-column file in the basic MTL format ("word \t label") into a dataframe
def read_dataframe(fn, label_to_index, task_id):
    # now build the actual dataframe for this dataset
    data = {'words': [], 'str_labels': [], 'input_ids': [], 'word_ids': [], 'labels': [], 'task_ids': []}
    with open(fn) as f:
        sent_words = []
        sent_labels = [] 
        for index, line in tqdm(enumerate(f)):
            line = line.strip()
            tokens = line.split()
            if tokens == []:
                data['words'].append(sent_words)
                data['str_labels'].append(sent_labels)
                
                # tokenize each sentence
                token_input = tokenizer(sent_words, is_split_into_words = True)  
                token_ids = token_input['input_ids']
                word_ids = token_input.word_ids(batch_index = 0)
                
                # map labels to the first token in each word
                token_labels = align_labels(word_ids, sent_labels, label_to_index)
                
                data['input_ids'].append(token_ids)
                data['word_ids'].append(word_ids)
                data['labels'].append(token_labels)
                data['task_ids'].append(task_id)
                sent_words = []
                sent_labels = [] 
            else:
                sent_words.append(tokens[0])
                sent_labels.append(tokens[1])
    return pd.DataFrame(data)


# In[16]:


class Task():
    def __init__(self, task_id, train_file_name, dev_file_name, test_file_name):
        self.task_id = task_id
        # we need an index of labels first
        self.labels = read_label_set(train_file_name)
        self.index_to_label = {i:t for i,t in enumerate(self.labels)}
        self.label_to_index = {t:i for i,t in enumerate(self.labels)}
        self.num_labels = len(self.index_to_label)
        # create data frames for the datasets
        self.train_df = read_dataframe(train_file_name, self.label_to_index, self.task_id)
        self.dev_df = read_dataframe(dev_file_name, self.label_to_index, self.task_id)
        self.test_df = read_dataframe(test_file_name, self.label_to_index, self.task_id)
                


# In[17]:


ner_task = Task(0, "data/conll-ner/train_small.txt", "data/conll-ner/dev.txt", "data/conll-ner/test.txt")
pos_task = Task(1, "data/pos/train_small.txt", "data/pos/dev.txt", "data/pos/test.txt")


# In[18]:


ner_task.train_df


# In[19]:


pos_task.train_df


# In[20]:


from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from transformers import PreTrainedModel
from transformers import AutoConfig, AutoModel

# This class is adapted from: https://towardsdatascience.com/how-to-create-and-train-a-multi-task-transformer-model-18c54a146240
class TokenClassificationModel(BertPreTrainedModel):    
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.output_heads = nn.ModuleDict() # these are initialized in add_heads
        self.init_weights()
        
    def add_heads(self, tasks):
        for task in tasks:
            head = TokenClassificationHead(self.bert.config.hidden_size, task.num_labels, config.hidden_dropout_prob)
            # ModuleDict requires keys to be strings
            self.output_heads[str(task.task_id)] = head
        return self
    
    def summarize_heads(self):
        print(f'Found {len(self.output_heads)} heads')
        for task_id in self.output_heads:
            self.output_heads[task_id].summarize(task_id)
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, task_ids=None, **kwargs):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        sequence_output = outputs[0]
        
        #print(f'batch size = {len(input_ids)}')
        #print(f'task_ids in this batch: {task_ids}')
        
        # generate specific predictions and losses for each task head
        unique_task_ids_list = torch.unique(task_ids).tolist()
        logits = None
        loss_list = []
        for unique_task_id in unique_task_ids_list:
            task_id_filter = task_ids == unique_task_id
            filtered_sequence_output = sequence_output[task_id_filter]
            filtered_labels = None if labels is None else labels[task_id_filter]
            filtered_attention_mask = None if attention_mask is None else attention_mask[task_id_filter]
            #print(f'size of batch for task {unique_task_id} is: {len(filtered_sequence_output)}')
            logits, task_loss = self.output_heads[str(unique_task_id)].forward(
                filtered_sequence_output, None,
                filtered_labels,
                filtered_attention_mask,
            )
            if filtered_labels is not None:
                loss_list.append(task_loss)
                
        loss = None if len(loss_list) == 0 else torch.stack(loss_list)
                    
        # logits are only used for eval, in which case we handle a single task at a time
        # TODO: allow all tasks in the forward pass at inference                     
        return TokenClassifierOutput(
            loss=loss.mean(),
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class TokenClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_p=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

        self._init_weights()

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()
            
    def summarize(self, task_id):
        print(f"Task {task_id} with {self.num_labels} labels.")
        print(f'Dropout is {self.dropout}')
        print(f'Classifier layer is {self.classifier}')

    def forward(self, sequence_output, pooled_output, labels=None, attention_mask=None, **kwargs):
        sequence_output_dropout = self.dropout(sequence_output)
        logits = self.classifier(sequence_output_dropout)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()            
            inputs = logits.view(-1, self.num_labels)
            targets = labels.view(-1)
            loss = loss_fn(inputs, targets)

        return logits, loss


# In[22]:


tasks = [ner_task, pos_task]
config = AutoConfig.from_pretrained(transformer_name)
model= TokenClassificationModel.from_pretrained(transformer_name, config=config).add_heads(tasks)
model.summarize_heads()


# In[23]:


from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    # gold labels
    label_ids = eval_pred.label_ids
    # predictions
    pred_ids = np.argmax(eval_pred.predictions, axis=-1)
    # collect gold and predicted labels, ignoring ignore_index label
    y_true, y_pred = [], []
    batch_size, seq_len = pred_ids.shape
    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != ignore_index:
                y_true.append(label_ids[i][j]) #index_to_label[label_ids[i][j]])
                y_pred.append(pred_ids[i][j]) #index_to_label[pred_ids[i][j]])
    # return computed metrics
    return {'accuracy': accuracy_score(y_true, y_pred)}


# In[24]:


from datasets import Dataset, DatasetDict

ds = DatasetDict()
ds['train'] = Dataset.from_pandas(pd.concat([ner_task.train_df, pos_task.train_df]))
ds['validation'] = Dataset.from_pandas(pd.concat([ner_task.dev_df, pos_task.dev_df]))
ds['test'] = Dataset.from_pandas(pd.concat([ner_task.test_df, pos_task.test_df]))

# these are no longer needed; discard them to save memory
ner_task.train_df = None
pos_task.train_df = None

ds


# In[29]:


from sklearn.metrics import classification_report

# compute accuracy
def evaluation_classification_report(trainer, task, name, useTest=False):
    print(f"Test classification report for task {name}:")
    num_labels = task.num_labels
    df = task.test_df if useTest == False else task.dev_df
    ds = Dataset.from_pandas(df)
    output = trainer.predict(ds)
    label_ids = output.label_ids.reshape(-1)
    predictions = output.predictions.reshape(-1, num_labels)
    predictions = np.argmax(predictions, axis=-1)
    mask = label_ids != ignore_index
    
    y_true = label_ids[mask]
    y_pred = predictions[mask]
    target_names = [task.index_to_label.get(ele, "") for ele in range(num_labels)]
    print(target_names)
    
    total = 0
    correct = 0
    for(t, p) in zip(y_true, y_pred):
        total = total + 1
        if t == p:
            correct = correct + 1
    accuracy = correct / total
    
    report = classification_report(
        y_true, y_pred,
        target_names=target_names
    )
    print(report)
    print(f'locally computed accuracy: {accuracy}')
    return accuracy

# compute loss and accuracy
def evaluate(trainer, task, name):
    print(f"Evaluating on the validation dataset for task {name}:")
    ds = Dataset.from_pandas(task.dev_df)
    scores = trainer.evaluate(ds)
    acc = evaluation_classification_report(trainer, task, name, useTest = False)
    return scores, acc


# In[30]:


from transformers import TrainingArguments
from transformers import Trainer
from transformers import DataCollatorForTokenClassification
import time
from datetime import timedelta

epochs = 2
batch_size = 128
weight_decay = 0.01
use_mps_device = True if str(device) == 'mps' else False
model_name = f'{transformer_name}-mtl'

data_collator = DataCollatorForTokenClassification(tokenizer)
last_checkpoint = None

for epoch in range(1, epochs + 1):
    print(f'STARTING EPOCH {epoch}')
    if last_checkpoint != None:
        print(f'Resuming from checkpoint {last_checkpoint}')
            
    training_args = TrainingArguments(
        output_dir=model_name,
        log_level='error',
        num_train_epochs=1, # one epoch at a time
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # evaluation_strategy='epoch',
        do_eval=False, # we will evaluate each task explicitly
        weight_decay=weight_decay,
        resume_from_checkpoint = last_checkpoint,
        use_mps_device = use_mps_device
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        train_dataset=ds['train'],
        # eval_dataset=ds['validation'],
        tokenizer=tokenizer
    )
    
    model.summarize_heads()

    start_time = time.monotonic()
    trainer.train()
    end_time = time.monotonic()
    print(f"Elapsed time for epoch {epoch}: {timedelta(seconds=end_time - start_time)}")

    ner_scores, ner_acc = evaluate(trainer, ner_task, "NER")
    pos_scores, pos_acc = evaluate(trainer, pos_task, "POS")
    macro_loss = (ner_scores['eval_loss'] + pos_scores['eval_loss'])/2
    print(f'DEV MACRO LOSS FOR EPOCH {epoch}: {macro_loss}\n\n')
    macro_acc = (ner_acc + pos_acc)/2
    print(f'DEV MACRO ACC FOR EPOCH {epoch}: {macro_acc}')

    last_checkpoint = training_args.output_dir + '/mtl_model_epoch' + str(epoch)
    trainer.save_model(last_checkpoint)


# In[13]:


#model = TokenClassificationModel.from_pretrained('bert-base-cased-mtl/mtl_model_epoch2', local_files_only=True)
#model.summarize_heads()


# In[32]:


ner_acc = evaluation_classification_report(trainer, ner_task, "NER", useTest = True)
pos_acc = evaluation_classification_report(trainer, pos_task, "POS", useTest = True)
macro_acc = (ner_acc + pos_acc)/2
print(f"MTL macro accuracy: {macro_acc}")


# In[ ]:




