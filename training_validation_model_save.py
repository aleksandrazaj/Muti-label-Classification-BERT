#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# In[19]:


from transformers import AdamW, BertConfig

model = BertForMultiLabelSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 17,    
    output_attentions = False, 
    output_hidden_states = False, 
)

#desc = model.cuda()
print ("Model loaded.")


# In[20]:


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 16

#DataLoaders
train_dataloader = DataLoader(
            train_dataset,  
            sampler = RandomSampler(train_dataset), 
            batch_size = batch_size 
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, 
            sampler = SequentialSampler(val_dataset), 
            batch_size = batch_size 
        )


# In[21]:


from transformers import AdamW, BertConfig
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, 
                  eps = 1e-8 
                )


# In[22]:


from transformers import get_linear_schedule_with_warmup


epochs = 4


total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)


# In[23]:


import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    
    elapsed_rounded = int(round((elapsed)))
    
    
    return str(datetime.timedelta(seconds=elapsed_rounded)) 


# In[24]:


def good_update_interval(total_iters, num_desired_updates):
    '''
    This function will try to pick an intelligent progress update interval 
    based on the magnitude of the total iterations.
    Parameters:
      `total_iters` - The number of iterations in the for-loop.
      `num_desired_updates` - How many times we want to see an update over the 
                              course of the for-loop.
    '''
    
    exact_interval = total_iters / num_desired_updates

   
    order_of_mag = len(str(total_iters)) - 1

    
    round_mag = order_of_mag - 1

    
    update_interval = int(round(exact_interval, -round_mag))

    
    if update_interval == 0:
        update_interval = 1

    return update_interval


# In[25]:


from sklearn.metrics import roc_auc_score

# A quick example...
true_labels = [0,1,0,0,1,0,0,0,0,0,0,0]
pred_labels = [0,1,0,0,0,0,0,0,0,0,0,0]

score = roc_auc_score(true_labels, pred_labels, average='macro')

print('Example ROC AUC score:', score)


# In[ ]:


import random
import numpy as np


seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

training_stats = []

total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    
    t0 = time.time()

    
    total_train_loss = 0

    model.train()

    update_interval = good_update_interval(
                total_iters = len(train_dataloader), 
                num_desired_updates = 10
            )

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):


        if (step % update_interval) == 0 and not step == 0:
            
            elapsed = format_time(time.time() - t0)
            
            
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))


        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)


        model.zero_grad()        

        loss, logits = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)

        total_train_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
        
 
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_loss = 0

    predictions, true_labels = [], []

    for batch in validation_dataloader:
        

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        with torch.no_grad():   
   
            (loss, logits) = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)

        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    # Combine the results across all batches. 
    flat_predictions = np.concatenate(predictions, axis=0)
    flat_true_labels = np.concatenate(true_labels, axis=0)
    
    try:
        val_accuracy = roc_auc_score(flat_true_labels, flat_predictions, average='macro')
    except ValueError:
        pass
        val_accuracy = 0
        
    print("  Accuracy: {0:.2f}".format(val_accuracy))

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


# In[ ]:


import pandas as pd

pd.set_option('precision', 2)
df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index('epoch')
df_stats


# In[26]:


#ADJUST COLUMNS TO FINAL DATASET
test = test1[["review_id","lineNo","sentence"]]
test_labels = test1[["review_id","lineNo","Negative","Neutral","Positive","Comfortability","Connectivity","Features","Not_Relevant","Others","Physical_Design","Power_Battery","Price","Product_Durability","Product_Support","Software_App","Sound_Quality_Rx","Sound_Quality_Tx","Usability"]]


# In[27]:


print('There are {:,} labeled test examples.'.format(len(test)))


# In[28]:


import torch

input_ids = []
attn_masks = []
labels = []

# ======== Encoding ========

print('Encoding all {:,} test samples...'.format(len(test)))

for (index, row) in test.iterrows():

    if ((len(input_ids) % 50) == 0):
        print('  Tokenized {:,} comments.'.format(len(input_ids)))

    encoded_dict = tokenizer.encode_plus(row['sentence'],  # The text to encode.
                                        max_length=max_len,    # Pad or truncate to this lenght.
                                        pad_to_max_length=True,
                                        truncation=True, 
                                        return_tensors='pt')   # Return objects as PyTorch tensors.

    # Add this example to our lists.
    input_ids.append(encoded_dict['input_ids'])
    attn_masks.append(encoded_dict['attention_mask'])
    
print('\nDONE. {:,} examples.'.format(len(input_ids)))

# ======== List of Examples --> Tensor ========

# Convert each Python list of Tensors into a 2D Tensor matrix.
input_ids = torch.cat(input_ids, dim=0)
attn_masks = torch.cat(attn_masks, dim=0)


# In[29]:


test_labels.head()


# In[30]:


# Remove the 'id' column.
del test_labels["review_id"]
del test_labels["lineNo"]

labels = test_labels.to_numpy().astype(float)

# Cast the labels list to a 2D Tensor.
labels = torch.tensor(labels)

# ======== Summary ========

print('\nData structure shapes:')
print('   input_ids:  {:}'.format(str(input_ids.shape)))
print('  attn_masks:  {:}'.format(str(attn_masks.shape)))
print('      labels:  {:}'.format(str(labels.shape)))


# In[31]:


from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

# Combine the training inputs into a TensorDataset.
test_dataset = TensorDataset(input_ids, attn_masks, labels)

# Specify our batch size.
batch_size = 16

# Create the DataLoader, which will select batches for us. For testing, the
test_dataloader = DataLoader(
            test_dataset,  # The training samples.
            sampler = SequentialSampler(test_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )


# In[ ]:


import time

# Prediction on test set

# Put model in evaluation mode
model.eval()

t0 = time.time()

# Tracking variables 
predictions , true_labels = [], []

print('Evaluating on {:,} test set batches...'.format(len(test_dataloader)))

# Predict 
for batch in test_dataloader:
    
    # Report progress.
    if ((len(predictions) % 50) == 0):
        print('  Batch {:>5,}  of  {:>5,}.'.format(len(predictions), len(test_dataloader)))

    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    
    with torch.no_grad():
        
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)

    logits = outputs[0]

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    # Store predictions and true labels
    predictions.append(logits)
    true_labels.append(label_ids)

print('    DONE.')

print('Evaluation took {:.0f} seconds.'.format(time.time() - t0))


# In[ ]:


# Combine the results across all batches. 
flat_predictions = np.concatenate(predictions, axis=0)
flat_true_labels = np.concatenate(true_labels, axis=0)


# In[ ]:


# For every test sample...
for test_i in range(0, len(test1)):

    # Break if it has at least one label set.
    if np.any(flat_true_labels[test_i, :]):
        break

print('Test sample: {:,}\n'.format(test_i))


# In[ ]:


# Print out model predictions vs. correct values for this test sample.
print('           Type   Output   Truth')
print('           ----   ------   -----')

for label_i in range(0, 10):
    print('{:>15}   {:>5.2f}      {:}'.format(
        label_cols[label_i], 
        flat_predictions[test_i, label_i], 
        int(flat_true_labels[test_i, label_i]))) 


# In[ ]:


try:
    score = roc_auc_score(flat_true_labels, flat_predictions)
except ValueError:
    pass
    score = 0
    

print('ROC AUC: {:.4f}'.format(score))


# In[ ]:


import os

output_dir = './model_save/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

model.save_pretrained(output_dir)

