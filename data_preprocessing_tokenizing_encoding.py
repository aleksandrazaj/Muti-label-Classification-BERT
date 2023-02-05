#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#PRE-PROCESS FOR TRAINING
#ADJUST TO APPROPRIATE TABLE COLUMNS FROM SQL - 

pivoted = df.pivot_table(df,index=["review_id","lineNo","sentence"],columns=["sentiment_label"], aggfunc='count', fill_value=0)
pivoted.head()
pivoted_sentiment = pivoted["ProductName"]
pivoted_sentiment.head()
pivoted2 = df.pivot_table(df,index=["review_id","lineNo","sentence"],columns=["performanceTopic"], aggfunc='count', fill_value=0)
pivoted_keypoint = pivoted2["ProductName"]
result = pd.merge(pivoted_sentiment, pivoted_keypoint, on=["review_id","lineNo","sentence"])
result1 = result.replace([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],1)
result2 = result1.reset_index()
result2.head()
result3 = result2[["review_id","lineNo","sentence","Mixed","Negative","Neutral","Positive","Com_","Conn_","Feat_","Not_","O_","P_","P1_","P2_","P3_","P4_","S_","S1_","S2_","U_"]]
rows = (result3.Mixed == 1) 
result4 = result3.loc[rows, ['Positive', 'Negative']] = 1
result5 = result4[["review_id","lineNo","sentence","Negative","Neutral","Positive","Com_","Conn_","Feat_","Not_","O_","P_","P1_","P2_","P3_","P4_","S_","S1_","S2_","U_"]]

#SPLIT FOR TRAIN AND FOR TEST
from sklearn.model_selection import train_test_split
train, test1 = train_test_split(result5, test_size=0.2)

print('There are {:,} training examples.'.format(len(train)))

#import textwrap
import random

# These are the 17 possible labels - remmeber to adjust
label_cols = ["Negative","Neutral","Positive","Com_","Conn_","Feat_","Not_","O_","P_","P1_","P2_","P3_","P4_","S_","S1_","S2_","U_"]

# Select just the labels (not the text), and for every row, check whether any
# of the labels are "1".
has_labels = train[label_cols].any(axis=1)

from transformers import BertTokenizer

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


import numpy as np
import logging

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

lengths = []

print('Tokenizing comments...')

# For every sentence...
for index, row in train.iterrows():
    
    # Report progress.
    if ((len(lengths) % 200) == 0):
        print('  Tokenized {:,} comments.'.format(len(lengths)))
    
    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    encoded_sent = tokenizer.encode(
                        row['sentence'],     # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                   )
    
    # Record the non-truncated length.
    lengths.append(len(encoded_sent))

print('DONE.')


# In[12]:


print('   Min length: {:,} tokens'.format(min(lengths)))
print('   Max length: {:,} tokens'.format(max(lengths)))
print('Median length: {:,} tokens'.format(int(np.median(lengths))))

import seaborn as sns

# Cast the list to a numpy array so we can use some numpy features.
lengths = np.asarray(lengths)

# Get the total number of comments.
num_comments = len(lengths)

# Check the following lengths:
max_lens = [100, 128, 256, 300, 400, 512]

print('How many comments will be truncated?\n')

# For each choice...
for max_len in max_lens:

    # Calculate how many comments will be truncacted.
    num_over = np.sum(lengths > max_len)

    prcnt_over = float(num_over) / float(num_comments)

    print('max_len = {:}  -->  {:>7,} of {:>7,}  ({:>5.1%})  '           'will be truncated '.format(
              max_len, num_over, num_comments, prcnt_over
          ))
# Set our sequence length to pad or truncate all of our samples to.
max_len = 256

import torch
import time

input_ids = []
attn_masks = []
labels = []

t0 = time.time()

# ======== Encoding ========

print('Encoding {:,} training examples...'.format(len(train)))

# For every training example...
for (index, row) in train.iterrows():

    # Report progress.
    if ((len(input_ids) % 150) == 0):
        print('  Encoded {:,} comments.'.format(len(input_ids)))

    # Convert sentence pairs to input IDs, with attention masks.
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

# ======== Prepare Labels ========

# Select the label columns for all examples.
labels = train[["Negative","Neutral","Positive","Com_","Conn_","Feat_","Not_","O_","P_","P1_","P2_","P3_","P4_","S_","S1_","S2_","U_"]]

labels = labels.to_numpy().astype(float)

# Cast the labels list to a 2D Tensor.
labels = torch.tensor(labels)

# ======== Summary ========

print('\nData structure shapes:')
print('   input_ids:  {:}'.format(str(input_ids.shape)))
print('  attn_masks:  {:}'.format(str(attn_masks.shape)))
print('      labels:  {:}'.format(str(labels.shape)))

print('\nEncoding took {:.0f} seconds'.format(time.time() - t0))

from torch.utils.data import TensorDataset, random_split

# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attn_masks, labels)

# Create a 90-10 train-validation split. Calculate the number of samples to 
# include in each set.
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

