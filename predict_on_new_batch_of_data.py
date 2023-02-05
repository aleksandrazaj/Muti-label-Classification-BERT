#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 

new_evaluate = pd.read_csv('data_to_evaluate.csv')

print('There are {:,} total test examples.'.format(len(new_evaluate)))


new_evaluate.info()


# In[42]:


max_len = 256


# In[38]:


import torch

input_ids = []
attn_masks = []
labels = []

# ======== Encoding ========

print('Encoding all {:,} test samples...'.format(len(new_evaluate)))

# For every test sample...
for (index, row) in new_evaluate.iterrows():

    # Report progress.
    if ((len(input_ids) % 5) == 0):
        print('  Tokenized {:,} comments.'.format(len(input_ids)))

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

input_ids = torch.cat(input_ids, dim=0)
attn_masks = torch.cat(attn_masks, dim=0)


# In[40]:


from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

# Combine the training inputs into a TensorDataset.
test_dataset = TensorDataset(input_ids, attn_masks)

# Specify our batch size.
batch_size = 16

#DataLoader
test_dataloader = DataLoader(
            test_dataset,  # The training samples.
            sampler = SequentialSampler(test_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )


import time

# Prediction on test set

# Put model in evaluation mode
model.from_pretrained(output_dir)

t0 = time.time()

# Tracking variables 
predictions , true_labels = [], []

print('Evaluating on {:,} test set batches...'.format(len(test_dataloader)))

# Predict 
for batch in test_dataloader:
    
    # Report progress.
    if ((len(predictions) % 5) == 0):
        print('  Batch {:>5,}  of  {:>5,}.'.format(len(predictions), len(test_dataloader)))

    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask = batch
    
    with torch.no_grad():
        
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)

    logits = outputs[0]

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    
    # Store predictions and true labels
    predictions.append(logits)

print('    DONE.')

print('Evaluation took {:.0f} seconds.'.format(time.time() - t0))

flat_predictions = np.concatenate(predictions, axis=0)

#FLAT PREDICTIONS
print(flat_predictions)

