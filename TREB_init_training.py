
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="5" 

import csv
output_csv = "cal_dataframe_result.csv"

from transformers import AutoTokenizer
from transformers import RobertaForMaskedLM
from transformers import RobertaConfig
from transformers import AdamW

from tqdm.auto import tqdm

import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings

    def __len__(self):
        # return the number of samples
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}
    

def read_csv_into_list(filename):
    """
    Reads a CSV file into a list, where each row (excluding the header) is an element.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        list: A list containing the rows of the CSV file.
    """

    data = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip the header row
        for row in reader:
            # print('row', row)
            data.append(row[0])
    return data

data_list = read_csv_into_list(output_csv)


text_data = []
file_count = 0

for sample in tqdm(data_list):
    text_data.append(sample.replace('\n', ''))
    if len(text_data) == 18_000:
        with open( output_csv.replace('.csv', '') + f'_part_{file_count}.txt', 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(text_data))
        text_data = []
        file_count += 1
        
with open(output_csv.replace('.csv', '') + f'_part_{file_count}.txt', 'w', encoding='utf-8') as fp:
    fp.write('\n'.join(text_data))


# tokenizer = AutoTokenizer.from_pretrained("the_tokenizer")
tokenizer = AutoTokenizer.from_pretrained("the_tokenizer_the")

with open('cal_dataframe_result_part_0.txt', 'r', encoding='utf-8') as fp:
    lines = fp.read().split('\n') 

piece = tokenizer(lines, max_length=512, padding='max_length', truncation=True)



labels = torch.tensor([x for x in piece['input_ids']])
mask = torch.tensor([x for x in piece['attention_mask']]) 


# make copy of labels tensor, this will be input_ids
input_ids = labels.detach().clone()

# create random array of floats with equal dims to input_ids
rand = torch.rand(input_ids.shape)

# mask random 15% where token is not 0 [PAD], 1 [CLS], or 2 [SEP]
# mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
# mask_arr = (rand < .15) * (input_ids > 2) 
mask_arr = (rand < .15) * (input_ids > 3) 

# loop through each row in input_ids tensor (cannot do in parallel)
for i in range(input_ids.shape[0]):
    # get indices of mask positions from mask array
    selection = torch.flatten(mask_arr[i].nonzero()).tolist()
    # mask input_ids
    # input_ids[i, selection] = 3  # our custom [MASK] token == 3 
    # input_ids[i, selection] = 4  # our custom [MASK] token == 4 
    input_ids[i, selection] = 103  # our custom [MASK] token == 103

encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels} 

dataset = Dataset(encodings) 
# loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True) 
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True) 



config = RobertaConfig(
    vocab_size=51100,  # we align this to the tokenizer vocab_size
    max_position_embeddings=514,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1
    ) 


model = RobertaForMaskedLM(config) 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device) 


# activate training mode
model.train()
# initialize optimizer
optim = AdamW(model.parameters(), lr=1e-4)

epochs = 500
# epochs = 2

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)   # The leave=True argument means that the last progress bar will be left visible after the loop is finished.
    for batch in loop:
        optim.zero_grad()

        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        
        loss = outputs.loss
        loss.backward()
        optim.step()
        
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item()) 

    if epoch % 10 == 0:
        model.save_pretrained('./the_TREB_model_the_' + str(epoch))

model.save_pretrained('./the_TREB_model_the_ultimate')  