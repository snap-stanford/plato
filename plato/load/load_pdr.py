import torch
import numpy as np
import os

from plato.load.provide import DatasetProvider
from plato.load.dataset import Dataset

RESPONSE_COL2TYPE = {"ln-ic50": "numeric", "auc": "numeric", "min-pct-tumor-growth": "numeric", "min-avg-pct-tumor-growth": "numeric", "response-type": "categorical", "OS.time": "numeric", "DSS.time": "numeric", "DFI.time": "numeric", "PFI.time": "numeric"}
CELL_RESPONSE_COL_OPTIONS = ["ln-ic50", "auc"]
MOUSE_RESPONSE_COL_OPTIONS = ["min-pct-tumor-growth", "min-avg-pct-tumor-growth", "response-type"]
PATIENT_RESPONSE_COL_OPTIONS = ["response-type", "OS.time", "DSS.time", "DFI.time", "PFI.time"]
QUANTIZE_OPTIONS = [True, False]
CENTER_AND_STANDARDIZE_OPTIONS = [True, False]    

class PDRDatasetProviderOptions():
    def __init__(self):
        pass

    def get_response_cols_valid(self, cell_response_col, mouse_response_col, patient_response_col):
        # Ensures the column options are within the df
        valid = (cell_response_col in CELL_RESPONSE_COL_OPTIONS) and (mouse_response_col in MOUSE_RESPONSE_COL_OPTIONS) and (patient_response_col in PATIENT_RESPONSE_COL_OPTIONS)
        
        return valid

    def get_numeric_modifications_valid(self, response_col2type, response_col, quantize, center_and_standardize):
        # Only numerical columns can have the quantize and center_and_standardize options
        assert(response_col2type[response_col] == "numeric")
        if (quantize):
            valid = not(center_and_standardize)
        else:
            valid = True

        return valid

    def get_all_numeric_modifications_valid(self, source2response_col, response_col2type, cell_quantize, mouse_quantize, cell_center_and_standardize, mouse_center_and_standardize):
        cell_valid = self.get_numeric_modifications_valid(response_col2type, source2response_col["cell"], cell_quantize, cell_center_and_standardize)
        mouse_valid = self.get_numeric_modifications_valid(response_col2type, source2response_col["mouse"], mouse_quantize, mouse_center_and_standardize)
        valid = cell_valid and mouse_valid

        return valid

class PDRDatasetProvider(DatasetProvider):    
    def __init__(self, load_from_scratch=False, skip_kg = False, embedding_model="ComplEx", cell_response_col = "ln-ic50", mouse_response_col = "min-avg-pct-tumor-growth", patient_response_col = "PFI.time", cell_quantize = False, mouse_quantize = False, patient_quantize = False, cell_center_and_standardize = True, mouse_center_and_standardize = True, patient_center_and_standardize = True, cache_file = None, cache_dir=None):
        '''
            - X_expression
                (num_entities, ~18000)
            - sourceint_of_entity_list
                (num_entities, )
            - y_dict
                - entity
                - drug
                - response
            - mapping_dict
                - entity
                - X_expression
                - drug
            - kg_mapping_dict
                - X_expression
                - drug

            Loads the dataset into a PyTorch compatible format. Multiple options are provided for loading including:
            1. What label to use for the cell line data of {"auc", "ln-ic50"}
            2. What label to use for the mouse data of {"min-pct-tumor-growth", "min-avg-pct-tumor-growth", "response-type"}
            3. Whether to "quantize" the label (i.e. if the label is a numerical value, whether to bin it into quartiles)?
            4. Whether to "center and standardize" the label (i.e. if the label is a numerical value, whether to make it have zero mean and unit standard deviation)?

            When options are passed, they are tested for validity according to rules given in PDRDatasetProviderOptions. PDRDatasetProviderOptions also has a function to generate all possible valid combinations of options.
        '''

        ## Response cols
        options = PDRDatasetProviderOptions()
        assert(options.get_response_cols_valid(cell_response_col, mouse_response_col, patient_response_col))
        self.source2response_col = {"cell": cell_response_col, "mouse": mouse_response_col, "patient": patient_response_col}
        self.response_col2type = RESPONSE_COL2TYPE

        ## Options for modifying numerics
        assert(options.get_all_numeric_modifications_valid(self.source2response_col, self.response_col2type, cell_quantize, mouse_quantize, cell_center_and_standardize, mouse_center_and_standardize))
        self.cell_quantize = cell_quantize
        self.mouse_quantize = mouse_quantize
        self.patient_quantize = patient_quantize

        self.cell_center_and_standardize = cell_center_and_standardize
        self.mouse_center_and_standardize = mouse_center_and_standardize
        self.patient_center_and_standardize = patient_center_and_standardize

        ## Cache file
        self.cache_file = cache_file
        self.cache_dir = cache_dir
        self.embedding_model = embedding_model
        
        # Load from scratch or load the cached file
        super().__init__(self.embedding_model, self.cache_file, "PDRDatasetProvider", load_from_scratch, skip_kg)

    def get_split_idx(self, split_type='random', index_by='pair', from_scratch = False, seed = 0):
        '''
            args
                - split_type: how to split the data
                - index_by: either entity or pair.
            return:
                - split_dict
                    'train': torch long tensor
                    'val': torch long tensor
                    'test': torch long tensor

        '''
        print("index_by: {}".format(index_by))
        split_type_candidate = ['random']
        index_by_candidate = ['entity', 'pair']

        assert(split_type in split_type_candidate)
        assert(index_by in index_by_candidate)

        SPLIT_DIR = os.path.join(self.cache_dir, 'split')
        os.makedirs(SPLIT_DIR, exist_ok=True)
        SPLIT_FILE = os.path.join(SPLIT_DIR, "pdr_" + split_type + str(seed) + '.pt')

        if not os.path.exists(SPLIT_FILE) or from_scratch:
            print(f'Generating split for {split_type}, seed {seed}')
            if split_type == 'random':
                torch.manual_seed(seed)
                num_entities = len(self.X_expression)
                perm = torch.randperm(num_entities)
                # entity-level indexing
                entity_train_idx = perm[:int(0.6*num_entities)]
                entity_valid_idx = perm[int(0.6*num_entities):int(0.8*num_entities)]
                entity_test_idx = perm[int(0.8*num_entities):]
            split_dict = {'train': entity_train_idx, 'val': entity_valid_idx, 'test': entity_test_idx}
            print('NOT saving split...')
            # torch.save(split_dict, SPLIT_FILE)
        else:
            # print('loading saved split index')
            # split_dict = torch.load(SPLIT_FILE)
            print("Split saving functionality not implemented for subtypes")
            assert(False)
        
        # entity-level index
        if index_by == 'entity':
            return split_dict
        elif index_by == 'pair':
            # turning into (entity, drug) pair-level indexing
            y_dict_entity = self.y_dict['entity'].numpy()
            # => y_dict_entity = np.array where each element is the index of an entity
            new_split_dict = {}
            for key in split_dict.keys():
                new_split_dict[f'{key}_entity'] = split_dict[key]
                entity_idx = split_dict[key].numpy()
                # split (entity, drug) pair-level by entity in entity_idx 
                new_split_dict[key] = torch.from_numpy(np.isin(y_dict_entity, entity_idx).nonzero()[0]).to(torch.long)
            
            assert(sum([len(new_split_dict[key]) for key in ['train', 'val', 'test']] )== len(self.y_dict['entity']))

            # make sure the disjoint entities across splits:
            # train_entities = np.unique(self.y_dict['entity'][new_split_dict['train']].numpy())
            # val_entities = np.unique(self.y_dict['entity'][new_split_dict['val']].numpy())
            # test_entities = np.unique(self.y_dict['entity'][new_split_dict['test']].numpy())

            # print(np.intersect1d(train_entities, val_entities))
            # print(np.intersect1d(train_entities, test_entities))
            # print(np.intersect1d(test_entities, val_entities))

            return new_split_dict

    def subset_idx_list_to_sourceint(self, idx_list, sourceint):
        assert(sourceint in [0, 1, 2])
        idx_list = idx_list[self.sourceint_of_entity_list[self.y_dict['entity'][idx_list]] == sourceint]

        return idx_list

    def get_dataset(self, idx_list, sourceint = None):
        # idx_list is torch tensor specifying the subset of the entire dataset
        # indexing is done at (entity, drug) pair
        # if sourceint is provided, then we only consider data with the specified type
        # sourceint can be list
        # sourceint can be an int (one type of data)

        if sourceint is not None:
            if type(sourceint) == int:
                idx_list = self.subset_idx_list_to_sourceint(idx_list, sourceint)
            elif type(sourceint) == list:
                tmp = []
                for elem in sourceint:
                    tmp.append(self.subset_idx_list_to_sourceint(idx_list, elem))
                idx_list = torch.cat(tmp)
                
        return PDRDataset(self.X_expression, self.y_dict, self.kg, self.sourceint_of_entity_list, idx_list, self.kg_mapping_dict, self.mapping_dict)

    def __repr__(self):
        return f'{str(self.__class__.__name__)}()'

class PDRDataset(Dataset):
    def __init__(self, X_expression, y_dict, kg, sourceint_of_entity_list, idx_list, kg_mapping_dict, mapping_dict):
        self.X_expression = X_expression
        self.y_dict = y_dict
        self.kg = kg
        self.kg_mapping_dict = kg_mapping_dict
        self.mapping_dict = mapping_dict
        self.sourceint_of_entity_list = sourceint_of_entity_list
        self.idx_list = idx_list
        self.drug_names = np.array([ self.kg.id2ent[x] for x in self.kg_mapping_dict['drug'] ])
        self.entity_ids = self.mapping_dict['entity']
        
    def __getitem__(self, i):
        idx = self.idx_list[i]
        entity_idx = self.y_dict['entity'][idx]
        drug_idxs = self.y_dict['drug_ids'][idx]
        X_expression = self.X_expression[entity_idx].to(torch.float32)
        X_drug = self.y_dict['drug'][idx].to(torch.float32)
        response = self.y_dict['response'][idx]

        # simply concatenate
        X = torch.cat([X_expression, X_drug])

        return X, response, self.sourceint_of_entity_list[entity_idx], entity_idx, drug_idxs