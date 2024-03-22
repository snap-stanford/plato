import os
import torch

from plato.load.load_kg import KGDataset

class DatasetProvider():
    def __init__(self, embedding_model, cache_file, name, load_from_scratch, skip_kg, load_dir=None, cache_dir=None):
        # Load KG
        self.cache_dir = cache_dir
        if not skip_kg:
            self._load_kg(embedding_model=embedding_model, load_dir=load_dir)

        # Either load from cache or load from raw
        if os.path.exists(cache_file) and (not load_from_scratch):
            print(f'{name}: Loading cached file at {cache_file}')
            load_dict = torch.load(cache_file)
            self.__dict__.update(load_dict)
        else:
            print("Cached file missing! Ensure arguments set correctly to point to file")
            assert(False)

    def _load_kg(self, embedding_model, load_dir=None):
        self.kg = KGDataset(load_from_scratch = False, embedding_model=embedding_model, load_dir=load_dir)

    def get_split_idx(self):
        raise NotImplementedError

    def subset_idx_list_to_sourceint(self):
        raise NotImplementedError

    def get_dataset(self):
        raise NotImplementedError