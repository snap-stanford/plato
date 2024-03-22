import torch
from torch_geometric.data import Data
import os
import pickle
import pdb
import numpy as np

class KGDataset():
    def __init__(self, load_from_scratch=False, save=False, embedding_model="ComplEx", load_dir=None):
        self.embedding_model = embedding_model
        self.load_dir = load_dir
        self.data = None
        self.networkx = None
        self.load_edge_set(load_from_scratch, save)
        self.num_entities = len(self.ent2id)
        self.num_relations = len(self.rel2id)

    def create_pyg_object(self):
        print("Creating PyG Object")
        edge_index = [ [ self.ent2id[str(edge[0])] for edge in self.edge_set ], [ self.ent2id[str(edge[2])] for edge in self.edge_set ] ]
        self.data = Data()
        self.data.edge_index = torch.LongTensor(edge_index)
        self.data.edge_attr = torch.LongTensor([ self.rel2id[str(edge[1])] for edge in self.edge_set ])

        try:
            print("Loading KG embedding")
            self.data.x = torch.Tensor(np.load(os.path.join(self.load_dir, "biokg_%s_entity.npy"%self.embedding_model)))
            self.data.relation_embedding = torch.Tensor(np.load(os.path.join(self.load_dir, "biokg_%s_relation.npy"%self.embedding_model)))
            assert self.data.x.shape[0] == len(self.ent2id), "entity embedding shape does not match, (%d, %d)" % (self.data.x.shape[0], len(self.ent2id))
            assert self.data.relation_embedding.shape[0] == len(self.rel2id), "relation embedding shape does not match, (%d, %d)" % (self.data.relation_embedding.shape[0], len(self.rel2id))
        except:
            print("Entity and relation embedding do not exist, randomly initialize them")
            self.data.x = torch.Tensor(np.random.rand(len(self.ent2id), 200))
            self.data.relation_embedding = torch.Tensor(np.random.rand(len(self.rel2id), 200))

    def load_edge_set(self, load_from_scratch=True, save = False):
        print("Loading KG...")
        if not load_from_scratch:
            print("Loading KG from cache")
            print("Loading KG id2ent, ent2id, rel2id, id2rel")
            self.ent2id = pickle.load(open(os.path.join(self.load_dir, "ent2id.pkl"), "rb")) 
            self.rel2id = pickle.load(open(os.path.join(self.load_dir, "rel2id.pkl"), "rb"))
            self.id2ent = pickle.load(open(os.path.join(self.load_dir, "id2ent.pkl"), "rb"))
            self.id2rel = pickle.load(open(os.path.join(self.load_dir, "id2rel.pkl"), "rb"))
            print("Loading indexed edges")
            self.edge_set = pickle.load(open(os.path.join(self.load_dir, "edge_set.pkl"), "rb"))
            self.create_pyg_object()
            return
        else:
            print("ERROR: Trying to load KG from scratch rather than cache!")
            assert(False)