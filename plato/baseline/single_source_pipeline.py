import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import numpy as np
import torch
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from plato.models.ggmlp import GGMLP
from plato.load.load_pdr_sub import PDRSubdatasetProvider
from plato.baseline.pipeline_utils import SPLITS, get_device, set_drug_representation, get_training_type2split2source2dataset_or_loader, print_dataset_sizes, train_loop, save_results, set_random_seeds
from torch_geometric.utils.subgraph import subgraph

def parse_m_layer_list(m_layer_list):
    m_layer_list_split = m_layer_list.split(",")
    if m_layer_list_split != ['']:
        m_layer_list_split = [int(i) for i in m_layer_list_split]
    else:
        m_layer_list_split = []
    return m_layer_list_split

def update_args(args):
    args.mlp_m_layer_list = parse_m_layer_list(args.mlp_m_layer_list)

    # Set subtype arguments
    args.mode = "subtype"
    if args.dataset_name in ["BC", "CH", "ME", "NSCLC", "SCLC"]:
        args.pretrain_sourceint_list = [0]
        args.finetune_sourceint_list = [0]
        args.subtype_category = "cancer-type"
    elif args.dataset_name in ["CRC", "PDAC", "BRCA", "MNSCLC", "CM"]:
        args.pretrain_sourceint_list = [1]
        args.finetune_sourceint_list = [1]
        args.subtype_category = "Tumor Type"
    else:
        assert(False)
    dataset_name2subtype_name = {"BC": "Breast Carcinoma", "CH": "Chondrosarcoma", "ME": "Melanoma", "NSCLC": "Non-Small Cell Lung Carcinoma", "SCLC": "Small Cell Lung Carcinoma", "CRC": "CRC", "PDAC": "PDAC", "BRCA": "BRCA", "MNSCLC": "NSCLC", "CM": "CM"}
    args.subtype_name = dataset_name2subtype_name[args.dataset_name]
    return args

def validate_args(args):
    if args.model == "GGMLP":
        assert(not(args.drugkg))
    if args.mode == "subtype":
        assert(args.pretrain_sourceint_list == args.finetune_sourceint_list)

def get_args():
    parser = argparse.ArgumentParser(description='Single source prediction pipeline')
    # General
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--filename', type=str, default='')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--save_step', type=int, default=19999)
    parser.add_argument('--selection_metric', type = str, default = 'neg_mse') # pearsonr
    parser.add_argument('--load_only', action='store_true', default = False)
    parser.add_argument('--tensorboard_dir', default = None)
    parser.add_argument('--dataset_name', choices = ['BC', 'CH', 'ME', 'NSCLC', 'SCLC', 'CRC', 'PDAC', 'BRCA', 'MNSCLC', 'CM'])
    
    # Training hyperparameter arguments
    parser.add_argument('--lr', type=float, default=0.001) 
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=10)
    
    # Pipeline arguments
    parser.add_argument('--sample_frac', type=float, default=1)
    parser.add_argument('--drugkg', action='store_true', default = False) # use drug kg embedding

    # KG Embedding arguments
    parser.add_argument('--kg_embed_size', type=int, default=200)
    parser.add_argument('--embedding_model', choices = ["ComplEx"], default = "ComplEx")

    # Model arguments
    parser.add_argument('--model', choices = ["GGMLP"], default = 'GGMLP')
    parser.add_argument('--skip_connection', action='store_true', default = False) # use drug kg embedding
    parser.add_argument('--use_bn', action='store_true', default = False) 
    parser.add_argument('--simple', action='store_true', default = False) 
    parser.add_argument('--train_meta', action='store_true', default = False) 
    parser.add_argument('--gene_nonlin', type = str, choices = ["none", "softmax", "relu", "leakyrelu", "tanh"], default = 'none')
    parser.add_argument('--drug_nonlin', type = str, choices = ["none", "softmax", "relu", "leakyrelu", "tanh"], default = 'none')
    parser.add_argument('--gene_div', type=float, default = 1.)
    parser.add_argument('--drug_div', type=float, default = 1.)
    parser.add_argument('--enlarge', type=int, default=20)
    parser.add_argument('--beta', type=float, default = 0.)
    parser.add_argument('--mp', action='store_true', default = False) 
    parser.add_argument('--scalar_attn', action='store_true', default = False) 
    parser.add_argument('--num_edges_sampled', type=int, default=50000)
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--l1_weight', type=float, default = 0.0)
    parser.add_argument('--l2_weight', type=float, default = 0.0)
    parser.add_argument('--mlp_m_layer_list', type = str, default = '32,32,1')

    # Output
    args = parser.parse_args()
    args = update_args(args)
    print(args)
    return args
        
if __name__ == "__main__":
    # Set up
    args = get_args()
    validate_args(args)
    device = get_device(args.device)
    assert(type(args.pretrain_sourceint_list) is list)
    assert(type(args.finetune_sourceint_list) is list)

    # Set seed
    set_random_seeds(args.seed)

    if os.path.exists(args.filename):
        print(f"************************{args.filename} already exists!************************")
    else:
        # Load Data
        if args.mode == "subtype":
            assert(len(args.finetune_sourceint_list) == 1)
            subtype_source = {0: "cell", 1: "mouse", 2: "patient"}[args.finetune_sourceint_list[0]]
            args.subtype_response_col = {"cell": "ln-ic50", "mouse": "min-avg-pct-tumor-growth", "patient": "PFI.time"}[subtype_source]
            provider = PDRSubdatasetProvider(subtype_source, args.subtype_category, args.subtype_name, args.subtype_response_col, load_dir=args.load_dir, cache_dir=args.cache_dir)
            provider.source2task = {"cell": "numeric", "mouse": "numeric", "patient": "numeric"}
        else:
            assert(False)
        print(provider.source2task)

        # Split Data
        split_dict = provider.get_split_idx(from_scratch = True, seed = args.seed)
        train_idx, val_idx, test_idx = split_dict['train'], split_dict['val'], split_dict['test']
        train_entity_idx, val_entity_idx, test_entity_idx = split_dict['train_entity'], split_dict['val_entity'], split_dict['test_entity']

        # Set up drug representation
        provider = set_drug_representation(provider, args.drugkg)
        gene_dim = provider.X_expression.size(1)
        drug_dim = provider.y_dict["drug"].size(1)
        print(provider.y_dict['drug'])

        # Set up datasets for training and pre-training
        training_type2split2source2dataset_or_loader = get_training_type2split2source2dataset_or_loader(provider, args.finetune_sourceint_list, args.pretrain_sourceint_list, train_idx, val_idx, test_idx, args.batch_size, args.sample_frac)
        print("Printing dataset sizes...")
        print_dataset_sizes(training_type2split2source2dataset_or_loader, args.pretrain_sourceint_list, args.finetune_sourceint_list)
        if args.mp:
            gene_edge_index, _ = subgraph(torch.LongTensor(provider.kg_mapping_dict['X_expression']), provider.kg.data.edge_index, relabel_nodes=True)
            drug_edge_index, _ = subgraph(torch.LongTensor(provider.kg_mapping_dict['drug']), provider.kg.data.edge_index, relabel_nodes=True)
        else:
            gene_edge_index = None
            drug_edge_index = None
        
        assert(args.model == "GGMLP")
        model = GGMLP(m_layer_list = args.mlp_m_layer_list, node_dim = args.kg_embed_size, gene_dim = gene_dim, drug_dim = drug_dim, provider = provider, device = device, skip_connection=args.skip_connection, use_bn=args.use_bn, simple=args.simple, train_meta=args.train_meta, drug_nonlin=args.drug_nonlin, gene_nonlin=args.gene_nonlin, drug_div=args.drug_div, gene_div=args.gene_div, l1_weight=args.l1_weight, l2_weight=args.l2_weight, enlarge=args.enlarge, mp=args.mp, beta=args.beta, gene_edge_index=gene_edge_index, drug_edge_index=drug_edge_index, num_edges_sampled=args.num_edges_sampled, scalar_attn=args.scalar_attn).to(device)
        print(model)
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))

        # Set up the tensorboard writer
        now = datetime.now()
        date_time = now.strftime("%m.%d.%Y.%H.%M.%S")

        # Exit if just loading data
        if args.load_only:
            assert(False)

        # For each run, do pretraining and then fine-tuning
        results_dict_list = []
        for run in range(args.runs):
            # Finish setting up the tensorboard writer
            split2writer = {"train": SummaryWriter(f"{args.tensorboard_dir}/{date_time}_model_{args.model}_run_{run}.train"), "val": SummaryWriter(f"{args.tensorboard_dir}/{date_time}_model_{args.model}_run_{run}.val"), "test": SummaryWriter(f"{args.tensorboard_dir}/{date_time}_model_{args.model}_run_{run}.test")}

            print(f'==============run{run}')
            model.reset_parameters()
            results_dict = dict()

            # Fine-tune
            model, results_dict = train_loop("finetune", args.finetune_sourceint_list, model, args, device, training_type2split2source2dataset_or_loader, provider.source2task, results_dict, split2writer)

            # Add to list of results_dict (i.e. one per run)
            results_dict_list.append(results_dict)
            
        # Save overall
        agg_results_dict = save_results(results_dict_list, training_type2split2source2dataset_or_loader, args, SPLITS)
        # torch.save(results_dict_list, args.filename.split(".pt")[0]+"_results_dict_list.pt")

        assert(len(args.finetune_sourceint_list) == 1)
        sourceint = args.finetune_sourceint_list[0]
        val_acc = np.array(agg_results_dict['finetune']['agg_source2split2metric2score'][sourceint]['val']['pearsonr'])
        print(f'Val pearsonr: {val_acc.mean()} pm {val_acc.std()}')
        test_acc = np.array(agg_results_dict['finetune']['agg_source2split2metric2score'][sourceint]['test']['pearsonr'])
        print(f'Test pearsonr: {test_acc.mean()} pm {test_acc.std()}')