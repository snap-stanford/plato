SPLITS = [ 'train', 'val', 'test']
SETTING2TRUE_OR_FALSE = {"True": True, "False": False}
INT2SOURCE = {0: "cell", 1: "mouse", 2: "patient"}

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import copy
import random
import math

from plato.utils.torch_utils import dt, get_l1_reg, get_l2_reg
from plato.baseline.evaluator import RegEval
from plato.utils.py_utils import flatten_nested_dict

def set_random_seeds(seed):
    print(f"seed = {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_device(device_in):
    device = f'cuda:{device_in}' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)
    device = torch.device(device)
    torch.set_num_threads(1)
    return device

def get_task(sourceint, source2task):
    return source2task[INT2SOURCE[sourceint]]

def train(model, device, train_loader, optimizer):
    # print("---Train---")
    model.train()

    for x, y, sourceint_list, __, __ in train_loader:
        # x (batch size, number of dimensions)
        # y (batch size)
        # sourceint_list (batch_size)
        optimizer.zero_grad()
        x, y, source = x.to(device), y.to(device), sourceint_list.to(device)
        batch_loss = model.compute_loss(x, y, sourceint_list)
        batch_loss.backward()
        optimizer.step()


@torch.no_grad()
def eval(model, device, loader, sourceint, source2task, writer, epoch):
    # A single source is passed in here with a corresponding loader object
    model.eval()

    # Prepare the vectors corresponding to the model predictions
    y_pred_list, y_prob_list, y_true_list = [], [], []
    drug_list, entity_list, source_list = [], [], []
    for x, y, loader_sourceint_list, entity_idx, drug_idxs in loader:
        assert torch.all(loader_sourceint_list == sourceint)
        x = x.to(device)

        pred = model(x, sourceint)
        assert(get_task(sourceint, source2task) == "numeric")
        y_pred_list.append(pred.view(-1,))
        y_prob_list.append(pred.view(-1))

        y_true_list.append(y)
        drug_list.append(drug_idxs)
        entity_list.append(entity_idx)
        source_list.append(loader_sourceint_list)

    y_pred = torch.cat(y_pred_list, dim = 0).cpu().detach()
    y_prob = torch.cat(y_prob_list, dim = 0).cpu().detach() # predictions for regression
    y_true = torch.cat(y_true_list, dim = 0)
    drug_list = torch.cat(drug_list, dim = 0)
    entity_list = torch.cat(entity_list, dim = 0)
    source_list = torch.cat(source_list, dim = 0)

    # Compute the corresponding eval_dict
    assert(get_task(sourceint, source2task) == "numeric")
    # dt(y_true, "y_true")
    # dt(y_pred, "y_pred")
    metric2score, _ = RegEval().evaluate_all(y_true.numpy(), y_pred.numpy()) # Doublecheck correctness for regression

    # Additionally compute the negative of the loss (positive means better for all metrics in metric2score)
    metric2score["neg_loss"] = 0
    n_samples = 0
    for x, y, sourceint_list, _, _ in loader:
        # x (batch size, number of dimensions)
        # y (batch size)
        # sourceint_list (batch_size)
        # Calculate batch loss
        x, y, source = x.to(device), y.to(device), sourceint_list.to(device)
        n_samples_in_batch = x.shape[0]
        batch_neg_loss_sum = n_samples_in_batch*model.compute_loss(x, y, sourceint_list)

        # Update
        metric2score["neg_loss"] -= batch_neg_loss_sum
        n_samples += n_samples_in_batch

    metric2score["neg_loss"] /= float(n_samples)
    metric2score["neg_loss"] = metric2score["neg_loss"].cpu().detach().item()

    # Add to writer
    for metric, score in metric2score.items():
        if "neg_" in metric:
            metric = metric.split("neg_")[1]
            score = -1*score
        writer.add_scalar(f'{metric}', score, epoch)

    return metric2score, y_prob, y_true, drug_list, entity_list, source_list

def set_drug_representation(provider, drugkg):
    if drugkg:
        print('Use KG embedding to represent drugs')
        # Sum pool over embeddings of drugs that are used
        provider.y_dict['drug'] = (provider.y_dict['drug'].to(torch.float32)).matmul(provider.kg.data.x[provider.kg_mapping_dict['drug']])

    return provider

def subsample_dataset(dataset, sample_frac):
    indices = torch.randint(low = 0, high = len(dataset), size = (int(sample_frac*len(dataset)),))
    dataset = torch.utils.data.Subset(dataset, indices)
    return dataset, indices

def get_split_dataset_and_loader(provider, sourceint_list, split_idx, batch_size, shuffle, sample_frac):
    split_dataset = provider.get_dataset(idx_list = split_idx, sourceint = sourceint_list)
    entity_ids = split_dataset.entity_ids
    drug_names = split_dataset.drug_names

    if sample_frac < 1:
        print("WARNING: sample_frac < 1 only applied to finetune dataset")
        split_dataset, indices = subsample_dataset(split_dataset, sample_frac)
        split_dataset.entity_ids = entity_ids # BUG: entity_ids not necessary correct for split_dataset; need to review
        split_dataset.drug_names = drug_names # BUG: drug_names not necessary correct for split_dataset; need to review

    split_loader = DataLoader(split_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = 0)

    return split_dataset, split_loader

def get_sourceint_list(training_type, finetune_sourceint_list, pretrain_sourceint_list):
    # Get sourceint
    if training_type == "finetune":
        sourceint_list = finetune_sourceint_list
    elif training_type == "pretrain":
        sourceint_list = pretrain_sourceint_list
    else:
        assert(False)

    return sourceint_list

def get_shuffle(split):
    # Get shuffle
    if split == "train":
        shuffle = True
    else:
        shuffle = False

    return shuffle

def get_training_type2split2source2dataset_or_loader(provider, finetune_sourceint_list, pretrain_sourceint_list, train_idx, val_idx, test_idx, batch_size, sample_frac):
    # Initialize
    training_type2split2source2dataset_or_loader = {"finetune": {"train": dict(), "val": dict(), "test": dict()}, "pretrain": {"train": dict(), "val": dict(), "test": dict()}}
    
    for training_type in ["finetune", "pretrain"]:
        sample_frac_to_use = {"finetune": sample_frac, "pretrain": 1}.get(training_type)

        # Get sourceint_list
        sourceint_list = get_sourceint_list(training_type, finetune_sourceint_list, pretrain_sourceint_list)

        # Populate by split
        for split, split_idx in zip(["train", "val", "test"], [train_idx, val_idx, test_idx]):
            # Get shuffle
            shuffle = get_shuffle(split)

            # Iterate over sourceint list
            for sourceint_i in sourceint_list:
                dataset_i, loader_i = get_split_dataset_and_loader(provider, sourceint_list = [sourceint_i], split_idx = split_idx, batch_size = batch_size, shuffle = shuffle, sample_frac = sample_frac_to_use)

                training_type2split2source2dataset_or_loader[training_type][split][sourceint_i] = {"dataset": dataset_i, "loader": loader_i}

    return training_type2split2source2dataset_or_loader

def print_dataset_sizes(training_type2split2source2dataset_or_loader, pretrain_sourceint_list, finetune_sourceint_list):
    for training_type in ["finetune"]: # ["pretrain", "finetune"]
        sourceint_list = get_sourceint_list(training_type, finetune_sourceint_list, pretrain_sourceint_list)
        for source in sourceint_list:
            dataset_split_size_list = np.array([len(training_type2split2source2dataset_or_loader[training_type]["train"][source]["dataset"]), 
                      len(training_type2split2source2dataset_or_loader[training_type]["val"][source]["dataset"]),
                      len(training_type2split2source2dataset_or_loader[training_type]["test"][source]["dataset"])])
            print("raw: {}".format(dataset_split_size_list))
            print("frac: {}".format(dataset_split_size_list/np.sum(dataset_split_size_list)))
            print()

def train_loop(training_type, sourceint_list, model, args, device, training_type2split2source2dataset_or_loader, source2task, results_dict, split2writer):
    # Originally written to enable pretraining on one tabular dataset and fine-tuning on a different tabular dataset. In the end, we just use it to finetune on a single tabular dataset
    # The pretraining mentioned in the PLATO manuscript refers to generation of the node embeddings from the KG; that is NOT the same pretraining described in the code below.
    assert(training_type == "finetune")

    # Set up optimizer
    print("{}...".format(training_type))
    print(f"lr: {args.lr}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Get initial performance
    assert(len(sourceint_list) == 1) # BUG: Will break pretraining (okay for now, just finetuning); need to review all sourceint mentions to restore pretraining
    sourceint = sourceint_list[0]
    # Get initial performance
    _, _, _, _, _, _ = eval(model, device, training_type2split2source2dataset_or_loader[training_type]["train"][sourceint]["loader"], sourceint = sourceint, source2task = source2task, writer = split2writer["train"], epoch = 0)
    _, _, _, _, _, _ = eval(model, device, training_type2split2source2dataset_or_loader[training_type]["test"][sourceint]["loader"], sourceint = sourceint, source2task = source2task, writer = split2writer["test"], epoch = 0)
    _, _, _, _, _, _ = eval(model, device, training_type2split2source2dataset_or_loader[training_type]["val"][sourceint]["loader"], sourceint = sourceint, source2task = source2task, writer = split2writer["val"], epoch = 0)

    # Main training loop
    best_avg_val_score_across_sources = -1*float('inf')
    pbar = tqdm(range(1, 1 + args.epochs))
    train_pr, val_pr = 0, 0
    for epoch in pbar:
        # Train model (one source only)
        assert(len(sourceint_list) == 1)
        pbar.set_description("epoch: %d, train PR: %.4f, val PR: %.4f" % (epoch, train_pr, val_pr))
        train(model, device, training_type2split2source2dataset_or_loader[training_type]["train"][sourceint_list[0]]["loader"], optimizer)

        # Validate (originally written to account for multiple sources but in the end, our problem setting is such that there's only ever a single source). Save performance
        if epoch % args.log_steps == 0:
            # Measure the performance of the model on the validation set
            # 1. Measure validation performance on each source separately
            val_source2metric2score = dict()
            for sourceint in sourceint_list:
                # Get performance for each on training and test (for tensorboard writer only)
                train_metric2score, _, _, _, _, _ = eval(model, device, training_type2split2source2dataset_or_loader[training_type]["train"][sourceint]["loader"], sourceint = sourceint, source2task = source2task, writer = split2writer["train"], epoch = epoch)
                # print("train loss, R2: {:.4f}, {:.4f}".format(-1*train_metric2score["neg_loss"], train_metric2score["pearsonr"]**2))
                train_pr = train_metric2score["pearsonr"]
                _, _, _, _, _, _ = eval(model, device, training_type2split2source2dataset_or_loader[training_type]["test"][sourceint]["loader"], sourceint = sourceint, source2task = source2task, writer = split2writer["test"], epoch = epoch)

                # Get performance for each source on validation
                metric2score, _, _, _, _, _ = eval(model, device, training_type2split2source2dataset_or_loader[training_type]["val"][sourceint]["loader"], sourceint = sourceint, source2task = source2task, writer = split2writer["val"], epoch = epoch)
                val_pr = metric2score["pearsonr"]
                val_source2metric2score[sourceint] = metric2score

            # 2. Select best model by taking average of validation score across the sources
            avg_val_loss_across_sources = np.mean([-1*metric2score["neg_loss"] for source, metric2score in val_source2metric2score.items()])
            avg_val_score_across_sources = np.mean([metric2score[args.selection_metric] for source, metric2score in val_source2metric2score.items()])
            # print('This Run\'s Average Val Loss, Score Across Sources: {:.2f}, {:.2f}'.format(avg_val_loss_across_sources, avg_val_score_across_sources))

            # If average validation score of the model across sources is the best seen so far, then...
            if avg_val_score_across_sources > best_avg_val_score_across_sources:
                # Save the model and best average validation score
                best_avg_val_score_across_sources = avg_val_score_across_sources
                best_model_state_dict = copy.deepcopy(model.state_dict())

                # print("New best_avg_val_score_across_sources: {}".format(best_avg_val_score_across_sources))
                
    # Load the best model
    model.load_state_dict(best_model_state_dict)

    # Measure performance of the best model on all sources and splits
    source2split2metric2score = dict()
    source2split2vec_name2vec = dict()
    for sourceint in sourceint_list:
        split2metric2score = dict()
        split2vec_name2vec = dict()
        for split in SPLITS:
            vec_name2vec = dict()
            metric2score, vec_name2vec["y_prob"], vec_name2vec["y_true"], vec_name2vec["drugs"], vec_name2vec["entities"], vec_name2vec["sources"] = eval(model, device, training_type2split2source2dataset_or_loader[training_type][split][sourceint]["loader"], sourceint = sourceint, source2task = source2task, writer = split2writer[split], epoch = epoch)
            split2metric2score[split] = metric2score
            split2vec_name2vec[split] = vec_name2vec
        source2split2metric2score[sourceint] = split2metric2score
        source2split2vec_name2vec[sourceint] = split2vec_name2vec

    # Print performance of best model
    print("Overall performance...")
    print(f'best avg val {args.selection_metric} across sources', best_avg_val_score_across_sources)

    # Update results_dict
    assert(not(training_type in results_dict))
    results_dict[training_type] = {"source2split2metric2score": source2split2metric2score, "source2split2vec_name2vec": source2split2vec_name2vec}

    return model, results_dict

def add_to_agg_source2split2metric2score(agg_source2split2metric2score, source, split, metric, score):
    if source in agg_source2split2metric2score:
        if split in agg_source2split2metric2score[source]:
            if metric in agg_source2split2metric2score[source][split]:
                agg_source2split2metric2score[source][split][metric].append(score)
            else:
                agg_source2split2metric2score[source][split][metric] = [score]
        else:
            agg_source2split2metric2score[source][split] = {metric : [score]}
    else:
        agg_source2split2metric2score[source] = {split: {metric: [score]}}
    return agg_source2split2metric2score

def agg_source2split2metric2score(results_dict_list, training_type):
    agg_source2split2metric2score = dict()
    for results_dict in results_dict_list:
        source2split2metric2score = results_dict[training_type]["source2split2metric2score"]
        for source, split2metric2score in source2split2metric2score.items():
            for split, metric2score in split2metric2score.items():
                for metric, score in metric2score.items():
                    agg_source2split2metric2score = add_to_agg_source2split2metric2score(agg_source2split2metric2score, source, split, metric, score)
    
    return agg_source2split2metric2score

def agg_split2y_pred(results_dict_list, training_type):
    split2y_pred_list = []
    for results_dict in results_dict_list:
        split2y_pred_list.append(results_dict[training_type]["split2y_pred"])
    return split2y_pred_list

def agg_split2const_list(results_dict_list, training_type, k):
    # Keys with const lists
    assert(k in ["split2y_true", "split2sources", "split2drugs", "split2entities"])

    # Ensure that all of them are truly the same
    const_list = None
    for results_dict in results_dict_list:
        assert(not(results_dict[training_type][k] is None))
        if const_list is None:
            const_list = results_dict[training_type][k]
        assert(const_list == results_dict[training_type][k])

    return const_list

def validate_keys_in_list_of_eq_dicts(dict_list):
    # Ensure that all of the results_dicts have the same keys (must flatten first to compare structure)
    comp_keys = None
    for dict_ in dict_list:
        keys_i = set(flatten_nested_dict(dict_).keys())
        if comp_keys is None:
            comp_keys = keys_i
        assert(comp_keys == keys_i)

def save_results(results_dict_list, training_type2split2source2dataset_or_loader, args, splits):
    # Ensure that all of the results_dicts have the same keys (must flatten first to compare structure)
    validate_keys_in_list_of_eq_dicts(results_dict_list)

    # For each key, iterate over the results dicts, aggregate, and populate agg_results_dict
    rep_results_dict = results_dict_list[0]
    agg_results_dict = {k: dict() for k in rep_results_dict.keys()}
    for training_type in agg_results_dict.keys():
        for k in rep_results_dict[training_type].keys():
            if k == "source2split2metric2score":
                agg_results_dict[training_type]["agg_" + k] = agg_source2split2metric2score(results_dict_list, training_type)
            elif k == "split2y_pred":
                agg_results_dict[training_type][k + "_list"] = agg_split2y_pred(results_dict_list, training_type)
            elif k in ["split2y_true", "split2sources", "split2drugs", "split2entities"]:
                agg_results_dict[training_type][k] = agg_split2const_list(results_dict_list, training_type, k)
            else:
                # TODO: Add functionality (used to print "MISSING FUNCTIONALITY")
                print(k)

    # Generate entity and drug mappings and add
    idx2entity_map = dict()
    idx2drug_map = dict()
    for split in splits:
        assert(len(args.finetune_sourceint_list) == 1)
        dataset = training_type2split2source2dataset_or_loader["finetune"][split][args.finetune_sourceint_list[0]]["dataset"]
        idx2entity_map[split] = dataset.entity_ids
        idx2drug_map[split] = dataset.drug_names
    agg_results_dict['idx2entity'] = idx2entity_map
    agg_results_dict['idx2drug'] = idx2drug_map

    # Add input arguments
    agg_results_dict["args"] = vars(args)

    # Save
    if args.filename != '':
        torch.save(agg_results_dict, args.filename)

    return agg_results_dict

def get_gene_ixs(x, gene_dim):
    return x[:, :gene_dim]

def get_drug_ixs(x, gene_dim):
    return x[:, gene_dim:]

def get_gene_embed(provider):
    # Matrix where every row corresponds to a gene embedding. Rows are ordered in the same way as the gene expression input vector
    gene_embed = provider.kg.data.x[provider.kg_mapping_dict['X_expression']] # (n_genes, d)
    # dt(gene_embed, "gene_embed")

    return gene_embed

def get_drug_embed(provider):
    # Matrix where every row corresponds to a drug embedding. Rows are ordered in the same way as in the multihot drug input vector
    drug_embed = provider.kg.data.x[provider.kg_mapping_dict['drug']]

    return drug_embed

class ParentDecoder(torch.nn.Module):
    def __init__(self):
        super(ParentDecoder, self).__init__()

    def get_loss_list(self):
        # Set loss according to regression task
        reg_criterion = torch.nn.MSELoss(reduction = 'mean')

        # Populate loss list
        loss_list = []
        for source in ["cell", "mouse", "patient"]:
            assert(self.source2task[source] == "numeric")
            loss_list.append(reg_criterion)

        return loss_list

    def compute_loss_without_reg(self, x, y, sourceint_list):
        '''
            Calculates loss across samples.

            sourceint_list (torch tensor): stores where i-th element comes from 
                - 0: cell line
                - 1: mouse
                - 2: patient
        '''
        loss_total = 0
        unique_sourceint_list = torch.unique(sourceint_list)

        # Handle one source and therefore one task (i.e. regression) at a time
        for sourceint in unique_sourceint_list:
            # true-false vector
            tf_idx = sourceint_list == sourceint

            # extracting features for sourceint
            x_sub, y_sub = x[tf_idx], y[tf_idx]

            pred = self(x_sub, sourceint)

            assert(get_task(sourceint.item(), self.source2task) == "numeric") # Need the .item() here because coming from Torch tensor
            pred = pred.view(-1,) # Convert from (n, 1) to (n) so loss computed correctly

            assert(pred.shape == y_sub.shape)
            loss = self.loss_list[int(sourceint.item())](pred, y_sub)
            
            loss_total += loss
        
        return loss_total

    def compute_loss(self, x, y, sourceint_list):
        # Get sum without regularization
        loss = self.compute_loss_without_reg(x, y, sourceint_list)
        
        # Get l1, l2 regularizations
        l1_loss = 0.0
        l2_loss = 0.0
        for name, parameter in self.named_parameters():
            l1_loss += get_l1_reg(parameter, self.l1_weight)
            l2_loss += get_l2_reg(parameter, self.l2_weight)

        # Sum
        loss = loss + l1_loss + l2_loss

        return loss
