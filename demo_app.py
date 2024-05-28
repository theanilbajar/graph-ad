import numpy as np
from sklearn.metrics import auc, roc_curve
import argparse
import load_data
import torch
import GCN_embedding
from torch.autograd import Variable
from graph_sampler import GraphSampler
import random
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
import mlflow
import networkx as nx
import streamlit as st
from util import sce_loss

from mapping import get_node_map_for_dataset

def arg_parse():
    parser = argparse.ArgumentParser(description='Arguments.')
    parser.add_argument('--datadir', dest='datadir', default ='dataset', help='Directory where benchmark is located')
    parser.add_argument('--DS', dest='DS', default ='BZR', help='dataset name')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int, default=0, help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--clip', dest='clip', default=0.1, type=float, help='Gradient clipping.')
    parser.add_argument('--num_epochs', dest='num_epochs', default=150, type=int, help='total epoch number')
    parser.add_argument('--batch-size', dest='batch_size', default=300, type=int, help='Batch size.')
    parser.add_argument('--hidden-dim', dest='hidden_dim', default=512, type=int, help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', default=256, type=int, help='Output dimension')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', default=3, type=int, help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const', const=False, default=True, help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', default=0.3, type=float, help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const', const=False, default=True, help='Whether to add bias. Default to True.')
    parser.add_argument('--feature', dest='feature', default='default', help='use what node feature')
    parser.add_argument('--seed', dest='seed', type=int, default=1, help='seed')
    return parser.parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def test(data_test_loader, model_teacher, model_student, args): 

    auroc_final = 0
    model_student.eval()   
    loss = []
    y=[]
    emb=[]
    
    for batch_idx, data in enumerate(data_test_loader):
  
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)
        h0 = Variable(data['feats'].float(), requires_grad=False).to(device)
        # st.write(data.keys())
        embed_node, embed = model_student(h0, adj)
        embed_teacher_node, embed_teacher = model_teacher(h0, adj)
        loss_node = torch.mean(sce_loss(embed_node, embed_teacher_node), dim=-1).mean(dim=-1)
        loss_graph = sce_loss(embed, embed_teacher).mean(dim=-1)
        # loss_node = torch.mean(F.mse_loss(embed_node, embed_teacher_node, reduction='none'), dim=2).mean(dim=1).mean(dim=0)
        # loss_graph = F.mse_loss(embed, embed_teacher, reduction='none').mean(dim=1).mean(dim=0)
        loss_ = loss_graph + loss_node
        loss_ = np.array(loss_.cpu().detach())
        loss.append(loss_)
        if data['label'] == 0:
            y.append(1)
        else:
            y.append(0)    
        emb.append(embed.cpu().detach().numpy())
                            
    label_test = []
    for loss_ in loss:
        label_test.append(loss_)
    label_test = np.array(label_test)
                            
    fpr_ab, tpr_ab, _ = roc_curve(y, label_test)       

    if st.button("Predict if it's a potential drug"):
        
        # st.write('Result in format: (original label, loss)')
        
        import pandas as pd
        data_res = pd.DataFrame(data=[(y, label_test)], columns=['Label', 'Anomaly Score'])
        # st.write(tuple(list(zip(y, label_test))))

        anomaly_score = label_test
        if label_test < 0.004:
            label_test = 0
        else:
            label_test = 1
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Actual Label", y[0])
        col2.metric("Predicted Label", label_test)
        col3.metric("Anomaly Score", round(anomaly_score[0], 5))

        if label_test == 1:
            st.write('### Anomalous or not a potential drug')
        else:
            st.write('### A potential drug')
    else:
        st.write("Click on predict button to get the results")
    
    # test_roc_ab = auc(fpr_ab, tpr_ab)   
    # st.write('semi-supervised abnormal detection: auroc_ab: {}'.format(test_roc_ab))
    return auroc_final

# mlflow.set_experiment("GraphAD")
# experiment = mlflow.get_experiment_by_name("GraphAD")

# import mlflow.pytorch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = arg_parse()
# DS = f'{args.DS}'

# print(f'DS: {DS}')
setup_seed(args.seed)

dataset_list = ('BZR', 'AIDS', 'DHFR', 'COX2')

dataset_name = st.sidebar.selectbox(
    "Select Dataset to check?",
    dataset_list
)
DS = dataset_name

st.write("# Graph Anomaly Detection with Dataset - "+ dataset_name)

node_map = get_node_map_for_dataset(dataset=dataset_name)

graphs_v = load_data.read_graphfile_viz('./dataset', dataname=dataset_name, node_map = node_map)

graphs = load_data.read_graphfile(args.datadir, dataname=dataset_name, max_nodes=args.max_nodes)  
datanum = len(graphs)
if args.max_nodes == 0:
    max_nodes_num = max([G.number_of_nodes() for G in graphs])
else:
    max_nodes_num = args.max_nodes

graph_index = st.number_input("Which graph do you want predict?", 0, datanum-1)

st.write(f'### Total graphs: {datanum}')

G = graphs_v[graph_index][0].copy()
labels = graphs_v[graph_index][1].copy()
only_labels = {k: v.split("-")[0] for k, v in labels.items()}

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

only_labels = {k: v.split("-")[0] for k, v in labels.items()}
nx.draw_kamada_kawai(G, labels=only_labels, with_labels = True)
plt.title(f'Structure of a {DS} molecule')
st.pyplot(fig)


graphs_label = [graph.graph['label'] for graph in graphs]

# just take one graph
graphs_label = [graphs[graph_index].graph['label']]
graphss = [graphs[graph_index]]


# kfd=StratifiedKFold(n_splits=1, random_state=args.seed, shuffle = True)
result_auc=[]
for k, (train_index,test_index) in enumerate(zip(graphss, graphs_label)):
    # graphs_train_ = [graphs[i] for i in train_index]
    graphs_test = graphss

    # graphs_train = []
    # for graph in graphs_train_:
    #     if graph.graph['label'] != 0:
    #         graphs_train.append(graph)
    

    # num_train = len(graphs_train)
    # num_test = len(graphs_test)
    # print(num_train, num_test)

    model_teacher = GCN_embedding.GcnEncoderGraph_teacher(3, args.hidden_dim, args.output_dim, 2,
                args.num_gc_layers, bn=args.bn, args=args).to(device)
    for param in model_teacher.parameters():
        param.requires_grad = False

    model_student = GCN_embedding.GcnEncoderGraph_student(3, args.hidden_dim, args.output_dim, 2,
                    args.num_gc_layers, bn=args.bn, args=args).to(device)
            

    model_student.load_state_dict(torch.load(f'./modelstd/{DS}_sce/student.pth'))
    model_student.eval()

    model_teacher.load_state_dict(torch.load(f'./modelstd/{DS}_sce/teacher.pth'))
    model_student.eval()

    dataset_sampler_test = GraphSampler(graphs_test, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)
    data_test_loader = torch.utils.data.DataLoader(dataset_sampler_test, 
                                                    shuffle=False,
                                                    batch_size=1)
    result = test(data_test_loader, model_teacher, model_student, args)     
    result_auc.append(result)
        
result_auc = np.array(result_auc)    
auc_avg = np.mean(result_auc)
auc_std = np.std(result_auc)
# st.write('auroc{}, average: {}, std: {}'.format(result_auc, auc_avg, auc_std))
