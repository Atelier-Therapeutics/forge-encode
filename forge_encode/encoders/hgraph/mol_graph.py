import torch
import rdkit
import rdkit.Chem as Chem
import networkx as nx
from .chemutils import *
from .nnutils import *
from tqdm.auto import tqdm

add = lambda x, y : x + y if type(x) is int else (x[0] + y, x[1] + y)

class MolGraph(object):

    BOND_LIST = [
        Chem.rdchem.BondType.SINGLE, 
        Chem.rdchem.BondType.DOUBLE, 
        Chem.rdchem.BondType.TRIPLE, 
        Chem.rdchem.BondType.AROMATIC
    ]
    
    MAX_POS = 20

    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = get_mol(smiles)

        # cluster based decomposition
        self.mol_graph = self.build_mol_graph()
        self.clusters, self.atom_cls = self.find_clusters()
        self.mol_tree = self.tree_decomp()
        self.order = self.label_tree()
    
    def build_mol_graph(self):
        mol = self.mol
        graph = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(mol))
        for atom in mol.GetAtoms():
            graph.nodes[atom.GetIdx()]['label'] = (atom.GetSymbol(), atom.GetFormalCharge())
        
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = MolGraph.BOND_LIST.index( bond.GetBondType() )
            graph[a1][a2]['label'] = btype
            graph[a2][a1]['label'] = btype
        return graph
    
    def find_clusters(self):
        mol = self.mol
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1: 
            return [(0,)], [[0]]
        
        clusters = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if not bond.IsInRing():
                clusters.append( (a1, a2) ) # if bond is not in ring, add its atoms to clusters list
        
        ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)]
        clusters.extend(ssr) # add all rings to clusters list

        # atom index 0 containing cluster is the root cluster (position 0)
        if 0 not in clusters[0]:
            for i, cls in enumerate(clusters):
                if 0 in cls:
                    clusters = [clusters[i]] + clusters[:i] + clusters[i+1:]
                    break
        
        # map atom index to cluster index
        atom_cls = [[] for i in range(n_atoms)]
        for i in range(len(clusters)):
            for atom in clusters[i]:
                atom_cls[atom].append(i)

        return clusters, atom_cls
    
    def tree_decomp(self):
        clusters = self.clusters
        graph = nx.empty_graph( len(clusters) )
        for atom, nei_cls in enumerate(self.atom_cls):
            if len(nei_cls) <= 1: continue
            bonds = [c for c in nei_cls if len(clusters[c]) == 2] # if atom has 2 neighbors, it is a bond
            rings = [c for c in nei_cls if len(clusters[c]) > 4] # if atom has more than 4 neighbors, it is a ring

            # if atom has more than 2 neighbors and has at least 2 bonds, it is a bond
            if len(nei_cls) > 2 and len(bonds) >= 2:
                clusters.append([atom])
                c2 = len(clusters) - 1
                graph.add_node(c2)
                for c1 in nei_cls:
                    graph.add_edge(c1, c2, weight=100)
            # if atom has more than 4 neighbors and has at least 2 rings, it is a ring
            elif len(rings) > 2:
                clusters.append([atom])
                c2 = len(clusters) - 1
                graph.add_node(c2)
                for c1 in nei_cls:
                    graph.add_edge(c1, c2, weight=100)
            else:
                for i, c1 in enumerate(nei_cls):
                    for c2 in nei_cls[i + 1:]:
                        inter = set(clusters[c1]) & set(clusters[c2])
                        graph.add_edge(c1, c2, weight = len(inter))
        n, m = len(graph.nodes), len(graph.edges)
        assert n - m <= 1
        return graph if n - m == 1 else nx.maximum_spanning_tree(graph)
    
    def label_tree(self):
        def dfs(order, pa, prev_sib, x, fa):
            pa[x] = fa
            sorted_child = sorted( [y for y in self.mol_tree[x] if y != fa])
            for idx, y in enumerate(sorted_child):
                self.mol_tree[x][y]['label'] = 0
                self.mol_tree[y][x]['label'] = idx + 1
                prev_sib[y] = sorted_child[:idx]
                prev_sib[y] += [x, fa] if fa >= 0 else [x]
                order.append( (x, y, 1) )
                dfs(order, pa, prev_sib, y, x)
                order.append( (y, x, 0) )
        
        order, pa = [], {}
        self.mol_tree = nx.DiGraph(self.mol_tree)
        prev_sib = [[] for i in range(len(self.clusters))]
        dfs(order, pa, prev_sib, 0, -1)

        order.append( (0, None, 0) )

        mol = get_mol(self.smiles)
        for a in mol.GetAtoms():
            a.SetAtomMapNum( a.GetIdx() + 1)
        
        tree = self.mol_tree
        for i,cls in enumerate(self.clusters):
            inter_atoms = set(cls) & set(self.clusters[pa[i]]) if pa[i] >= 0 else set([0])
            cmol, inter_label = get_inter_label(mol, cls, inter_atoms)
            tree.nodes[i]['ismiles'] = ismiles = get_smiles(cmol)
            tree.nodes[i]['inter_label'] = inter_label
            tree.nodes[i]['smiles'] = smiles = get_smiles(set_atommap(cmol))
            tree.nodes[i]['label'] = (smiles, ismiles) if len(cls) > 1 else (smiles, smiles)
            tree.nodes[i]['cluster'] = cls 
            tree.nodes[i]['assm_cands'] = []

            if pa[i] >= 0 and len(self.clusters[ pa[i] ]) > 2: #uncertainty occurs in assembly
                hist = [a for c in prev_sib[i] for a in self.clusters[c]] 
                pa_cls = self.clusters[ pa[i] ]
                tree.nodes[i]['assm_cands'] = get_assm_cands(mol, hist, inter_label, pa_cls, len(inter_atoms)) 

                child_order = tree[i][pa[i]]['label']
                diff = set(cls) - set(pa_cls)
                for fa_atom in inter_atoms:
                    for ch_atom in self.mol_graph[fa_atom]:
                        if ch_atom in diff:
                            label = self.mol_graph[ch_atom][fa_atom]['label']
                            if type(label) is int: #in case one bond is assigned multiple times
                                self.mol_graph[ch_atom][fa_atom]['label'] = (label, child_order)
        
        return order
    
    @staticmethod
    def tensorize(mol_batch, vocab, avocab, show_progress=True):
        if show_progress:
            mol_batch = [MolGraph(x) for x in tqdm(mol_batch, desc="Building MolGraph", leave=False)]
        else:
            mol_batch = [MolGraph(x) for x in mol_batch]
        tree_tensors, tree_batchG = MolGraph.tensorize_graph([x.mol_tree for x in mol_batch], vocab)
        graph_tensors, graph_batchG = MolGraph.tensorize_graph([x.mol_graph for x in mol_batch], avocab)
        tree_scope = tree_tensors[-1]
        graph_scope = graph_tensors[-1]

        max_cls_size = max( [len(c) for x in mol_batch for c in x.clusters] )
        cgraph = torch.zeros(len(tree_batchG) + 1, max_cls_size).int()
        for v, attr in tree_batchG.nodes(data=True):
            bid = attr['batch_id']
            offset = graph_scope[bid][0]
            tree_batchG.nodes[v]['inter_label'] = inter_label = [(x + offset, y) for x,y in attr['inter_label']]
            tree_batchG.nodes[v]['cluster'] = cls = [x + offset for x in attr['cluster']]
            tree_batchG.nodes[v]['assm_cands'] = [add(x, offset) for x in attr['assm_cands']]
            cgraph[v, :len(cls)] = torch.IntTensor(cls)
        
        all_orders = []
        for i,hmol in enumerate(mol_batch):
            offset = tree_scope[i][0]
            order = [(x + offset, y + offset, z) for x,y,z in hmol.order[:-1]] + [(hmol.order[-1][0] + offset, None, 0)]
            all_orders.append(order)

        tree_tensors = tree_tensors[:4] + (cgraph, tree_scope)
        return (tree_batchG, graph_batchG), (tree_tensors, graph_tensors), all_orders
    
    @staticmethod
    def tensorize_graph(graph_batch, vocab):
        fnode, fmess = [None],[(0,0,0,0)]
        agraph, bgraph = [[]], [[]]
        scope = []
        edge_dict = {}
        all_G = []

        for bid, G in enumerate(graph_batch):
            offset = len(fnode)
            scope.append( (offset, len(G)) )
            G = nx.convert_node_labels_to_integers(G, first_label = offset)
            all_G.append(G)
            fnode.extend( [None for v in G.nodes] )

            for v, attr in G.nodes(data='label'):
                G.nodes[v]['batch_id'] = bid
                fnode[v] = vocab[attr]
                agraph.append([])
            
            for u, v, attr in G.edges(data='label'):
                if type(attr) is tuple:
                    fmess.append( (u, v, attr[0], attr[1]) )
                else:
                    fmess.append( (u, v, attr, 0) )
                edge_dict[(u, v)] = eid = len(edge_dict) + 1
                G[u][v]['mess_idx'] = eid
                agraph[v].append(eid)
                bgraph.append([])

            for u, v in G.edges:
                eid = edge_dict[(u, v)]
                for w in G.predecessors(u):
                    if w == v: continue
                    bgraph[eid].append( edge_dict[(w, u)] )
        
        fnode[0] = fnode[1]
        fnode = torch.IntTensor(fnode)
        fmess = torch.IntTensor(fmess)
        agraph = create_pad_tensor(agraph)
        bgraph = create_pad_tensor(bgraph)

        return (fnode, fmess, agraph, bgraph, scope), nx.union_all(all_G)

if __name__ == "__main__":
    smiles_batch = [
        'O=C(C1CCN(C2=NC=NC3=C2N=C2CCCCCN23)CC1)N1CCN(C2=CC=CC=C2F)CC1', 
        'O=[N+]([O-])C1=CC=CC(C2=NC3=CC(C4=CC=CC=C4)=NN3C(C(F)(F)F)=C2)=C1',
        'CCS(=O)(=O)N1CCCn2c(nc(Cn3cncn3)cc2=O)C1', 
        'Cc1ccc(-c2cn3c(n2)CCCC3)cc1', 
        'CN(C)CCn1nc2c3c(c(NCCc4c[nH]c5ccccc45)ccc31)C(=O)c1ccncc1-2', 
        'O=C(c1cncc2nnc(-c3ccc(OC(F)F)cc3)n12)N1Cc2ccccc2C1', 
        'CCCCNC(=O)c1ccc2nc(CC)c(N(CC)CCOC)n2c1', 
        'COC(=O)c1cc(C(=O)C2CC2)n2c(Br)cccc12', 
        'O=C(c1cccc(Cl)c1Cl)N1CCn2c(nnc2-c2cnccn2)C1', 
        'CC(C)c1nnc2n1CC(CNCC=Cc1ccccc1)CC2', 
        'C=CC(=O)NCCN(C)CCCC', 
        'COC(=O)Cc1c[nH]c(=O)c2cc(Br)c(Br)n12', 
        'CC(C)CCNC(=O)C(CC(C)C)NC(=O)CC(O)C(CC(C)C)NC(=O)C(NC(=O)CC(C)C)C(C)C'
    ]

    