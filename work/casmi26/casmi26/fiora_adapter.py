"""Pinned author FIORA inference with graph caching, without fitting or pickled models.

All graphs are single-molecule batches. Molecular embeddings do not depend on
collision-energy covariates; cached and full paths must pass numerical parity.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np

MODEL_HASH='83221e187991f116a8aed1bf272dd656cf31721a177dcbb0239f414c8df31a0e'
PARAMS_HASH='c214195f945b1273b7b350cfe4bd964b9a410316433966e74c0a8bf9acdb86c5'


def verify_model_files(path):
    path=Path(path)
    for suffix,digest in (('_state.pt',MODEL_HASH),('_params.json',PARAMS_HASH)):
        file=path.with_name(path.stem+suffix)
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=digest:
            raise ValueError('Pinned model file hash mismatch: '+file.name)


class ForwardModel:
    def __init__(self,path,device='cpu'):
        verify_model_files(path)
        import torch
        from fiora.GNN.FioraModel import FioraModel
        from fiora.GNN.CovariateFeatureEncoder import CovariateFeatureEncoder
        from fiora.MS.SimulationFramework import SimulationFramework
        self.device=device
        self.model=FioraModel.load_from_state_dict(str(path)).to(device).eval()
        self.framework=SimulationFramework(self.model,dev=device)
        sets=self.model.model_params.get('setup_features_categorical_set')
        self.setup=CovariateFeatureEncoder(feature_list=['collision_energy','molecular_weight','precursor_mode','instrument','element_composition'],sets_overwrite=sets)
        self.rt=CovariateFeatureEncoder(feature_list=['molecular_weight','precursor_mode','instrument'],sets_overwrite=sets)
        self.setup.normalize_features['collision_energy']['max']=100.
        self.setup.normalize_features['molecular_weight']['max']=1000.
        self.rt.normalize_features['molecular_weight']['max']=1000.
        self.transform='square' if self.model.model_params.get('training_label')=='compiled_probsSQRT' else 'None'
        self.model.requires_grad_(False)

    def predict(self,smiles,modes,energies,*,cache_graph=True):
        import pandas as pd
        import torch
        from torch_geometric.data import Batch
        from fiora.cli.predict import build_metabolites
        modes=sorted(set(modes));energies=sorted(set(map(float,energies)))
        if not modes or any(m not in ('[M+H]+','[M-H]-') for m in modes):raise ValueError('Unsupported forward adduct')
        if not energies or not np.isfinite(energies).all() or min(energies)<0 or max(energies)>100:
            raise ValueError('Unsupported forward collision energy')
        initial=pd.DataFrame([{'Name':'candidate','SMILES':smiles,'Precursor_type':modes[0],'CE':energies[0],'Instrument_type':'HCD'}])
        data,invalid=build_metabolites(initial,self.model.model_params)
        if len(data)!=1:raise ValueError('Candidate not supported by forward model')
        molecule=data.iloc[0]['Metabolite'];outputs={};embeddings=None
        with torch.inference_mode():
            for mode in modes:
                for energy in energies:
                    molecule.add_metadata({'name':'candidate','collision_energy':energy,'instrument':'HCD','precursor_mode':mode},self.setup,self.rt)
                    batch=Batch.from_data_list([molecule.as_geometric_data(with_labels=False)]).to(self.device)
                    if cache_graph:
                        if embeddings is None:
                            batch['node_embedding']=self.model.node_embedding(batch['x'])
                            batch['edge_embedding']=self.model.edge_embedding(batch['edge_attr'])
                            x=self.model.GNN_module(batch)
                            embeddings=(batch['node_embedding'],batch['edge_embedding'],x)
                        else:batch['node_embedding'],batch['edge_embedding'],x=embeddings
                        edge=self.model.edge_module(x,batch);precursor=self.model.precursor_module(x,batch)
                        probs,_=self.model._compile_output(edge,precursor,batch)
                    else:probs=self.model(batch)['fragment_probs']
                    if not torch.isfinite(probs).all():raise ValueError('Nonfinite forward prediction')
                    molecule._casmi_probs=probs.cpu()
                    peaks=self.framework.simulate_spectrum(molecule,'_casmi_probs',precursor_mode=mode,
                            transform_prob=self.transform,min_intensity=.001)
                    array=np.column_stack((peaks['mz'],peaks['intensity'])).astype('f8')
                    if not np.isfinite(array).all() or np.any(array[:,0]<=0) or np.any(array[:,1]<0):raise ValueError('Invalid simulated spectrum')
                    outputs[(mode,energy)]=array
        return outputs
