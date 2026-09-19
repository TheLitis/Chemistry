import numpy as np
import pytest
pd=pytest.importorskip('pandas')
torch=pytest.importorskip('torch')
from casmi26.conditioned_encoder import model_logits,condition_groups

class Recorder(torch.nn.Module):
 def __init__(self):super().__init__();self.calls=[]
 def forward(self,mz,it,pad,prec,ad,ins,ce,mode):
  self.calls.append((mz.clone(),prec.clone(),ad.clone(),mode.clone()))
  return torch.stack([prec+ad,mode+it.sum(1)],axis=1)

def prep(mz,it,prec):
 a=np.asarray(mz,'f4');b=np.asarray(it,'f4');mask=b>0;return a[mask],b[mask]
def merge(frame):
 return np.concatenate(frame.ms2_mzs.to_list()),np.concatenate(frame.ms2_normalized_intensities.to_list())
def frame():
 return pd.DataFrame([{'ms2_mzs':[20.], 'ms2_normalized_intensities':[v],'precursor_mz':p,'adduct':a,
    'instrument_type':'timsTOF','collision_energy_ev':[25.],'ionization_mode':m} for v,p,a,m in [
    (0,101,'[M+H]+','positive'),(1,99,'[M-H]-','negative'),(2,101,'[M+H]+','positive')]])
AD={'[M+H]+':0,'[M-H]-':1,'<unk>':2}

def test_metadata_tracks_filtered_spectrum():
 net=Recorder();df=frame();out=model_logits(df,([net],[],'cpu',2),prep,merge,AD,lambda x:0)
 calls=net.calls[0];assert len(calls[1])==2
 for mass,ad in zip(calls[1],calls[2]):assert (float(mass),int(ad)) in [(99,1),(101,0)]

def test_no_mixed_mode_and_group_order_invariant():
 a=Recorder();b=Recorder();f=frame();x=model_logits(f,([a],[b],'cpu',2),prep,merge,AD,lambda x:0)
 x2=model_logits(f.iloc[::-1],([a],[b],'cpu',2),prep,merge,AD,lambda x:0)
 assert np.array_equal(x,x2)
 for _,p,ad,m in b.calls:assert abs(float(m[0]))==1 and float(p[0]) in (99,101)

def test_groups_do_not_mix_adducts():
 groups=condition_groups(frame(),lambda x:0);assert len(groups)==2
 for g in groups:assert g.adduct.nunique()==1 and g.ionization_mode.nunique()==1

def test_labels_not_read():
 f=frame();a=Recorder();b=Recorder();before=model_logits(f,([a],[b],'cpu',2),prep,merge,AD,lambda x:0)
 f['normalized_smiles']='THIS IS NOT AN INPUT';f['molecular_formula']='NOT AN INPUT'
 assert np.array_equal(before,model_logits(f,([a],[b],'cpu',2),prep,merge,AD,lambda x:0))

def test_empty_input_returns_none():
 f=frame().iloc[:1];assert model_logits(f,([Recorder()],[Recorder()],'cpu',2),prep,merge,AD,lambda x:0) is None

def test_unknown_polarity_rejected():
 f=frame();f.loc[0,'ionization_mode']='other'
 with pytest.raises(ValueError):condition_groups(f,lambda x:0)
