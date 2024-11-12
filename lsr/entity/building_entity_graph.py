import numpy as np
import torch
from tqdm import tqdm
emb_path = "data/wiki_embs.pkl.npy"
embs = torch.from_numpy(np.load(emb_path))
embs = torch.nn.functional.normalize(embs).to("cuda")
top_k = []
batch_size = 50
for idx in tqdm(range(0, embs.size(0), batch_size)):
    batch_embs = embs[idx: idx+batch_size]
    sim_scores = batch_embs @ embs.T
    batch_top_k = torch.topk(sim_scores, dim=1, k=10,
                             largest=False, sorted=True).indices
    top_k.append(batch_top_k.to("cpu"))
top_k = torch.cat(top_k, dim=0).numpy()
np.save("data/wiki_ent_graph.npy", top_k)
