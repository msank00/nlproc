import torch
from torch import nn
from tqdm import tqdm
from annoy import AnnoyIndex
import numpy as np
from typing import List, Set, Dict, Tuple, Optional
from netpool import pretrainedEmbedding as pe

emb_file = "data/embeddinggs/glove.6B.100d.txt"

embeddings = pe.PreTrainedEmbeddings.from_embedding_file(emb_file)
embeddings.compute_and_print_analogy('man', 'he', 'woman')