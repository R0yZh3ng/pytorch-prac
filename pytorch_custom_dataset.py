#NOTE: we've used some datasets with pytorch before, but how do you get your own data into pytorch, use custom datasets

#domain libraries, depending on what you're workiing on , vision, text, audio, recommendation, you'll want to look into each of the PyTroch domain libraries for existing data loading functions and customizabel data loading functions.

import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

