import torch
class argparse:
    def __init__(self):
        self.epochs, self.learning_rate, self.patience = [300, 0.001, 4]
        self.hidden_size, self.input_size = [40, 30]
        self.device, = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), ]
        self.image_size,self.patch_size=[32,4]
        self.num_classes = 10
        self.dim = 512
        self.depth = 4
        self.heads = 8
        self.mlp_dim = 256