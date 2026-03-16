from kldm.model import Model
from kldm.data import MyDataset

def train():
    dataset = MyDataset("data/mp_20/train.pt")
    model = Model()
    # add rest of your training code here

if __name__ == "__main__":
    train()
