from kldm.model import Model
from kldm.data import DNGTask

def train():
    dataset = DNGTask().fit_dataset("data/mp_20/train.pt")
    model = Model()
    # add rest of your training code here

if __name__ == "__main__":
    train()
