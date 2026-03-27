from kldm.kldm import ModelKLDM
from kldm.data import DNGTask

def train():
    dataset = DNGTask().fit_dataset("data/mp_20", split="train")
    model = ModelKLDM()

    #For now use ipynb for development, but goal is to implement lightning capabilities here.

if __name__ == "__main__":
    train()
