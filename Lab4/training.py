import time
import torch.optim
from dataset import MNISTMetricDataset
from torch.utils.data import DataLoader
from model import SimpleMetricEmbedding, IdentityModel
from utils import train, evaluate, compute_representations

EVAL_ON_TRAIN = False
EVAL_ON_TEST = True

USE_IDENTITY_MODEL = False
REMOVE_CLASSES = True

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(f"= Using device {device}")

    path = "./mnist/"

    ds_train_without_class = MNISTMetricDataset(path, split='train', remove_classes=[0])
    ds_train = MNISTMetricDataset(path, split='train', remove_classes=None)
    ds_test = MNISTMetricDataset(path, split='test', remove_classes=None)
    ds_traineval = MNISTMetricDataset(path, split='traineval', remove_classes=None)

    num_classes = 10

    print(f"> Loaded {len(ds_train)} training images!")
    print(f"> Loaded {len(ds_test)} validation images!")

    train_loader_without_class = DataLoader(
        ds_train_without_class,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
    )

    test_loader = DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    traineval_loader = DataLoader(
        ds_traineval,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    emb_size = 784 if USE_IDENTITY_MODEL else 32
    model = IdentityModel().to(device) if USE_IDENTITY_MODEL else SimpleMetricEmbedding(1, emb_size).to(device)
    optimizer = None if USE_IDENTITY_MODEL else torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 3

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        t0 = time.time_ns()

        if not USE_IDENTITY_MODEL:
            if REMOVE_CLASSES:
                train_loss = train(model, optimizer, train_loader_without_class, device, epoch)
            else:
                train_loss = train(model, optimizer, train_loader, device, epoch)
            print(f"Mean Loss in Epoch {epoch}: {train_loss:.3f}")

        if EVAL_ON_TEST or EVAL_ON_TRAIN:
            print("Computing mean representations for evaluation...")
            representations = compute_representations(model, train_loader, num_classes, emb_size, device)

        if EVAL_ON_TRAIN:
            print("Evaluating on training set...")
            accuracy = evaluate(model, representations, traineval_loader, device)
            print(f"Epoch {epoch}: Train Accuracy: {accuracy * 100:.2f}%")

        if EVAL_ON_TEST:
            print("Evaluating on test set...")
            accuracy = evaluate(model, representations, test_loader, device)
            print(f"Epoch {epoch}: Test Accuracy: {accuracy * 100:.2f}%")

        t1 = time.time_ns()
        print(f"Epoch time (sec): {(t1 - t0) / 10 ** 9:.1f}")

    if not USE_IDENTITY_MODEL:
        torch.save(model.state_dict(), "./parameters/task3/testing.pt")
