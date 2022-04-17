from AE_data import AE_data
from convAE import SegNet

import argparse
import tqdm
from time import time
import numpy as np
import pickle
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def parse_arguments():
    """Arguments for running the baseline.

    Returns:
        parsed arguments

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",
                        type=int,
                        default=512,
                        help="Batch size")
    parser.add_argument("--model_path",
                        required=False,
                        type=str,
                        help="path to the saved model")
    parser.add_argument("--city",
                        required=True,
                        type=str,
                        help="city name")
    parser.add_argument("--test",
                        action="store_true",
                        help="If true, only run the inference")
    parser.add_argument("--end_epoch",
                        type=int,
                        default=150,
                        help="Last epoch")
    parser.add_argument("--gpu",
                        type=int,
                        default=-1,
                        help="GPU to use")
    parser.add_argument("--get_embeddings",
                        action="store_true",
                        help="Use the model to save embeddings")
    return parser.parse_args()

def train(model, optimizer, loader, device):
    model.train()
    rec_loss = nn.MSELoss()
    losses = []

    for i_batch, (x, path, city) in enumerate(tqdm.tqdm(loader)):
        x = x.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = rec_loss(y_hat, x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return np.mean(np.asarray(losses))


def validate(model, loader, device):
    model.eval()
    rec_loss = nn.MSELoss()
    losses = []

    for i_batch, (x, path, city) in enumerate(tqdm.tqdm(loader)):
        x = x.to(device)
        y_hat = model(x)
        loss = rec_loss(y_hat, x)
        losses.append(loss.item())

    return np.mean(np.asarray(losses))

def test():
    pass

def main():

    args = parse_arguments()

    # get data:
    train_dataset = AE_data(mode="train", city=args.city, get_mean=False)
    val_dataset = AE_data(mode="val", city=args.city, get_mean=False)
    test_dataset = AE_data(mode="test", city=args.city, get_mean=False)

    n_workers = 10
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=n_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=n_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=n_workers, shuffle=False)

    if args.gpu >= 0:
        print("Using CUDA")
        device = torch.device("cuda:"+str(args.gpu))
    else:
        device = torch.device("cpu")




    if args.get_embeddings:
        # load the model:
        model = SegNet()
        model.load_state_dict(torch.load("./EDN.pth"))
        model.to(device)
        print("Getting embeddings")
        train_embeddings = get_embeddings(model, train_dataset, batch_size=args.batch_size, device=device)
        pickle.dump(train_embeddings, open("./embeddings/embeddings_train_"+args.city+".p", "wb"))
        val_embeddings = get_embeddings(model, val_dataset, batch_size=args.batch_size, device=device)
        pickle.dump(val_embeddings, open("./embeddings/embeddings_val_" + args.city + ".p", "wb"))
        test_embeddings = get_embeddings(model, test_dataset, batch_size=args.batch_size, device=device)
        pickle.dump(test_embeddings, open("./embeddings/embeddings_test_" + args.city + ".p", "wb"))

        return

    # get the model:
    model = SegNet()
    model = model.to(device)
    step_size = 10
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    best_loss = 10000

    for epoch in range(args.end_epoch):
        start = time()
        # Train epoch:
        train_loss = train(model, optimizer, train_loader, device)
        scheduler.step()
        print("Training epoch:", epoch, "is done. Loss:", round(train_loss, 5), "Time elapsed:", round(time()-start, 2))

        start = time()
        # Val epoch:
        val_loss = validate(model, val_loader, device)
        print("Validation:", epoch, "is done. Loss:", round(val_loss, 5), "Time elapsed:", round(time() - start, 2))
        if val_loss < best_loss:
            print("Improvement from:", best_loss, "to:", val_loss, " Saving the model...")
            best_loss = val_loss
            torch.save(model.state_dict(), "./EDN.pth")


def get_embeddings(model, dataset, batch_size, device):
    n_workers = 10
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, shuffle=False)
    model.eval()

    embeddings = []
    paths = []
    cities = []

    for i_batch, (x, path, city) in enumerate(tqdm.tqdm(loader)):
        x = x.to(device)
        y_hat = model(x)

        embeddings.extend(model.latent_representation.detach().cpu().numpy())
        paths.extend(path)
        cities.extend(city)
    return [embeddings, paths, cities]



if __name__=="__main__":
    main()