import pandas as pd
import numpy as np
import argparse
import subprocess
import random
import time
from model import KGCN
from data_loader import DataLoader
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

DATA_PATH = "./data/movie"

RATINGS_COLS = ["userId", "movieId", "rating", "timestamp"]
MOVIE_COLS = ["movieId", "title", "genres"]


def download_zip(url):
    zip_fn = url.split("/")[-1]
    download_dir = "./data"
    subprocess.call(["wget", url, "-P", download_dir])
    subprocess.call(["unzip", "-j", f"{download_dir}/{zip_fn}", "-d", download_dir])
    subprocess.call(["cp", f"{download_dir}/ratings.dat", f"{download_dir}/movies.dat", f"{DATA_PATH}/."])
    print("Download and transformation done")

def convert_dat(dat_path, cols):
    df = pd.read_csv(dat_path, sep="::", names=cols)
    output_fn = dat_path.split("/")[-1].split(".")[0]+".csv"
    output_path = f"{DATA_PATH}/{output_fn}"
    df.to_csv(output_path, index=False)
    print(f"{output_path} saved")


class KGCNDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.arr = df.to_numpy()

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, idx):
        user_id,item_id, label = self.arr[idx]
        return user_id, item_id, np.float32(label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_url', type=str, default='http://files.grouplens.org/datasets/movielens/ml-10m.zip', help='which dataset to use')
    parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
    parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
    parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
    parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
    parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
    parser.add_argument('--n_iter', type=int, default=1,
                        help='number of iterations when computing entity representation')
    parser.add_argument('--batch_size', type=int, default=65536, help='batch size')
    parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--ratio', type=float, default=0.8, help='size of training dataset')

    args = parser.parse_args(['--l2_weight', '1e-4'])

    # download & transform data
    download_zip(args.data_url)
    convert_dat(f"{DATA_PATH}/ratings.dat", RATINGS_COLS)
    convert_dat(f"{DATA_PATH}/movies.dat", MOVIE_COLS)

    # prepare dataset
    data_loader = DataLoader(args.dataset)
    kg = data_loader.load_kg()
    df_dataset = data_loader.load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(df_dataset,
                                                        df_dataset['label'],
                                                        test_size=1 - args.ratio,
                                                        shuffle=False,
                                                        random_state=999)
    train_dataset = KGCNDataset(x_train)
    test_dataset = KGCNDataset(x_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True)

    # prepare network, loss function, optimizer
    num_user, num_entity, num_relation = data_loader.get_num()
    user_encoder, entity_encoder, relation_encoder = data_loader.get_encoders()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = KGCN(num_user, num_entity, num_relation, kg, args, device).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(net.parameters(),
                           lr=args.lr,
                           weight_decay=args.l2_weight)

    # train
    loss_list = []
    test_loss_list = []
    auc_score_list = []

    for epoch in range(args.n_epochs):
        print(f"start {epoch} epoch")
        running_loss = 0.0
        for i, (user_ids, item_ids, labels) in enumerate(train_loader):
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(user_ids, item_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # print train loss per every epoch
        print('[Epoch {}]train_loss: '.format(epoch + 1), running_loss / len(train_loader))
        loss_list.append(running_loss / len(train_loader))

        # evaluate per every epoch
        with torch.no_grad():
            test_loss = 0
            total_roc = 0
            for user_ids, item_ids, labels in test_loader:
                user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
                outputs = net(user_ids, item_ids)
                test_loss += criterion(outputs, labels).item()
                total_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            print('[Epoch {}]test_loss: '.format(epoch + 1), test_loss / len(test_loader))
            test_loss_list.append(test_loss / len(test_loader))
            auc_score_list.append(total_roc / len(test_loader))

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(loss_list)
    ax1.plot(test_loss_list)
    ax2.plot(auc_score_list)

    plt.tight_layout()

    # save model
    ts = int(time.time())
    model_path = f"./kgcn_{ts}.pt"
    torch.save(net.state_dict(), model_path)
    print(f"model saved : {model_path}")


