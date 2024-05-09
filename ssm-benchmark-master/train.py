import argparse
import torch
import wandb
from tqdm import tqdm
import yaml
from .dataloaders import SequenceDataset

from .models import Mamba

def train_mamba(seed, trainloader, testloader, wandb_config, train_config, model_config):
    torch.manual_seed(seed)
    device = "cuda"
    model = Mamba(**model_config).to(device)
    nr_params = sum(p.numel() for p in model.parameters())
    print("Nr. of parameters: {0}".format(nr_params))
    if wandb_config is not None:
        wandb.log({"params": nr_params})
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["lr"], weight_decay=train_config["wd"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config["num_epochs"], eta_min=5e-6)
    running_loss = 0.0
    for epoch in range(train_config["num_epochs"]):
        for X, y, _ in tqdm(trainloader):
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            y_hat = model(X)
            loss = torch.nn.functional.cross_entropy(y_hat, y)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = running_loss / len(trainloader)
        print("Loss: {0:.3f}".format(train_loss))
        scheduler.step()

        model.eval()
        running_accuracy = 0.0
        with torch.no_grad():
            for X, y, _ in tqdm(trainloader):
                X = X.to(device)
                y = y.to(device)
                y_hat = model(X)
                accuracy = (y_hat.argmax(dim=1) == y).float().sum() / len(y)
                running_accuracy += accuracy
        train_acc = running_accuracy / len(trainloader)
        print("Train accuracy: {0:.4f}".format(train_acc))

        running_accuracy = 0.0
        running_loss = 0.0
        with torch.no_grad():
            for X, y, _ in tqdm(testloader):
                X = X.to(device)
                y = y.to(device)
                y_hat = model(X)
                loss = torch.nn.functional.cross_entropy(y_hat, y)
                running_loss += loss.item()
                accuracy = (y_hat.argmax(dim=1) == y).float().sum() / len(y)
                running_accuracy += accuracy
        test_loss = running_loss / len(testloader)
        test_acc = running_accuracy / len(testloader)
        print("Test accuracy: {0:.4f}\n".format(test_acc))

        if wandb_config is not None:
            wandb.log(
                {"train acc": train_acc,
                 "test acc": test_acc,
                 "train loss": train_loss,
                 "test loss": test_loss,
                 "lr": optimizer.param_groups[0]['lr']}
            )
        model.train()

def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0 - val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="cifar-10.yaml", help="experiment config file")
    config = parser.parse_args().config
    print("\nUsing config {0}".format(config))

    # get GPU info
    if not torch.cuda.is_available():
        raise NotImplementedError("Cannot run on CPU!")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_type = torch.cuda.get_device_name(0)
    print("Running on {0}".format(gpu_type))

    # get args
    with open("ssm-benchmark-master/configs/" + config) as stream:
        try:
            args = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise RuntimeError(exc)

    args["GPU"] = gpu_type

    # get wandb config
    if "wandb" in args:
        wandb_config = args.pop("wandb")
    else:
        wandb_config = None

    print("\nCONFIG:")
    print(yaml.dump(args))

    # split configs
    data_config = args["dataset"]
    train_config = args["train"]
    model_config = args["model"]

    # start wandb logging
    if wandb_config is not None:
        wandb.login(key=wandb_config["key"])
        wandb.init(
            entity=wandb_config["entity"],
            project=wandb_config["project"],
            config=args,
            job_type="train",
            name=args["model"]["ssm_type"]
        )

    # prepare dataset
    data_config.pop("name")  # remove logging name
    dataset = SequenceDataset.registry[data_config["_name_"]](**data_config)
    dataset.setup()

    # dataloaders
    trainloader = dataset.train_dataloader(batch_size=train_config["batch_size"], shuffle=True)
    testloader = dataset.test_dataloader(batch_size=train_config["batch_size"], shuffle=False)
    if type(testloader) is dict:
        testloader = testloader[None]

    # extract model class [mamba | hawk]
    layer = model_config.pop("layer")

    # start train loop
    if layer == "mamba":
        train_mamba(
            args["seed"],
            trainloader,
            testloader,
            wandb_config,
            train_config,
            model_config
        )
    else:
        raise RuntimeError("{0} is not a valid model option".format(layer))

    if wandb_config is not None:
        wandb.finish()