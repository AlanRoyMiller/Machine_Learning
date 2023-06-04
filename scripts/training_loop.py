import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def training_loop(
        network: torch.nn.Module,
        train_data: torch.utils.data.Dataset,
        eval_data: torch.utils.data.Dataset,
        num_epochs: int,
        show_progress: bool = True
):  # -> tuple[list, list]:

    optimizer = torch.optim.SGD(network.parameters(), lr=0.00001)
    loss_function = torch.nn.MSELoss()

    train_dataloader = DataLoader(train_data, shuffle=False, batch_size=32, num_workers=0)
    eval_dataloader = DataLoader(eval_data, shuffle=False, batch_size=32, num_workers=0)

    losses_train_dataloader = []
    losses_eval_dataloader = []

    if show_progress:
        progress_bar = tqdm(total=num_epochs, desc="epochs")

    for epoch in range(num_epochs):
        network.train()
        average_batch_loss = []

        for input_tensor, target_tensor in train_dataloader:
            output = network(input_tensor)
            loss = loss_function(output, target_tensor)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            average_batch_loss.append(loss.item())

        losses_train_dataloader.append(sum(average_batch_loss) / len(average_batch_loss))

        network.eval()
        with torch.no_grad():
            average_batch_loss_eval = []
            for input_tensor, target_tensor in eval_dataloader:
                output = network(input_tensor)
                loss = loss_function(output, target_tensor)
                average_batch_loss_eval.append(loss.item())
        loss_eval = sum(average_batch_loss_eval) / len(average_batch_loss_eval)
        losses_eval_dataloader.append(loss_eval)


        if show_progress:
            progress_bar.update()

    if show_progress:
        progress_bar.close()

    return losses_train_dataloader, losses_eval_dataloader



if __name__ == "__main__":
    from simple_network import SimpleNetwork
    from dataset import get_dataset

    torch.random.manual_seed(0)
    train_data, eval_data = get_dataset()
    network = SimpleNetwork(32, 128, 1)
    train_losses, eval_losses = training_loop(network, train_data, eval_data, num_epochs=10)
    for epoch, (tl, el) in enumerate(zip(train_losses, eval_losses)):
        print(f"Epoch: {epoch} --- Train loss: {tl:7.2f} --- Eval loss: {el:7.2f}")

