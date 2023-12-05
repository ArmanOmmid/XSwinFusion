
import torch

def run_epoch(
    model,
    criterion,
    optimizer,
    dataloader,
    device,
    learn
):
    epoch_loss = 0.
    dataset_size = len(dataloader.dataset)

    for iter, (inputs, labels) in enumerate(dataloader):
        with torch.set_grad_enabled(learn):

            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.size(0)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            model.train(False)

            avg_batch_loss = loss.detach().item()
            epoch_loss += avg_batch_loss * batch_size

    avg_epoch_loss = epoch_loss / dataset_size
    return avg_epoch_loss
