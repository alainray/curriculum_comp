import torch
from utils import timing, AverageMeter, get_prefix
from torch.nn import LogSigmoid, Sigmoid

@timing
def train(experiment, args, model, loader, opt, criterion, epoch):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    model.train()
    acc_data = ""
    total_batches = len(loader)
    metrics = {'loss': None, 'acc': None}
    # confusion matrix
    y_pred = []
    y_true = []

    for n_batch, (x, label) in enumerate(loader):
        opt.zero_grad()
        x = x.to(args.device)
        bs = x.shape[0]
        label = label.to(args.device)
        logits = model(x)

        preds = logits.argmax(dim=1)
        y_pred.extend(preds.detach().cpu().numpy())
        y_true.extend(label.detach().cpu().numpy())
        correct = (preds == label).cpu().sum()
        acc_meter.update(correct.cpu() / float(bs), bs)  
        acc_data = f"Acc: {100 * correct.float() / bs:.1f}% Cum. Acc: {100 * acc_meter.avg:.1f}%"
        metrics['acc'] = float(100*acc_meter.avg)
        loss = criterion(logits, label)
        loss.backward()
        # Update stats
        opt.step()
        cur_loss = loss.detach().cpu()
        metrics['loss'] = float(cur_loss)
        loss_meter.update(cur_loss, bs)
        loss_data = f" Loss (Current): {cur_loss:.3f} Cum. Loss: {loss_meter.avg:.3f}"
        training_iteration = total_batches*(epoch-1) + n_batch + 1
        experiment.log_metrics(metrics, prefix='train', step=training_iteration, epoch=epoch)

        print(f"\r[TRAIN] Epoch {epoch}: {n_batch + 1}/{total_batches}: {loss_data} {acc_data}", end="", flush=True)
    
    if args.label_type != "score":
        experiment.log_confusion_matrix(y_true,
                                    y_pred,
                                    step=epoch, 
                                    title=f"Confusion Matrix TRAIN, Epoch {epoch}",
                                    file_name=f"cf_{get_prefix(args)}_train_{epoch}.json")
    
    return model, [loss_meter, acc_meter]

#@timing
def test(experiment, args, model, loader, criterion, epoch, prefix="test"):

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    model.eval()
    acc_data = ""
    total_batches = len(loader)
    metrics = {'loss': None, 'acc': None}
    # confusion matrix
    y_pred = []
    y_true = []
    with torch.no_grad():
        for n_batch, (x, label) in enumerate(loader):
            x = x.to(args.device)
            bs = x.shape[0]
            label = label.to(args.device)
            logits = model(x)

            preds = logits.argmax(dim=1)
            y_pred.extend(preds.detach().cpu().numpy())
            y_true.extend(label.detach().cpu().numpy())
            correct = (preds == label).cpu().sum()
            acc_meter.update(correct.cpu() / float(bs), bs)  
            acc_data = f"Acc: {100 * acc_meter.avg:.2f}%"
            metrics['acc'] = float(100*acc_meter.avg)
            loss = criterion(logits, label)
            # Update stats
            cur_loss = loss.detach().cpu()
            metrics['loss'] = float(cur_loss)
            loss_meter.update(cur_loss, bs)
            loss_data = f"Loss: {loss_meter.avg:.3f}"
            experiment.log_metrics(metrics, prefix=prefix, step=epoch, epoch=epoch)

            print(f"\r [{prefix.upper()}] Iter: {epoch}-{loss_data} {acc_data}", end="", flush=True)
    '''experiment.log_confusion_matrix(y_true,
                                    y_pred,
                                    step=epoch, 
                                    title=f"Confusion Matrix {prefix.upper()}, Epoch {epoch}",
                                    file_name=f"cf_{get_prefix(args)}_{prefix}_{epoch}_.json")'''
    return model, [loss_meter, acc_meter]



if __name__ == "__main__":
    torch.manual_seed(123)
    bs = 10
    scale = 1
    a = scale*torch.randn((bs,1)).float()
    b = scale*torch.randn((bs,1)).float()


