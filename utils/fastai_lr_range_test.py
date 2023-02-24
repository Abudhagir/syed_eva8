from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, Accuracy
from ignite.contrib.handlers import FastaiLRFinder, ProgressBar

def range_test(model, optimizer, criterion, device, trainloader, testloader, start_lr = 1e-4, end_lr = 10, max_epochs = 10):
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    ProgressBar(persist=True).attach(trainer, output_transform=lambda x: {"batch loss": x})

    lr_finder = FastaiLRFinder()
    to_save={'model': model, 'optimizer': optimizer}
    with lr_finder.attach(trainer, to_save, start_lr = start_lr, end_lr = end_lr, diverge_th=1.5) as trainer_with_lr_finder:
        trainer_with_lr_finder.run(trainloader)
        
    trainer.run(trainloader, max_epochs=10)

    evaluator = create_supervised_evaluator(model, metrics={"acc": Accuracy(), "loss": Loss(criterion)}, device=device)
    evaluator.run(testloader)

    print(evaluator.state.metrics)

    return lr_finder


'''



from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, Accuracy
from ignite.contrib.handlers import FastaiLRFinder, ProgressBar

def range_test(model, optimizer, criterion, device, train_loader, test_loader, start_lr = 1e-4, end_lr = 10, max_epochs = 200):
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    ProgressBar(persist=True).attach(trainer, output_transform=lambda x: {"batch loss": x})

    lr_finder = FastaiLRFinder()
    to_save={'model': model, 'optimizer': optimizer}
    with lr_finder.attach(trainer, to_save, start_lr = start_lr, end_lr = end_lr, diverge_th=1.5) as trainer_with_lr_finder:
        trainer_with_lr_finder.run(train_loader)
        
    trainer.run(train_loader, max_epochs=10)

    evaluator = create_supervised_evaluator(model, metrics={"acc": Accuracy(), "loss": Loss(criterion)}, device=device)
    evaluator.run(test_loader)

    print(evaluator.state.metrics)

    return lr_finder

'''