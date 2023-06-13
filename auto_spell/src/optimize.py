import time
import torch

class optimize:
    def __init__(self, train_dataloader, test_dataloader, model, criterion, optimizer):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self):
        self.model.train()
        total_acc, total_count = 0, 0
        log_interval = 500
        start_time = time.time()

        for idx, (label, text, offsets) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            predicted_label = self.model(text, offsets)
            loss = self.criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            print(total_acc)
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                    '| accuracy {:8.3f}'.format(epoch, idx, len(self.train_dataloader),
                                                total_acc/total_count))
                total_acc, total_count = 0, 0
                start_time = time.time()

    def evaluate(self):
        self.model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(self.test_dataloader):
                predicted_label = self.model(text, offsets)
                loss = self.criterion(predicted_label, label)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        return total_acc/total_count
    
    def hypertuning(self, epochs):
        total_accu = None
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            self.train()
            accu_val = self.evaluate()
            if total_accu is not None and total_accu > accu_val:
                scheduler.step()
            else:
                total_accu = accu_val
            print('-' * 59)
            print('| end of epoch {:3d} | time: {:5.2f}s | '
                'valid accuracy {:8.3f} '.format(epoch,
                                                time.time() - epoch_start_time,
                                                accu_val))
            print('-' * 59)

    def predict(self, text, textpipeline):
        with torch.no_grad():
            text = torch.tensor(textpipeline(text))
            output = self.model(text, torch.tensor([0]))
            return output.argmax(1).item() + 1