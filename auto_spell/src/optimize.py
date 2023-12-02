import time
import torch

class optimize:
    def __init__(self, train_dataloader, test_dataloader, model, criterion, optimizer, padding):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.padding = padding

    def train(self, epoch):
        self.model.train()
        total_acc, total_count = 0, 0
        log_interval = 500
        start_time = time.time()
        running_loss = 0.0
        if self.padding:
            for idx, (label, text) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
#                 print(f'text size is: {text.size(1)}')
#                 print(f'label is: {label}')
                predicted_label = self.model(text)
                loss = self.criterion(predicted_label, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()
#                 for name, param in self.model.named_parameters():
#                     print(name, param.mean())
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
#                 print(total_acc)
                running_loss =+ loss.item() * text.size(0)
                if idx % log_interval == 0 and idx > 0:
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d}/{:5d} batches '
                        '| accuracy {:8.3f}'.format(epoch, idx, len(self.train_dataloader),
                                                    total_acc/total_count))
                    total_acc, total_count = 0, 0
                    start_time = time.time()
        else:
            for idx, (label, text, offsets) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                predicted_label = self.model(text, offsets)
                loss = self.criterion(predicted_label, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()
#                 for name, param in self.model.named_parameters():
#                     print(name, param.mean())
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
#                 print(total_acc)
                running_loss =+ loss.item() * text.size(0)
                if idx % log_interval == 0 and idx > 0:
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d}/{:5d} batches '
                        '| accuracy {:8.3f}'.format(epoch, idx, len(self.train_dataloader),
                                                    total_acc/total_count))
                    total_acc, total_count = 0, 0
                    start_time = time.time()
        return running_loss / len(self.train_dataloader)

    def evaluate(self):
        self.model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            if self.padding:
                for idx, (label, text) in enumerate(self.test_dataloader):
                    predicted_label = self.model(text)
                    loss = self.criterion(predicted_label, label)
                    total_acc += (predicted_label.argmax(1) == label).sum().item()
                    total_count += label.size(0)            
            else:
                for idx, (label, text, offsets) in enumerate(self.test_dataloader):
                    predicted_label = self.model(text, offsets)
                    loss = self.criterion(predicted_label, label)
                    total_acc += (predicted_label.argmax(1) == label).sum().item()
                    total_count += label.size(0)
        return total_acc/total_count
    
    def hypertuning(self, epochs):
        total_accu = None
        loss_values = []
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            new_loss = self.train(epoch)
            loss_values.append(new_loss)
            print(f'Loss is: {new_loss}')
            accu_val = self.evaluate()
            if total_accu is not None and total_accu > accu_val:
                self.optimizer.step()
            else:
                total_accu = accu_val
            print('-' * 59)
            print('| end of epoch {:3d} | time: {:5.2f}s | '
                'valid accuracy {:8.3f} '.format(epoch,
                                                time.time() - epoch_start_time,
                                                accu_val))
            print('-' * 59)

    def predict(self, text, textpipeline):
#         print(text)
        with torch.no_grad():
            if self.padding:
                text = torch.tensor([textpipeline(text)])
#                 print(f'text is: {text}')
                output = self.model(text)
            else:
                text = torch.tensor(textpipeline(text))
                output = self.model(text, torch.tensor([0]))
            return output.argmax(1).item() + 1