from matplotlib import pyplot as plt

class LossHistory:
    def __init__(self, path) -> None:
        self.path = path
        self.data = {'train_loss': [], 'val_loss': []}


    def append(self, train_loss, val_loss):

        self.data['train_loss'].append(train_loss)
        self.data['val_loss'].append(val_loss)

    def plot(self, title):
        
        plt.figure()
        plt.semilogy(self.data['train_loss'], label='Train')
        plt.semilogy(self.data['val_loss'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.grid()
        plt.legend()
        plt.title(title)
        plt.show()