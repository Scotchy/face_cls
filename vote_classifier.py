from sklearn.linear_model import LogisticRegression
import torch

class VoteClassifier():

    def __init__(self, models):
        self.models = models
        self.lr = LogisticRegression(C=1e5, n_jobs=12, max_iter=100000)

    def fit_lr(self, train_loader):
        full_output, labels_train = self.aggregated_forward(train_loader)
        self.lr.fit(full_output, labels_train)

    def aggregated_forward(self, loader): 

        # Generate labels numpy array
        all_labels = []
        for _, labels in loader:
            all_labels += [labels.detach().cpu()]
        all_labels = torch.cat(all_labels, dim=0).numpy()

        # Generate a (N_samples, N_models) numpy array containing output of each model for each sample
        full_output = []
        for model in self.models:
            model = model.cuda()
            model.eval()

            pre_labels = []
            for images, labels in loader:
                out = model(images.cuda())
                pre_labels += [out.detach().cpu()]

            pre_labels = torch.cat(pre_labels, dim=0)
            full_output += [pre_labels]
        full_output = torch.cat(full_output, dim=1).numpy()
        
        return full_output, all_labels
    
    def forward(self, loader):
        full_output, _ = self.aggregated_forward(loader)
        return self.lr.predict(full_output)