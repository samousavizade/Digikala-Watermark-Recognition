import os

import mlflow
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models


TRAIN_STR = 'train'
IMAGE_SIZE_STR = 'size'
BATCH_SIZE_STR = 'batch_size'
DROPOUT_PROB_STR = 'dropout'
LEARNING_RATE_STR = 'lr'
ACCURACY_STR = 'Accuracy'
LOSS_STR = 'Loss'
EPOCHS_NUMBER_STR = 'epoch_number'
WEIGHT_DECAY_STR = 'weight_decay'
SCHEDULER_GAMMA_STR = 'scheduler_gamma'

MLF_EXPERIMENT_NAME = 'Watermark Recognition'
DATASET_PATH = 'dk-dataset/dataset'

MLF_EXPERIMENT_ID = None
TRAIN_VALIDATION_RATIO = 0.8

TRAIN_AUC_STR = 'train_AUC'
TRAIN_ACCURACY_STR = 'train_accuracy'
TRAIN_F1SCORE_STR = 'train_f1score'
TRAIN_LOSS_STR = 'train_loss'

VALIDATION_AUC_STR = 'validation_AUC'
VALIDATION_ACCURACY_STR = 'validation_accuracy'
VALIDATION_F1SCORE_STR = 'validation_f1score'
VALIDATION_LOSS_STR = 'validation_loss'


class HyperParameterConfigs:

    def __init__(self, params):
        self.epochs_number = params[EPOCHS_NUMBER_STR]
        self.image_data_size = params[IMAGE_SIZE_STR]
        self.learning_rate = params[LEARNING_RATE_STR]
        self.batch_size = params[BATCH_SIZE_STR]
        self.weight_decay = params[WEIGHT_DECAY_STR]
        self.scheduler_gamma = params[SCHEDULER_GAMMA_STR]

    def set_metrics(self):
        pass

    def to_dict(self):
        return {
            EPOCHS_NUMBER_STR: self.epochs_number,
            IMAGE_SIZE_STR: self.image_data_size,
            LEARNING_RATE_STR: self.learning_rate,
            BATCH_SIZE_STR: self.batch_size,
            WEIGHT_DECAY_STR: self.weight_decay,
            SCHEDULER_GAMMA_STR: self.scheduler_gamma,
        }

    def __str__(self):
        return f'Epochs Number : {self.epochs_number}, ' \
               f'Image Size : ({self.image_data_size},{self.image_data_size}), ' \
               f'Learning Rate : {self.learning_rate}, ' \
               f'Batch Size : {self.batch_size}, ' \
               f'Weight Decay: {self.weight_decay}, ' \
               f'Scheduler Gamma: {self.scheduler_gamma}, '


class ModelInitializer:

    def __init__(self, data_directory):
        self.data_directory = data_directory

        self.train_subset, self.validation_subset = None, None

        self.train_dataloader = None

        self.train_dataloader, self.validation_dataloader = None, None

        self.model = None
        self.loss_criterion = None
        self.optimizer = None
        self.epochs = None
        self.device = None
        self.lr_scheduler = None

        self.number_of_classes = 0

    def read_data(self, ratio=0.8, to_delete_size=8000):
        train_directory = f'{self.data_directory}/{TRAIN_STR}/'

        dataset = torchvision.datasets.ImageFolder(train_directory)

        self.number_of_classes = len(dataset.classes)

        total_set_size = len(dataset) - to_delete_size
        train_set_size = int(ratio * total_set_size)
        validation_set_size = total_set_size - train_set_size

        print(len(dataset))
        print(train_set_size, validation_set_size, to_delete_size)

        # ratio = 0.8
        # train_set_size = int(ratio * len(dataset))
        # validation_set_size = len(dataset) - train_set_size
        #
        # train_dataset, validation_dataset, _ = torch.utils.data.random_split(dataset, [train_set_size, validation_set_size, to_delete_size])

        self.train_subset, self.validation_subset, _ = torch.utils.data.random_split(dataset, [train_set_size, validation_set_size, to_delete_size])

    def initialize_with_hyper_parameters(self, hp_configs: HyperParameterConfigs):
        # print('Hyper Parameters : \n')
        # print(hp_configs)

        image_data_shape = (hp_configs.image_data_size, hp_configs.image_data_size)

        mean, standard_deviation = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        probability = 0.5
        train_transforms = transforms.Compose([
            transforms.Resize(size=image_data_shape),
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(p=probability),
            # transforms.RandomPerspective(p=probability, distortion_scale=.5),
            transforms.Normalize(mean, standard_deviation)
        ])

        validation_transforms = transforms.Compose([
            transforms.Resize(size=image_data_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean, standard_deviation)
        ])

        class DatasetFromSubset(torch.utils.data.Dataset):
            def __init__(self, subset, transform=None):
                self.subset = subset
                self.transform = transform

            def __getitem__(self, index):
                x, y = self.subset[index]
                if self.transform:
                    x = self.transform(x)
                return x, y

            def __len__(self):
                return len(self.subset)

        train_dataset = DatasetFromSubset(self.train_subset, transform=train_transforms)
        validation_dataset = DatasetFromSubset(self.validation_subset, transform=validation_transforms)

        # come from hyper parameter configs object as input
        batch_size = hp_configs.batch_size
        shuffle = True
        drop_last = False
        num_workers = 3

        self.train_dataloader = data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                drop_last=drop_last,
                                                num_workers=num_workers,
                                                pin_memory=True)

        self.validation_dataloader = data.DataLoader(validation_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=shuffle,
                                                     drop_last=drop_last,
                                                     num_workers=num_workers,
                                                     pin_memory=True)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        lr = hp_configs.learning_rate
        weight_decay = hp_configs.weight_decay
        scheduler_gamma = hp_configs.scheduler_gamma

        number_of_classes = self.number_of_classes

        model: models.Inception3 = models.inception_v3(pretrained=True, progress=True)
        last_fc_layer_input_number = model.fc.in_features
        model.fc = nn.Linear(last_fc_layer_input_number, number_of_classes)
        model = model.to(self.device)
        self.model = model

        self.optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_criterion = nn.CrossEntropyLoss()
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=scheduler_gamma)

        return self


class ModelTrainer:

    def __init__(self, epochs=30):

        self.train_data_loader = None
        self.validation_data_loader = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.device = None

        self.epochs = epochs

        self.test_predicts_proba = None
        self.test_true_labels = None

    def train_loop(self):
        data_loader = self.train_data_loader
        self.model.train()

        targets, predicts_scores = list(), list()

        # sum_loss = 0
        # sum_correct = 0
        for i, (fields, target) in enumerate(data_loader):
            fields, target = fields.to(self.device), target.to(self.device)

            output: torch.Tensor = self.model(fields)
            output = output[0]

            predicts_probabilities = torch.softmax(output, dim=1)
            # predicts_proba, _ = torch.max(predicts_proba, dim=1)

            loss = self.criterion(predicts_probabilities, target)

            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()

            # predicts = self.get_predicts_label(predicts_proba)

            # sum_loss += loss.item()
            # sum_correct += predicts.eq(target).sum().item()
            targets.extend(target.tolist())
            predicts_scores.extend(predicts_probabilities.tolist())

        targets = torch.Tensor(targets)
        predicts_scores = torch.Tensor(predicts_scores)

        # n_batches = len(data_loader)
        # train_loss = sum_loss / n_batches
        # train_accuracy = sum_correct / n_batches

        return targets, predicts_scores

    def test_loop(self):
        data_loader = self.validation_data_loader
        self.model.eval()

        targets, predicted_targets = list(), list()

        # sum_loss = 0
        # sum_correct = 0
        with torch.no_grad():
            for fields, target in data_loader:
                fields, target = fields.to(self.device), target.to(self.device)

                output = self.model(fields)
                # output = output[0]
                #
                # print(output)

                predicts_probabilities = torch.softmax(output, dim=1)
                loss = self.criterion(predicts_probabilities, target)

                # sum_loss += loss.item()

                # predicts = self.get_predicts_label(predicts_proba)

                # sum_correct += predicts.eq(target).sum().item()

                targets.extend(target.tolist())
                predicted_targets.extend(output.tolist())

        targets = torch.Tensor(targets)
        predicted_targets = torch.Tensor(predicted_targets)

        # n_batches = len(data_loader)
        # test_loss = sum_loss / n_batches
        # test_accuracy = sum_correct / n_batches

        return targets, predicted_targets

    def calculate_metrics(self, trues, predicted_targets):
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        # Yt_train.type(torch.LongTensor)
        # trues = trues.type(torch.LongTensor)

        trues = trues.type(torch.LongTensor)

        predicted_probas = torch.softmax(predicted_targets, dim=1)
        predicted_probas, predicted_labels = torch.max(predicted_probas, dim=1)

        loss = self.criterion(predicted_targets, trues).item()

        auc = roc_auc_score(trues, predicted_probas)

        accuracy = accuracy_score(trues, predicted_labels)
        f_score = f1_score(trues, predicted_labels)
        return auc, accuracy, f_score, loss

    def fit_model_with_setup(self,
                             train_data_loader,
                             validation_data_loader,
                             model,
                             criterion,
                             optimizer,
                             device,
                             ):

        metrics_tracker = []

        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.device = device

        print("{:<15}{:<25}{:<25}{:<25}{:<25}{:<25}{:<25}".format('Epoch', 'Train Loss', 'Validation Loss', 'Train F-Score', 'Validation F-Score', 'Train Accuracy', 'Validation Accuracy'))
        print('-' * 160)

        for epoch_i in range(self.epochs):
            train_true_labels, train_predicted_scores = self.train_loop()
            test_true_labels, test_predicts_proba = self.test_loop()

            train_auc, train_accuracy, train_f_score, train_loss = self.calculate_metrics(train_true_labels, train_predicted_scores)
            validation_auc, validation_accuracy, validation_f_score, validation_loss = self.calculate_metrics(test_true_labels, test_predicts_proba)

            current_epoch_metrics_dictionary = {
                TRAIN_AUC_STR: train_auc,
                TRAIN_ACCURACY_STR: train_accuracy,
                TRAIN_F1SCORE_STR: train_f_score,
                TRAIN_LOSS_STR: train_loss,

                VALIDATION_AUC_STR: validation_auc,
                VALIDATION_ACCURACY_STR: validation_accuracy,
                VALIDATION_F1SCORE_STR: validation_f_score,
                VALIDATION_LOSS_STR: validation_loss,
            }

            metrics_tracker.insert(epoch_i, current_epoch_metrics_dictionary)

            mlflow.log_metrics(current_epoch_metrics_dictionary, step=epoch_i + 1)

            print("{:<15}{:<25}{:<25}{:<25}{:<25}{:<25}{:<25}".format(epoch_i + 1, train_loss, validation_loss, train_f_score, validation_f_score, train_accuracy, validation_accuracy))
            print('-' * 160)

        metrics_dataframe = pd.DataFrame(metrics_tracker)

        return metrics_dataframe


import optuna


class ModelEvaluator:
    class UnlabeledTestDataset(torch.utils.data.Dataset):
        def __init__(self, root_dir, hp_configs: HyperParameterConfigs):
            self.root_dir = root_dir
            self.paths = [f for f in os.listdir(self.root_dir)]

            size = (hp_configs.image_data_size, hp_configs.image_data_size)

            mean, standard_deviation = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            to_apply_transforms = transforms.Compose([
                transforms.Resize(size=size),
                transforms.ToTensor(),
                transforms.Normalize(mean, standard_deviation)
            ])

            self.transform = to_apply_transforms

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, index):
            if torch.is_tensor(index):
                index = index.tolist()

            image_path = os.path.join(self.root_dir,
                                      self.paths[index])

            from PIL import Image

            image = Image.open(image_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            image_path = str.replace(image_path, self.root_dir, '')

            return image_path, image

    def __init__(self, test_directory):
        self.test_directory = test_directory
        self.test_dataloader = None
        self.device = None

    def initialize(self, hp_configs: HyperParameterConfigs):
        test_dataset = ModelEvaluator.UnlabeledTestDataset(self.test_directory, hp_configs)
        shuffle = True
        drop_last = False
        batch_size = hp_configs.batch_size
        num_workers = 3

        self.test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, )

    def evaluate(self, model):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        IMAGE_PATH_STR = 'image_path'
        PREDICTED_STR = 'predicted'
        columns = [IMAGE_PATH_STR, PREDICTED_STR]
        total_result_df = pd.DataFrame(columns=columns)

        with torch.no_grad():
            for i, data in enumerate(self.test_dataloader):
                paths, images = data
                images = images.to(device)

                model.eval()
                outputs = model(images)

                predicts_probabilities = torch.softmax(outputs, dim=1)
                _, predicted_labels = torch.max(predicts_probabilities, dim=1)

                result_dict = {IMAGE_PATH_STR: list(paths), PREDICTED_STR: predicted_labels.cpu()}
                result = pd.DataFrame(result_dict, columns=columns, )
                total_result_df = total_result_df.append(result)

        return total_result_df


class HyperParameterOptimization:

    def __init__(self, model_initializer, model_trainer, to_optimize_metric_name='accuracy', direction='maximize', n_trial=5):
        # to_optimize_metric_name: 'accuracy' | 'f1score' | 'auc'
        # direction: 'maximize' | 'minimize'

        self.n_trial = n_trial
        self.run_counter = 0

        self.model_initializer: ModelInitializer = model_initializer
        self.model_initializer.read_data(ratio=.9, )

        self.model_trainer: ModelTrainer = model_trainer

        to_optimize_metric_name: str

        # CHECK NAME OF METRICS_STR IN CONSTANT STRINGS IN THE TOP OF FILE

        self.to_optimize_metric_column = eval(f'VALIDATION_{to_optimize_metric_name.upper()}_STR')

    def hyper_parameter_optimization_objective_function(self, trial: optuna.Trial):
        trial_params = {
            LEARNING_RATE_STR: trial.suggest_discrete_uniform(LEARNING_RATE_STR, 5 * 1e-5, 25 * 1e-5, q=5 * 1e-5),
            IMAGE_SIZE_STR: trial.suggest_categorical(IMAGE_SIZE_STR, [299]),
            EPOCHS_NUMBER_STR: trial.suggest_categorical(EPOCHS_NUMBER_STR, [1]),
            BATCH_SIZE_STR: trial.suggest_categorical(BATCH_SIZE_STR, [12]),
            WEIGHT_DECAY_STR: trial.suggest_categorical(WEIGHT_DECAY_STR, [1 * 10 ** i for i in range(-7, -5)]),
            SCHEDULER_GAMMA_STR: trial.suggest_discrete_uniform(SCHEDULER_GAMMA_STR, 0.82, 0.95, q=0.03)
        }

        self.run_counter += 1
        with mlflow.start_run(run_name=f'Run {self.run_counter}'):
            hyperparameters_configs = HyperParameterConfigs(trial_params)

            mlflow.log_params(hyperparameters_configs.to_dict())
            print(hyperparameters_configs)

            self.model_initializer.initialize_with_hyper_parameters(hyperparameters_configs)

            metrics_dataframe = self.model_trainer.fit_model_with_setup(self.model_initializer.train_dataloader,
                                                                        self.model_initializer.validation_dataloader,
                                                                        self.model_initializer.model,
                                                                        self.model_initializer.loss_criterion,
                                                                        self.model_initializer.optimizer,
                                                                        self.model_initializer.device)

            to_optimize_metric_tracks: pd.Series = metrics_dataframe[self.to_optimize_metric_column]
            metrics_max_in_tracks = to_optimize_metric_tracks.max()

            return metrics_max_in_tracks

    def tune(self):
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())

        study.optimize(self.hyper_parameter_optimization_objective_function, n_trials=self.n_trial)

        best_trial = study.best_trial

        params = best_trial.params

        print('*' * 160)
        for k, v in params.items():
            print("{:<15}{:<25}".format(k, str(v)))

        print('*' * 160)

        hyperparameters_configs = HyperParameterConfigs(params)
        hyperparameters_configs.set_metrics()

        self.model_initializer.initialize_with_hyper_parameters(hyperparameters_configs)

        _ = self.model_trainer.fit_model_with_setup(self.model_initializer.train_dataloader,
                                                    self.model_initializer.validation_dataloader,
                                                    self.model_initializer.model,
                                                    self.model_initializer.loss_criterion,
                                                    self.model_initializer.optimizer,
                                                    self.model_initializer.device)

        fine_tuned_model = self.model_initializer.model

        return fine_tuned_model, hyperparameters_configs


def main():
    try:
        MLF_EXPERIMENT_ID = mlflow.get_experiment_by_name(MLF_EXPERIMENT_NAME).experiment_id
    except:
        MLF_EXPERIMENT_ID = mlflow.create_experiment(MLF_EXPERIMENT_NAME)

    mlflow.set_experiment(experiment_id=MLF_EXPERIMENT_ID)

    data_directory = os.path.abspath(DATASET_PATH)
    model_initializer = ModelInitializer(data_directory)
    model_trainer = ModelTrainer(epochs=15)

    hpo = HyperParameterOptimization(model_initializer, model_trainer, n_trial=10)
    fine_tuned_model, to_fine_tune_hyperparameters_configs = hpo.tune()

    data_directory = 'dk-dataset/dataset/test/'
    evaluator = ModelEvaluator(data_directory)

    evaluator.initialize(to_fine_tune_hyperparameters_configs)

    total_result_df = evaluator.evaluate(fine_tuned_model)

    pd.DataFrame.to_csv(total_result_df, 'output.csv', index_label=False, index=False)


if __name__ == '__main__':
    main()
