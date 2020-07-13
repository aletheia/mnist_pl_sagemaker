import argparse
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from MNISTClassifier import MNISTClassifier

if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--gpus', type=int, default=1)

    # Data, model, and output directories
    parser.add_argument('-o','--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('-m','--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('-tr','--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('-te','--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args, _ = parser.parse_known_args()
    print(args)

    mnistTrainer=pl.Trainer(gpus=args.gpus, max_epochs=args.epochs, default_root_dir=args.output_data_dir)

    model = MNISTClassifier(
        batch_size=args.batch_size, 
        train_data_dir=args.train, 
        test_data_dir=args.test
        )
    mnistTrainer.fit(model)

    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)
