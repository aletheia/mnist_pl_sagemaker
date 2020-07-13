# MNIST on SageMaker with PyTorch Lightning
import json
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch


sagemaker_session = sagemaker.Session()

bucket = 's3://dataset.mnist'
role = 'arn:aws:iam::682411330166:role/SageMakerRole_MNIST'

estimator = PyTorch(entry_point='train.py',
                    source_dir='code',
                    role=role,
                    framework_version='1.4.0',
                    train_instance_count=1,
                    train_instance_type='ml.p2.xlarge',
                    hyperparameters={
                        'epochs': 6,
                        'batch-size': 128,
                    })

estimator.fit({
    'train': bucket+'/training',
    'test': bucket+'/testing'
})