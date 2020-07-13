# MNIST on SageMaker with PyTorch Lightning
import sagemaker
from sagemaker.pytorch import PyTorch

sagemaker_session = sagemaker.Session()

bucket = 's3://dataset.mnist'
role = sagemaker.get_execution_role()

estimator = PyTorch(entry_point='train.py',
                    role=role,
                    framework_version='1.4.0',
                    train_instance_count=1,
                    train_instance_type='ml.c4.xlarge',
                    hyperparameters={
                        'epochs': 6,
                        'batch-size': 128,
                    })

estimator.fit({
    'train': bucket+'/training',
    'test': bucket+'/testing'
})