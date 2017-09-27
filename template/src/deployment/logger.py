import logging
from ...logs import setup_logger


api_logger = setup_logger('APIDeploy', 'deployment/', logging.DEBUG)
train_logger = setup_logger('TrainDeploy', 'deployment/', logging.DEBUG)

