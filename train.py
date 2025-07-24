import torch
import argparse
import logging
import datetime
from tqdm import tqdm
def main(args):
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # model = get_model(args = args, logger = logger)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Bayesian Neural Network')
    args = parser.parse_args()
    
    main(args)