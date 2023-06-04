import argparse
from time import sleep, time
import yaml
from . import trainer
from src.utils import utils
from src import trainer , inference

parser = argparse.ArgumentParser()
parser.add_argument("--config" , type=str  , required=True , help="path to yaml config")
args = parser.parse_args()

with open(args.config , mode="r") as stream:
        config = yaml.safe_load(stream)
parser = utils.Utils(config)
data = parser.load_datasets()
model , tokenizer = parser.load_models()
trainer(data , model , tokenizer, config)


print(parser.load_models())