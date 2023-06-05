import argparse
from time import sleep, time
import yaml

from src.Inference import inference 

parser = argparse.ArgumentParser()
parser.add_argument("--config" , type=str  , required=True , help="path to yaml config")
args = parser.parse_args()

with open(args.config , mode="r") as stream:
        config = yaml.safe_load(stream)
        
input="Les médicaments couramment utilisés pour traiter : Affections de la peau chez un patient présentant les symptômes suivants : ['anxiété et nervosité', 'dépression', 'essoufflement','symptômes dépressifs ou psychotiques', 'douleurs thoraciques aiguës','étourdissements', 'insomnie','mouvements involontaires anormaux', 'pression artérielle', 'palpitations','battements de cœur irréguliers','respiration rapide'],"

if __name__=="__main__":
    inference(input,"./weights_/llm")