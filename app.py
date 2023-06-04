from src.Inference import inference 
input="Les médicaments couramment utilisés pour traiter : Affections de la peau chez un patient présentant les symptômes suivants : ['anxiété et nervosité', 'dépression', 'essoufflement','symptômes dépressifs ou psychotiques', 'douleurs thoraciques aiguës','étourdissements', 'insomnie','mouvements involontaires anormaux', 'pression artérielle', 'palpitations','battements de cœur irréguliers','respiration rapide'],"

if __name__=="__main__":
    inference(input,"./weights_/llm")