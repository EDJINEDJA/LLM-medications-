init:
	echo "git initialization ..."
	git init
	sleep 1
	git add .
	sleep 1
	git commit -m "Code clean, Nothing to commit #(0000)"
	sleep 1
	git branch -M main
	sleep 1
	git remote add origin https://github.com/EDJINEDJA/LLM-medications-.git
	sleep 1
	git push -u origin main

push:
	git add .
	git commit -m $(var)
	git push -u orgin main

setup:
	pipenv install 
	pipenv shell
train:
	python trainer.py --config config/config.yaml
run:
	python app.py --config config/config.yaml