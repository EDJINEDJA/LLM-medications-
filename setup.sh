#!/bin/bash
echo "Hardward updating ..."
sudo apt-get update
sudo apt-get upgrade
sudo apt update 
sudo apt upgrade

echo "Software installation .."
pip install --upgrade pip
pip install -r requirements.txt