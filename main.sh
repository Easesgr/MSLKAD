#!/bin/bash

nohup python main.py > "./logs/$(date +%Y%m%d_%H%M%S)_train.log" 2>&1 &
