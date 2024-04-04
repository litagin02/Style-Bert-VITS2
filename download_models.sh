#!/bin/bash

aws s3 cp s3://style-bert-vits2-model/bert/ ./bert --recursive
aws s3 cp s3://style-bert-vits2-model/model_assets/ ./model_assets --recursive
aws s3 cp s3://style-bert-vits2-model/pretrained_jp_extra/ ./pretrained_jp_extra --recursive
aws s3 cp s3://style-bert-vits2-model/pretrained/ ./pretrained --recursive
aws s3 cp s3://style-bert-vits2-model/slm/ ./slm --recursive
