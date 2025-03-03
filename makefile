##make file for model compression
PYTHON := python3
##PRUNNING section

PRUNING_MODEL := resnet50

pruning:
	$(PYTHON) train_pruning.py \
		--model $(PRUNING_MODEL) \
		--pruning-steps 1 \
		--device cuda
		--pruning-ratio 0.4 \

