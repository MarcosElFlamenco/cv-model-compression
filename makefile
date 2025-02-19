##make file for model compression
PYTHON := python3
##PRUNNING section

PRUNING_MODEL := 
pruning:
	$(PYTHON) train_pruning.py \
		--model legacy_seresnet33ts \
		--pruning-steps 1 \
		--pruning-ratio 0.4 \


##this should only be on branch prunning runs

