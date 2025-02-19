##make file for model compression
PYTHON := python3
##PRUNNING section

PRUNING_MODEL := 
pruning:
	$(PYTHON) train_pruning.py \
		--model legacy_seresnet33ts \
		--pruning-steps 1 \
		--pruning-ratio 0.4 \


##now lets see when this disapears
