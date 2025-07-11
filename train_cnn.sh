python train.py \
--batch_size 1 \
--gradient_accumulation_steps 2 \
--num_workers 4 \
--num_epochs 5 \
--net_name test_cnn \
--model_name utils.model.cnn.CNN \
--data_path_train ../Data/train/ \
--data_path_val ../Data/val/ \
--lr 1e-3 \
--seed 430 \
# --restart_from_checkpoint ../result/test_promptmr/checkpoints/0707_165655-9zbeok11/best_model.pt \
# --continue_lr_scheduler True \