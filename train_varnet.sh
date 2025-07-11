python train.py \
--batch_size 1 \
--gradient_accumulation_steps 2 \
--num_workers 4 \
--num_epochs 5 \
--net_name test_varnet \
--model_name utils.model.varnet.VarNet \
--data_path_train ../Data/train/ \
--data_path_val ../Data/val/ \
--lr 1e-3 \
--cascade 1 \
--chans 9 \
--pools 4 \
--sens_chans 4 \
--sens_pools 4 \
--input_key kspace \
--target_key image_label \
--max_key max \
--seed 430 \
--acceleration 4 \
--task brain \
# --volume_sample_rate 0.3 \
# --restart_from_checkpoint ../result/test_varnet/checkpoints/0707_203020-tz597zye/best_model.pt \
# --continue_lr_scheduler True \