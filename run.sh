python train_HCAN.py --data_dir=../data --xlnet_model=xlnet-large-cased --output_dir=xlnet_dream --task=dream --max_no_sent=10 --max_sent_len=20 --d_lstm=512  --lstm_layers=1 
# --do_train --do_eval --train_batch_size=32 --eval_batch_size=2 --learning_rate=2e-5 --num_train_epochs=3 --warmup_steps=120 --weight_decay=0.0 --adam_epsilon=1e-6 --gradient_accumulation_steps=16 && /root/shutdown.sh
