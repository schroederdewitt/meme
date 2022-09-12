# MEME
Contains the code for the paper ["Communicating via Markov Decision Processes" 
by Samuel Sokota*, Christian Schroeder de Witt*, Maximilian Igl, 
Luisa Zintgraf, Philip Torr, Martin Strohmeier, Zico Kolter, Shimon Whiteson, and Jakob Foerster (ICML 2022)](https://icml.cc/virtual/2022/poster/16975).

Please create issues for questions and feedback.

## Train gridworld 4x4 MaxEnt

```
python3 train_rec.py --exp-name "maxent_4x4_rec" --net "mlp_lstm_deep" --net-device "cuda:0" \
    --log-betas-range-rollout 0.1 7.5 --log-betas-eval 0.1 1 2 3 5 7 --env-arg-grid-dim 4 4 \
    --env-arg-max-steps 8 --lr 0.0005 --gamma 1.0 --epsilon-final 0.1 --bs_learn 64 --learn-n-times 256
```

## Train gridworld 4x4 RL+PBS baseline

```
python3 train_rec_token.py --exp-name "final_4x4_pbs" --net "mlp_lstm_deep" --net-device "cuda:0" \
    --log-betas-range-rollout 10 10 --log-betas-eval 10 --env-arg-grid-dim 4 4 --env-arg-max-steps 4 \
    --buffer-sampling-mode episodes --lr 0.0001 --reward-weight-task 0.5 --token-n 2 --token-pred-fn "pbs" \
    --gamma 1.0 --learn-n-times 100 --bs_learn 128
```

## Train gridworld 4x4 IQL baseline

```
python3 train_rec_token.py --exp-name "iql" --net "mlp_lstm_deep" --net-device "cuda:0" \
    --log-betas-range-rollout 10 10 --log-betas-eval 10 --env-arg-grid-dim 4 4 --env-arg-max-steps 8 \
    --buffer-sampling-mode episodes --lr 0.00001 --reward-weight-task 0.5 --token-n 16 --token-pred-fn "iql" \
    --gamma 1.0 --learn-n-times 100 --bs_learn 128
```

## Train Pong MaxEnt

```
python3 train_ff.py --exp-name "pong21" --net "pong3" --net-device "cuda:0" --log-betas-range-rollout 0.1 10\
    --log-betas-eval 0.1 1.0 2.0 3.0 5.0 10.0 --lr 0.0005 --env "pong" --bs_rollout 1 --n-vec-envs 1 \
    --n-episode-rollout-per-outer-loop 1600 --buffer-max-n-episodes 50 --buffer-device cpu --epsilon-timescale 100 \
    --n-episodes-eval 10 --gamma 1.0 --learn-n-times 64 --epsilon-final 0.05 --bs_learn 256 --input-t False \
    --eval-every-x-episodes 200
```

## Train Pong MaxEnt (Restricted Action Space)

```
python3 train_ff.py --exp-name "pongres" --net "pongres" --net-device "cuda:0" --log-betas-range-rollout 0.1 10\
    --log-betas-eval 0.1 1.0 2.0 3.0 5.0 10.0 --lr 0.0005 --env "pongres" --bs_rollout 1 --n-vec-envs 1 \
    --n-episode-rollout-per-outer-loop 1600 --buffer-max-n-episodes 50 --buffer-device cpu --epsilon-timescale 100 \
    --n-episodes-eval 10 --gamma 1.0 --learn-n-times 64 --epsilon-final 0.05 --bs_learn 256 --input-t False \
    --eval-every-x-episodes 200
```

## Train Breakout MaxEnt

```
python3 train_ff.py --exp-name "breakout" --net "breakout" --net-device "cuda:0" --log-betas-range-rollout 10 10 --log-betas-eval \
10.0 --lr 0.0005 --env "breakout" --bs_rollout 1 --n-vec-envs 1 --n-episode-rollout-per-outer-loop 1600 \
--buffer-max-n-episodes 50 --buffer-device cpu --epsilon-timescale 100 --n-episodes-eval 10 --gamma 1.0 \
--learn-n-times 64 --epsilon-final 0.05 --bs_learn 256 --input-t False --eval-every-x-episodes 200 
```

## Evaluate policy

```
python3 eval_ff.py --exp-name "bs1_reload_pong21_" --net "pong3" --net-device "cuda:0"\
    --log-betas-range-rollout 0.1 10 --log-betas-eval 6.5 --lr 0.0005 --env "pong" --bs_rollout 1 \
    --n-vec-envs 1 --n-episode-rollout-per-outer-loop 1600 --buffer-max-n-episodes 50 --buffer-device cpu \
    --epsilon-timescale 100 --n-episodes-eval 1 --gamma 1.0 --learn-n-times 64 --epsilon-final 0.05 --bs_learn 256\
    --input-t False --eval-every-x-episodes 1 --pretrain-weights path-to-policy-weights
```

## Encode message token in pretrained MaxEnt policy (Pong)

```
python3 minimum_entropy.py --model-path "models/pong21_neurips21/pong21_maxent_lb6.5_ep423493.pt"
```

## Encode image token in pretrained MaxEnt policy (Pong21)

```
python3 minimum_entropy_token.py --exp-name "met_pong21" --model-path "models/pong21_neurips21/pong21_maxent_lb6.5_ep423493"
```

## Video encode

```
ffmpeg -framerate 30  -i ./CodePong_b6a0_1/pic%d.png -c:v libx264 -pix_fmt yuv420p out_pongres_b6a0.mp4
```