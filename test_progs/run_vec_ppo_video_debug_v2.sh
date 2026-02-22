export MINIWOB_URL=http://127.0.0.1:8000/miniwob/

python vec_ppo_video_debug_v2.py \
  --env-id browsergym/miniwob.click-button \
  --num-envs  1\
  --steps-per-env 128 \
  --ppo-epochs 2 \
  --total-iters 200 \
  --headless --disable-checker \
  --train-mp4-dir ./runs/train_videos --train-mp4-every 5 \
  --traj-print --traj-print-topk 5 --traj-print-at-end \
  --traj-jsonl ./runs/trajectories.jsonl
