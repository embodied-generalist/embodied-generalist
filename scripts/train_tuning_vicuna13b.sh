python launch.py --name leo_tuning \
                 --qos lv0b \
                 --mem_per_gpu 100 \
                 --time 48 \
                 --config configs/default.yaml \
                 --port 2100 \
                 --gpu_per_node 4 \
                 --num_nodes 1 \
                 --partition HGX \
                 task=tuning_noact \
                 note=tuning_vicuna13b \
                 llm=vicuna13b \
                 pretrained_ckpt_path={TBD} \
