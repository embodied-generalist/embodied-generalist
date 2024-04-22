python launch.py --name leo_scaling \
                 --qos lv0b \
                 --mem_per_gpu 100 \
                 --time 48 \
                 --config configs/default.yaml \
                 --port 2120 \
                 --gpu_per_node 4 \
                 --num_nodes 1 \
                 --partition HGX \
                 trainer=LeoScaler \
                 task=scaling_law \
                 note=scaling_law \
                 pretrained_ckpt_path={TBD} \
