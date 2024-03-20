python launch.py --name leo_align \
                 --qos lv0b \
                 --mem_per_gpu 100 \
                 --time 48 \
                 --config configs/default.yaml \
                 --port 2030 \
                 --gpu_per_node 4 \
                 --num_nodes 1 \
                 --partition HGX \
                 task=align \
                 note=align_pointbert \
                 vision3d/backbone=pointbert \
