python launch.py --name leo_align \
                 --qos lv0b \
                 --mem_per_gpu 100 \
                 --time 48 \
                 --config configs/default.yaml \
                 --port 2040 \
                 --gpu_per_node 4 \
                 --num_nodes 1 \
                 --partition HGX \
                 task=align \
                 note=align_vicuna13b \
                 llm=vicuna13b \
                 dataloader.train.batchsize=2 \
                 dataloader.train.num_workers=2 \
                 dataloader.eval.batchsize=2 \
                 dataloader.eval.num_workers=2 \
