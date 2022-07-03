from tqdm import tqdm
from time import sleep

# bar = tqdm(total=10)
# for j in range(5):
#     for i in range(10):
#         bar.update(1)
#         sleep(0.2)
#     bar.reset()

# pbar = tqdm(total=200, ncols=150)
# for j in range(5):
#     # pbar = tqdm(total=200, ncols=150)
#     for i in range(10):
#         # pbar.set_description_str(desc="Reached {}".format(i))
#         pbar.set_postfix_str(s="Reacheddddddddd {}".format(i))
#         pbar.update(20)
#         sleep(0.1)
#     pbar.reset()
# pbar.close()
# print("fff", len(pbar))

# from collections import defaultdict
# losses_per_phase = defaultdict(list)
# losses_per_phase['train'].append(1)
# losses_per_phase['train'].append(2)
# print(losses_per_phase['train'])
# a = defaultdict(float)
# a['train_loss'] = 0.2
# a['val_loss'] = 0.3

# for k in a.keys():
#     print(k.startswith('train'))
#     print(k.startswith('val'))

# import time
# from tqdm import tqdm

# #initializing progress bar objects
# outer_loop=tqdm(range(3))
# inner_loop=tqdm(range(5))

# for i in range(len(outer_loop)):
#     inner_loop.refresh()  #force print final state
#     inner_loop.reset()  #reuse bar
#     outer_loop.update() #update outer tqdm

#     for j in range(len(inner_loop)):
#         inner_loop.update() #update inner tqdm
#         time.sleep(1)

# import time
# from tqdm import tqdm

# for i in tqdm(range(3)):
#     for j in tqdm(range(5), leave=False):
#         time.sleep(1)