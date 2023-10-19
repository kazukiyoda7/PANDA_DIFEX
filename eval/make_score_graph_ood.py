import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', type=str, default=42)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--id_class', type=int, required=True)
    parser.add_argument('--ood_class', type=int, default=1)
    parser.add_argument('--eval_domain', type=str, default='all')
    args = parser.parse_args()
    return args
    
args = get_args()

corruption_list = ['clean', 'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'gaussian_blur', 'snow', 
                    'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', 'saturate', 'spatter']

cifar10_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 

id_class = args.id_class
ood_class = args.ood_class

corruption_names = args.eval_domain.split('-')
if 'all' in corruption_names:
    corruption_names = corruption_list
else:
    assert all(c in corruption_list for c in corruption_names), "corruption name is incorrect"
print(corruption_names)

corruption_list = corruption_names

save_dir = os.path.join(args.output_dir, 'graph', 'covariate', str(id_class))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
for s in range(1, 6):
    severity = s

    score_list = {}
    
    path = os.path.join(args.input_dir, str(s), str(id_class), 'clean.npy')
    arr = np.load(path)
    score_list['id_clean'] = arr

    for k in corruption_list:
        path = os.path.join(args.input_dir, str(s), str(ood_class), f'{k}.npy')
        arr = np.load(path)
        score_list[k] = arr

    # box ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    data_to_plot = [score_list[key] for key in score_list.keys()]
    plt.boxplot(data_to_plot, vert=True, patch_artist=True, labels=score_list.keys())
    plt.title(f"Boxplot of anomaly score (ID class = {cifar10_class[id_class]})")
    plt.ylabel("Value")
    plt.grid(True)
    plt.xticks(rotation=90)
    plt.tight_layout()
    box_save_dir = os.path.join(save_dir, 'box')
    if not os.path.exists(box_save_dir):
        os.makedirs(box_save_dir)
    plt.savefig(os.path.join(box_save_dir, f'box-ood{ood_class}-severity{severity}.png'))


    # median --------------------------------------------------------------
    medians = [np.median(score_list[key]) for key in score_list.keys()]

    plt.figure(figsize=(10, 6))

    y_values = np.linspace(0, len(medians) - 1, len(medians))

    # 赤い点のインデックスと青い点のインデックスを取得
    red_indices = [i for i, txt in enumerate(score_list.keys()) if txt in corruption_list]
    green_indices = [i for i, txt in enumerate(score_list.keys()) if txt == 'id_clean']

    # 赤い点をプロット
    plt.scatter([medians[i] for i in red_indices], [y_values[i] for i in red_indices], color='red', s=100)
    # 緑の点をプロット
    plt.scatter([medians[i] for i in green_indices], [y_values[i] for i in green_indices], color='green', s=100)

    # ラベルを付ける
    for i, txt in enumerate(score_list.keys()):
        y_offset = 0.2 if i in red_indices else -0.2  # 赤点のラベルは上に、青点のラベルは下に配置
        plt.annotate(txt, (medians[i], y_values[i] + y_offset), ha='center', va='bottom', rotation=0)

    plt.title(f"Medians of anomaly score (ID class = {cifar10_class[id_class]})")
    plt.xlabel("Median")
    plt.yticks([])  # y軸の目盛りを非表示にする
    plt.grid(axis='x')
    plt.tight_layout()
    median_save_dir = os.path.join(save_dir, 'median')
    if not os.path.exists(median_save_dir):
        os.makedirs(median_save_dir)
    plt.savefig(os.path.join(median_save_dir, f'median-ood{ood_class}-severity{severity}.png'))

    # # mean-fix --------------------------------------------------------------
    # means = [np.mean(score_list[key]) for key in score_list.keys()]

    # plt.figure(figsize=(10, 6))

    # y_values = np.linspace(0, len(means) - 1, len(means))

    # # 赤い点のインデックスと青い点のインデックスを取得
    # red_indices = [i for i, txt in enumerate(score_list.keys()) if txt in corruption_list]
    # blue_indices = [i for i, txt in enumerate(score_list.keys()) if txt in cifar10_class]
    # green_indices = [i for i, txt in enumerate(score_list.keys()) if txt == 'clean']

    # # 赤い点をプロット
    # plt.scatter([means[i] for i in red_indices], [y_values[i] for i in red_indices], color='red', s=100)
    # # 青い点をプロット
    # plt.scatter([means[i] for i in blue_indices], [y_values[i] for i in blue_indices], color='skyblue', s=100)
    # # 緑の点をプロット
    # plt.scatter([means[i] for i in green_indices], [y_values[i] for i in green_indices], color='green', s=100)

    # # ラベルを付ける
    # for i, txt in enumerate(score_list.keys()):
    #     y_offset = 0.2 if i in red_indices else -0.2  # 赤点のラベルは上に、青点のラベルは下に配置
    #     plt.annotate(txt, (means[i], y_values[i] + y_offset), ha='center', va='bottom', rotation=0)

    # plt.title(f"Means of anomaly score (ID class = {cifar10_class[id_class]})")
    # plt.xlabel("Mean")
    # plt.yticks([])  # y軸の目盛りを非表示にする
    # plt.grid(axis='x')
    # plt.tight_layout()
    # plt.savefig('result_score/graph_corr_class/mean.png')




