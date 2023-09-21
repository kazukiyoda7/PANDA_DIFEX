import matplotlib.pyplot as plt
import numpy as np
import os

corruption_list = ['clean', 'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'gaussian_blur', 'snow', 
                    'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', 'saturate', 'spatter']

cifar10_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 

for i in range(1):
    id_class = i

    for s in range(1, 6):
        ood_class = list(filter(lambda x: x!=id_class, range(10)))
        severity = s

        score_list = {}

        for i in ood_class:
            path = f'result_score/resnet18/{id_class}/ood/clean_{i}.npy'
            arr = np.load(path)
            score_list[cifar10_class[i]] = arr


        for k in corruption_list:
            path = f'result_score/resnet18/{id_class}/{severity}/{k}.npy'
            arr = np.load(path)
            score_list[k] = arr
            
        # # histgram -----------------------------------------------------------
        # plt.figure(figsize=(10, 6))
        # # 各numpy配列のヒストグラムを描画
        # for key in score_list.keys():
        #     arr = score_list[key]
        #     plt.hist(arr, bins=10, alpha=0.5, label=key, density=True)
        # plt.title("Histograms of anomaly score")
        # plt.xlabel("Value")
        # plt.ylabel("Frequency")
        # plt.legend()
        # plt.grid(True)
        # plt.savefig('result_score/graph_corr_class/hist.png')


        # # box ------------------------------------------------------------------
        # plt.figure(figsize=(10, 6))
        # data_to_plot = [score_list[key] for key in score_list.keys()]
        # plt.boxplot(data_to_plot, vert=True, patch_artist=True, labels=score_list.keys())
        # plt.title("Boxplot of anomaly score")
        # plt.ylabel("Value")
        # plt.grid(True)
        # plt.xticks(rotation=90)
        # plt.tight_layout()
        # plt.savefig('result_score/graph_corr_class/box.png') # この行で箱ひげ図を保存します


        # # median --------------------------------------------------------------
        # # 中央値を抽出
        # medians = [np.median(score_list[key]) for key in score_list.keys()]

        # plt.figure(figsize=(10, 6))

        # # y値として1の定数配列を使用して、すべての中央値を同じy=1の位置にプロット
        # plt.scatter(medians, [1] * len(medians), color='skyblue', s=100)  # sはマーカーサイズ

        # # ラベルを付ける
        # for i, txt in enumerate(score_list.keys()):
        #     plt.annotate(txt, (medians[i], 1.02), ha='center', va='bottom', rotation=90)

        # plt.title("Medians of anomaly score")
        # plt.xlabel("Median ")
        # plt.yticks([])  # y軸の目盛りを非表示にする
        # plt.grid(axis='x')
        # plt.tight_layout()
        # plt.savefig('result_score/graph_corr_class/median.png')

        # median-fix --------------------------------------------------------------
        medians = [np.median(score_list[key]) for key in score_list.keys()]

        plt.figure(figsize=(10, 6))

        y_values = np.linspace(0, len(medians) - 1, len(medians))

        # 赤い点のインデックスと青い点のインデックスを取得
        red_indices = [i for i, txt in enumerate(score_list.keys()) if txt in corruption_list]
        blue_indices = [i for i, txt in enumerate(score_list.keys()) if txt in cifar10_class]
        green_indices = [i for i, txt in enumerate(score_list.keys()) if txt == 'clean']

        # 赤い点をプロット
        plt.scatter([medians[i] for i in red_indices], [y_values[i] for i in red_indices], color='red', s=100)
        # 青い点をプロット
        plt.scatter([medians[i] for i in blue_indices], [y_values[i] for i in blue_indices], color='skyblue', s=100)
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
        
        save_dir = os.path.join('result_score/graph_corr_class/', str(id_class))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'median-severity{severity}.png'))

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




