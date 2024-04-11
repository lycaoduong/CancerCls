import yaml
import matplotlib.pyplot as plt
import numpy as np
import os


class YamlRead:
    def __init__(self, params_path):
        self.params = yaml.safe_load(open(params_path, encoding='utf-8').read())

    def update(self, dictionary):
        self.params = dictionary

    def __getattr__(self, item):
        return self.params.get(item, None)


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha='center')
def plot_data(data_dis, cls_name, save_dir="./", save_name="fg.png"):
    plt.rc('font', size=20)
    fig = plt.figure(figsize=(20, 10))
    x_pos = np.arange(len(cls_name))
    plt.bar(cls_name, data_dis[0], align='center', alpha=0.5)
    addlabels(x_pos, data_dis[0])
    plt.xticks(x_pos, cls_name, rotation=45)
    plt.ylabel('Total images', fontsize=30)
    plt.title('Cancer Images Distribution', fontsize=25)
    # plt.show()
    save_path = os.path.join(save_dir, save_name)
    plt.tight_layout()
    plt.savefig(save_path)
    