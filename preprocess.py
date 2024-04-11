from src.dataprocessor import DataProcessor


cls_name = {"breast_cancer": 0, "breast_normal": 1, "thyroid_cancer": 2, "thyroid_normal": 3, "lung_cancer": 4, "lung_normal": 5}
thyroid_tirads = {'2': 'thyroid_normal', '3': 'thyroid_normal', '4a': 'thyroid_normal', '4b': 'thyroid_cancer', '4c': 'thyroid_cancer', '5': 'thyroid_cancer'}


if __name__=='__main__':
    root = 'D:/lycaoduong/workspace/datasets/others/cancer/dataset'

    engine = DataProcessor(root=root)

    # data_name = 'breastcancer'
    # csv = 'full/{}.csv'.format(data_name)
    # ratio = [0.8, 0.1, 0.1]
    # savename = data_name
    # engine.split_train_val_test(csv, ratio, savename)

    # split_name = 'test'
    # ldf_list = ['filter/breastcancer_{}.csv'.format(split_name), 'filter/thyroidcancer_{}.csv'.format(split_name), 'filter/lungcancer_{}.csv'.format(split_name)]
    # engine.merge_csv(ldf_list, save_name='{}.csv'.format(split_name))

    # folder = 'breastcancer'
    # sub = 'benign'
    # engine.makeCSVfromFolder(parent_folder=folder, folder=sub, label=1)

    # folder = 'thyroidcancer'
    # engine.makeCSVfromXML(folder=folder, tirads=thyroid_tirads)
