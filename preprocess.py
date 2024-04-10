from src.dataprocessor import DataProcessor


cls_name = {0: "breast_cancer", 1: "breast_normal", 2: "thyroid_cancer", 3: "thyroid_normal", 4: "lung_cancer", 5: "lung_normal"}
thyroid_tirads = {'2': 'thyroid_normal', '3': 'thyroid_normal', '4a': 'thyroid_normal', '4b': 'thyroid_cancer', '4c': 'thyroid_cancer', '5': 'thyroid_cancer'}


if __name__=='__main__':
    root = '../dataset'
    engine = DataProcessor(root=root)
    # folder = 'malignant'
    # engine.makeCSVfromFolder(folder=folder, label=4)
    engine.makeCSVfromXML('thyroidcancer', tirads=thyroid_tirads)
