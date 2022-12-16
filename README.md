# CS675-Machine-Learning-Project

Alzheimer’s disease is the most common type of dementia, which involves parts of the brain that control thought, memory, and language. Early diagnosis of Alzheimer’s disease with appropriate treatment could slow down the development of symptoms. Like disease prediction problem mentioned in lectures, this dataset is also imbalanced. We proposed a method that uses ResNet18 to predict our target which is a alternative of CNN model introduced in lecture. The problem combined image data with numerical data which makes this problem unique. Alzheimer’s disease itself is sensitive to genders and ages thus a ethical problem could be the fairness dealing with the accuracy of different groups of genders and age ranges.

## How to use the code

1. Download the OASIS-1 dataset (MRI scans and CSV file) from their website (http://www.oasis-brains.org/)
2. Use `python dataprep.py --dataset_path YOURPATHTODATASET --slice_ways coronal` to process the raw MRI scan data
3. Make sure your file hierarchy is correct:
```├─test
├─train
├─val
└─oasis_cross-sectional.csv
```
4. Run train.py to train the model `python trainNet.py --dataset_path YOURPATHTODATASET --model pretrained`

For trainNet.py, you can adjust the training by adding or changing arguments.

* --dataset_dir Directory for storing data_set\\
* --model model used for training (pretrained or age or attention)
* --model_path Path for storing model
* --batch_size Batch size, number of speakers per batch
* --weighted_loss Whether to use weighted cross entropy loss
* --lr Initial learning rate
* --momentum Momentum in SGD
* --epoches Maximum number of epochs
