# Dataset

>In the  torch  library, the  Dataset  class is used as a base class to create custom datasets for machine learning tasks. The  Dataset  class provides an interface for accessing samples and their corresponding labels. You can subclass this base class and implement the  __getitem__  and  __len__  methods to provide access to your data samples. 
Here's an example of how you can create a custom dataset using the  Dataset  class:

> 在  torch  库中， Dataset  类用作创建用于机器学习任务的自定义数据集的基类。 Dataset  类提供了一种访问样本及其相应标签的接口。您可以子类化此基类并实现  __getitem__  和  __len__  方法，以提供对您的数据样本的访问。以下是使用  Dataset  类创建自定义数据集的示例:


    import torch
    from torch.utils.data import Dataset
     class CustomDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels
        def __getitem__(self, index):
            x = self.data[index]
            y = self.labels[index]
            return x, y
        def __len__(self):
            return len(self.data)
            
> In this example, we're creating a custom dataset that takes two arguments:  data  and  labels . We initialize these variables in the  __init__  method.  
>The  __getitem__  method takes an index and returns the data sample and corresponding label at that index.  
>The  __len__  method returns the number of samples in the dataset. 

> 在这个例子中，我们创建了一个自定义数据集，需要两个参数： data  和  labels 。我们在  __init__  方法中初始化这些变量。 __getitem__  方法接受一个索引，并返回该索引处的数据样本和相应的标签。 __len__  方法返回数据集中的样本数量。

## 代码逻辑详细梳理
 该代码是一个自定义数据集类  `MyData` ，继承自  `torch.utils.data.Dataset` 。该数据集类用于读取指定目录中的图像和标签数据。
 ###  `__init__(self, pic_dir, label_dir)` 
 该方法接受两个参数： `pic_dir` 和 `label_dir` 。分别表示存储图像和标签数据的目录。
 在该方法中，首先将  `pic_dir`  和  `label_dir`  存储起来，以备后用。然后，通过  `os.listdir()`  方法获取  `pic_dir`  和  `label_dir`  目录下的所有文件列表，分别存储在  `image_list`  和  `label_list`  属性中。
 ###  `__getitem__(self, item)` 
 该方法接受一个参数  `item` ，表示要获取的数据在数据集中的索引。在该方法中，首先通过  `image_list`  和  `label_list`  获取图像和标签文件名。然后，使用  `PIL`  库中的  `Image.open()`  方法打开图像文件。接着，使用  `open()`  方法读取标签文件中的文本数据。
 最后，将读取到的图像和标签数据返回。
 ###  `__len__(self)` 
 该方法返回数据集的长度，即图像和标签数据的数量。
 ### 创建  `MyData`  类的实例
 在代码的最后，创建了一个  `MyData`  类的实例  `mydata` ，并传入了图像和标签数据存储的目录参数。