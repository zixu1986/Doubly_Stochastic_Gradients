1. Download the mnist8m dataset at 
http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#mnist8m
Please download the un-scaled one: mnist8m.bz2.

2. Unzip it by typing "bzip2 -d mnist8m.bz2" in the directory that contains the downloaded dataset. 
It takes about twenty minutes to decompress.

3. Transform the dataset to Matlab format batches using the script
transform_8m_dataset.m. Please fill in paths for your downloaded dataset and
the output files. It takes about an hour to transform the data.

4. Modify the path variable train_datapath_pattern in preprocess_8m_data.m to
the same as the variable output_pattern in transform_8m_dataset.m

5. Run the script doubly_sgd_pcd_lr_8m.m. After about 100 iterations, the test error should reach about 0.73%.
