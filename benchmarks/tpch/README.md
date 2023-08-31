This python script allows you to benchmark TPC-H with one command line.

# TPC-H Dataset 
This script requires the following directory structure for the TPC-H dataset:
```bash
<tpch_dataset_dir>
├── customer
│   ├── part-xxx.pq
├── lineitem
│   ├── part-xxxx.pq
├── nation
├── orders
│   ├── part-xxx.pq
├── part
│   ├── part-xxx.pq
├── partsupp
│   ├── part-xxx.pq
├── region
└── supplier
```
Note that first-level directories and files are without ``.pq`` or ``.parquet`` suffixes.

# Execution
The following commands require the current directory to be the Xorbits home directory.

## Run all queries
```bash
python benchmarks/tpch/run_queries.py --data_set <your_tpch_dataset_dir>
```

## Run some specific queries
For example, if you just want to run the ``q01`` and ``q03``:
```bash
python benchmarks/tpch/run_queries.py --data_set <your_tpch_dataset_dir> --queries 1 3
```

## Run queries using an existing Xorbits cluster
When you have an existed Xorbits cluster, and you want to run queries directly in that cluster:
```bash
python benchmarks/tpch/run_queries.py --data_set <your_tpch_dataset_dir> --endpoint <your_Xorbits_endpoint>
```

## Run queries using dataset on the cloud storage
When your dataset is stored on the cloud storage, 
You need to save that ``storage_option`` as a ``json`` file and then specify it in the command:
```bash
python benchmarks/tpch/run_queries.py --data_set <your_tpch_dataset_dir> --storage_options <your_storage_option_file_path>
```

## Run queries using GPU
```bash
python benchmarks/tpch/run_queries.py --data_set <your_tpch_dataset_dir> --gpu
```
Note that you can specify ``--cuda_devices`` to decide which GPUs to use. For example, use ``gpu0`` and ``gpu1`` (If you have two or more GPUs on your machine):
```bash
python benchmarks/tpch/run_queries.py --data_set <your_tpch_dataset_dir> --gpu --cuda_devices 0 1
```

## Run queries using MMAP backend
```bash
python benchmarks/tpch/run_queries.py --data_set <your_tpch_dataset_dir> --mmap_root_dir <your_dir_for_mmap_files>
```
