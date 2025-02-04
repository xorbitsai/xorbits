We derived these queries from the TPC-H benchmarks. 
More information about TPC-H can be found [here](https://www.tpc.org/tpch/).

# Generating Data in Parquet Format
1. Download and Install tpch-dbgen
```bash
git clone https://github.com/Bodo-inc/tpch-dbgen
cd tpch-dbgen
make
cd ../
```

2. Generate Data
```bash
usage: python generate_data_pq.py [-h] --folder FOLDER [--SF N] [--validate_dataset]

    -h, --help        Show this help message and exit
    folder FOLDER:    output folder name (can be local folder or S3 bucket)
    SF N:             data size number in GB (Default 1)
    validate_dataset: Validate each parquet dataset with pyarrow.parquet.ParquetDataset (Default True)
```
Example:

Generate 1GB data locally:
```bash
python generate_data_pq.py --SF 1 --folder SF1
```

Generate 1TB data and upload to S3 bucket:
```bash
python generate_data_pq.py --SF 1000 --folder s3://bucket-name/
```

NOTES:
- This script assumes tpch-dbgen is in the ``same`` directory. If you downloaded it at another location, make sure to update ``tpch_dbgen_location`` in the script with the new location.
- If using S3 bucket, install ``s3fs`` and add your AWS credentials.

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

## Run queries with specific dtype backend
You can specify which dtype backend to use when reading the data:
```bash
# Use pyarrow dtype backend
python benchmarks/tpch/run_queries.py --data_set <your_tpch_dataset_dir> --dtype-backend pyarrow

# Use numpy_nullable dtype backend (default)
python benchmarks/tpch/run_queries.py --data_set <your_tpch_dataset_dir> --dtype-backend numpy_nullable
```

## Run queries using MMAP backend
```bash
python benchmarks/tpch/run_queries.py --data_set <your_tpch_dataset_dir> --mmap_root_dir <your_dir_for_mmap_files>
```
