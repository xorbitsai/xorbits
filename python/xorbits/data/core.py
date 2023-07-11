from .block import Block, BlockData
from .dataset import Dataset, DatasetData
from .backends.huggingface.core import HuggingfaceDataset, HuggingfaceDatasetData
from .._mars.core.entity.output_types import OutputType, register_output_types


print("register_output_types data")
register_output_types(
    OutputType.huggingface_data,
    (HuggingfaceDataset, HuggingfaceDatasetData),
    (Block, BlockData),
)
