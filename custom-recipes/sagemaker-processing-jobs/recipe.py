# -*- coding: utf-8 -*-
import dataiku
from dataiku.customrecipe import get_recipe_config, get_input_names_for_role, get_output_names_for_role
from pandas.io.json import json_normalize
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput
from session_utils import get_instance_role, get_sagemaker_client, get_region

SAGEMAKER_ROLE = get_instance_role()

# ==============================================================================
# DATAIKU SETUP
# ==============================================================================
input_names = get_input_names_for_role('input')
input_datasets = [dataiku.Dataset(name) for name in input_names]
output_names = get_output_names_for_role('output')
output_datasets = [dataiku.Dataset(name) for name in output_names]
# ==============================================================================
# TRAINING JOB PARAMETERS
# ==============================================================================
container = str(get_recipe_config()['container'])
input_folder = str(get_recipe_config()['input_folder'])
output_folder = str(get_recipe_config()['output_folder'])
s3_input = str(get_recipe_config()['s3_input'])
s3_output = str(get_recipe_config()['s3_output'])
instance_type = str(get_recipe_config()['instance_type'])
instance_count = int(get_recipe_config()['instance_count'])

# ==============================================================================
# TRAINING
# ==============================================================================
data_processor = Processor(role=SAGEMAKER_ROLE,
                           image_uri=container,
                           instance_count=instance_count,
                           instance_type=instance_type,
                           volume_size_in_gb=30,
                           max_runtime_in_seconds=3600,
                           base_job_name='dku-processing')

data_processor.run(
    arguments=[
        f'--input={input_folder}',
        f'--output={output_folder}'
    ],
    inputs=[
        ProcessingInput(
            input_name='input',
            source=s3_input,
            destination=input_folder
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name='preprocessed',
            source=output_folder,
            destination=s3_output
        )
    ]
)

# ==============================================================================
# OUTPUT
# ==============================================================================
job_description = data_processor.latest_job.describe()
output = json_normalize(job_description)

output_dataset = output_datasets[0]
output_dataset.write_with_schema(output)
