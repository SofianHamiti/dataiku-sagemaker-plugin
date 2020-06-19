# -*- coding: utf-8 -*-
import json
import dataiku
from dataiku.customrecipe import get_recipe_config, get_input_names_for_role, get_output_names_for_role
from pandas.io.json import json_normalize
from sagemaker.estimator import Estimator
from sagemaker.amazon.amazon_estimator import get_image_uri
from session_utils import get_instance_role, get_sagemaker_client, get_region

SAGEMAKER_ROLE = get_instance_role()
sagemaker = get_sagemaker_client()
region = get_region()

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
algorithm = str(get_recipe_config()['algorithm'])
hyperparameters = json.loads(get_recipe_config()['hyperparameters'])
instance_type = str(get_recipe_config()['instance_type'])
instance_count = int(get_recipe_config()['instance_count'])
subnet = []
security_group_id = []
metric_definitions = []
s3_inputs = json.loads(get_recipe_config()['s3_inputs'])
s3_output = str(get_recipe_config()['s3_output'])

# ==============================================================================
# TRAINING
# ==============================================================================
container = get_image_uri(region, algorithm, 'latest')

estimator = Estimator(image_name=container,
                                base_job_name=f'dku',
                                role=SAGEMAKER_ROLE,
                                train_instance_count=instance_count,
                                train_instance_type=instance_type,
                                subnets=subnet,
                                security_group_ids=security_group_id,
                                metric_definitions=metric_definitions,
                                train_volume_size=30,
                                input_mode='File',
                                hyperparameters=hyperparameters,
                                output_path=s3_output)
estimator.fit(inputs=s3_inputs, logs=True)

# ==============================================================================
# OUTPUT
# ==============================================================================
job_description = estimator.latest_training_job.describe()
output = json_normalize(job_description)

output_dataset = output_datasets[0]
output_dataset.write_with_schema(output)
