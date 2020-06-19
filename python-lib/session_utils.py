import boto3


def get_instance_role():
    client = boto3.client('sts')
    response = client.get_caller_identity()
    account_id = response['Account']
    assumed_role_name = response['Arn'].split('/')[1]
    role_arn = f'arn:aws:iam::{account_id}:role/{assumed_role_name}'
    return role_arn


def get_region():
    boto3_session = boto3.session.Session()
    return boto3_session.region_name


def get_sagemaker_client():
    return boto3.client('sagemaker', region_name=get_region())
