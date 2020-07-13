#!/usr/bin/env bash

# This script creates a role named SageMakerRole
# that can be used by SageMaker and has Full access to S3.

ROLE_NAME=SageMakerRole_MNIST

# WARNING: this policy gives full S3 access to container that
# is running in SageMaker. You can change this policy to a more
# restrictive one, or create your own policy.
POLICY=arn:aws:iam::aws:policy/AmazonS3FullAccess

# Creates a AWS policy that allows the role to interact
# with ANY S3 bucket
cat <<EOF > ./assume-role-policy-document.json
{
	"Version": "2012-10-17",
	"Statement": [{
		"Effect": "Allow",
		"Principal": {
			"Service": "sagemaker.amazonaws.com"
		},
		"Action": "sts:AssumeRole"
	}]
}
EOF

# Creates the role
aws iam create-role --role-name ${ROLE_NAME} --assume-role-policy-document file://./assume-role-policy-document.json

# attaches the S3 full access policy to the role
aws iam attach-role-policy --policy-arn ${POLICY}  --role-name ${ROLE_NAME}