"""AWS infrastructure validator for production readiness.

This validator checks that all AWS infrastructure is properly configured,
including credentials, VPC, IAM roles, API Gateway, Bedrock access, S3 buckets,
and CloudFormation templates.
"""

import json
import logging
from typing import List, Optional

from .base_validator import BaseValidator
from .models import ValidationResult, ValidationStatus


logger = logging.getLogger(__name__)


class AWSInfrastructureValidator(BaseValidator):
    """Validates AWS infrastructure for production deployment.
    
    Checks for:
    - AWS credentials and permissions
    - VPC configuration
    - IAM roles and policies
    - API Gateway configuration
    - Bedrock model access
    - S3 bucket configuration
    - CloudFormation template validity
    """
    
    def __init__(
        self,
        aws_region: Optional[str] = None,
        vpc_id: Optional[str] = None,
        bedrock_model_id: Optional[str] = None,
        required_iam_roles: Optional[List[str]] = None,
        required_s3_buckets: Optional[List[str]] = None
    ):
        """Initialize the AWS infrastructure validator.
        
        Args:
            aws_region: AWS region for resources
            vpc_id: VPC ID to validate
            bedrock_model_id: Bedrock model ID to validate access
            required_iam_roles: List of IAM role names that must exist
            required_s3_buckets: List of S3 bucket names that must exist
        """
        super().__init__(name="AWSInfrastructureValidator")
        self.aws_region = aws_region
        self.vpc_id = vpc_id
        self.bedrock_model_id = bedrock_model_id
        self.required_iam_roles = required_iam_roles or [
            "AgentScrivenerLambdaExecutionRole",
            "AgentScrivenerECSTaskRole"
        ]
        self.required_s3_buckets = required_s3_buckets or []
        
    async def validate(self) -> List[ValidationResult]:
        """Execute all AWS infrastructure validation checks.
        
        Returns:
            List of validation results
        """
        self.log_validation_start()
        
        results = []
        
        # Validate each AWS infrastructure aspect
        results.append(await self.validate_aws_credentials())
        results.append(await self.validate_vpc_configuration())
        results.append(await self.validate_iam_roles())
        results.append(await self.validate_api_gateway_configuration())
        results.append(await self.validate_bedrock_access())
        results.append(await self.validate_s3_buckets())
        results.append(await self.validate_cloudformation_templates())
        
        self.log_validation_complete(results)
        return results
    
    async def validate_aws_credentials(self) -> ValidationResult:
        """Validate AWS credentials and permissions using boto3 STS.
        
        Checks for:
        - Valid AWS credentials are configured
        - Credentials have necessary permissions for Bedrock, CloudWatch, Secrets Manager
        - Account ID and user/role information
        
        Returns:
            ValidationResult for AWS credentials
        """
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            
            try:
                # Create STS client to verify credentials
                if self.aws_region:
                    sts_client = boto3.client('sts', region_name=self.aws_region)
                else:
                    sts_client = boto3.client('sts')
                
                # Get caller identity
                identity = sts_client.get_caller_identity()
                
                account_id = identity.get('Account')
                user_arn = identity.get('Arn')
                user_id = identity.get('UserId')
                
                # Test permissions for required services
                permission_checks = []
                
                # Check Bedrock permissions
                try:
                    bedrock_client = boto3.client('bedrock', region_name=self.aws_region or 'us-east-1')
                    bedrock_client.list_foundation_models()
                    permission_checks.append(("Bedrock", True, None))
                except ClientError as e:
                    error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                    permission_checks.append(("Bedrock", False, error_code))
                
                # Check CloudWatch permissions
                try:
                    cloudwatch_client = boto3.client('cloudwatch', region_name=self.aws_region or 'us-east-1')
                    cloudwatch_client.list_metrics(MaxRecords=1)
                    permission_checks.append(("CloudWatch", True, None))
                except ClientError as e:
                    error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                    permission_checks.append(("CloudWatch", False, error_code))
                
                # Check Secrets Manager permissions
                try:
                    secrets_client = boto3.client('secretsmanager', region_name=self.aws_region or 'us-east-1')
                    secrets_client.list_secrets(MaxResults=1)
                    permission_checks.append(("SecretsManager", True, None))
                except ClientError as e:
                    error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                    permission_checks.append(("SecretsManager", False, error_code))
                
                # Check for failed permissions
                failed_permissions = [
                    (service, error) for service, success, error in permission_checks if not success
                ]
                
                if failed_permissions:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"AWS credentials lack permissions for: {', '.join([s for s, _ in failed_permissions])}",
                        details={
                            "account_id": account_id,
                            "user_arn": user_arn,
                            "user_id": user_id,
                            "permission_checks": permission_checks,
                            "failed_permissions": failed_permissions,
                            "region": self.aws_region or "default"
                        },
                        remediation_steps=[
                            "Grant necessary AWS permissions to the IAM user/role",
                            "Required permissions:",
                            "  - bedrock:ListFoundationModels, bedrock:InvokeModel",
                            "  - cloudwatch:PutMetricData, cloudwatch:ListMetrics",
                            "  - secretsmanager:GetSecretValue, secretsmanager:ListSecrets",
                            "Update IAM policy to include these permissions",
                            f"Failed services: {', '.join([f'{s} ({e})' for s, e in failed_permissions])}"
                        ]
                    )
                
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message="AWS credentials are valid with all required permissions",
                    details={
                        "account_id": account_id,
                        "user_arn": user_arn,
                        "user_id": user_id,
                        "permission_checks": permission_checks,
                        "region": self.aws_region or "default"
                    }
                )
                
            except NoCredentialsError:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message="AWS credentials not configured",
                    details={"reason": "no_credentials"},
                    remediation_steps=[
                        "Configure AWS credentials using one of these methods:",
                        "  1. AWS CLI: aws configure",
                        "  2. Environment variables: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY",
                        "  3. AWS credentials file: ~/.aws/credentials",
                        "  4. IAM role (if running on EC2/ECS/Lambda)",
                        "Ensure credentials have necessary permissions for Bedrock, CloudWatch, and Secrets Manager"
                    ]
                )
            
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Failed to validate AWS credentials: {error_code}",
                    details={
                        "error": str(e),
                        "error_code": error_code
                    },
                    remediation_steps=[
                        "Check AWS credentials are valid and not expired",
                        "Verify IAM permissions for sts:GetCallerIdentity",
                        "Check network connectivity to AWS services",
                        f"Error: {error_code}"
                    ]
                )
                
        except ImportError:
            return self.create_result(
                status=ValidationStatus.SKIP,
                message="boto3 not installed, skipping AWS credentials validation",
                details={"reason": "boto3_not_installed"},
                remediation_steps=[
                    "Install boto3: pip install boto3",
                    "Or add boto3 to your requirements.txt"
                ]
            )
    
    async def validate_vpc_configuration(self) -> ValidationResult:
        """Validate VPC configuration.
        
        Checks for:
        - VPC exists if vpc_id is specified
        - VPC has correct subnet configuration
        - Subnets are in multiple availability zones
        
        Returns:
            ValidationResult for VPC configuration
        """
        if not self.vpc_id:
            return self.create_result(
                status=ValidationStatus.SKIP,
                message="No VPC ID specified, skipping VPC validation",
                details={"reason": "no_vpc_id"}
            )
        
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            
            try:
                # Create EC2 client
                if self.aws_region:
                    ec2_client = boto3.client('ec2', region_name=self.aws_region)
                else:
                    ec2_client = boto3.client('ec2')
                
                # Describe VPC
                vpc_response = ec2_client.describe_vpcs(VpcIds=[self.vpc_id])
                
                if not vpc_response.get('Vpcs'):
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"VPC {self.vpc_id} not found",
                        details={
                            "vpc_id": self.vpc_id,
                            "region": self.aws_region or "default"
                        },
                        remediation_steps=[
                            f"Create VPC with ID {self.vpc_id} or update configuration with correct VPC ID",
                            "Use AWS CLI: aws ec2 create-vpc --cidr-block 10.0.0.0/16",
                            "Or configure VPC in AWS CDK/CloudFormation",
                            f"Verify VPC exists in region: {self.aws_region or 'default'}"
                        ]
                    )
                
                vpc = vpc_response['Vpcs'][0]
                vpc_state = vpc.get('State')
                
                if vpc_state != 'available':
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"VPC {self.vpc_id} is in state '{vpc_state}', expected 'available'",
                        details={
                            "vpc_id": self.vpc_id,
                            "vpc_state": vpc_state,
                            "region": self.aws_region or "default"
                        },
                        remediation_steps=[
                            f"Wait for VPC to become available",
                            "Check VPC configuration for issues",
                            "Verify VPC is not being deleted"
                        ]
                    )
                
                # Check subnets
                subnets_response = ec2_client.describe_subnets(
                    Filters=[{'Name': 'vpc-id', 'Values': [self.vpc_id]}]
                )
                
                subnets = subnets_response.get('Subnets', [])
                
                if not subnets:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"VPC {self.vpc_id} has no subnets configured",
                        details={
                            "vpc_id": self.vpc_id,
                            "subnet_count": 0,
                            "region": self.aws_region or "default"
                        },
                        remediation_steps=[
                            "Create subnets in the VPC",
                            "Recommended: Create subnets in at least 2 availability zones",
                            "Use AWS CLI: aws ec2 create-subnet --vpc-id <vpc-id> --cidr-block <cidr> --availability-zone <az>",
                            "Or configure subnets in AWS CDK/CloudFormation"
                        ]
                    )
                
                # Check availability zones
                availability_zones = set(subnet.get('AvailabilityZone') for subnet in subnets)
                
                if len(availability_zones) < 2:
                    return self.create_result(
                        status=ValidationStatus.WARNING,
                        message=f"VPC {self.vpc_id} has subnets in only {len(availability_zones)} availability zone(s), recommended: 2+",
                        details={
                            "vpc_id": self.vpc_id,
                            "subnet_count": len(subnets),
                            "availability_zones": list(availability_zones),
                            "region": self.aws_region or "default"
                        },
                        remediation_steps=[
                            "Add subnets in additional availability zones for high availability",
                            "Recommended: At least 2 availability zones",
                            "This improves fault tolerance and availability"
                        ]
                    )
                
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message=f"VPC {self.vpc_id} is properly configured with {len(subnets)} subnet(s) across {len(availability_zones)} availability zone(s)",
                    details={
                        "vpc_id": self.vpc_id,
                        "vpc_state": vpc_state,
                        "subnet_count": len(subnets),
                        "availability_zones": list(availability_zones),
                        "region": self.aws_region or "default"
                    }
                )
                
            except NoCredentialsError:
                return self.create_result(
                    status=ValidationStatus.SKIP,
                    message="AWS credentials not configured, skipping VPC validation",
                    details={"reason": "no_credentials"}
                )
            
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Failed to validate VPC configuration: {error_code}",
                    details={
                        "vpc_id": self.vpc_id,
                        "error": str(e),
                        "error_code": error_code
                    },
                    remediation_steps=[
                        "Check AWS credentials have EC2 permissions",
                        "Required permissions: ec2:DescribeVpcs, ec2:DescribeSubnets",
                        "Verify the VPC ID is correct",
                        f"Check if VPC exists in region: {self.aws_region or 'default'}",
                        f"Error: {error_code}"
                    ]
                )
                
        except ImportError:
            return self.create_result(
                status=ValidationStatus.SKIP,
                message="boto3 not installed, skipping VPC validation",
                details={"reason": "boto3_not_installed"}
            )
    
    async def validate_iam_roles(self) -> ValidationResult:
        """Validate IAM roles and policies.
        
        Checks for:
        - Required IAM roles exist
        - Roles have correct trust policies
        - Roles have necessary permissions
        
        Returns:
            ValidationResult for IAM roles
        """
        if not self.required_iam_roles:
            return self.create_result(
                status=ValidationStatus.SKIP,
                message="No IAM roles specified for validation",
                details={"reason": "no_required_roles"}
            )
        
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            
            try:
                # Create IAM client
                iam_client = boto3.client('iam')
                
                missing_roles = []
                found_roles = []
                role_details = {}
                
                for role_name in self.required_iam_roles:
                    try:
                        role_response = iam_client.get_role(RoleName=role_name)
                        role = role_response.get('Role', {})
                        found_roles.append(role_name)
                        
                        # Store role details
                        role_details[role_name] = {
                            "arn": role.get('Arn'),
                            "created": str(role.get('CreateDate')),
                            "has_policies": True
                        }
                        
                    except ClientError as e:
                        if e.response.get('Error', {}).get('Code') == 'NoSuchEntity':
                            missing_roles.append(role_name)
                        else:
                            raise
                
                if missing_roles:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"Missing required IAM roles: {', '.join(missing_roles)}",
                        details={
                            "missing_roles": missing_roles,
                            "found_roles": found_roles,
                            "role_details": role_details
                        },
                        remediation_steps=[
                            f"Create missing IAM roles: {', '.join(missing_roles)}",
                            "For Lambda execution role:",
                            "  - Trust policy: Allow lambda.amazonaws.com to assume role",
                            "  - Permissions: AWSLambdaBasicExecutionRole, Bedrock access, CloudWatch access",
                            "For ECS task role:",
                            "  - Trust policy: Allow ecs-tasks.amazonaws.com to assume role",
                            "  - Permissions: Bedrock access, CloudWatch access, Secrets Manager access",
                            "Use AWS CLI: aws iam create-role --role-name <name> --assume-role-policy-document file://trust-policy.json",
                            "Or configure roles in AWS CDK/CloudFormation"
                        ]
                    )
                
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message=f"All required IAM roles exist: {', '.join(found_roles)}",
                    details={
                        "found_roles": found_roles,
                        "role_details": role_details
                    }
                )
                
            except NoCredentialsError:
                return self.create_result(
                    status=ValidationStatus.SKIP,
                    message="AWS credentials not configured, skipping IAM roles validation",
                    details={"reason": "no_credentials"}
                )
            
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Failed to validate IAM roles: {error_code}",
                    details={
                        "error": str(e),
                        "error_code": error_code
                    },
                    remediation_steps=[
                        "Check AWS credentials have IAM permissions",
                        "Required permissions: iam:GetRole, iam:ListAttachedRolePolicies",
                        "Verify IAM role names are correct",
                        f"Error: {error_code}"
                    ]
                )
                
        except ImportError:
            return self.create_result(
                status=ValidationStatus.SKIP,
                message="boto3 not installed, skipping IAM roles validation",
                details={"reason": "boto3_not_installed"}
            )
    
    async def validate_api_gateway_configuration(self) -> ValidationResult:
        """Validate API Gateway configuration.
        
        Checks for:
        - REST API and WebSocket API definitions exist
        - APIs are properly configured
        - APIs have correct stages deployed
        
        Returns:
            ValidationResult for API Gateway configuration
        """
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            
            try:
                # Create API Gateway clients
                if self.aws_region:
                    apigw_client = boto3.client('apigateway', region_name=self.aws_region)
                    apigwv2_client = boto3.client('apigatewayv2', region_name=self.aws_region)
                else:
                    apigw_client = boto3.client('apigateway')
                    apigwv2_client = boto3.client('apigatewayv2')
                
                # Check for REST APIs
                rest_apis_response = apigw_client.get_rest_apis()
                rest_apis = rest_apis_response.get('items', [])
                
                # Look for Agent Scrivener APIs
                agent_scrivener_rest_apis = [
                    api for api in rest_apis
                    if 'agent' in api.get('name', '').lower() or 'scrivener' in api.get('name', '').lower()
                ]
                
                # Check for WebSocket APIs
                websocket_apis_response = apigwv2_client.get_apis()
                websocket_apis = [
                    api for api in websocket_apis_response.get('Items', [])
                    if api.get('ProtocolType') == 'WEBSOCKET'
                ]
                
                agent_scrivener_websocket_apis = [
                    api for api in websocket_apis
                    if 'agent' in api.get('Name', '').lower() or 'scrivener' in api.get('Name', '').lower()
                ]
                
                if not agent_scrivener_rest_apis and not agent_scrivener_websocket_apis:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message="No API Gateway APIs found for Agent Scrivener",
                        details={
                            "total_rest_apis": len(rest_apis),
                            "total_websocket_apis": len(websocket_apis),
                            "region": self.aws_region or "default"
                        },
                        remediation_steps=[
                            "Create API Gateway REST API for Agent Scrivener",
                            "Create API Gateway WebSocket API for real-time updates",
                            "Use AWS CDK/CloudFormation to define API Gateway resources",
                            "Configure API endpoints, methods, and integrations",
                            "Deploy APIs to a stage (e.g., 'prod')"
                        ]
                    )
                
                # Check if APIs have stages deployed
                apis_without_stages = []
                
                for api in agent_scrivener_rest_apis:
                    stages_response = apigw_client.get_stages(restApiId=api['id'])
                    if not stages_response.get('item'):
                        apis_without_stages.append(api['name'])
                
                if apis_without_stages:
                    return self.create_result(
                        status=ValidationStatus.WARNING,
                        message=f"Some APIs have no stages deployed: {', '.join(apis_without_stages)}",
                        details={
                            "rest_apis": [api['name'] for api in agent_scrivener_rest_apis],
                            "websocket_apis": [api['Name'] for api in agent_scrivener_websocket_apis],
                            "apis_without_stages": apis_without_stages,
                            "region": self.aws_region or "default"
                        },
                        remediation_steps=[
                            "Deploy APIs to a stage",
                            "Use AWS CLI: aws apigateway create-deployment --rest-api-id <id> --stage-name prod",
                            "Or configure deployment in AWS CDK/CloudFormation"
                        ]
                    )
                
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message=f"API Gateway configured with {len(agent_scrivener_rest_apis)} REST API(s) and {len(agent_scrivener_websocket_apis)} WebSocket API(s)",
                    details={
                        "rest_apis": [{"name": api['name'], "id": api['id']} for api in agent_scrivener_rest_apis],
                        "websocket_apis": [{"name": api['Name'], "id": api['ApiId']} for api in agent_scrivener_websocket_apis],
                        "region": self.aws_region or "default"
                    }
                )
                
            except NoCredentialsError:
                return self.create_result(
                    status=ValidationStatus.SKIP,
                    message="AWS credentials not configured, skipping API Gateway validation",
                    details={"reason": "no_credentials"}
                )
            
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Failed to validate API Gateway configuration: {error_code}",
                    details={
                        "error": str(e),
                        "error_code": error_code
                    },
                    remediation_steps=[
                        "Check AWS credentials have API Gateway permissions",
                        "Required permissions: apigateway:GET",
                        "Verify the AWS region is correct",
                        f"Error: {error_code}"
                    ]
                )
                
        except ImportError:
            return self.create_result(
                status=ValidationStatus.SKIP,
                message="boto3 not installed, skipping API Gateway validation",
                details={"reason": "boto3_not_installed"}
            )
    
    async def validate_bedrock_access(self) -> ValidationResult:
        """Validate Bedrock model access.
        
        Checks for:
        - Bedrock service is accessible
        - Specified model ID is available
        - Credentials have permission to invoke model
        
        Returns:
            ValidationResult for Bedrock access
        """
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            
            try:
                # Create Bedrock client
                bedrock_region = self.aws_region or 'us-east-1'
                bedrock_client = boto3.client('bedrock', region_name=bedrock_region)
                
                # List available foundation models
                models_response = bedrock_client.list_foundation_models()
                available_models = models_response.get('modelSummaries', [])
                
                if not available_models:
                    return self.create_result(
                        status=ValidationStatus.WARNING,
                        message="No Bedrock foundation models available in this region",
                        details={
                            "region": bedrock_region,
                            "model_count": 0
                        },
                        remediation_steps=[
                            f"Bedrock may not be available in region {bedrock_region}",
                            "Try a different region (e.g., us-east-1, us-west-2)",
                            "Check AWS Bedrock service availability",
                            "Verify your AWS account has access to Bedrock"
                        ]
                    )
                
                # If a specific model ID is specified, check if it's available
                if self.bedrock_model_id:
                    model_ids = [model.get('modelId') for model in available_models]
                    
                    if self.bedrock_model_id not in model_ids:
                        return self.create_result(
                            status=ValidationStatus.FAIL,
                            message=f"Bedrock model '{self.bedrock_model_id}' not found in region {bedrock_region}",
                            details={
                                "requested_model_id": self.bedrock_model_id,
                                "available_models": model_ids[:10],  # Show first 10
                                "total_available": len(model_ids),
                                "region": bedrock_region
                            },
                            remediation_steps=[
                                f"Model '{self.bedrock_model_id}' is not available",
                                "Check the model ID is correct",
                                "Verify the model is available in your region",
                                "Request access to the model if needed (some models require approval)",
                                f"Available models in {bedrock_region}: {', '.join(model_ids[:5])}..."
                            ]
                        )
                    
                    return self.create_result(
                        status=ValidationStatus.PASS,
                        message=f"Bedrock model '{self.bedrock_model_id}' is accessible in {bedrock_region}",
                        details={
                            "model_id": self.bedrock_model_id,
                            "region": bedrock_region,
                            "total_available_models": len(available_models)
                        }
                    )
                
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message=f"Bedrock is accessible with {len(available_models)} foundation model(s) in {bedrock_region}",
                    details={
                        "region": bedrock_region,
                        "model_count": len(available_models),
                        "sample_models": [model.get('modelId') for model in available_models[:5]]
                    }
                )
                
            except NoCredentialsError:
                return self.create_result(
                    status=ValidationStatus.SKIP,
                    message="AWS credentials not configured, skipping Bedrock validation",
                    details={"reason": "no_credentials"}
                )
            
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Failed to validate Bedrock access: {error_code}",
                    details={
                        "error": str(e),
                        "error_code": error_code,
                        "region": bedrock_region
                    },
                    remediation_steps=[
                        "Check AWS credentials have Bedrock permissions",
                        "Required permissions: bedrock:ListFoundationModels, bedrock:InvokeModel",
                        "Verify Bedrock is available in your region",
                        "Some regions may not have Bedrock service",
                        f"Error: {error_code}"
                    ]
                )
                
        except ImportError:
            return self.create_result(
                status=ValidationStatus.SKIP,
                message="boto3 not installed, skipping Bedrock validation",
                details={"reason": "boto3_not_installed"}
            )
    
    async def validate_s3_buckets(self) -> ValidationResult:
        """Validate S3 bucket configuration.
        
        Checks for:
        - Required S3 buckets exist
        - Buckets have correct permissions
        - Buckets have lifecycle policies configured
        
        Returns:
            ValidationResult for S3 buckets
        """
        if not self.required_s3_buckets:
            return self.create_result(
                status=ValidationStatus.SKIP,
                message="No S3 buckets specified for validation",
                details={"reason": "no_required_buckets"}
            )
        
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            
            try:
                # Create S3 client
                if self.aws_region:
                    s3_client = boto3.client('s3', region_name=self.aws_region)
                else:
                    s3_client = boto3.client('s3')
                
                missing_buckets = []
                found_buckets = []
                buckets_without_lifecycle = []
                
                for bucket_name in self.required_s3_buckets:
                    try:
                        # Check if bucket exists
                        s3_client.head_bucket(Bucket=bucket_name)
                        found_buckets.append(bucket_name)
                        
                        # Check for lifecycle policy
                        try:
                            s3_client.get_bucket_lifecycle_configuration(Bucket=bucket_name)
                        except ClientError as e:
                            if e.response.get('Error', {}).get('Code') == 'NoSuchLifecycleConfiguration':
                                buckets_without_lifecycle.append(bucket_name)
                            else:
                                raise
                        
                    except ClientError as e:
                        if e.response.get('Error', {}).get('Code') in ['404', 'NoSuchBucket']:
                            missing_buckets.append(bucket_name)
                        else:
                            raise
                
                if missing_buckets:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"Missing required S3 buckets: {', '.join(missing_buckets)}",
                        details={
                            "missing_buckets": missing_buckets,
                            "found_buckets": found_buckets,
                            "region": self.aws_region or "default"
                        },
                        remediation_steps=[
                            f"Create missing S3 buckets: {', '.join(missing_buckets)}",
                            "Use AWS CLI: aws s3 mb s3://<bucket-name>",
                            "Or configure buckets in AWS CDK/CloudFormation",
                            "Configure bucket policies for appropriate access",
                            "Enable versioning and encryption for production buckets"
                        ]
                    )
                
                if buckets_without_lifecycle:
                    return self.create_result(
                        status=ValidationStatus.WARNING,
                        message=f"Some S3 buckets have no lifecycle policies: {', '.join(buckets_without_lifecycle)}",
                        details={
                            "found_buckets": found_buckets,
                            "buckets_without_lifecycle": buckets_without_lifecycle,
                            "region": self.aws_region or "default"
                        },
                        remediation_steps=[
                            "Configure lifecycle policies to manage storage costs",
                            "Example: Transition old objects to Glacier after 90 days",
                            "Example: Delete incomplete multipart uploads after 7 days",
                            "Use AWS CLI: aws s3api put-bucket-lifecycle-configuration --bucket <name> --lifecycle-configuration file://lifecycle.json"
                        ]
                    )
                
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message=f"All required S3 buckets exist: {', '.join(found_buckets)}",
                    details={
                        "found_buckets": found_buckets,
                        "region": self.aws_region or "default"
                    }
                )
                
            except NoCredentialsError:
                return self.create_result(
                    status=ValidationStatus.SKIP,
                    message="AWS credentials not configured, skipping S3 buckets validation",
                    details={"reason": "no_credentials"}
                )
            
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Failed to validate S3 buckets: {error_code}",
                    details={
                        "error": str(e),
                        "error_code": error_code
                    },
                    remediation_steps=[
                        "Check AWS credentials have S3 permissions",
                        "Required permissions: s3:ListBucket, s3:GetBucketLifecycleConfiguration",
                        "Verify bucket names are correct",
                        f"Error: {error_code}"
                    ]
                )
                
        except ImportError:
            return self.create_result(
                status=ValidationStatus.SKIP,
                message="boto3 not installed, skipping S3 buckets validation",
                details={"reason": "boto3_not_installed"}
            )
    
    async def validate_cloudformation_templates(self) -> ValidationResult:
        """Validate CloudFormation template syntax.
        
        Checks for:
        - CloudFormation templates are syntactically valid
        - Templates can be validated by AWS
        
        Returns:
            ValidationResult for CloudFormation templates
        """
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            import os
            import glob
            
            # Look for CloudFormation templates in common locations
            template_patterns = [
                "*.template.json",
                "*.template.yaml",
                "*.template.yml",
                "cloudformation/*.json",
                "cloudformation/*.yaml",
                "cloudformation/*.yml",
                "cdk.out/*.template.json"
            ]
            
            template_files = []
            for pattern in template_patterns:
                template_files.extend(glob.glob(pattern, recursive=True))
            
            if not template_files:
                return self.create_result(
                    status=ValidationStatus.SKIP,
                    message="No CloudFormation templates found",
                    details={
                        "searched_patterns": template_patterns,
                        "reason": "no_templates_found"
                    },
                    remediation_steps=[
                        "CloudFormation templates not found in common locations",
                        "If using AWS CDK, run 'cdk synth' to generate templates",
                        "Templates should be in cdk.out/ directory after synthesis"
                    ]
                )
            
            try:
                # Create CloudFormation client
                if self.aws_region:
                    cfn_client = boto3.client('cloudformation', region_name=self.aws_region)
                else:
                    cfn_client = boto3.client('cloudformation')
                
                valid_templates = []
                invalid_templates = []
                
                for template_file in template_files:
                    try:
                        # Read template content
                        with open(template_file, 'r') as f:
                            template_body = f.read()
                        
                        # Validate template
                        cfn_client.validate_template(TemplateBody=template_body)
                        valid_templates.append(template_file)
                        
                    except ClientError as e:
                        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                        error_message = e.response.get('Error', {}).get('Message', str(e))
                        invalid_templates.append({
                            "file": template_file,
                            "error_code": error_code,
                            "error_message": error_message
                        })
                    except Exception as e:
                        invalid_templates.append({
                            "file": template_file,
                            "error_code": "ReadError",
                            "error_message": str(e)
                        })
                
                if invalid_templates:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"Invalid CloudFormation templates found: {len(invalid_templates)}",
                        details={
                            "valid_templates": valid_templates,
                            "invalid_templates": invalid_templates,
                            "total_templates": len(template_files)
                        },
                        remediation_steps=[
                            "Fix CloudFormation template syntax errors",
                            "Check template structure and resource definitions",
                            "Use AWS CLI to validate: aws cloudformation validate-template --template-body file://<template>",
                            "Review error messages for specific issues",
                            f"Invalid templates: {', '.join([t['file'] for t in invalid_templates])}"
                        ]
                    )
                
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message=f"All {len(valid_templates)} CloudFormation template(s) are valid",
                    details={
                        "valid_templates": valid_templates,
                        "total_templates": len(template_files)
                    }
                )
                
            except NoCredentialsError:
                return self.create_result(
                    status=ValidationStatus.SKIP,
                    message="AWS credentials not configured, skipping CloudFormation validation",
                    details={"reason": "no_credentials"}
                )
            
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Failed to validate CloudFormation templates: {error_code}",
                    details={
                        "error": str(e),
                        "error_code": error_code,
                        "template_files": template_files
                    },
                    remediation_steps=[
                        "Check AWS credentials have CloudFormation permissions",
                        "Required permissions: cloudformation:ValidateTemplate",
                        "Verify templates are readable",
                        f"Error: {error_code}"
                    ]
                )
                
        except ImportError:
            return self.create_result(
                status=ValidationStatus.SKIP,
                message="boto3 not installed, skipping CloudFormation validation",
                details={"reason": "boto3_not_installed"}
            )
