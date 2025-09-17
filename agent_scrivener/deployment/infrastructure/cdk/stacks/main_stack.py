"""Main CDK stack for Agent Scrivener infrastructure."""

from aws_cdk import (
    Stack,
    aws_iam as iam,
    aws_apigateway as apigateway,
    aws_secretsmanager as secretsmanager,
    aws_ssm as ssm,
    aws_s3 as s3,
    aws_dynamodb as dynamodb,
    aws_sns as sns,
    aws_resourcegroups as resourcegroups,
    RemovalPolicy,
    Duration,
    CfnOutput
)
from constructs import Construct
from typing import Dict, Any, Optional


class AgentScrivenerMainStack(Stack):
    """Main infrastructure stack for Agent Scrivener."""
    
    def __init__(
        self, 
        scope: Construct, 
        construct_id: str, 
        config: Dict[str, Any],
        networking_stack: Optional[Stack] = None,
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        self.config = config
        self.networking_stack = networking_stack
        
        # Create core resources
        self._create_iam_roles()
        self._create_storage_resources()
        self._create_secrets()
        self._create_parameters()
        self._create_api_gateway()
        self._create_resource_groups()
        self._create_outputs()
    
    def _create_iam_roles(self):
        """Create IAM roles for AgentCore and individual agents."""
        # AgentCore execution role
        self.agentcore_role = iam.Role(
            self,
            "AgentCoreExecutionRole",
            role_name=f"{self.config['project_name']}-agentcore-execution-{self.config['environment']}",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("bedrock.amazonaws.com"),
                iam.ServicePrincipal("lambda.amazonaws.com"),
                iam.ServicePrincipal("ecs-tasks.amazonaws.com")
            ),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole"),
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonECSTaskExecutionRolePolicy")
            ]
        )
        
        # Add Bedrock permissions
        self.agentcore_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock:ListFoundationModels",
                    "bedrock:GetFoundationModel"
                ],
                resources=[
                    f"arn:aws:bedrock:{self.region}::foundation-model/*",
                    f"arn:aws:bedrock:{self.region}:{self.account}:model/*"
                ]
            )
        )
        
        # Individual agent roles
        self.agent_roles = {}
        agent_types = ["planner", "research", "api", "analysis", "drafting", "citation"]
        
        for agent_type in agent_types:
            role = iam.Role(
                self,
                f"{agent_type.title()}AgentRole",
                role_name=f"{self.config['project_name']}-{agent_type}-agent-{self.config['environment']}",
                assumed_by=iam.ArnPrincipal(self.agentcore_role.role_arn)
            )
            
            # Add basic Bedrock permissions
            role.add_to_policy(
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=["bedrock:InvokeModel"],
                    resources=[f"arn:aws:bedrock:{self.region}::foundation-model/anthropic.claude-*"]
                )
            )
            
            self.agent_roles[agent_type] = role
    
    def _create_storage_resources(self):
        """Create S3 bucket and DynamoDB table for storage."""
        # Temporary data bucket
        self.temp_bucket = s3.Bucket(
            self,
            "TempDataBucket",
            bucket_name=f"{self.config['project_name']}-temp-data-{self.config['environment']}-{self.account}",
            versioned=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="DeleteTempFiles",
                    enabled=True,
                    expiration=Duration.days(7)
                )
            ],
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=RemovalPolicy.DESTROY if self.config['environment'] != 'production' else RemovalPolicy.RETAIN
        )
        
        # Memory store DynamoDB table
        self.memory_table = dynamodb.Table(
            self,
            "MemoryTable",
            table_name=f"{self.config['project_name']}-memory-{self.config['environment']}",
            partition_key=dynamodb.Attribute(name="session_id", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="memory_key", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="ttl",
            point_in_time_recovery=True,
            removal_policy=RemovalPolicy.DESTROY if self.config['environment'] != 'production' else RemovalPolicy.RETAIN
        )
        
        # Add global secondary index
        self.memory_table.add_global_secondary_index(
            index_name="timestamp-index",
            partition_key=dynamodb.Attribute(name="memory_key", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="timestamp", type=dynamodb.AttributeType.NUMBER)
        )
        
        # Grant permissions to AgentCore role
        self.temp_bucket.grant_read_write(self.agentcore_role)
        self.memory_table.grant_read_write_data(self.agentcore_role)
    
    def _create_secrets(self):
        """Create secrets in AWS Secrets Manager."""
        # JWT secret
        self.jwt_secret = secretsmanager.Secret(
            self,
            "JWTSecret",
            secret_name=f"{self.config['project_name']}/jwt-secret",
            description="JWT secret key for Agent Scrivener authentication",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                secret_string_template="{}",
                generate_string_key="secret",
                password_length=64,
                exclude_characters='"@/\\'
            )
        )
        
        # Database credentials
        self.db_secret = secretsmanager.Secret(
            self,
            "DatabaseCredentials",
            secret_name=f"{self.config['project_name']}/database-credentials",
            description="Database credentials for Agent Scrivener",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                secret_string_template='{"username": "agent_user"}',
                generate_string_key="password",
                password_length=32,
                exclude_characters='"@/\\'
            )
        )
        
        # External API keys
        self.api_keys_secret = secretsmanager.Secret(
            self,
            "ExternalAPIKeys",
            secret_name=f"{self.config['project_name']}/external-api-keys",
            description="External API keys for Agent Scrivener",
            secret_string_value=secretsmanager.SecretStringValueBeta1.from_token(
                '{"semantic_scholar": "your-semantic-scholar-api-key", "pubmed": "your-pubmed-api-key", "arxiv": "your-arxiv-api-key"}'
            )
        )
        
        # Grant read access to secrets
        for secret in [self.jwt_secret, self.db_secret, self.api_keys_secret]:
            secret.grant_read(self.agentcore_role)
    
    def _create_parameters(self):
        """Create SSM parameters for configuration."""
        # Bedrock model ID
        ssm.StringParameter(
            self,
            "BedrockModelParameter",
            parameter_name=f"/{self.config['project_name']}/bedrock/model-id",
            string_value="anthropic.claude-3-sonnet-20240229-v1:0",
            description="Bedrock model ID for Agent Scrivener"
        )
        
        # Max concurrency
        max_concurrency = "20" if self.config['environment'] == 'production' else "10"
        ssm.StringParameter(
            self,
            "MaxConcurrencyParameter",
            parameter_name=f"/{self.config['project_name']}/config/max-concurrency",
            string_value=max_concurrency,
            description="Maximum concurrent sessions"
        )
        
        # Session timeout
        ssm.StringParameter(
            self,
            "SessionTimeoutParameter",
            parameter_name=f"/{self.config['project_name']}/config/session-timeout",
            string_value="28800",  # 8 hours
            description="Session timeout in seconds"
        )
    
    def _create_api_gateway(self):
        """Create API Gateway for external access."""
        # REST API
        self.api = apigateway.RestApi(
            self,
            "AgentScrivenerAPI",
            rest_api_name=f"{self.config['project_name']}-api-{self.config['environment']}",
            description="Agent Scrivener Research Platform API",
            endpoint_configuration=apigateway.EndpointConfiguration(
                types=[apigateway.EndpointType.REGIONAL]
            ),
            deploy_options=apigateway.StageOptions(
                stage_name=self.config['environment'],
                logging_level=apigateway.MethodLoggingLevel.INFO,
                data_trace_enabled=True,
                metrics_enabled=True,
                throttling_burst_limit=100,
                throttling_rate_limit=50
            )
        )
        
        # Health check resource
        health_resource = self.api.root.add_resource("health")
        health_resource.add_method(
            "GET",
            apigateway.HttpIntegration(
                f"{self.config['agentcore_endpoint']}/health",
                http_method="GET"
            ),
            authorization_type=apigateway.AuthorizationType.NONE
        )
        
        # Research resource
        research_resource = self.api.root.add_resource("research")
        research_resource.add_method(
            "POST",
            apigateway.HttpIntegration(
                f"{self.config['agentcore_endpoint']}/research",
                http_method="POST"
            ),
            authorization_type=apigateway.AuthorizationType.IAM
        )
        
        # Usage plan and API key
        usage_plan = self.api.add_usage_plan(
            "UsagePlan",
            name=f"{self.config['project_name']}-{self.config['environment']}-plan",
            description=f"Usage plan for Agent Scrivener {self.config['environment']} environment",
            throttle=apigateway.ThrottleSettings(
                burst_limit=100,
                rate_limit=50
            ),
            quota=apigateway.QuotaSettings(
                limit=10000,
                period=apigateway.Period.DAY
            )
        )
        
        self.api_key = self.api.add_api_key(
            "APIKey",
            api_key_name=f"{self.config['project_name']}-{self.config['environment']}-key",
            description=f"API Key for Agent Scrivener {self.config['environment']} environment"
        )
        
        usage_plan.add_api_key(self.api_key)
    
    def _create_resource_groups(self):
        """Create resource groups for cost allocation."""
        self.alert_topic = sns.Topic(
            self,
            "AlertTopic",
            topic_name=f"{self.config['project_name']}-alerts-{self.config['environment']}",
            display_name=f"Agent Scrivener {self.config['environment']} Alerts"
        )
        
        # Resource group for cost tracking
        resourcegroups.CfnGroup(
            self,
            "ResourceGroup",
            name=f"{self.config['project_name']}-{self.config['environment']}-resources",
            description=f"Resource group for {self.config['project_name']} {self.config['environment']} environment",
            resource_query=resourcegroups.CfnGroup.ResourceQueryProperty(
                type="TAG_FILTERS_1_0",
                query=resourcegroups.CfnGroup.QueryProperty(
                    resource_type_filters=["AWS::AllSupported"],
                    tag_filters=[
                        resourcegroups.CfnGroup.TagFilterProperty(
                            key="Project",
                            values=[self.config['project_name']]
                        ),
                        resourcegroups.CfnGroup.TagFilterProperty(
                            key="Environment",
                            values=[self.config['environment']]
                        )
                    ]
                )
            )
        )
    
    def _create_outputs(self):
        """Create CloudFormation outputs."""
        CfnOutput(
            self,
            "APIGatewayURL",
            value=self.api.url,
            description="API Gateway endpoint URL",
            export_name=f"{self.stack_name}-APIGatewayURL"
        )
        
        CfnOutput(
            self,
            "APIKey",
            value=self.api_key.key_id,
            description="API Key for accessing the service",
            export_name=f"{self.stack_name}-APIKey"
        )
        
        CfnOutput(
            self,
            "AgentCoreExecutionRoleArn",
            value=self.agentcore_role.role_arn,
            description="AgentCore execution role ARN",
            export_name=f"{self.stack_name}-AgentCoreExecutionRoleArn"
        )
        
        CfnOutput(
            self,
            "TempDataBucketName",
            value=self.temp_bucket.bucket_name,
            description="Temporary data bucket name",
            export_name=f"{self.stack_name}-TempDataBucketName"
        )
        
        CfnOutput(
            self,
            "MemoryTableName",
            value=self.memory_table.table_name,
            description="Memory store DynamoDB table name",
            export_name=f"{self.stack_name}-MemoryTableName"
        )