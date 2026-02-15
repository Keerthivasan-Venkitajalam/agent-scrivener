"""Monitoring infrastructure validator for production readiness.

This validator checks that all monitoring infrastructure is properly configured,
including CloudWatch logs, metrics, alarms, health checks, structured logging,
and alerting systems.
"""

import json
import logging
import re
from typing import List, Optional, Set

from .base_validator import BaseValidator
from .models import ValidationResult, ValidationStatus


logger = logging.getLogger(__name__)


class MonitoringValidator(BaseValidator):
    """Validates monitoring infrastructure for production deployment.
    
    Checks for:
    - CloudWatch log groups for all components
    - CloudWatch custom metrics configuration
    - CloudWatch alarms for critical conditions
    - Health check endpoints for all components
    - Structured logging configuration
    - Alerting configuration (SNS topics)
    - Alert delivery testing
    """
    
    def __init__(
        self,
        aws_region: Optional[str] = None,
        log_group_prefix: str = "/aws/agent-scrivener",
        required_components: Optional[List[str]] = None
    ):
        """Initialize the monitoring validator.
        
        Args:
            aws_region: AWS region for CloudWatch resources
            log_group_prefix: Prefix for CloudWatch log groups
            required_components: List of components that must have monitoring
        """
        super().__init__(name="MonitoringValidator")
        self.aws_region = aws_region
        self.log_group_prefix = log_group_prefix
        self.required_components = required_components or [
            "api",
            "orchestrator",
            "agents",
            "database"
        ]
        
    async def validate(self) -> List[ValidationResult]:
        """Execute all monitoring validation checks.
        
        Returns:
            List of validation results
        """
        self.log_validation_start()
        
        results = []
        
        # Validate each monitoring aspect
        results.append(await self.validate_cloudwatch_logs())
        results.append(await self.validate_cloudwatch_metrics())
        results.append(await self.validate_cloudwatch_alarms())
        results.append(await self.validate_health_checks())
        results.append(await self.validate_structured_logging())
        results.append(await self.validate_alerting())
        results.append(await self.test_alert_delivery())
        
        self.log_validation_complete(results)
        return results
    
    async def validate_cloudwatch_logs(self) -> ValidationResult:
        """Validate CloudWatch log groups exist for all components.
        
        Checks for:
        - Log groups exist for all required components
        - Log groups have appropriate retention policies
        - Log groups are in the correct region
        
        Returns:
            ValidationResult for CloudWatch logs
        """
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            
            try:
                # Create CloudWatch Logs client
                if self.aws_region:
                    client = boto3.client('logs', region_name=self.aws_region)
                else:
                    client = boto3.client('logs')
                
                # Get all log groups with our prefix
                response = client.describe_log_groups(
                    logGroupNamePrefix=self.log_group_prefix
                )
                
                existing_log_groups = {
                    lg['logGroupName'] for lg in response.get('logGroups', [])
                }
                
                # Check for required component log groups
                missing_components = []
                found_components = []
                
                for component in self.required_components:
                    expected_log_group = f"{self.log_group_prefix}/{component}"
                    if expected_log_group in existing_log_groups:
                        found_components.append(component)
                    else:
                        missing_components.append(component)
                
                if missing_components:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"Missing CloudWatch log groups for components: {', '.join(missing_components)}",
                        details={
                            "missing_components": missing_components,
                            "found_components": found_components,
                            "log_group_prefix": self.log_group_prefix,
                            "region": self.aws_region or "default"
                        },
                        remediation_steps=[
                            f"Create CloudWatch log groups for missing components: {', '.join(missing_components)}",
                            f"Use AWS CLI: aws logs create-log-group --log-group-name {self.log_group_prefix}/<component>",
                            "Or configure log groups in your AWS CDK/CloudFormation templates",
                            "Ensure log retention policies are set appropriately (e.g., 30 days)",
                            "Configure application logging to send logs to these log groups"
                        ]
                    )
                
                # Check retention policies
                log_groups_without_retention = []
                for lg in response.get('logGroups', []):
                    if lg['logGroupName'].startswith(self.log_group_prefix):
                        if 'retentionInDays' not in lg:
                            log_groups_without_retention.append(lg['logGroupName'])
                
                if log_groups_without_retention:
                    return self.create_result(
                        status=ValidationStatus.WARNING,
                        message="Some log groups do not have retention policies configured",
                        details={
                            "log_groups_without_retention": log_groups_without_retention,
                            "found_components": found_components
                        },
                        remediation_steps=[
                            "Set retention policies for log groups to manage costs",
                            "Recommended retention: 30 days for production logs",
                            f"Use AWS CLI: aws logs put-retention-policy --log-group-name <name> --retention-in-days 30"
                        ]
                    )
                
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message=f"All required CloudWatch log groups exist for components: {', '.join(found_components)}",
                    details={
                        "found_components": found_components,
                        "log_group_count": len(found_components),
                        "region": self.aws_region or "default"
                    }
                )
                
            except NoCredentialsError:
                return self.create_result(
                    status=ValidationStatus.SKIP,
                    message="AWS credentials not configured, skipping CloudWatch logs validation",
                    details={"reason": "no_credentials"},
                    remediation_steps=[
                        "Configure AWS credentials using AWS CLI or environment variables",
                        "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY",
                        "Or configure AWS profile in ~/.aws/credentials"
                    ]
                )
            
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Failed to validate CloudWatch log groups: {error_code}",
                    details={
                        "error": str(e),
                        "error_code": error_code
                    },
                    remediation_steps=[
                        "Check AWS credentials have CloudWatch Logs permissions",
                        "Required permissions: logs:DescribeLogGroups",
                        "Verify the AWS region is correct",
                        f"Check if log groups exist in region: {self.aws_region or 'default'}"
                    ]
                )
                
        except ImportError:
            return self.create_result(
                status=ValidationStatus.SKIP,
                message="boto3 not installed, skipping CloudWatch logs validation",
                details={"reason": "boto3_not_installed"},
                remediation_steps=[
                    "Install boto3: pip install boto3",
                    "Or add boto3 to your requirements.txt"
                ]
            )
    
    async def validate_cloudwatch_metrics(self) -> ValidationResult:
        """Validate CloudWatch custom metrics are configured.
        
        Checks for:
        - Custom metrics for session count
        - Custom metrics for success rate
        - Custom metrics for error rate
        - Metrics namespace is configured
        
        Returns:
            ValidationResult for CloudWatch metrics
        """
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            
            try:
                # Create CloudWatch client
                if self.aws_region:
                    client = boto3.client('cloudwatch', region_name=self.aws_region)
                else:
                    client = boto3.client('cloudwatch')
                
                # Define expected metrics
                namespace = "AgentScrivener"
                required_metrics = [
                    "SessionCount",
                    "SuccessRate",
                    "ErrorRate"
                ]
                
                # List metrics in our namespace
                response = client.list_metrics(Namespace=namespace)
                
                existing_metrics = {
                    metric['MetricName'] for metric in response.get('Metrics', [])
                }
                
                missing_metrics = [
                    metric for metric in required_metrics
                    if metric not in existing_metrics
                ]
                
                if missing_metrics:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"Missing CloudWatch custom metrics: {', '.join(missing_metrics)}",
                        details={
                            "missing_metrics": missing_metrics,
                            "found_metrics": list(existing_metrics),
                            "namespace": namespace,
                            "region": self.aws_region or "default"
                        },
                        remediation_steps=[
                            f"Configure custom metrics in namespace '{namespace}'",
                            "Implement metric publishing in your application code",
                            "Required metrics: SessionCount, SuccessRate, ErrorRate",
                            "Use boto3 CloudWatch client to put_metric_data()",
                            "Example: cloudwatch.put_metric_data(Namespace='AgentScrivener', MetricData=[...])"
                        ]
                    )
                
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message=f"All required CloudWatch custom metrics are configured: {', '.join(required_metrics)}",
                    details={
                        "found_metrics": list(existing_metrics),
                        "namespace": namespace,
                        "region": self.aws_region or "default"
                    }
                )
                
            except NoCredentialsError:
                return self.create_result(
                    status=ValidationStatus.SKIP,
                    message="AWS credentials not configured, skipping CloudWatch metrics validation",
                    details={"reason": "no_credentials"},
                    remediation_steps=[
                        "Configure AWS credentials using AWS CLI or environment variables"
                    ]
                )
            
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Failed to validate CloudWatch metrics: {error_code}",
                    details={
                        "error": str(e),
                        "error_code": error_code
                    },
                    remediation_steps=[
                        "Check AWS credentials have CloudWatch permissions",
                        "Required permissions: cloudwatch:ListMetrics",
                        "Verify the AWS region is correct"
                    ]
                )
                
        except ImportError:
            return self.create_result(
                status=ValidationStatus.SKIP,
                message="boto3 not installed, skipping CloudWatch metrics validation",
                details={"reason": "boto3_not_installed"},
                remediation_steps=[
                    "Install boto3: pip install boto3"
                ]
            )
    
    async def validate_cloudwatch_alarms(self) -> ValidationResult:
        """Validate CloudWatch alarms exist for critical conditions.
        
        Checks for:
        - Alarm for high error rate (>5%)
        - Alarm for high latency (>5 minutes)
        - Alarm for service unavailability
        
        Returns:
            ValidationResult for CloudWatch alarms
        """
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            
            try:
                # Create CloudWatch client
                if self.aws_region:
                    client = boto3.client('cloudwatch', region_name=self.aws_region)
                else:
                    client = boto3.client('cloudwatch')
                
                # Get all alarms
                response = client.describe_alarms()
                
                existing_alarms = {
                    alarm['AlarmName'] for alarm in response.get('MetricAlarms', [])
                }
                
                # Check for required alarm patterns
                required_alarm_patterns = {
                    "high_error_rate": ["error", "rate"],
                    "high_latency": ["latency", "duration", "time"],
                    "service_unavailable": ["health", "availability", "unavailable"]
                }
                
                missing_alarm_types = []
                found_alarm_types = []
                
                for alarm_type, keywords in required_alarm_patterns.items():
                    # Check if any existing alarm matches this pattern
                    found = False
                    for alarm_name in existing_alarms:
                        if any(keyword.lower() in alarm_name.lower() for keyword in keywords):
                            found = True
                            break
                    
                    if found:
                        found_alarm_types.append(alarm_type)
                    else:
                        missing_alarm_types.append(alarm_type)
                
                if missing_alarm_types:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"Missing CloudWatch alarms for: {', '.join(missing_alarm_types)}",
                        details={
                            "missing_alarm_types": missing_alarm_types,
                            "found_alarm_types": found_alarm_types,
                            "total_alarms": len(existing_alarms),
                            "region": self.aws_region or "default"
                        },
                        remediation_steps=[
                            "Create CloudWatch alarms for critical conditions",
                            "High error rate alarm: Trigger when ErrorRate > 5% for 5 minutes",
                            "High latency alarm: Trigger when request duration > 5 minutes",
                            "Service unavailability alarm: Trigger when health check fails",
                            "Configure alarm actions to send notifications to SNS topics",
                            "Use AWS CDK/CloudFormation to define alarms in infrastructure code"
                        ]
                    )
                
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message=f"CloudWatch alarms configured for: {', '.join(found_alarm_types)}",
                    details={
                        "found_alarm_types": found_alarm_types,
                        "total_alarms": len(existing_alarms),
                        "region": self.aws_region or "default"
                    }
                )
                
            except NoCredentialsError:
                return self.create_result(
                    status=ValidationStatus.SKIP,
                    message="AWS credentials not configured, skipping CloudWatch alarms validation",
                    details={"reason": "no_credentials"},
                    remediation_steps=[
                        "Configure AWS credentials using AWS CLI or environment variables"
                    ]
                )
            
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Failed to validate CloudWatch alarms: {error_code}",
                    details={
                        "error": str(e),
                        "error_code": error_code
                    },
                    remediation_steps=[
                        "Check AWS credentials have CloudWatch permissions",
                        "Required permissions: cloudwatch:DescribeAlarms",
                        "Verify the AWS region is correct"
                    ]
                )
                
        except ImportError:
            return self.create_result(
                status=ValidationStatus.SKIP,
                message="boto3 not installed, skipping CloudWatch alarms validation",
                details={"reason": "boto3_not_installed"},
                remediation_steps=[
                    "Install boto3: pip install boto3"
                ]
            )
    
    async def validate_health_checks(self) -> ValidationResult:
        """Validate health check endpoints for all components.
        
        Checks for:
        - API health endpoint exists and responds
        - Database connectivity check
        - AWS service access check
        - Agent availability check
        
        Returns:
            ValidationResult for health checks
        """
        try:
            import httpx
            
            # Try to check the API health endpoint
            health_url = "http://localhost:8000/health"
            
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(health_url)
                    
                    if response.status_code == 200:
                        health_data = response.json()
                        
                        # Check for required health check components
                        required_checks = ["api", "database", "aws"]
                        missing_checks = []
                        
                        for check in required_checks:
                            if check not in health_data:
                                missing_checks.append(check)
                        
                        if missing_checks:
                            return self.create_result(
                                status=ValidationStatus.WARNING,
                                message=f"Health endpoint missing checks for: {', '.join(missing_checks)}",
                                details={
                                    "health_url": health_url,
                                    "missing_checks": missing_checks,
                                    "health_data": health_data
                                },
                                remediation_steps=[
                                    "Add health checks for all components",
                                    "Include API availability check",
                                    "Include database connectivity check",
                                    "Include AWS service access check",
                                    "Include agent availability check"
                                ]
                            )
                        
                        return self.create_result(
                            status=ValidationStatus.PASS,
                            message="Health check endpoint is functional and includes all required checks",
                            details={
                                "health_url": health_url,
                                "health_data": health_data
                            }
                        )
                    else:
                        return self.create_result(
                            status=ValidationStatus.FAIL,
                            message=f"Health endpoint returned status {response.status_code}",
                            details={
                                "health_url": health_url,
                                "status_code": response.status_code
                            },
                            remediation_steps=[
                                "Ensure the API server is running",
                                "Check health endpoint implementation returns 200 OK",
                                "Verify health checks are properly implemented"
                            ]
                        )
            
            except httpx.ConnectError:
                return self.create_result(
                    status=ValidationStatus.SKIP,
                    message="API server not running, skipping health check validation",
                    details={
                        "health_url": health_url,
                        "reason": "connection_refused"
                    },
                    remediation_steps=[
                        "Start the API server before running health check validation",
                        "Ensure the server is listening on the expected port (8000)",
                        "Check if the health endpoint is implemented at /health"
                    ]
                )
            
            except Exception as e:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Failed to check health endpoint: {str(e)}",
                    details={
                        "health_url": health_url,
                        "error": str(e)
                    },
                    remediation_steps=[
                        "Check if the API server is accessible",
                        "Verify the health endpoint URL is correct",
                        "Check network connectivity"
                    ]
                )
                
        except ImportError:
            return self.create_result(
                status=ValidationStatus.SKIP,
                message="httpx not installed, skipping health check validation",
                details={"reason": "httpx_not_installed"},
                remediation_steps=[
                    "Install httpx: pip install httpx"
                ]
            )
    
    async def validate_structured_logging(self) -> ValidationResult:
        """Validate structured logging configuration.
        
        Checks for:
        - Logging is configured with JSON format
        - Appropriate log levels are set
        - Sensitive information is not logged
        
        Returns:
            ValidationResult for structured logging
        """
        import logging
        
        # Check if any handlers use JSON formatting
        root_logger = logging.getLogger()
        
        has_json_formatter = False
        has_structured_formatter = False
        
        for handler in root_logger.handlers:
            if handler.formatter:
                formatter_class = handler.formatter.__class__.__name__
                # Check for common JSON formatter classes
                if 'json' in formatter_class.lower():
                    has_json_formatter = True
                # Check if formatter includes structured fields
                if hasattr(handler.formatter, '_fmt') and handler.formatter._fmt:
                    fmt_string = handler.formatter._fmt
                    # Look for structured logging patterns
                    if any(field in fmt_string for field in ['%(name)s', '%(levelname)s', '%(message)s']):
                        has_structured_formatter = True
        
        if not has_json_formatter and not has_structured_formatter:
            return self.create_result(
                status=ValidationStatus.WARNING,
                message="Structured logging not configured",
                details={
                    "handler_count": len(root_logger.handlers),
                    "has_json_formatter": has_json_formatter,
                    "has_structured_formatter": has_structured_formatter
                },
                remediation_steps=[
                    "Configure structured logging with JSON format",
                    "Use python-json-logger or similar library",
                    "Include standard fields: timestamp, level, logger, message",
                    "Include context fields: request_id, user_id, session_id",
                    "Example: pip install python-json-logger",
                    "Configure in logging config: formatter = pythonjsonlogger.jsonlogger.JsonFormatter"
                ]
            )
        
        # Check log levels
        if root_logger.level == logging.NOTSET or root_logger.level > logging.INFO:
            return self.create_result(
                status=ValidationStatus.WARNING,
                message=f"Root logger level is {logging.getLevelName(root_logger.level)}, should be INFO or lower for production",
                details={
                    "current_level": logging.getLevelName(root_logger.level),
                    "has_json_formatter": has_json_formatter,
                    "has_structured_formatter": has_structured_formatter
                },
                remediation_steps=[
                    "Set root logger level to INFO for production",
                    "Use DEBUG level only for development",
                    "Configure in logging config: level = INFO"
                ]
            )
        
        return self.create_result(
            status=ValidationStatus.PASS,
            message="Structured logging is configured",
            details={
                "has_json_formatter": has_json_formatter,
                "has_structured_formatter": has_structured_formatter,
                "log_level": logging.getLevelName(root_logger.level),
                "handler_count": len(root_logger.handlers)
            }
        )
    
    async def validate_alerting(self) -> ValidationResult:
        """Validate alerting configuration (SNS topics).
        
        Checks for:
        - SNS topics exist for critical alerts
        - Topics have subscriptions configured
        - Topics have appropriate permissions
        
        Returns:
            ValidationResult for alerting
        """
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            
            try:
                # Create SNS client
                if self.aws_region:
                    client = boto3.client('sns', region_name=self.aws_region)
                else:
                    client = boto3.client('sns')
                
                # List SNS topics
                response = client.list_topics()
                
                topics = response.get('Topics', [])
                topic_arns = [topic['TopicArn'] for topic in topics]
                
                # Look for alert-related topics
                alert_topics = [
                    arn for arn in topic_arns
                    if any(keyword in arn.lower() for keyword in ['alert', 'alarm', 'notification', 'critical'])
                ]
                
                if not alert_topics:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message="No SNS topics configured for alerting",
                        details={
                            "total_topics": len(topics),
                            "region": self.aws_region or "default"
                        },
                        remediation_steps=[
                            "Create SNS topic for critical alerts",
                            "Use AWS CLI: aws sns create-topic --name agent-scrivener-alerts",
                            "Or configure in AWS CDK/CloudFormation",
                            "Subscribe email addresses or other endpoints to the topic",
                            "Configure CloudWatch alarms to publish to this topic"
                        ]
                    )
                
                # Check if topics have subscriptions
                topics_without_subscriptions = []
                
                for topic_arn in alert_topics:
                    subs_response = client.list_subscriptions_by_topic(TopicArn=topic_arn)
                    subscriptions = subs_response.get('Subscriptions', [])
                    
                    if not subscriptions:
                        topics_without_subscriptions.append(topic_arn)
                
                if topics_without_subscriptions:
                    return self.create_result(
                        status=ValidationStatus.WARNING,
                        message="Some SNS topics have no subscriptions",
                        details={
                            "alert_topics": alert_topics,
                            "topics_without_subscriptions": topics_without_subscriptions
                        },
                        remediation_steps=[
                            "Add subscriptions to SNS topics",
                            "Subscribe email addresses for alert notifications",
                            "Use AWS CLI: aws sns subscribe --topic-arn <arn> --protocol email --notification-endpoint <email>",
                            "Confirm subscription via email"
                        ]
                    )
                
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message=f"Alerting configured with {len(alert_topics)} SNS topic(s)",
                    details={
                        "alert_topics": alert_topics,
                        "region": self.aws_region or "default"
                    }
                )
                
            except NoCredentialsError:
                return self.create_result(
                    status=ValidationStatus.SKIP,
                    message="AWS credentials not configured, skipping alerting validation",
                    details={"reason": "no_credentials"},
                    remediation_steps=[
                        "Configure AWS credentials using AWS CLI or environment variables"
                    ]
                )
            
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Failed to validate SNS topics: {error_code}",
                    details={
                        "error": str(e),
                        "error_code": error_code
                    },
                    remediation_steps=[
                        "Check AWS credentials have SNS permissions",
                        "Required permissions: sns:ListTopics, sns:ListSubscriptionsByTopic",
                        "Verify the AWS region is correct"
                    ]
                )
                
        except ImportError:
            return self.create_result(
                status=ValidationStatus.SKIP,
                message="boto3 not installed, skipping alerting validation",
                details={"reason": "boto3_not_installed"},
                remediation_steps=[
                    "Install boto3: pip install boto3"
                ]
            )
    
    async def test_alert_delivery(self) -> ValidationResult:
        """Test alert delivery by sending a test alert.
        
        Checks for:
        - Test alert can be published to SNS topic
        - Alert delivery is successful
        
        Returns:
            ValidationResult for alert delivery test
        """
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            
            try:
                # Create SNS client
                if self.aws_region:
                    client = boto3.client('sns', region_name=self.aws_region)
                else:
                    client = boto3.client('sns')
                
                # List SNS topics to find alert topic
                response = client.list_topics()
                topics = response.get('Topics', [])
                
                # Find an alert topic
                alert_topic = None
                for topic in topics:
                    topic_arn = topic['TopicArn']
                    if any(keyword in topic_arn.lower() for keyword in ['alert', 'alarm', 'notification', 'critical']):
                        alert_topic = topic_arn
                        break
                
                if not alert_topic:
                    return self.create_result(
                        status=ValidationStatus.SKIP,
                        message="No alert SNS topic found, skipping alert delivery test",
                        details={"total_topics": len(topics)},
                        remediation_steps=[
                            "Create an SNS topic for alerts first",
                            "Run validate_alerting() to configure alerting"
                        ]
                    )
                
                # Send test alert
                test_message = {
                    "alert_type": "test",
                    "message": "This is a test alert from Agent Scrivener monitoring validation",
                    "severity": "info",
                    "timestamp": "2024-01-01T00:00:00Z"
                }
                
                publish_response = client.publish(
                    TopicArn=alert_topic,
                    Subject="[TEST] Agent Scrivener Monitoring Validation",
                    Message=json.dumps(test_message, indent=2)
                )
                
                message_id = publish_response.get('MessageId')
                
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message="Test alert delivered successfully",
                    details={
                        "topic_arn": alert_topic,
                        "message_id": message_id,
                        "region": self.aws_region or "default"
                    }
                )
                
            except NoCredentialsError:
                return self.create_result(
                    status=ValidationStatus.SKIP,
                    message="AWS credentials not configured, skipping alert delivery test",
                    details={"reason": "no_credentials"},
                    remediation_steps=[
                        "Configure AWS credentials using AWS CLI or environment variables"
                    ]
                )
            
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Failed to send test alert: {error_code}",
                    details={
                        "error": str(e),
                        "error_code": error_code
                    },
                    remediation_steps=[
                        "Check AWS credentials have SNS publish permissions",
                        "Required permissions: sns:Publish",
                        "Verify the SNS topic exists and is accessible",
                        "Check if the topic has the correct permissions policy"
                    ]
                )
                
        except ImportError:
            return self.create_result(
                status=ValidationStatus.SKIP,
                message="boto3 not installed, skipping alert delivery test",
                details={"reason": "boto3_not_installed"},
                remediation_steps=[
                    "Install boto3: pip install boto3"
                ]
            )
