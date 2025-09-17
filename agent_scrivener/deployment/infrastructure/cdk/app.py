#!/usr/bin/env python3
"""AWS CDK application for Agent Scrivener infrastructure."""

import os
from aws_cdk import App, Environment
from stacks.main_stack import AgentScrivenerMainStack
from stacks.monitoring_stack import AgentScrivenerMonitoringStack
from stacks.networking_stack import AgentScrivenerNetworkingStack


def main():
    """Main CDK application entry point."""
    app = App()
    
    # Get environment configuration
    environment = os.getenv("ENVIRONMENT", "development")
    aws_account = os.getenv("CDK_DEFAULT_ACCOUNT")
    aws_region = os.getenv("CDK_DEFAULT_REGION", "us-east-1")
    
    # CDK environment
    env = Environment(account=aws_account, region=aws_region)
    
    # Stack configuration
    config = {
        "environment": environment,
        "project_name": "agent-scrivener",
        "alert_email": os.getenv("ALERT_EMAIL", "admin@example.com"),
        "agentcore_endpoint": os.getenv("AGENTCORE_ENDPOINT", "https://your-agentcore-endpoint.com"),
        "domain_name": os.getenv("DOMAIN_NAME", ""),
        "certificate_arn": os.getenv("CERTIFICATE_ARN", "")
    }
    
    # Create stacks
    if environment == "production":
        # Networking stack for production
        networking_stack = AgentScrivenerNetworkingStack(
            app,
            f"AgentScrivenerNetworking-{environment}",
            config=config,
            env=env
        )
        
        # Main stack with networking dependencies
        main_stack = AgentScrivenerMainStack(
            app,
            f"AgentScrivenerMain-{environment}",
            config=config,
            networking_stack=networking_stack,
            env=env
        )
        
        # Monitoring stack
        monitoring_stack = AgentScrivenerMonitoringStack(
            app,
            f"AgentScrivenerMonitoring-{environment}",
            config=config,
            main_stack=main_stack,
            env=env
        )
    else:
        # Simplified stack for development/staging
        main_stack = AgentScrivenerMainStack(
            app,
            f"AgentScrivenerMain-{environment}",
            config=config,
            env=env
        )
        
        monitoring_stack = AgentScrivenerMonitoringStack(
            app,
            f"AgentScrivenerMonitoring-{environment}",
            config=config,
            main_stack=main_stack,
            env=env
        )
    
    app.synth()


if __name__ == "__main__":
    main()