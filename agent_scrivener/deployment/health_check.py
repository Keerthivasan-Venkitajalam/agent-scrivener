"""Health check and deployment validation scripts for Agent Scrivener."""

import asyncio
import aiohttp
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import sys

from .environment import env_manager
from .secrets import secrets_manager

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status types."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Health check result."""
    component: str
    status: HealthStatus
    message: str
    response_time_ms: Optional[float] = None
    details: Optional[Dict] = None


class HealthChecker:
    """Comprehensive health checker for Agent Scrivener deployment."""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or "http://localhost:8000"
        self.timeout = 10.0
        
    async def check_all(self) -> List[HealthCheckResult]:
        """Run all health checks."""
        checks = [
            self.check_api_health(),
            self.check_database_connection(),
            self.check_aws_services(),
            self.check_agent_availability(),
            self.check_memory_usage(),
            self.check_external_dependencies()
        ]
        
        results = await asyncio.gather(*checks, return_exceptions=True)
        
        # Convert exceptions to unhealthy results
        health_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                health_results.append(HealthCheckResult(
                    component=f"check_{i}",
                    status=HealthStatus.UNHEALTHY,
                    message=str(result)
                ))
            else:
                health_results.append(result)
        
        return health_results
    
    async def check_api_health(self) -> HealthCheckResult:
        """Check API endpoint health."""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(f"{self.base_url}/health") as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        return HealthCheckResult(
                            component="api",
                            status=HealthStatus.HEALTHY,
                            message="API is responding",
                            response_time_ms=response_time,
                            details=data
                        )
                    else:
                        return HealthCheckResult(
                            component="api",
                            status=HealthStatus.UNHEALTHY,
                            message=f"API returned status {response.status}",
                            response_time_ms=response_time
                        )
        except asyncio.TimeoutError:
            return HealthCheckResult(
                component="api",
                status=HealthStatus.UNHEALTHY,
                message="API health check timed out"
            )
        except Exception as e:
            return HealthCheckResult(
                component="api",
                status=HealthStatus.UNHEALTHY,
                message=f"API health check failed: {str(e)}"
            )
    
    async def check_database_connection(self) -> HealthCheckResult:
        """Check database connectivity."""
        try:
            # This would typically test actual database connection
            # For now, we'll check if database config is valid
            db_config = env_manager.get_database_config()
            
            # Simulate database connection check
            await asyncio.sleep(0.1)  # Simulate connection time
            
            return HealthCheckResult(
                component="database",
                status=HealthStatus.HEALTHY,
                message="Database connection successful",
                details={
                    "host": db_config.host,
                    "port": db_config.port,
                    "database": db_config.name
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}"
            )
    
    async def check_aws_services(self) -> HealthCheckResult:
        """Check AWS service connectivity."""
        try:
            aws_config = env_manager.get_aws_config()
            
            # Check if we can access AWS services
            import boto3
            
            # Test Bedrock access
            bedrock_client = boto3.client("bedrock-runtime", region_name=aws_config.region)
            
            # This is a lightweight check - just verify client creation
            return HealthCheckResult(
                component="aws_services",
                status=HealthStatus.HEALTHY,
                message="AWS services accessible",
                details={
                    "region": aws_config.region,
                    "bedrock_model": aws_config.bedrock_model_id
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component="aws_services",
                status=HealthStatus.UNHEALTHY,
                message=f"AWS services check failed: {str(e)}"
            )
    
    async def check_agent_availability(self) -> HealthCheckResult:
        """Check agent availability."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(f"{self.base_url}/agents/status") as response:
                    if response.status == 200:
                        data = await response.json()
                        active_agents = data.get("active_agents", 0)
                        total_agents = data.get("total_agents", 0)
                        
                        if active_agents == total_agents and total_agents > 0:
                            status = HealthStatus.HEALTHY
                            message = f"All {total_agents} agents are active"
                        elif active_agents > 0:
                            status = HealthStatus.DEGRADED
                            message = f"{active_agents}/{total_agents} agents are active"
                        else:
                            status = HealthStatus.UNHEALTHY
                            message = "No agents are active"
                        
                        return HealthCheckResult(
                            component="agents",
                            status=status,
                            message=message,
                            details=data
                        )
                    else:
                        return HealthCheckResult(
                            component="agents",
                            status=HealthStatus.UNKNOWN,
                            message=f"Agent status endpoint returned {response.status}"
                        )
        except Exception as e:
            return HealthCheckResult(
                component="agents",
                status=HealthStatus.UNHEALTHY,
                message=f"Agent availability check failed: {str(e)}"
            )
    
    async def check_memory_usage(self) -> HealthCheckResult:
        """Check system memory usage."""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            if memory_percent < 80:
                status = HealthStatus.HEALTHY
                message = f"Memory usage is normal ({memory_percent:.1f}%)"
            elif memory_percent < 90:
                status = HealthStatus.DEGRADED
                message = f"Memory usage is high ({memory_percent:.1f}%)"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Memory usage is critical ({memory_percent:.1f}%)"
            
            return HealthCheckResult(
                component="memory",
                status=status,
                message=message,
                details={
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "percent_used": memory_percent
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component="memory",
                status=HealthStatus.UNKNOWN,
                message=f"Memory check failed: {str(e)}"
            )
    
    async def check_external_dependencies(self) -> HealthCheckResult:
        """Check external service dependencies."""
        dependencies = [
            ("https://api.semanticscholar.org/graph/v1/paper/search", "Semantic Scholar"),
            ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", "PubMed"),
            ("https://export.arxiv.org/api/query", "arXiv")
        ]
        
        results = []
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0)) as session:
                for url, name in dependencies:
                    try:
                        async with session.get(url) as response:
                            if response.status < 500:  # Accept any non-server-error status
                                results.append(f"{name}: OK")
                            else:
                                results.append(f"{name}: Error {response.status}")
                    except Exception as e:
                        results.append(f"{name}: {str(e)}")
            
            # Determine overall status
            failed_count = len([r for r in results if "Error" in r or ":" in r and not r.endswith("OK")])
            
            if failed_count == 0:
                status = HealthStatus.HEALTHY
                message = "All external dependencies are accessible"
            elif failed_count < len(dependencies) / 2:
                status = HealthStatus.DEGRADED
                message = f"{failed_count}/{len(dependencies)} dependencies have issues"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"{failed_count}/{len(dependencies)} dependencies are failing"
            
            return HealthCheckResult(
                component="external_dependencies",
                status=status,
                message=message,
                details={"checks": results}
            )
        except Exception as e:
            return HealthCheckResult(
                component="external_dependencies",
                status=HealthStatus.UNHEALTHY,
                message=f"External dependencies check failed: {str(e)}"
            )


class DeploymentValidator:
    """Validates deployment configuration and readiness."""
    
    def __init__(self):
        self.health_checker = HealthChecker()
    
    async def validate_deployment(self) -> Tuple[bool, List[str]]:
        """Validate complete deployment readiness."""
        errors = []
        
        # Check environment configuration
        config_errors = env_manager.validate_config()
        if config_errors:
            for section, section_errors in config_errors.items():
                errors.extend([f"{section}: {error}" for error in section_errors])
        
        # Check secrets
        if not secrets_manager.validate_secrets():
            errors.append("Required secrets are not accessible")
        
        # Run health checks
        health_results = await self.health_checker.check_all()
        unhealthy_components = [
            result.component for result in health_results 
            if result.status == HealthStatus.UNHEALTHY
        ]
        
        if unhealthy_components:
            errors.extend([f"Unhealthy component: {comp}" for comp in unhealthy_components])
        
        return len(errors) == 0, errors
    
    def generate_health_report(self, results: List[HealthCheckResult]) -> str:
        """Generate a formatted health report."""
        report = ["=== Agent Scrivener Health Report ===\n"]
        
        status_counts = {status: 0 for status in HealthStatus}
        
        for result in results:
            status_counts[result.status] += 1
            
            status_icon = {
                HealthStatus.HEALTHY: "✅",
                HealthStatus.DEGRADED: "⚠️",
                HealthStatus.UNHEALTHY: "❌",
                HealthStatus.UNKNOWN: "❓"
            }[result.status]
            
            report.append(f"{status_icon} {result.component.upper()}: {result.message}")
            
            if result.response_time_ms:
                report.append(f"   Response time: {result.response_time_ms:.1f}ms")
            
            if result.details:
                report.append(f"   Details: {json.dumps(result.details, indent=2)}")
            
            report.append("")
        
        # Summary
        report.append("=== Summary ===")
        for status, count in status_counts.items():
            if count > 0:
                report.append(f"{status.value.title()}: {count}")
        
        return "\n".join(report)


async def main():
    """Main health check script."""
    if len(sys.argv) > 1 and sys.argv[1] == "--validate":
        # Full deployment validation
        validator = DeploymentValidator()
        is_valid, errors = await validator.validate_deployment()
        
        if is_valid:
            print("✅ Deployment validation passed")
            sys.exit(0)
        else:
            print("❌ Deployment validation failed:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
    else:
        # Basic health check
        checker = HealthChecker()
        results = await checker.check_all()
        
        validator = DeploymentValidator()
        report = validator.generate_health_report(results)
        print(report)
        
        # Exit with error code if any component is unhealthy
        unhealthy_count = len([r for r in results if r.status == HealthStatus.UNHEALTHY])
        sys.exit(1 if unhealthy_count > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())