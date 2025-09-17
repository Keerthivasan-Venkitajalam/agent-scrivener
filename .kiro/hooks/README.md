# Agent Scrivener - Kiro Agent Hooks

This directory contains Agent Hooks configurations that automate various development and operational tasks for the Agent Scrivener project.

## Available Hooks

### 1. Auto Test Runner (`test-on-save.yml`)

**Purpose**: Automatically run relevant unit tests when agent code is modified.

**Trigger**: File save events for any Python file in the `agent_scrivener/` directory

**Actions**:
- Runs pytest for the corresponding test file
- Triggers agent execution to review test results and suggest improvements

**Benefits**:
- Immediate feedback on code changes
- Catches regressions early in development
- Reduces manual testing overhead

### 2. Documentation Synchronizer (`doc-sync.yml`)

**Purpose**: Keep documentation in sync with specification changes.

**Trigger**: File save events for spec files (requirements.md, design.md, tasks.md)

**Actions**:
- Automatically updates API documentation to reflect spec changes
- Ensures all new endpoints and models are documented

**Benefits**:
- Maintains documentation consistency
- Reduces documentation debt
- Ensures specs and docs stay aligned

### 3. Performance Benchmark (`performance-check.yml`)

**Purpose**: Run comprehensive performance tests and generate reports.

**Trigger**: Manual button click in Kiro interface

**Actions**:
- Executes performance test suite with report generation
- Analyzes results and suggests optimizations

**Benefits**:
- On-demand performance monitoring
- Identifies bottlenecks before they become problems
- Tracks performance trends over time

### 4. Deployment Validator (`deployment-validation.yml`)

**Purpose**: Validate deployment configurations and run health checks.

**Trigger**: File save events for deployment configuration files

**Actions**:
- Validates deployment configuration syntax and completeness
- Reviews validation results and suggests fixes

**Benefits**:
- Prevents deployment failures due to configuration errors
- Ensures deployment best practices are followed
- Catches issues before they reach production

## Hook Configuration Format

Each hook is defined using YAML format with the following structure:

```yaml
name: "Hook Name"
description: "Description of what the hook does"
trigger:
  type: "file_save" | "manual"
  pattern: "glob pattern for files" # for file_save triggers
  button_text: "Button Text" # for manual triggers
actions:
  - type: "shell_command" | "agent_execution"
    command: "shell command to run" # for shell_command type
    prompt: "prompt for agent" # for agent_execution type
```

## Trigger Types

### File Save Triggers

Monitor file system events and trigger when matching files are saved.

**Configuration**:
```yaml
trigger:
  type: "file_save"
  pattern: "path/pattern/**/*.ext"
```

**Supported Patterns**:
- `**/*.py` - All Python files recursively
- `agent_scrivener/**/*.py` - Python files in agent_scrivener directory
- `.kiro/specs/**/*.md` - Markdown files in specs directory
- `**/deployment/**/*.yml` - YAML files in any deployment directory

### Manual Triggers

Create buttons in the Kiro interface for on-demand execution.

**Configuration**:
```yaml
trigger:
  type: "manual"
  button_text: "Run Performance Tests"
```

## Action Types

### Shell Commands

Execute shell commands directly.

**Configuration**:
```yaml
- type: "shell_command"
  command: "python -m pytest tests/unit/test_${filename}.py -v"
```

**Variable Substitution**:
- `${filename}` - Name of the file that triggered the hook (without extension)
- `${filepath}` - Full path of the triggering file
- `${directory}` - Directory containing the triggering file

### Agent Execution

Trigger Kiro agent execution with a specific prompt.

**Configuration**:
```yaml
- type: "agent_execution"
  prompt: "Review test results and suggest improvements if any tests failed"
```

## Best Practices

### Hook Design

1. **Single Responsibility**: Each hook should have one clear purpose
2. **Fast Execution**: Avoid long-running operations in file save hooks
3. **Error Handling**: Ensure hooks fail gracefully and provide useful feedback
4. **Idempotent Actions**: Hooks should be safe to run multiple times

### File Patterns

1. **Specific Patterns**: Use specific patterns to avoid unnecessary triggers
2. **Exclude Patterns**: Consider excluding temporary files and build artifacts
3. **Test Patterns**: Test patterns thoroughly to ensure they match intended files

### Action Sequencing

1. **Logical Order**: Arrange actions in logical execution order
2. **Dependencies**: Ensure earlier actions complete before dependent actions run
3. **Failure Handling**: Consider what happens if an action fails

### Performance Considerations

1. **Debouncing**: File save hooks are automatically debounced to prevent rapid-fire execution
2. **Concurrent Execution**: Multiple hooks can run concurrently
3. **Resource Usage**: Monitor resource usage of hook actions

## Testing Hooks

### Manual Testing

1. **File Save Hooks**: Save a file matching the pattern and verify the hook executes
2. **Manual Hooks**: Click the button in Kiro interface and verify execution
3. **Action Results**: Check that all actions complete successfully

### Debugging

1. **Hook Logs**: Check Kiro logs for hook execution details
2. **Action Output**: Review output from shell commands and agent executions
3. **Error Messages**: Look for error messages in hook execution logs

### Validation

1. **YAML Syntax**: Ensure YAML files are valid
2. **Pattern Testing**: Test file patterns with sample files
3. **Command Testing**: Test shell commands independently

## Integration with Agent Scrivener

### Development Workflow

The hooks integrate seamlessly with the Agent Scrivener development workflow:

1. **Code Changes**: Auto test runner provides immediate feedback
2. **Spec Updates**: Documentation synchronizer keeps docs current
3. **Performance Monitoring**: Performance benchmark tracks system health
4. **Deployment Safety**: Deployment validator prevents configuration errors

### Continuous Integration

Hooks complement CI/CD pipelines by providing:

1. **Local Validation**: Catch issues before committing code
2. **Documentation Consistency**: Ensure docs are updated with code changes
3. **Performance Awareness**: Monitor performance impact of changes
4. **Configuration Validation**: Verify deployment configs locally

## Customization

### Adding New Hooks

1. Create a new YAML file in `.kiro/hooks/`
2. Define appropriate triggers and actions
3. Test the hook thoroughly
4. Document the hook purpose and usage

### Modifying Existing Hooks

1. Edit the YAML configuration file
2. Test changes with sample triggers
3. Update documentation if needed
4. Consider backward compatibility

### Hook Templates

Use these templates for common hook patterns:

**Test Runner Template**:
```yaml
name: "Test Runner for [Component]"
description: "Run tests when [component] code changes"
trigger:
  type: "file_save"
  pattern: "[component]/**/*.py"
actions:
  - type: "shell_command"
    command: "python -m pytest tests/[component]/ -v"
```

**Documentation Sync Template**:
```yaml
name: "Sync [Document Type]"
description: "Update [document] when [source] changes"
trigger:
  type: "file_save"
  pattern: "[source_pattern]"
actions:
  - type: "agent_execution"
    prompt: "Update [document] to reflect changes in [source]"
```

## Troubleshooting

### Common Issues

1. **Hook Not Triggering**: Check file pattern and trigger configuration
2. **Action Failures**: Verify shell commands and agent prompts
3. **Performance Issues**: Review hook frequency and action complexity
4. **Permission Errors**: Ensure proper file and command permissions

### Support

For issues with Agent Hooks:

1. Check Kiro documentation for hook configuration
2. Review hook logs for error details
3. Test hooks manually to isolate issues
4. Consult the Kiro community for advanced use cases