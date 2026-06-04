#!/usr/bin/env python3
"""
Task Executor - Script that runs on SLURM nodes
Repeatedly extracts and executes tasks from the Task Database
"""
import os
import sys
import time
import json
import subprocess
import shlex
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Import project modules
from task_db import TaskDB
from config_manager import Config


def setup_logging(log_dir: str, slurm_job_id: str):
    """Setup logging for task executor"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"executor_{slurm_job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def check_completion_file(completion_file: str) -> bool:
    """Check if completion file exists"""
    if not completion_file:
        return False
    return os.path.exists(completion_file)


def execute_task(task: dict, config: Config, logger: logging.Logger, slurm_job_id: str) -> bool:
    """
    Execute a single task
    Returns True if task completed successfully, False otherwise
    """
    task_id = task['id']
    task_type = task['task_type']
    arguments = task['arguments']
    
    logger.info(f"Executing task {task_id} (type: {task_type})")
    
    try:
        # Get task type configuration
        task_config = config.get_task_type(task_type)
        script_path = task_config['script_path']
        python_env = task_config.get('python_env')
        command_type = task_config.get('command_type', 'python').lower()
        module_loads = task_config.get('module_loads', [])
        env_overrides = task_config.get('env', {})
        
        # Build command arguments
        cmd_args = [script_path]
        script_args_exclude = set(task_config.get('script_args_exclude', []))
        if command_type == 'bash':
            if isinstance(arguments, dict) and 'args' in arguments and isinstance(arguments['args'], list):
                cmd_args.extend([str(v) for v in arguments['args']])
            else:
                for key, value in arguments.items():
                    if key in script_args_exclude:
                        continue
                    cmd_args.append(f"--{key}")
                    cmd_args.append(str(value))
        else:
            for key, value in arguments.items():
                if key in script_args_exclude:
                    continue
                cmd_args.append(f"--{key}")
                cmd_args.append(str(value))

        # Build the actual command based on task type
        if command_type == 'bash':
            cmd = ['bash', script_path] + cmd_args[1:]
        elif command_type == 'torchrun':
            if not python_env:
                raise ValueError(f"Task type '{task_type}' missing python_env")
            nproc = task_config.get('nproc_per_node', 8)
            cmd = [python_env, '-m', 'torch.distributed.run', '--nproc_per_node', str(nproc)] + cmd_args
        else:
            if not python_env:
                raise ValueError(f"Task type '{task_type}' missing python_env")
            cmd = [python_env] + cmd_args
        
        # Create a shell script that loads modules and runs the task
        if module_loads:
            # Build shell script with module loads
            shell_script = "#!/bin/bash\n"
            shell_script += "set -e\n\n"  # Exit on error
            
            # Add module loads
            for module in module_loads:
                shell_script += f"module load {module}\n"
            
            shell_script += "\n"
            
            # Add env exports (if any)
            if env_overrides:
                for key, value in env_overrides.items():
                    shell_script += f"export {key}={shlex.quote(str(value))}\n"
                shell_script += "\n"
            # Add the actual command
            shell_script += f"{shlex.join(cmd)}\n"
            
            # Write to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
                temp_script = f.name
                f.write(shell_script)
            
            # Make executable
            os.chmod(temp_script, 0o755)
            
            logger.info(f"Running task with modules: {', '.join(module_loads)}")
            logger.info(f"Script: {temp_script}")
            
            # Execute the shell script
            result = subprocess.run(
                ['bash', temp_script],
                env={**os.environ, **{k: str(v) for k, v in env_overrides.items()}},
                capture_output=True,
                text=True
            )
            
            # Clean up temp script
            try:
                os.unlink(temp_script)
            except:
                pass
        else:
            # No modules to load, run directly
            logger.info(f"Running command: {shlex.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                env={**os.environ, **{k: str(v) for k, v in env_overrides.items()}},
                capture_output=True,
                text=True
            )
        
        # Log output
        if result.stdout:
            logger.info(f"Task {task_id} stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Task {task_id} stderr:\n{result.stderr}")
        
        if result.returncode != 0:
            logger.error(f"Task {task_id} failed with return code {result.returncode}")
            return False
        
        logger.info(f"Task {task_id} executed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error executing task {task_id}: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="CHPC Task Executor")
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--slurm-job-id', required=True, help='SLURM job ID')
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Setup logging
    log_dir = config.task_executor_settings.get('log_dir', './task_logs')
    logger = setup_logging(log_dir, args.slurm_job_id)
    
    logger.info(f"Task Executor started (SLURM Job ID: {args.slurm_job_id})")
    
    # Connect to database
    db = TaskDB(config.database_path)
    
    try:
        task_count = 0
        
        while True:
            # Get next idle task
            task = db.get_next_idle_task()
            
            if task is None:
                logger.info("No more idle tasks available. Exiting.")
                break
            
            task_id = task['id']
            completion_file = task['completion_file']
            
            # Check if task is already done (completion file exists)
            if check_completion_file(completion_file):
                logger.info(f"Task {task_id} already completed (completion file exists). Marking as done.")
                db.update_task_status(task_id, 'done')
                continue
            
            # Update task status to running
            status = f"running:{args.slurm_job_id}"
            db.update_task_status(task_id, status, args.slurm_job_id)
            logger.info(f"Task {task_id} status updated to {status}")
            
            # Execute the task
            success = execute_task(task, config, logger, args.slurm_job_id)
            
            # Check completion after execution
            if success and check_completion_file(completion_file):
                logger.info(f"Task {task_id} completed successfully")
                db.update_task_status(task_id, 'done')
                task_count += 1
            elif success:
                # Task script ran successfully but no completion file
                # This might be intentional or an error - log warning
                logger.warning(
                    f"Task {task_id} script completed but completion file not found: {completion_file}"
                )
                # Mark as done anyway since script succeeded
                db.update_task_status(task_id, 'done')
                task_count += 1
            else:
                # Task failed - reset to idle for retry
                logger.warning(f"Task {task_id} failed. Resetting to idle.")
                db.update_task_status(task_id, 'idle')
            
            # Small delay between tasks
            time.sleep(1)
        
        logger.info(f"Task Executor finished. Completed {task_count} tasks.")
    
    except Exception as e:
        logger.error(f"Fatal error in task executor: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        db.close()


if __name__ == '__main__':
    main()
